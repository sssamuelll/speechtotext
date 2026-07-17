# src/speechtotext/evaluation/runner.py
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from speechtotext.asr.base import VerifiedLocalAsrBackend
from speechtotext.asr.types import TranscriptionRequest
from speechtotext.audio.gate import QualityThresholds, evaluate_pre_inference
from speechtotext.audio.io import decode_audio
from speechtotext.audio.level import apply_fixed_gain
from speechtotext.audio.quality import compute_audio_quality
from speechtotext.audio.types import AudioClip, AudioView, AudioViews
from speechtotext.audio.fingerprint import PipelineProvenance, PipelineStep
from speechtotext.confidence.calibration import LogisticCalibrator
from speechtotext.confidence.features import extract_asr_features
from speechtotext.evaluation.environment import PACKAGES, process_memory_bytes
from speechtotext.evaluation.privacy import protected_ref
from speechtotext.evaluation.manifest import (
    CorpusEntry,
    CorpusManifest,
)
from speechtotext.evaluation.filesystem import (
    CorpusAssetLease,
    lease_corpus_asset,
)
from speechtotext.evaluation.retention import (
    DatasetSecurityEvidence,
)
from speechtotext.evaluation.training import (
    LabeledFeatureExample,
    LabeledFeaturePartition,
)
from speechtotext.evaluation.metrics import (
    brier_score,
    character_error,
    expected_calibration_error,
    cluster_bootstrap_error_upper,
    cluster_bootstrap_percentile_upper,
    one_sided_success_lower,
    percentile,
    risk_coverage_curve,
    word_error,
)
from speechtotext.evaluation.splits import DatasetSplit
from speechtotext.models.manifest import VerifiedModelArtifact

PartitionName = Literal["development", "calibration", "holdout"]
REQUIRED_ACCEPTANCE_CONDITIONS = frozenset({"clean", "noise", "silence"})


@dataclass(frozen=True)
class EvaluationConfig:
    as_of: date
    expected_language: str = "es"
    sample_rate: int = 16000
    usable_max_wer: float = 0.10
    confidence_bins: int = 10
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, date):
            raise TypeError("as_of debe ser date")
        if not self.expected_language.strip() or self.sample_rate <= 0:
            raise ValueError("lenguaje/sample_rate invalidos")
        if not 0.0 <= self.usable_max_wer <= 1.0 or self.confidence_bins <= 0:
            raise ValueError("config de evaluacion invalida")
        if not math.isfinite(self.gain_db) or self.gain_db > 18.0:
            raise ValueError("gain_db debe ser finito y <= 18 dB")


def preflight_evaluation(
    manifest: CorpusManifest,
    split: DatasetSplit,
    partition: PartitionName,
    config: EvaluationConfig,
    *,
    today: Callable[[], date] = date.today,
) -> tuple[tuple[CorpusEntry, ...], date, date]:
    checked_on = today()
    if not isinstance(checked_on, date):
        raise TypeError("today debe devolver date")
    entries = split.partition(partition, manifest)
    if not entries:
        raise ValueError(f"particion vacia: {partition}")
    effective_as_of = max(config.as_of, checked_on)
    if any(entry.retention_until < effective_as_of for entry in entries):
        raise ValueError("corpus_retention_expired")
    return entries, checked_on, effective_as_of


def load_entry_clip(
    entry: CorpusEntry,
    lease: CorpusAssetLease,
    expected_pipeline: PipelineProvenance,
    sample_rate: int,
    gain_db: float,
) -> AudioClip:
    capture = decode_audio(
        lease.stream,
        sample_rate=sample_rate,
    )
    gain = apply_fixed_gain(capture.samples, gain_db)
    analysis = AudioView.derive(
        capture,
        gain.samples,
        steps=(PipelineStep(
            "fixed-gain",
            "1",
            {"gain_db": gain_db, "max_gain_db": 18.0, "peak_limit_dbfs": -1.0},
        ),),
    )
    if analysis.provenance != expected_pipeline:
        raise ValueError("pipeline construido no coincide con el artifact verificado")
    decoded_duration_ms = round(capture.duration_s * 1000)
    tolerance_ms = max(50, round(decoded_duration_ms * 0.01))
    if abs(entry.duration_ms - decoded_duration_ms) > tolerance_ms:
        raise ValueError("duration_ms del manifest fuera de tolerancia")
    quality = compute_audio_quality(
        capture.samples,
        analysis.samples,
        capture.sample_rate,
        entry.speech_regions,
        requested_gain_db=gain.requested_gain_db,
        applied_gain_db=gain.applied_gain_db,
    )
    return AudioClip(
        started_at=0.0,
        ended_at=capture.duration_s,
        source_id=entry.source_id,
        speech_regions=entry.speech_regions,
        quality=quality,
        views=AudioViews(capture=capture, analysis=analysis, asr=analysis),
    )


def _corpus_evidence(
    entries: Sequence[CorpusEntry],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    duration_ms = sum(int(row["decoded_duration_ms"]) for row in rows)
    declared_duration_ms = sum(entry.duration_ms for entry in entries)
    days = {entry.recorded_on for entry in entries}
    sessions = {entry.session_id for entry in entries}
    observed_conditions = {entry.condition for entry in entries}
    missing_conditions = sorted(REQUIRED_ACCEPTANCE_CONDITIONS - observed_conditions)
    reasons: list[str] = []
    if duration_ms < 1_800_000:
        reasons.append("duration_below_30m")
    if len(days) < 3:
        reasons.append("recording_days_below_3")
    if len(sessions) < 3:
        reasons.append("recording_sessions_below_3")
    if missing_conditions:
        reasons.append("required_conditions_missing")
    return {
        "evaluated_clips": len(entries),
        "duration_source": "decoded_leased_asset",
        "decoded_duration_ms": duration_ms,
        "declared_duration_ms_diagnostic": declared_duration_ms,
        "minimum_duration_passed": duration_ms >= 1_800_000,
        "target_duration_range_passed": 1_800_000 <= duration_ms <= 2_700_000,
        "recording_days": len(days),
        "recording_sessions": len(sessions),
        "observed_conditions": sorted(observed_conditions),
        "required_conditions": sorted(REQUIRED_ACCEPTANCE_CONDITIONS),
        "missing_conditions": missing_conditions,
        "sufficient_evidence": not reasons,
        "insufficient_reason_codes": reasons,
    }


def _error_report(
    rows: list[dict[str, object]],
    *,
    error_field: str,
    units_field: str,
    block_field: str,
    corpus_sufficient: bool,
) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        errors = row[error_field]
        units = row[units_field]
        if errors is not None and units:
            counts = grouped[str(row[block_field])]
            counts[0] += int(errors)
            counts[1] += int(units)
    blocks = [tuple(values) for _, values in sorted(grouped.items())]
    errors = sum(pair[0] for pair in blocks)
    units = sum(pair[1] for pair in blocks)
    macro_rate = (
        sum(errors_i / units_i for errors_i, units_i in blocks) / len(blocks)
        if blocks
        else None
    )
    sufficient = corpus_sufficient and len(blocks) >= 3
    return {
        "errors": errors,
        "reference_units": units,
        "rate": macro_rate,
        "rate_aggregation": "macro_by_independent_block",
        "micro_rate_diagnostic": errors / units if units else None,
        "upper_95": (
            cluster_bootstrap_error_upper(blocks) if sufficient else None
        ),
        "blocks": len(blocks),
        "block_field": block_field,
        "sufficient_evidence": sufficient,
    }


def _error_reports_by_group(
    rows,
    group_field,
    *,
    error_field,
    units_field,
    block_field,
    corpus_sufficient,
):
    groups = sorted({str(row[group_field]) for row in rows})
    return {
        group: _error_report(
            [row for row in rows if str(row[group_field]) == group],
            error_field=error_field,
            units_field=units_field,
            block_field=block_field,
            corpus_sufficient=corpus_sufficient,
        )
        for group in groups
    }


def _engine_latency_report(
    rows: list[dict[str, object]],
    *,
    corpus_evidence: Mapping[str, object],
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        latency = row["latency_ms"]
        if latency is not None:
            grouped[str(row["day_ref"])].append(float(latency))
    blocks = [values for _, values in sorted(grouped.items())]
    values = [value for block in blocks for value in block]
    sufficient = bool(corpus_evidence["sufficient_evidence"]) and len(blocks) >= 3
    return {
        "scope": "backend_engine_only",
        "provisional": True,
        "acceptance_gate": False,
        "count": len(values),
        "p50": percentile(values, 50.0) if values else None,
        "p95": percentile(values, 95.0) if values else None,
        "p95_upper_95": (
            cluster_bootstrap_percentile_upper(blocks) if sufficient else None
        ),
        "blocks": len(blocks),
        "block_field": "day_ref",
        "sufficient_evidence": sufficient,
    }


def _acceptance_gate(
    *,
    corpus: Mapping[str, object],
    wer_by_condition: Mapping[str, Mapping[str, object]],
    silence: Mapping[str, object],
) -> dict[str, object]:
    insufficient = list(corpus["insufficient_reason_codes"])
    thresholds = {"wer_clean_upper_95": 0.05, "wer_noise_upper_95": 0.10}
    for condition in ("clean", "noise"):
        metric = wer_by_condition.get(condition)
        if metric is None or not metric["sufficient_evidence"]:
            insufficient.append(f"wer_{condition}_insufficient_evidence")
    if int(silence["trials"]) == 0:
        insufficient.append("silence_insufficient_evidence")

    failures: list[str] = []
    if not insufficient:
        if float(wer_by_condition["clean"]["upper_95"]) > 0.05:
            failures.append("wer_clean_above_limit")
        if float(wer_by_condition["noise"]["upper_95"]) > 0.10:
            failures.append("wer_noise_above_limit")
        if int(silence["errors"]) != 0:
            failures.append("silence_transcript_detected")
    status = (
        "insufficient_evidence"
        if insufficient
        else "failed"
        if failures
        else "passed"
    )
    return {
        "status": status,
        "passed": status == "passed",
        "sufficient_evidence": not insufficient,
        "insufficient_reason_codes": sorted(set(insufficient)),
        "failure_reason_codes": failures,
        "thresholds": thresholds,
    }


_ENVIRONMENT_FIELDS = frozenset({
    "schema_version", "git_ref", "python", "implementation", "platform",
    "machine", "processor", "executable_name", "memory", "packages",
})
_RAW_DIGEST = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def _safe_environment_text(name: str, value: object, *, allow_empty=False) -> str:
    if not isinstance(value, str) or len(value) > 256:
        raise ValueError(f"environment.{name} invalido")
    if not allow_empty and not value:
        raise ValueError(f"environment.{name} vacio")
    if any(char in value for char in ("/", "\\", ":", "\n", "\r")):
        raise ValueError(f"environment.{name} contiene ruta")
    if _RAW_DIGEST.search(value):
        raise ValueError(f"environment.{name} contiene digest crudo")
    return value


def _validated_environment(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment schema invalido")
    if value["schema_version"] != "speechtotext.environment/v1":
        raise ValueError("environment version invalida")
    git_ref = value["git_ref"]
    if not isinstance(git_ref, str) or not re.fullmatch(
        r"git-revision:[0-9a-f]{32}", git_ref
    ):
        raise ValueError("environment git_ref debe estar protegido")
    memory = value["memory"]
    packages = value["packages"]
    if not isinstance(memory, Mapping) or set(memory) != {"rss", "peak_rss"}:
        raise ValueError("environment memory invalido")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in memory.values()
    ):
        raise ValueError("environment memory exige enteros no negativos")
    if not isinstance(packages, Mapping) or set(packages) != set(PACKAGES):
        raise ValueError("environment packages invalido")
    clean = {
        "schema_version": value["schema_version"],
        "git_ref": git_ref,
        "python": _safe_environment_text("python", value["python"]),
        "implementation": _safe_environment_text(
            "implementation", value["implementation"]
        ),
        "platform": _safe_environment_text("platform", value["platform"]),
        "machine": _safe_environment_text("machine", value["machine"]),
        "processor": _safe_environment_text(
            "processor", value["processor"], allow_empty=True
        ),
        "executable_name": _safe_environment_text(
            "executable_name", value["executable_name"]
        ),
        "memory": {key: int(memory[key]) for key in ("rss", "peak_rss")},
        "packages": {
            name: _safe_environment_text(f"packages.{name}", packages[name])
            for name in PACKAGES
        },
    }
    return clean


def _validate_result_identity(result, backend: VerifiedLocalAsrBackend) -> None:
    if (result.backend, result.model, result.model_version) != (
        backend.backend_id,
        backend.model_id,
        backend.model_version,
    ):
        raise ValueError("resultado ASR incompatible con backend verificado")


def collect_labeled_feature_partition(
    *,
    manifest: CorpusManifest,
    split: DatasetSplit,
    partition: Literal["development", "calibration"],
    dataset_root: Path,
    backend: VerifiedLocalAsrBackend,
    request: TranscriptionRequest,
    config: EvaluationConfig,
    thresholds: QualityThresholds,
    pipeline: PipelineProvenance,
    security_evidence: DatasetSecurityEvidence,
    clip_loader: Callable = load_entry_clip,
    asset_lease_factory: Callable = lease_corpus_asset,
    today: Callable[[], date] = date.today,
) -> LabeledFeaturePartition:
    if partition not in {"development", "calibration"}:
        raise ValueError("collector nunca acepta holdout")
    entries, _, _ = preflight_evaluation(
        manifest, split, partition, config, today=today,
    )
    if request.language != config.expected_language:
        raise ValueError("request/lenguaje esperado incompatibles")
    if not isinstance(backend, VerifiedLocalAsrBackend) or not isinstance(
        backend.model_artifact, VerifiedModelArtifact
    ):
        raise TypeError("collector local exige VerifiedLocalAsrBackend")
    backend.model_artifact.require_active()
    security_evidence.require_for(dataset_root, manifest)
    backend.warm()
    examples: list[LabeledFeatureExample] = []
    for entry in entries:
        if entry.kind != "speech":
            continue
        with asset_lease_factory(entry.primary_audio, dataset_root) as lease:
            security_evidence.require_asset(entry.primary_audio, lease)
            clip = clip_loader(
                entry,
                lease,
                pipeline,
                config.sample_rate,
                config.gain_db,
            )
        gate = evaluate_pre_inference(clip.quality, thresholds)
        if not gate.eligible:
            raise ValueError("calibration_clip_ineligible")
        result = backend.transcribe(clip, request)
        _validate_result_identity(result, backend)
        error = word_error(entry.transcript, result.text)
        examples.append(
            LabeledFeatureExample(
                clip_id=entry.clip_id,
                session_id=entry.session_id,
                recorded_on=entry.recorded_on,
                features=extract_asr_features(
                    result,
                    clip.quality,
                    expected_language=config.expected_language,
                ),
                label=int(error.rate <= config.usable_max_wer),
            )
        )
    return LabeledFeaturePartition.from_split(
        partition, manifest, split, tuple(examples)
    )


def _evaluation_job_ref(
    *,
    ref_key: bytes,
    manifest: CorpusManifest,
    split: DatasetSplit,
    partition: PartitionName,
    backend: VerifiedLocalAsrBackend,
    request: TranscriptionRequest,
    config: EvaluationConfig,
    thresholds: QualityThresholds,
    pipeline: PipelineProvenance,
    calibrator: LogisticCalibrator | None,
    environment: Mapping[str, object],
) -> str:
    git_ref = environment.get("git_ref")
    if not isinstance(git_ref, str) or not git_ref:
        raise ValueError("environment exige git_ref protegido")
    config_payload = {**asdict(config), "as_of": config.as_of.isoformat()}
    payload = {
        "schema_version": "speechtotext.evaluation-job/v1",
        "git_ref": git_ref,
        "manifest": manifest.version,
        "split": split.fingerprint,
        "partition": partition,
        "backend_artifact_kind": backend.backend_artifact_kind,
        "model": backend.backend_artifact_fingerprint,
        "backend_config": backend.config_fingerprint,
        "pipeline": pipeline.fingerprint,
        "request": request.fingerprint,
        "calibrator": None if calibrator is None else calibrator.artifact.version,
        "config": config_payload,
        "quality_thresholds": asdict(thresholds),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return protected_ref(ref_key, "evaluation-job", canonical)


def run_evaluation(
    *,
    manifest: CorpusManifest,
    split: DatasetSplit,
    partition: PartitionName,
    dataset_root: Path,
    backend: VerifiedLocalAsrBackend,
    request: TranscriptionRequest,
    config: EvaluationConfig,
    thresholds: QualityThresholds,
    pipeline: PipelineProvenance,
    output_path: Path,
    calibrator: LogisticCalibrator | None,
    clip_loader: Callable[
        [CorpusEntry, CorpusAssetLease, PipelineProvenance, int, float], AudioClip
    ] = load_entry_clip,
    asset_lease_factory: Callable = lease_corpus_asset,
    security_evidence: DatasetSecurityEvidence,
    environment: Mapping[str, object],
    report_ref_key: bytes,
    memory_probe: Callable[[], dict[str, int]] = process_memory_bytes,
    today: Callable[[], date] = date.today,
) -> dict[str, object]:
    entries, retention_checked_on, effective_as_of = preflight_evaluation(
        manifest, split, partition, config, today=today,
    )
    # Valida la clave antes de cualquier side effect o acceso al corpus.
    protected_ref(report_ref_key, "manifest", manifest.version)
    if not isinstance(backend, VerifiedLocalAsrBackend) or not isinstance(
        backend.model_artifact, VerifiedModelArtifact
    ):
        raise TypeError("runner local exige VerifiedLocalAsrBackend")
    backend.model_artifact.require_active()
    security_evidence.require_for(dataset_root, manifest)
    if calibrator is not None:
        calibrator.validate_for(
            backend=backend,
            pipeline=pipeline,
            request=request,
            expected_language=config.expected_language,
            usable_max_wer=config.usable_max_wer,
        )
    environment = _validated_environment(environment)
    job_ref = _evaluation_job_ref(
        ref_key=report_ref_key,
        manifest=manifest,
        split=split,
        partition=partition,
        backend=backend,
        request=request,
        config=config,
        thresholds=thresholds,
        pipeline=pipeline,
        calibrator=calibrator,
        environment=environment,
    )
    backend.warm()
    memory_after_warm = memory_probe()
    memory_samples = [memory_after_warm]
    rows: list[dict[str, object]] = []
    total_word_errors = total_reference_words = 0
    total_char_errors = total_reference_chars = 0
    eligible = transcribed = 0
    probabilities: list[float] = []
    labels: list[int] = []
    for entry in entries:
        with asset_lease_factory(entry.primary_audio, dataset_root) as lease:
            security_evidence.require_asset(entry.primary_audio, lease)
            clip = clip_loader(
                entry,
                lease,
                pipeline,
                config.sample_rate,
                config.gain_db,
            )
        gate = evaluate_pre_inference(clip.quality, thresholds)
        text = ""
        latency_ms: int | None = None
        confidence: float | None = None
        result = None
        if gate.eligible:
            eligible += 1
            result = backend.transcribe(
                clip,
                request,
            )
            _validate_result_identity(result, backend)
            transcribed += 1
            text = result.text
            latency_ms = result.latency_ms
        memory_samples.append(memory_probe())
        has_reference_speech = entry.kind in {"speech", "other_voice", "replay", "tts"}
        wer = word_error(entry.transcript, text) if has_reference_speech else None
        cer = character_error(entry.transcript, text) if has_reference_speech else None
        usable: bool | None = None
        if wer is not None:
            total_word_errors += wer.errors
            total_reference_words += wer.reference_units
            total_char_errors += cer.errors
            total_reference_chars += cer.reference_units
            usable = wer.rate <= config.usable_max_wer
        if result is not None and calibrator is not None:
            features = extract_asr_features(
                result,
                clip.quality,
                expected_language=config.expected_language,
            )
            calibrated = calibrator.apply(
                result,
                features,
                backend=backend,
                view=clip.view("asr"),
                request=request,
                expected_language=config.expected_language,
                usable_max_wer=config.usable_max_wer,
            )
            confidence = calibrated.calibrated_confidence
            if usable is not None:
                probabilities.append(float(confidence))
                labels.append(int(usable))
        silence_trial = int(entry.kind in {"silence", "noise"})
        silence_error = int(silence_trial and bool(text.strip()))
        rows.append(
            {
                "clip_ref": protected_ref(report_ref_key, "clip", entry.clip_id),
                "session_ref": protected_ref(
                    report_ref_key, "session", entry.session_id
                ),
                "day_ref": protected_ref(
                    report_ref_key, "day", entry.recorded_on.isoformat()
                ),
                "condition": entry.condition,
                "kind": entry.kind,
                "decoded_duration_ms": clip.quality.duration_ms,
                "gate_eligible": gate.eligible,
                "gate_reasons": list(gate.reason_codes),
                "wer": None if wer is None else wer.rate,
                "word_errors": None if wer is None else wer.errors,
                "reference_words": None if wer is None else wer.reference_units,
                "cer": None if cer is None else cer.rate,
                "character_errors": None if cer is None else cer.errors,
                "reference_characters": None if cer is None else cer.reference_units,
                "segment_usable": usable,
                "calibrated_confidence": confidence,
                "latency_ms": latency_ms,
                "silence_trial": silence_trial,
                "silence_error": silence_error,
            }
        )
    word_rate = (
        total_word_errors / total_reference_words
        if total_reference_words
        else 0.0
    )
    char_rate = (
        total_char_errors / total_reference_chars
        if total_reference_chars
        else 0.0
    )
    calibration: dict[str, object] | None = None
    if probabilities:
        curve = risk_coverage_curve(probabilities, labels)
        calibration = {
            "brier": brier_score(probabilities, labels),
            "ece": expected_calibration_error(
                probabilities,
                labels,
                bins=config.confidence_bins,
            ),
            "risk_coverage": [asdict(point) for point in curve],
            "examples": len(probabilities),
        }
    corpus_evidence = _corpus_evidence(entries, rows)
    corpus_sufficient = bool(corpus_evidence["sufficient_evidence"])
    wer_overall = _error_report(
        rows,
        error_field="word_errors",
        units_field="reference_words",
        block_field="session_ref",
        corpus_sufficient=corpus_sufficient,
    )
    cer_overall = _error_report(
        rows,
        error_field="character_errors",
        units_field="reference_characters",
        block_field="session_ref",
        corpus_sufficient=corpus_sufficient,
    )
    asr_report = {
        "word_errors": total_word_errors,
        "reference_words": total_reference_words,
        "wer": wer_overall["rate"],
        "wer_micro_diagnostic": word_rate,
        "character_errors": total_char_errors,
        "reference_characters": total_reference_chars,
        "cer": cer_overall["rate"],
        "cer_micro_diagnostic": char_rate,
        "wer_overall": wer_overall,
        "cer_overall": cer_overall,
        "wer_by_condition": _error_reports_by_group(
            rows,
            "condition",
            error_field="word_errors",
            units_field="reference_words",
            block_field="session_ref",
            corpus_sufficient=corpus_sufficient,
        ),
        "wer_by_session_diagnostic": _error_reports_by_group(
            rows,
            "session_ref",
            error_field="word_errors",
            units_field="reference_words",
            block_field="clip_ref",
            corpus_sufficient=corpus_sufficient,
        ),
    }
    silence_rate = _error_report(
        rows,
        error_field="silence_error",
        units_field="silence_trial",
        block_field="session_ref",
        corpus_sufficient=corpus_sufficient,
    )
    silence_report = dict(silence_rate)
    silence_report["trials"] = silence_report.pop("reference_units")
    engine_latency_report = _engine_latency_report(
        rows, corpus_evidence=corpus_evidence
    )
    report: dict[str, object] = {
        "schema_version": "speechtotext.evaluation/v1",
        "job_ref": job_ref,
        "dataset_ref": protected_ref(
            report_ref_key, "dataset", manifest.dataset_id
        ),
        "corpus": {
            "manifest_clips": len(manifest.entries),
            "manifest_ref": protected_ref(
                report_ref_key, "manifest", manifest.version
            ),
            **corpus_evidence,
        },
        "partition": partition,
        "split_ref": protected_ref(report_ref_key, "split", split.fingerprint),
        "pipeline_ref": protected_ref(
            report_ref_key, "pipeline", pipeline.fingerprint
        ),
        "request_ref": protected_ref(
            report_ref_key, "request", request.fingerprint
        ),
        "backend": {
            "backend_ref": protected_ref(
                report_ref_key, "backend", backend.backend_id
            ),
            "model_ref": protected_ref(
                report_ref_key,
                "model",
                backend.backend_artifact_fingerprint,
            ),
            "config_ref": protected_ref(
                report_ref_key, "backend-config", backend.config_fingerprint
            ),
        },
        "calibrator_ref": (
            protected_ref(
                report_ref_key,
                "calibrator",
                calibrator.artifact.version,
            )
            if calibrator is not None
            else None
        ),
        "config": {
            **asdict(config),
            "as_of": config.as_of.isoformat(),
            "retention_checked_on": retention_checked_on.isoformat(),
            "effective_as_of": effective_as_of.isoformat(),
        },
        "quality_thresholds": asdict(thresholds),
        "dataset_security": security_evidence.to_dict(),
        "environment": dict(environment),
        "counts": {
            "clips": len(entries),
            "eligible": eligible,
            "blocked_before_asr": len(entries) - eligible,
            "transcribed": transcribed,
        },
        "quality_gate": {
            "eligible": eligible,
            "trials": len(entries),
            "lower_95": one_sided_success_lower(eligible, len(entries)),
        },
        "asr": asr_report,
        "safety": {
            "silence_transcripts": silence_report,
        },
        "engine_latency_ms": engine_latency_report,
        "memory_bytes": {
            "rss_after_warm": memory_after_warm["rss"],
            "peak_after_warm": memory_after_warm["peak_rss"],
            "max_rss_observed": max(sample["rss"] for sample in memory_samples),
            "max_peak_rss_observed": max(
                sample["peak_rss"] for sample in memory_samples
            ),
        },
        "calibration": calibration,
        "clips": rows,
    }
    report["acceptance_gate"] = _acceptance_gate(
        corpus=corpus_evidence,
        wer_by_condition=asr_report["wer_by_condition"],
        silence=silence_report,
    )
    security_evidence.write_json_report(output_path, report)
    return report
