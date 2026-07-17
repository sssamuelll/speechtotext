# src/speechtotext/evaluation/__main__.py
from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Sequence

from speechtotext.asr.faster_whisper import (
    FasterWhisperBackend,
    FasterWhisperConfig,
)
from speechtotext.asr.types import TranscriptionRequest
from speechtotext.audio.fingerprint import PipelineProvenance, PipelineStep
from speechtotext.audio.gate import QualityThresholds
from speechtotext.confidence.calibration import (
    LogisticCalibrator,
    parse_calibrator_artifact_bytes,
)
from speechtotext.evaluation.environment import collect_environment
from speechtotext.evaluation.filesystem import (
    default_corpus_filesystem,
    lease_corpus_asset,
)
from speechtotext.evaluation.privacy import protected_ref
from speechtotext.evaluation.manifest import load_corpus_manifest
from speechtotext.evaluation.retention import (
    audit_dataset_security,
    initialize_report_ref_key,
    list_retention,
    purge_expired,
    renew_retention,
)
from speechtotext.evaluation.runner import (
    collect_labeled_feature_partition,
    EvaluationConfig,
    preflight_evaluation,
    run_evaluation,
)
from speechtotext.evaluation.splits import split_by_recording_day
from speechtotext.evaluation.training import fit_segment_usable_calibrator
from speechtotext.models.filesystem import default_model_filesystem
from speechtotext.models.manifest import load_model_manifest, verify_model_files

GATE_STATUSES = frozenset({"passed", "failed", "insufficient_evidence"})
CORPUS_SCHEMA_STATUSES = frozenset({
    ("speechtotext.retention-list/v1", "listed"),
    ("speechtotext.retention-renew/v1", "complete"),
    ("speechtotext.retention-receipt/v1", "planned"),
    ("speechtotext.retention-receipt/v1", "complete"),
    ("speechtotext.retention-receipt/v1", "partial"),
    ("speechtotext.retention-receipt/v1", "failed"),
    ("speechtotext.report-key-init/v1", "complete"),
})
_GATE_FIELDS = frozenset({
    "status", "passed", "sufficient_evidence", "insufficient_reason_codes",
    "failure_reason_codes", "thresholds",
})
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _validated_clip_count(payload: Mapping[str, object]) -> int:
    counts = payload.get("counts")
    clips = counts.get("clips") if isinstance(counts, Mapping) else None
    if isinstance(clips, bool) or not isinstance(clips, int) or clips < 0:
        raise ValueError("counts.clips invalido")
    return clips


def _validated_corpus_payload(payload: object, command: str) -> tuple[str, int]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload corpus invalido")
    schema = payload.get("schema_version")
    status = payload.get("status")
    if not isinstance(schema, str) or not isinstance(status, str):
        raise ValueError("schema/status corpus invalidos")
    if (schema, status) not in CORPUS_SCHEMA_STATUSES:
        raise ValueError("schema/status corpus incompatibles")
    expected_schema = {
        "init-report-key": "speechtotext.report-key-init/v1",
        "list": "speechtotext.retention-list/v1",
        "renew": "speechtotext.retention-renew/v1",
        "purge-expired": "speechtotext.retention-receipt/v1",
    }.get(command)
    if schema != expected_schema:
        raise ValueError("schema no corresponde al comando corpus")
    return status, _validated_clip_count(payload)


def _validated_training_payload(payload: object) -> tuple[int, int, str]:
    if not isinstance(payload, Mapping) or payload.get(
        "schema_version"
    ) != "speechtotext.calibrator-training/v1" or payload.get(
        "status"
    ) != "complete":
        raise ValueError("payload de training invalido")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "development", "calibration"
    }:
        raise ValueError("counts de training invalidos")
    values = (counts["development"], counts["calibration"])
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("training exige ejemplos positivos")
    fingerprint = payload.get("artifact_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ValueError("artifact_fingerprint de training invalido")
    return int(values[0]), int(values[1]), fingerprint


def _validated_gate(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _GATE_FIELDS:
        raise ValueError("acceptance_gate schema invalido")
    status = value["status"]
    passed = value["passed"]
    sufficient = value["sufficient_evidence"]
    insufficient = value["insufficient_reason_codes"]
    failures = value["failure_reason_codes"]
    thresholds = value["thresholds"]
    if status not in GATE_STATUSES or not isinstance(passed, bool) or not isinstance(
        sufficient, bool
    ):
        raise ValueError("acceptance_gate tipos invalidos")
    if not all(
        isinstance(items, list)
        and all(isinstance(item, str) and _REASON_CODE.fullmatch(item) for item in items)
        for items in (insufficient, failures)
    ):
        raise ValueError("acceptance_gate reason codes invalidos")
    if not isinstance(thresholds, Mapping) or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in thresholds.values()
    ):
        raise ValueError("acceptance_gate thresholds invalidos")
    if passed is not (status == "passed") or sufficient is not (
        status != "insufficient_evidence"
    ):
        raise ValueError("acceptance_gate estado inconsistente")
    if passed and (insufficient or failures):
        raise ValueError("acceptance_gate passed con razones")
    if status == "failed" and not failures:
        raise ValueError("acceptance_gate failed sin razones")
    if status == "insufficient_evidence" and not insufficient:
        raise ValueError("acceptance_gate insuficiente sin razones")
    return value


def _validated_evaluation_report(
    report: object,
) -> tuple[int, Mapping[str, object]]:
    if not isinstance(report, Mapping) or report.get(
        "schema_version"
    ) != "speechtotext.evaluation/v1":
        raise ValueError("reporte de evaluacion invalido")
    return _validated_clip_count(report), _validated_gate(report.get("acceptance_gate"))


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "ERROR code=invalid_arguments\n")


def _validate_calibrator_options(args: argparse.Namespace) -> None:
    path = getattr(args, "calibrator", None)
    fingerprint = getattr(args, "calibrator_fingerprint", None)
    if (path is None) != (fingerprint is None):
        raise ValueError("calibrator_path_fingerprint_pair_required")
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise ValueError("calibrator_fingerprint_invalid")


def _validate_planned_model_identity(args: argparse.Namespace) -> None:
    if (
        not isinstance(args.model_id, str)
        or not args.model_id.strip()
        or len(args.model_id) > 256
    ):
        raise ValueError("planned_model_id_invalid")
    if not isinstance(args.model_revision, str) or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})",
        args.model_revision,
    ):
        raise ValueError("planned_model_revision_invalid")
    if not isinstance(args.model_manifest_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", args.model_manifest_fingerprint
    ):
        raise ValueError("planned_model_fingerprint_invalid")


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="python -m speechtotext.evaluation",
        description="Replay local y reproducible del corpus privado de audio.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("development", "calibration", "holdout"),
        required=True,
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest-fingerprint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-ref-key-file", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--calibrator-fingerprint")
    parser.add_argument("--language", default="es")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--usable-max-wer", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--min-effective-voice-ms", type=int, required=True)
    parser.add_argument("--min-rms-dbfs", type=float, required=True)
    parser.add_argument("--min-snr-db", type=float, required=True)
    parser.add_argument("--max-clipping-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--require-acceptance",
        action="store_true",
        help="sale 2 salvo que los limites upper-95 y la evidencia aprueben",
    )
    return parser


def build_training_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="python -m speechtotext.evaluation train-calibrator"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest-fingerprint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--min-precision-lower-95", type=float, default=0.99)
    parser.add_argument("--language", default="es")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--usable-max-wer", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--min-effective-voice-ms", type=int, required=True)
    parser.add_argument("--min-rms-dbfs", type=float, required=True)
    parser.add_argument("--min-snr-db", type=float, required=True)
    parser.add_argument("--max-clipping-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser


def _add_corpus_location_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--report-ref-key-file", type=Path, required=True)


def build_corpus_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(prog="python -m speechtotext.evaluation corpus")
    commands = parser.add_subparsers(
        dest="corpus_command",
        required=True,
        parser_class=SafeArgumentParser,
    )
    init_key = commands.add_parser("init-report-key")
    init_key.add_argument("--dataset-root", type=Path, required=True)
    init_key.add_argument("--repo-root", type=Path, required=True)
    init_key.add_argument("--output", type=Path, required=True)
    listing = commands.add_parser("list")
    _add_corpus_location_args(listing)
    renew = commands.add_parser("renew")
    _add_corpus_location_args(renew)
    renew.add_argument("--clip-id", action="append", required=True)
    renew.add_argument("--until", type=date.fromisoformat, required=True)
    renew.add_argument("--confirm", action="store_true", required=True)
    purge = commands.add_parser("purge-expired")
    _add_corpus_location_args(purge)
    purge.add_argument("--receipt", type=Path, required=True)
    purge.add_argument("--confirm", action="store_true")
    return parser


def execute_corpus_job(args: argparse.Namespace) -> dict[str, object]:
    filesystem = default_corpus_filesystem()
    if args.corpus_command == "init-report-key":
        initialize_report_ref_key(
            args.dataset_root,
            args.repo_root,
            args.output,
            filesystem=filesystem,
        )
        return {
            "schema_version": "speechtotext.report-key-init/v1",
            "status": "complete",
            "counts": {"clips": 0},
        }
    manifest = load_corpus_manifest(
        args.manifest,
        dataset_root=args.dataset_root,
        repo_root=args.repo_root,
        filesystem=filesystem,
    )
    with audit_dataset_security(
        args.dataset_root,
        args.repo_root,
        args.manifest,
        manifest,
        filesystem=filesystem,
    ) as security:
        security.require_for(args.dataset_root, manifest)
        report_ref_key = security.read_report_ref_key(args.report_ref_key_file)
        if args.corpus_command == "list":
            items = list_retention(
                manifest,
                security_evidence=security,
                ref_key=report_ref_key,
            )
            return {
                "schema_version": "speechtotext.retention-list/v1",
                "status": "listed",
                "counts": {"clips": len(items)},
                "items": [item.to_dict() for item in items],
            }
        if args.corpus_command == "renew":
            renewed = renew_retention(
                args.manifest,
                args.dataset_root,
                clip_ids=tuple(args.clip_id),
                until=args.until,
                confirm=args.confirm,
                filesystem=filesystem,
                security_evidence=security,
            )
            return {
                "schema_version": "speechtotext.retention-renew/v1",
                "status": "complete",
                "counts": {"clips": len(args.clip_id)},
                "manifest_ref": protected_ref(
                    report_ref_key, "manifest", renewed.version
                ),
            }
        receipt = purge_expired(
            manifest,
            args.dataset_root,
            confirm=args.confirm,
            manifest_path=args.manifest,
            receipt_path=args.receipt,
            filesystem=filesystem,
            security_evidence=security,
            ref_key=report_ref_key,
        )
        return receipt.to_dict()


def execute_training_job(args: argparse.Namespace) -> dict[str, object]:
    if (
        not math.isfinite(args.min_precision_lower_95)
        or not 0.0 < args.min_precision_lower_95 <= 1.0
        or not args.artifact_version.strip()
    ):
        raise ValueError("politica/version de calibrador invalida")
    filesystem = default_corpus_filesystem()
    manifest = load_corpus_manifest(
        args.manifest,
        dataset_root=args.dataset_root,
        repo_root=args.repo_root,
        filesystem=filesystem,
    )
    split = split_by_recording_day(manifest, seed=args.seed)
    config = EvaluationConfig(
        as_of=args.as_of,
        expected_language=args.language,
        sample_rate=args.sample_rate,
        usable_max_wer=args.usable_max_wer,
        gain_db=args.gain_db,
    )
    trusted_today = date.today()
    development_entries, _, _ = preflight_evaluation(
        manifest,
        split,
        "development",
        config,
        today=lambda: trusted_today,
    )
    calibration_entries, _, _ = preflight_evaluation(
        manifest,
        split,
        "calibration",
        config,
        today=lambda: trusted_today,
    )
    selected = development_entries + calibration_entries
    model_filesystem = default_model_filesystem()
    model_manifest = load_model_manifest(
        args.model_manifest,
        model_root=args.model_root,
        expected_fingerprint=args.model_manifest_fingerprint,
        filesystem=model_filesystem,
    )
    with (
        audit_dataset_security(
            args.dataset_root,
            args.repo_root,
            args.manifest,
            manifest,
            assets=tuple(asset for entry in selected for asset in entry.assets),
            filesystem=filesystem,
        ) as security,
        verify_model_files(
            model_manifest,
            args.model_root,
            filesystem=model_filesystem,
        ) as model_artifact,
    ):
        security.require_for(args.dataset_root, manifest)
        if model_manifest.sample_rate != args.sample_rate:
            raise ValueError("sample rate de modelo incompatible")
        thresholds = QualityThresholds(
            args.min_effective_voice_ms,
            args.min_rms_dbfs,
            args.min_snr_db,
            args.max_clipping_ratio,
        )
        capture_pipeline = PipelineProvenance.capture(
            sample_rate=args.sample_rate,
            step=PipelineStep(
                "pyav-decode",
                "1",
                {"dtype": "float32", "layout": "mono",
                 "sample_rate": args.sample_rate},
            ),
        )
        pipeline = PipelineProvenance.derive(
            capture_pipeline,
            sample_rate=args.sample_rate,
            steps=(PipelineStep(
                "fixed-gain",
                "1",
                {"gain_db": args.gain_db, "max_gain_db": 18.0,
                 "peak_limit_dbfs": -1.0},
            ),),
        )
        backend = FasterWhisperBackend(
            FasterWhisperConfig(
                device=args.device,
                compute_type=args.compute_type,
            ),
            model_artifact,
        )
        request = TranscriptionRequest(language=args.language)

        def collect(partition):
            return collect_labeled_feature_partition(
                manifest=manifest,
                split=split,
                partition=partition,
                dataset_root=args.dataset_root,
                backend=backend,
                request=request,
                config=config,
                thresholds=thresholds,
                pipeline=pipeline,
                security_evidence=security,
                asset_lease_factory=lambda asset, root: lease_corpus_asset(
                    asset, root, filesystem=filesystem
                ),
                today=lambda: trusted_today,
            )

        development = collect("development")
        calibration = collect("calibration")
        artifact = fit_segment_usable_calibrator(
            development,
            calibration,
            backend=backend,
            pipeline=pipeline,
            request=request,
            usable_max_wer=config.usable_max_wer,
            min_precision_lower_95=args.min_precision_lower_95,
            artifact_version=args.artifact_version,
        )
        security.write_json_report(args.output, artifact.to_dict())
        return {
            "schema_version": "speechtotext.calibrator-training/v1",
            "status": "complete",
            "artifact_fingerprint": artifact.version,
            "counts": {
                "development": len(development.examples),
                "calibration": len(calibration.examples),
            },
        }


def _build_replay_inputs(args: argparse.Namespace):
    thresholds = QualityThresholds(
        min_effective_voice_ms=args.min_effective_voice_ms,
        min_processed_rms_dbfs=args.min_rms_dbfs,
        min_snr_db=args.min_snr_db,
        max_clipping_ratio=args.max_clipping_ratio,
    )
    capture_pipeline = PipelineProvenance.capture(
        sample_rate=args.sample_rate,
        step=PipelineStep(
            "pyav-decode",
            "1",
            {"dtype": "float32", "layout": "mono", "sample_rate": args.sample_rate},
        ),
    )
    asr_input_pipeline = PipelineProvenance.derive(
        capture_pipeline,
        sample_rate=args.sample_rate,
        steps=(
            PipelineStep(
                "fixed-gain",
                "1",
                {"gain_db": args.gain_db, "max_gain_db": 18.0, "peak_limit_dbfs": -1.0},
            ),
        ),
    )
    request = TranscriptionRequest(language=args.language)
    backend_config = FasterWhisperConfig(
        device=args.device,
        compute_type=args.compute_type,
    )
    return thresholds, asr_input_pipeline, request, backend_config


def execute_job(args: argparse.Namespace) -> dict[str, object]:
    _validate_calibrator_options(args)
    _validate_planned_model_identity(args)
    filesystem = default_corpus_filesystem()
    manifest = load_corpus_manifest(
        args.manifest,
        dataset_root=args.dataset_root,
        repo_root=args.repo_root,
        filesystem=filesystem,
    )
    split = split_by_recording_day(manifest, seed=args.seed)
    config = EvaluationConfig(
        as_of=args.as_of,
        expected_language=args.language,
        sample_rate=args.sample_rate,
        usable_max_wer=args.usable_max_wer,
        gain_db=args.gain_db,
    )
    trusted_today = date.today()
    selected_entries, _, _ = preflight_evaluation(
        manifest,
        split,
        args.partition,
        config,
        today=lambda: trusted_today,
    )
    thresholds, pipeline, request, backend_config = _build_replay_inputs(args)
    with audit_dataset_security(
        args.dataset_root,
        args.repo_root,
        args.manifest,
        manifest,
        assets=tuple(
            asset for entry in selected_entries for asset in entry.assets
        ),
        filesystem=filesystem,
    ) as security_evidence:
        security_evidence.require_for(args.dataset_root, manifest)
        calibrator = None
        if args.calibrator is not None:
            calibrator = LogisticCalibrator(
                parse_calibrator_artifact_bytes(
                    security_evidence.read_report_bytes(
                        args.calibrator,
                        max_bytes=1_000_000,
                    ),
                    expected_fingerprint=args.calibrator_fingerprint,
                )
            )
            calibrator.validate_binding_for_identity(
                backend=FasterWhisperBackend.backend_id,
                model=args.model_id,
                model_version=args.model_revision,
                backend_artifact_kind="local_model_manifest",
                backend_artifact_fingerprint=args.model_manifest_fingerprint,
                backend_config_fingerprint=backend_config.fingerprint,
                pipeline=pipeline,
                request=request,
                expected_language=config.expected_language,
                usable_max_wer=config.usable_max_wer,
            )
        model_filesystem = default_model_filesystem()
        model_manifest = load_model_manifest(
            args.model_manifest,
            model_root=args.model_root,
            expected_fingerprint=args.model_manifest_fingerprint,
            filesystem=model_filesystem,
        )
        if (
            model_manifest.model_id,
            model_manifest.revision,
            model_manifest.fingerprint,
        ) != (
            args.model_id,
            args.model_revision,
            args.model_manifest_fingerprint,
        ):
            raise ValueError("modelo activo no coincide con identidad planeada")
        if model_manifest.sample_rate != args.sample_rate:
            raise ValueError("sample rate de modelo incompatible")
        with verify_model_files(
            model_manifest,
            args.model_root,
            filesystem=model_filesystem,
        ) as model_artifact:
            backend = FasterWhisperBackend(backend_config, model_artifact)
            if calibrator is not None:
                calibrator.validate_for(
                    backend=backend,
                    pipeline=pipeline,
                    request=request,
                    expected_language=config.expected_language,
                    usable_max_wer=config.usable_max_wer,
                )
            report_ref_key = security_evidence.read_report_ref_key(
                args.report_ref_key_file
            )
            return run_evaluation(
                manifest=manifest,
                split=split,
                partition=args.partition,
                dataset_root=args.dataset_root,
                backend=backend,
                request=request,
                config=config,
                thresholds=thresholds,
                pipeline=pipeline,
                output_path=args.output,
                calibrator=calibrator,
                asset_lease_factory=lambda asset, root: lease_corpus_asset(
                    asset, root, filesystem=filesystem
                ),
                security_evidence=security_evidence,
                environment=collect_environment(
                    args.repo_root, ref_key=report_ref_key
                ),
                report_ref_key=report_ref_key,
                today=lambda: trusted_today,
            )


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: Callable[[argparse.Namespace], dict[str, object]] = execute_job,
    execute_corpus: Callable[
        [argparse.Namespace], dict[str, object]
    ] = execute_corpus_job,
    execute_training: Callable[
        [argparse.Namespace], dict[str, object]
    ] = execute_training_job,
) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["train-calibrator"]:
        training_args = build_training_parser().parse_args(raw_args[1:])
        try:
            payload = execute_training(training_args)
            development, calibration, fingerprint = _validated_training_payload(
                payload
            )
            print(
                "OK calibrator "
                f"development={development} calibration={calibration} "
                f"artifact_fingerprint={fingerprint} artifact_written=true"
            )
            return 0
        except Exception:
            print("ERROR code=calibrator_training_failed", file=sys.stderr)
            return 1
    if raw_args[:1] == ["corpus"]:
        corpus_args = build_corpus_parser().parse_args(raw_args[1:])
        try:
            payload = execute_corpus(corpus_args)
            status, clips = _validated_corpus_payload(
                payload, corpus_args.corpus_command
            )
            print(f"OK corpus status={status} clips={clips}")
            return 2 if status == "partial" else 1 if status == "failed" else 0
        except Exception:
            print("ERROR code=corpus_failed", file=sys.stderr)
            return 1
    args = build_parser().parse_args(raw_args)
    try:
        report = execute(args)
        clips, gate = _validated_evaluation_report(report)
        if args.require_acceptance:
            if gate["passed"] is not True:
                status = str(gate["status"])
                print(
                    f"BLOCKED status={status} report_written=true",
                    file=sys.stderr,
                )
                return 2
        print(f"OK clips={clips} report_written=true")
        return 0
    except Exception:
        print("ERROR code=evaluation_failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
