# tests/test_evaluation_cli.py
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import speechtotext.evaluation.__main__ as cli_module
from speechtotext.evaluation.__main__ import (
    build_corpus_parser,
    build_parser,
    build_training_parser,
    execute_job,
    main,
)
from speechtotext.confidence.calibration import (
    CalibratorArtifact,
    serialize_calibrator_artifact,
)
from speechtotext.evaluation.splits import DatasetSplit
from speechtotext.statistics import one_sided_success_lower


ARGS = [
    "--manifest", r"D:\AudioBench\aurelius-2026\manifest.json",
    "--dataset-root", r"D:\AudioBench\aurelius-2026",
    "--repo-root", r"C:\src\speechtotext",
    "--partition", "development",
    "--as-of", "2026-07-16",
    "--model-manifest", r"D:\Models\fw-small\manifest.json",
    "--model-root", r"D:\Models\fw-small",
    "--model-manifest-fingerprint", "a" * 64,
    "--model-id", "faster-whisper-small",
    "--model-revision", "b" * 40,
    "--output", r"D:\AudioBench\aurelius-2026\reports\dev.json",
    "--report-ref-key-file", r"D:\AudioBench\aurelius-2026\secrets\report-ref.key",
    "--min-effective-voice-ms", "160",
    "--min-rms-dbfs", "-45",
    "--min-snr-db", "6",
    "--max-clipping-ratio", "0.01",
]
TRAINING_ARGS = [
    "--manifest", r"D:\AudioBench\aurelius-2026\manifest.json",
    "--dataset-root", r"D:\AudioBench\aurelius-2026",
    "--repo-root", r"C:\src\speechtotext",
    "--model-manifest", r"D:\Models\fw-small\manifest.json",
    "--model-root", r"D:\Models\fw-small",
    "--model-manifest-fingerprint", "a" * 64,
    "--output", r"D:\AudioBench\aurelius-2026\reports\calibrator.json",
    "--as-of", "2026-07-16",
    "--artifact-version", "fw-small-es-v1",
    "--min-effective-voice-ms", "160",
    "--min-rms-dbfs", "-45",
    "--min-snr-db", "6",
    "--max-clipping-ratio", "0.01",
]


def _report(status="passed"):
    insufficient = ["duration_below_30m"] if status == "insufficient_evidence" else []
    failures = ["wer_noise_above_limit"] if status == "failed" else []
    return {
        "schema_version": "speechtotext.evaluation/v1",
        "counts": {"clips": 3},
        "acceptance_gate": {
            "status": status,
            "passed": status == "passed",
            "sufficient_evidence": status != "insufficient_evidence",
            "insufficient_reason_codes": insufficient,
            "failure_reason_codes": failures,
            "thresholds": {
                "wer_clean_upper_95": 0.05,
                "wer_noise_upper_95": 0.10,
            },
        },
    }


def test_parser_exige_particion_y_thresholds():
    parser = build_parser()
    args = parser.parse_args(ARGS)
    assert args.partition == "development"
    assert args.as_of == date(2026, 7, 16)
    assert args.min_effective_voice_ms == 160
    assert args.min_rms_dbfs == -45.0
    assert args.compute_type == "int8"
    assert args.gain_db == 0.0
    assert args.model_manifest_fingerprint == "a" * 64
    assert args.model_id == "faster-whisper-small"
    assert args.model_revision == "b" * 40
    assert args.require_acceptance is False
    assert args.report_ref_key_file.name == "report-ref.key"
    assert args.calibrator is None and args.calibrator_fingerprint is None


def test_calibrador_exige_path_y_trust_anchor_externo_juntos():
    for path, fingerprint in ((Path("calibrator.json"), None), (None, "a" * 64)):
        with pytest.raises(ValueError, match="pair_required"):
            cli_module._validate_calibrator_options(SimpleNamespace(
                calibrator=path,
                calibrator_fingerprint=fingerprint,
            ))
    cli_module._validate_calibrator_options(SimpleNamespace(
        calibrator=Path("calibrator.json"),
        calibrator_fingerprint="a" * 64,
    ))


def test_train_calibrator_parsea_y_despacha_sin_holdout(capsys):
    parsed = build_training_parser().parse_args(TRAINING_ARGS)
    assert parsed.artifact_version == "fw-small-es-v1"
    assert parsed.min_precision_lower_95 == 0.99
    calls = []

    def fake_execute(args):
        calls.append(args)
        return {
            "schema_version": "speechtotext.calibrator-training/v1",
            "status": "complete",
            "artifact_fingerprint": "b" * 64,
            "counts": {"development": 12, "calibration": 8},
        }

    assert main(
        ["train-calibrator", *TRAINING_ARGS], execute_training=fake_execute
    ) == 0
    assert len(calls) == 1 and not hasattr(calls[0], "partition")
    output = capsys.readouterr().out
    assert "development=12 calibration=8" in output
    assert "artifact_fingerprint=" + "b" * 64 in output
    assert str(parsed.output) not in output


def test_main_delega_y_reporta_output(capsys):
    calls = []

    def fake_execute(args):
        calls.append(args)
        return _report()

    assert main(ARGS, execute=fake_execute) == 0
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert "clips=3" in output
    assert str(ARGS[ARGS.index("--output") + 1]) not in output


def test_main_falla_cerrado_si_se_exige_aceptacion_sin_evidencia(capsys):
    def fake_execute(args):
        return _report("insufficient_evidence")

    assert main(ARGS + ["--require-acceptance"], execute=fake_execute) == 2
    error = capsys.readouterr().err
    assert "insufficient_evidence" in error
    assert str(ARGS[ARGS.index("--output") + 1]) not in error


def test_parser_rechaza_particion_invalida_sin_reflejar_valor(capsys):
    parser = build_parser()
    bad = list(ARGS)
    private = r"D:\AudioBench\secret\clip.wav"
    bad[bad.index("development")] = private
    with pytest.raises(SystemExit):
        parser.parse_args(bad)
    assert private not in capsys.readouterr().err


def test_corpus_parser_tiene_dry_run_y_confirmacion_explicita():
    parser = build_corpus_parser()
    base = [
        "--manifest", r"D:\AudioBench\aurelius-2026\manifest.json",
        "--dataset-root", r"D:\AudioBench\aurelius-2026",
        "--repo-root", r"C:\src\speechtotext",
        "--report-ref-key-file", r"D:\AudioBench\aurelius-2026\secrets\report-ref.key",
        "--receipt", r"D:\AudioBench\aurelius-2026\reports\purge.json",
    ]
    planned = parser.parse_args(["purge-expired", *base])
    confirmed = parser.parse_args(
        ["purge-expired", *base, "--confirm"]
    )
    assert planned.confirm is False
    assert confirmed.confirm is True
    assert not hasattr(planned, "as_of")


def test_init_report_key_despacha_factory_segura_una_vez(monkeypatch):
    calls = []
    filesystem = object()
    monkeypatch.setattr(
        cli_module, "default_corpus_filesystem", lambda: filesystem
    )
    monkeypatch.setattr(
        cli_module,
        "initialize_report_ref_key",
        lambda root, repo, output, *, filesystem: calls.append(
            (root, repo, output, filesystem)
        ),
    )
    args = build_corpus_parser().parse_args([
        "init-report-key",
        "--dataset-root", r"D:\AudioBench\aurelius-2026",
        "--repo-root", r"C:\src\speechtotext",
        "--output", r"D:\AudioBench\aurelius-2026\secrets\report-ref.key",
    ])
    payload = cli_module.execute_corpus_job(args)
    assert payload == {
        "schema_version": "speechtotext.report-key-init/v1",
        "status": "complete",
        "counts": {"clips": 0},
    }
    assert len(calls) == 1 and calls[0][3] is filesystem


def test_main_despacha_corpus_sin_cargar_modelo(capsys):
    calls = []

    def fake_execute_corpus(args):
        calls.append(args)
        return {"schema_version": "speechtotext.retention-list/v1",
                "status": "listed", "counts": {"clips": 1}}

    assert main(
        ["corpus", "list", "--manifest", "m.json", "--dataset-root", "private",
         "--repo-root", "repo", "--report-ref-key-file", "private.key"],
        execute_corpus=fake_execute_corpus,
    ) == 0
    assert calls[0].corpus_command == "list"


def test_main_sanitiza_excepcion_sin_filtrar_ruta_hash_o_mensaje(capsys):
    private = r"D:\AudioBench\secret\clip.wav"
    digest = "a" * 64

    def fake_execute(args):
        raise RuntimeError(f"fallo leyendo {private} sha256={digest}")

    assert main(ARGS, execute=fake_execute) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR code=evaluation_failed\n"
    assert private not in captured.err
    assert digest not in captured.err


def test_main_sanitiza_excepcion_corpus_sin_filtrar_mensaje(capsys):
    private = r"D:\AudioBench\secret\clip.wav"

    def fake_execute_corpus(args):
        raise RuntimeError(f"fallo corpus {private}")

    assert main(
        ["corpus", "list", "--manifest", "m.json", "--dataset-root", "private",
         "--repo-root", "repo", "--report-ref-key-file", "private.key"],
        execute_corpus=fake_execute_corpus,
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR code=corpus_failed\n"
    assert private not in captured.err


def test_main_rechaza_payloads_inconsistentes_aunque_declaren_pass(capsys):
    invalid_gate = _report()
    invalid_gate["acceptance_gate"]["status"] = "invented"
    assert main(ARGS + ["--require-acceptance"], execute=lambda _args: invalid_gate) == 1
    assert capsys.readouterr().err == "ERROR code=evaluation_failed\n"

    assert main(
        ["corpus", "list", "--manifest", "m.json", "--dataset-root", "private",
         "--repo-root", "repo", "--report-ref-key-file", "private.key"],
        execute_corpus=lambda _args: {
            "schema_version": "speechtotext.retention-list/v1",
            "status": "invalid_report",
            "counts": {"clips": 0},
        },
    ) == 1
    assert capsys.readouterr().err == "ERROR code=corpus_failed\n"


# --------------------------------------------------------------------------
# Composition-root tests para execute_job. Verifican el orden fail-closed del
# wiring del CLI sin decode real ni faster-whisper (prohibidos en tests): cada
# camino aborta antes de tocar assets, modelo o inferencia. Las garantias del
# happy-path (warm/lease/decode/inferencia, holdout nunca abierto) las cubre
# tests/test_evaluation_runner.py sobre run_evaluation/collect_labeled_*.
# --------------------------------------------------------------------------


def _job_args(corpus, **overrides):
    values = {
        "--manifest": str(corpus.manifest_path),
        "--dataset-root": str(corpus.root),
        "--repo-root": str(corpus.repo),
        "--partition": "development",
        "--as-of": "2026-07-16",
        "--model-manifest": str(corpus.root / "model" / "manifest.json"),
        "--model-root": str(corpus.root / "model"),
        "--model-manifest-fingerprint": "a" * 64,
        "--model-id": "faster-whisper-small",
        "--model-revision": "b" * 40,
        "--output": str(corpus.root / "reports" / "dev.json"),
        "--report-ref-key-file": str(corpus.root / "secrets" / "report-ref.key"),
        "--min-effective-voice-ms": "160",
        "--min-rms-dbfs": "-45",
        "--min-snr-db": "6",
        "--max-clipping-ratio": "0.01",
    }
    values.update(overrides)
    argv = []
    for key, value in values.items():
        argv.append(key)
        if value is not None:
            argv.append(value)
    return build_parser().parse_args(argv)


def _single_partition_split(manifest, seed):
    # Coloca la unica entry del corpus en development; deja calibration/holdout
    # vacios para poder ejercitar preflight sin exigir nueve fechas.
    return DatasetSplit.create(manifest, manifest.entries, (), (), seed)


def _wire_spies(monkeypatch, calls, filesystem):
    monkeypatch.setattr(cli_module, "default_corpus_filesystem", lambda: filesystem)
    monkeypatch.setattr(
        cli_module,
        "split_by_recording_day",
        lambda manifest, seed: _single_partition_split(manifest, seed),
    )
    for name in (
        "default_model_filesystem",
        "load_model_manifest",
        "verify_model_files",
        "lease_corpus_asset",
        "run_evaluation",
        "collect_environment",
    ):
        monkeypatch.setattr(
            cli_module,
            name,
            lambda *a, __name=name, **k: calls.append(__name),
        )


class _NoBackend:
    backend_id = "faster-whisper"

    def __init__(self, *args, **kwargs):
        raise AssertionError("backend construido antes de tiempo")


def test_execute_job_expirado_no_audita_assets_ni_verifica_modelo(
    monkeypatch, corpus, fs_adapter
):
    calls = []
    _wire_spies(monkeypatch, calls, fs_adapter)
    monkeypatch.setattr(cli_module, "audit_dataset_security", lambda *a, **k: calls.append("audit"))
    monkeypatch.setattr(cli_module, "FasterWhisperBackend", _NoBackend)

    class _FutureDate(date):
        @classmethod
        def today(cls):
            return cls(2028, 1, 1)  # despues de retention_until 2027-01-12

    monkeypatch.setattr(cli_module, "date", _FutureDate)

    with pytest.raises(ValueError, match="corpus_retention_expired"):
        execute_job(_job_args(corpus))
    # Solo se leyo manifest/split; audit, modelo, leases, run y environment: cero.
    assert calls == []


def _incompatible_calibrator(corpus):
    correct, accepted, total = 1, 1, 1
    artifact = CalibratorArtifact(
        schema_version="speechtotext.calibrator/v1",
        artifact_version="cal-v1",
        target="segment_usable",
        usable_max_wer=0.10,
        expected_language="es",
        feature_names=("feat",),
        feature_means=(0.0,),
        feature_scales=(1.0,),
        coefficients=(0.5,),
        intercept=0.0,
        operating_threshold=0.5,
        selection_correct=correct,
        selection_accepted=accepted,
        selection_total=total,
        precision_lower_95=one_sided_success_lower(correct, accepted),
        backend="faster-whisper",
        model="modelo-que-no-coincide",
        model_version="c" * 40,
        backend_artifact_kind="local_model_manifest",
        backend_artifact_fingerprint="0" * 64,
        backend_config_fingerprint="0" * 64,
        fit_split_fingerprint="0" * 64,
        calibration_split_fingerprint="0" * 64,
        pipeline_fingerprint="0" * 64,
        request_fingerprint="0" * 64,
    )
    path = corpus.root / "reports" / "cal.json"
    path.write_bytes(serialize_calibrator_artifact(artifact))
    return path, artifact.version


def test_execute_job_calibrador_incompatible_falla_antes_de_audio_jit(
    monkeypatch, corpus, fs_adapter
):
    fs_adapter.configure_security(acl_ok=True, encryption_ok=True)
    (corpus.root / "secrets" / "report-ref.key").write_bytes(b"k" * 32)
    calibrator_path, fingerprint = _incompatible_calibrator(corpus)
    calls = []
    # audit real: debe correr (root+manifest+lectura protegida del calibrador).
    monkeypatch.setattr(cli_module, "default_corpus_filesystem", lambda: fs_adapter)
    monkeypatch.setattr(
        cli_module,
        "split_by_recording_day",
        lambda manifest, seed: _single_partition_split(manifest, seed),
    )
    for name in (
        "default_model_filesystem",
        "load_model_manifest",
        "verify_model_files",
        "lease_corpus_asset",
        "run_evaluation",
        "collect_environment",
    ):
        monkeypatch.setattr(
            cli_module, name, lambda *a, __name=name, **k: calls.append(__name)
        )
    monkeypatch.setattr(cli_module, "FasterWhisperBackend", _NoBackend)

    args = _job_args(
        corpus,
        **{
            "--calibrator": str(calibrator_path),
            "--calibrator-fingerprint": fingerprint,
        },
    )
    with pytest.raises(ValueError, match="incompatible"):
        execute_job(args)
    # El binding contra la identidad planeada externa falla antes de abrir el
    # model manifest/root, construir backend, tomar leases o inferir.
    assert calls == []


def test_execute_job_modelo_activo_distinto_a_identidad_planeada_falla_antes_de_verify(
    monkeypatch, corpus, fs_adapter
):
    fs_adapter.configure_security(acl_ok=True, encryption_ok=True)
    calls = []
    monkeypatch.setattr(cli_module, "default_corpus_filesystem", lambda: fs_adapter)
    monkeypatch.setattr(
        cli_module,
        "split_by_recording_day",
        lambda manifest, seed: _single_partition_split(manifest, seed),
    )
    monkeypatch.setattr(
        cli_module, "default_model_filesystem", lambda: object()
    )
    # El manifest co-local declara otra identidad que la planeada externa.
    monkeypatch.setattr(
        cli_module,
        "load_model_manifest",
        lambda *a, **k: SimpleNamespace(
            model_id="otro-modelo",
            revision="b" * 40,
            fingerprint="a" * 64,
            sample_rate=16000,
        ),
    )
    for name in ("verify_model_files", "lease_corpus_asset", "run_evaluation",
                 "collect_environment"):
        monkeypatch.setattr(
            cli_module, name, lambda *a, __name=name, **k: calls.append(__name)
        )
    monkeypatch.setattr(cli_module, "FasterWhisperBackend", _NoBackend)

    with pytest.raises(ValueError, match="identidad planeada"):
        execute_job(_job_args(corpus))
    # verify_model_files/backend/leases/run nunca corren tras el mismatch.
    assert calls == []
