from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tomllib

import pytest

from speechtotext.evaluation.environment import (
    _parse_linux_statm,
    collect_environment,
    process_memory_bytes,
    write_environment_report,
)


def test_av_es_dependencia_directa_y_evaluation_es_opcional():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11"
    dependencies = data["project"]["dependencies"]
    assert any(dep.startswith("av>=") for dep in dependencies)
    assert data["project"]["optional-dependencies"]["evaluation"] == [
        "scikit-learn>=1.9,<2",
        "scipy>=1.17.1,<1.18",
    ]
    constraints = Path("constraints/windows-cpu.txt").read_text(encoding="utf-8")
    for dependency in (
        "ctranslate2==4.8.1",
        "huggingface-hub==1.22.0",
        "onnxruntime==1.27.0",
        "tokenizers==0.23.1",
    ):
        assert dependency in constraints.splitlines()


def test_environment_report_es_json_atomico(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._package_version",
        lambda name: {"speechtotext": "0.3.0", "av": "18.0.0"}.get(name, "test"),
    )
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_revision",
        lambda repo: "abc123",
    )
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_status",
        lambda repo: "",
    )
    output = tmp_path / "baseline.json"
    report = write_environment_report(tmp_path, output, ref_key=b"r" * 32)
    assert report["schema_version"] == "speechtotext.environment/v1"
    assert report["git_ref"].startswith("git-revision:")
    assert "abc123" not in json.dumps(report)
    assert "git_revision" not in report
    assert "executable" not in report
    assert str(tmp_path.resolve()) not in json.dumps(report)
    assert report["packages"]["av"] == "18.0.0"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_suffix(".json.tmp").exists()


def test_process_memory_reporta_rss_y_peak_positivos():
    memory = process_memory_bytes()
    assert memory["rss"] > 0
    assert memory["peak_rss"] >= memory["rss"]


def test_linux_statm_reporta_residente_actual_en_bytes():
    assert _parse_linux_statm("100 25 10 5 0 20 0\n", page_size=4096) == 102_400


def test_environment_rechaza_worktree_sucio_antes_de_leer_revision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_status",
        lambda repo: " M src/module.py",
    )
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_revision",
        lambda repo: pytest.fail("no debe atribuir un worktree sucio a HEAD"),
    )
    with pytest.raises(ValueError, match="worktree limpio"):
        collect_environment(tmp_path, ref_key=b"r" * 32)


def test_writers_concurrentes_usan_temporales_independientes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_status", lambda repo: ""
    )
    monkeypatch.setattr(
        "speechtotext.evaluation.environment._git_revision", lambda repo: "abc123"
    )
    monkeypatch.setattr(
        "speechtotext.evaluation.environment.process_memory_bytes",
        lambda: {"rss": 1, "peak_rss": 1},
    )
    output = tmp_path / "baseline.json"
    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = tuple(
            executor.map(
                lambda key: write_environment_report(
                    tmp_path, output, ref_key=bytes([key]) * 32
                ),
                (1, 2),
            )
        )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted in reports
    assert not tuple(tmp_path.glob(".baseline.json.*.tmp"))


def test_environment_exige_clave_hmac_suficiente(tmp_path):
    with pytest.raises(ValueError, match="32 bytes"):
        collect_environment(tmp_path, ref_key=b"short")
