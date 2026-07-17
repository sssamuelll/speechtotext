import hashlib
import json
import math

import pytest
from hypothesis import given, strategies as st

from speechtotext.audio.fingerprint import PipelineProvenance, PipelineStep
from speechtotext.models.filesystem import FakeModelFilesystem
from speechtotext.models.manifest import load_model_manifest, verify_model_files


def test_fingerprint_es_determinista_ante_orden_de_claves():
    first = PipelineProvenance.capture(
        sample_rate=16000,
        step=PipelineStep("gain", "1", {"max_db": 18.0, "mode": "profile"}),
        models=(),
        thresholds={"vad_end": 0.35, "vad_start": 0.60},
    )
    second = PipelineProvenance.capture(
        sample_rate=16000,
        step=PipelineStep("gain", "1", {"mode": "profile", "max_db": 18.0}),
        models=(),
        thresholds={"vad_start": 0.60, "vad_end": 0.35},
    )
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_fingerprint_cambia_si_cambia_orden_o_threshold():
    gain = PipelineStep("gain", "1", {"db": 6.0})
    resample = PipelineStep("resample", "1", {"to": 16000})
    parent = PipelineProvenance.capture(sample_rate=16000, step=gain)
    base = PipelineProvenance.derive(
        parent, sample_rate=16000, steps=(resample,), thresholds={"min_voice_ms": 160}
    )
    reordered = PipelineProvenance.derive(
        PipelineProvenance.capture(sample_rate=16000, step=resample),
        sample_rate=16000,
        steps=(gain,),
        thresholds={"min_voice_ms": 160},
    )
    assert reordered.fingerprint != base.fingerprint


def test_fingerprint_rechaza_nan():
    with pytest.raises(ValueError, match="JSON finito"):
        PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("gain", "1", {"db": math.nan}),
        )


@given(
    st.dictionaries(
        st.text(min_size=1),
        st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(),
            st.booleans(),
            st.none(),
        ),
        max_size=12,
    )
)
def test_fingerprint_es_determinista_para_json_finito(parameters):
    step = PipelineStep("property", "1", parameters)
    assert PipelineProvenance.capture(
        sample_rate=16000, step=step
    ).fingerprint == PipelineProvenance.capture(
        sample_rate=16000, step=step
    ).fingerprint


def test_from_dict_rechaza_fingerprint_autoafirmado():
    provenance = PipelineProvenance.capture(
        sample_rate=16000, step=PipelineStep("decode", "1", {"mono": True})
    )
    payload = provenance.to_dict()
    payload["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="no coincide"):
        PipelineProvenance.from_dict(payload, parent=None, models=())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["steps"][0].__setitem__("name", 7),
        lambda payload: payload["steps"][0].__setitem__("extra", True),
        lambda payload: payload.__setitem__("fingerprint", 7),
    ],
)
def test_from_dict_rechaza_coercion_o_campos_de_step(mutation):
    provenance = PipelineProvenance.capture(
        sample_rate=16000, step=PipelineStep("decode", "1", {"mono": True})
    )
    payload = provenance.to_dict()
    mutation(payload)
    with pytest.raises(ValueError, match="schema"):
        PipelineProvenance.from_dict(payload, parent=None, models=())


@pytest.mark.parametrize("sample_rate", [True, 0, -1, 16000.5, "16000"])
def test_pipeline_rechaza_sample_rate_no_entero_positivo(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        PipelineProvenance.capture(
            sample_rate=sample_rate,
            step=PipelineStep("decode", "1", {}),
        )


def test_pipeline_rechaza_objeto_que_solo_imita_un_modelo_verificado():
    class FakeVerifiedModel:
        fingerprint = "0" * 64

    with pytest.raises(TypeError, match="VerifiedModelArtifact"):
        PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("denoise", "1", {}),
            models=(FakeVerifiedModel(),),
        )


def test_pipeline_rechaza_step_y_parent_simulados():
    fake_step = type("FakeStep", (), {"to_dict": lambda self: {}})()
    fake_parent = type("FakeParent", (), {"fingerprint": "0" * 64})()
    with pytest.raises(TypeError, match="PipelineStep"):
        PipelineProvenance.capture(sample_rate=16000, step=fake_step)
    with pytest.raises(TypeError, match="PipelineProvenance"):
        PipelineProvenance.derive(
            fake_parent,
            sample_rate=16000,
            steps=(PipelineStep("gain", "1", {}),),
        )


def test_pipeline_factory_privado_no_es_una_ruta_publica():
    with pytest.raises(TypeError, match="factory"):
        PipelineProvenance._create(
            16000,
            None,
            (PipelineStep("capture", "1", {}),),
            (),
            {},
        )


def test_fingerprint_desacopla_y_congela_json_anidado():
    parameters = {"frontend": {"bands": [1, 2]}}
    thresholds = {"vad": {"start": 0.6}}
    provenance = PipelineProvenance.capture(
        sample_rate=16000,
        step=PipelineStep("analysis", "1", parameters),
        thresholds=thresholds,
    )
    fingerprint = provenance.fingerprint
    exported = provenance.to_dict()
    parameters["frontend"]["bands"].append(3)
    thresholds["vad"]["start"] = 0.1
    exported["steps"][0]["parameters"]["frontend"]["bands"].append(4)
    assert provenance.fingerprint == fingerprint
    with pytest.raises(TypeError):
        provenance.thresholds["vad"] = {}
    with pytest.raises(AttributeError):
        provenance.steps[0].parameters["frontend"]["bands"].append(5)


def test_pipeline_provenance_no_tiene_constructor_publico():
    with pytest.raises(TypeError):
        PipelineProvenance(
            16000,
            None,
            (PipelineStep("capture", "1", {}),),
            (),
            {},
        )


def _verified_model(tmp_path):
    model_fs = FakeModelFilesystem(root_read_only=True)
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = {
        "schema_version": "speechtotext.model/v1",
        "model_id": "denoise-small",
        "source": "https://example.invalid/denoise-small",
        "revision_kind": "git_commit",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "license": "MIT",
        "format": "onnx",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{"path": "model.bin", "sha256": hashlib.sha256(b"weights").hexdigest()}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            data, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    manifest = load_model_manifest(
        manifest_path,
        model_root=tmp_path,
        expected_fingerprint=expected_fingerprint,
        filesystem=model_fs,
    )
    return verify_model_files(manifest, tmp_path, filesystem=model_fs)


def test_modelo_verificado_de_denoise_cambia_el_fingerprint(tmp_path):
    with _verified_model(tmp_path) as artifact:
        without_model = PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("denoise", "1", {}),
        )
        with_model = PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("denoise", "1", {}),
            models=(artifact,),
        )
        assert with_model.fingerprint != without_model.fingerprint
