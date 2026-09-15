import math

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from speechtotext.audio.fingerprint import ModelRef, PipelineProvenance, PipelineStep


def test_fingerprint_is_deterministic_regardless_of_key_order():
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


def test_fingerprint_changes_if_step_order_or_a_threshold_changes():
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


def test_fingerprint_rejects_nan():
    with pytest.raises(ValueError, match="finite JSON"):
        PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("gain", "1", {"db": math.nan}),
        )


# too_slow measures how long Hypothesis takes to GENERATE inputs, not the property. In
# the full suite (and on CI runners, which are slower than any laptop), the `st.text()`
# generator without a bounded alphabet exceeds the budget and fails the test even though
# nothing is wrong: it once failed in 2 out of 3 full runs here. The property being
# checked—the fingerprint does not depend on key order—is the same whether the health
# check is enabled or disabled.
@settings(suppress_health_check=[HealthCheck.too_slow])
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
def test_fingerprint_is_deterministic_for_finite_json(parameters):
    step = PipelineStep("property", "1", parameters)
    assert PipelineProvenance.capture(
        sample_rate=16000, step=step
    ).fingerprint == PipelineProvenance.capture(
        sample_rate=16000, step=step
    ).fingerprint


def test_from_dict_rejects_a_self_asserted_fingerprint():
    provenance = PipelineProvenance.capture(
        sample_rate=16000, step=PipelineStep("decode", "1", {"mono": True})
    )
    payload = provenance.to_dict()
    payload["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        PipelineProvenance.from_dict(payload, parent=None, models=())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["steps"][0].__setitem__("name", 7),
        lambda payload: payload["steps"][0].__setitem__("extra", True),
        lambda payload: payload.__setitem__("fingerprint", 7),
    ],
)
def test_from_dict_rejects_coercion_or_step_fields(mutation):
    provenance = PipelineProvenance.capture(
        sample_rate=16000, step=PipelineStep("decode", "1", {"mono": True})
    )
    payload = provenance.to_dict()
    mutation(payload)
    with pytest.raises(ValueError, match="schema"):
        PipelineProvenance.from_dict(payload, parent=None, models=())


@pytest.mark.parametrize("sample_rate", [True, 0, -1, 16000.5, "16000"])
def test_pipeline_rejects_a_sample_rate_that_is_not_a_positive_integer(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        PipelineProvenance.capture(
            sample_rate=sample_rate,
            step=PipelineStep("decode", "1", {}),
        )


def test_pipeline_rejects_a_model_that_is_not_a_model_ref():
    class Impostor:
        model_id = "denoise"
        fingerprint = "0" * 64

    with pytest.raises(TypeError, match="ModelRef"):
        PipelineProvenance.capture(
            sample_rate=16000,
            step=PipelineStep("denoise", "1", {}),
            models=(Impostor(),),
        )


def test_pipeline_rejects_mocked_step_and_parent_objects():
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


def test_pipeline_private_factory_is_not_a_public_entry_point():
    with pytest.raises(TypeError, match="factory"):
        PipelineProvenance._create(
            16000,
            None,
            (PipelineStep("capture", "1", {}),),
            (),
            {},
        )


def test_fingerprint_decouples_and_freezes_nested_json():
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


def test_pipeline_provenance_has_no_public_constructor():
    with pytest.raises(TypeError):
        PipelineProvenance(
            16000,
            None,
            (PipelineStep("capture", "1", {}),),
            (),
            {},
        )


def test_model_ref_requires_an_id_and_a_64_character_hex_fingerprint():
    with pytest.raises(ValueError, match="fingerprint"):
        ModelRef("denoise", "abc")
    with pytest.raises(ValueError, match="fingerprint"):
        ModelRef("denoise", "G" * 64)
    with pytest.raises(ValueError, match="model_id"):
        ModelRef("  ", "0" * 64)


def test_a_referenced_model_changes_the_fingerprint():
    step = PipelineStep("denoise", "1", {})
    without_model = PipelineProvenance.capture(sample_rate=16000, step=step)
    with_model = PipelineProvenance.capture(
        sample_rate=16000, step=step, models=(ModelRef("denoise-small", "a" * 64),),
    )
    assert with_model.fingerprint != without_model.fingerprint
    assert with_model.model_fingerprints == ("a" * 64,)
