from speechtotext.audio.evidence import VoiceEvidence, compute_voice_evidence
from speechtotext.audio.fingerprint import ModelRef, PipelineProvenance, PipelineStep
from speechtotext.audio.gate import (
    PreInferenceDecision,
    QualityReason,
    QualityThresholds,
    evaluate_pre_inference,
)
from speechtotext.audio.io import AudioDecodeError, decode_audio
from speechtotext.audio.level import GainResult, apply_fixed_gain
from speechtotext.audio.quality import compute_audio_quality
from speechtotext.audio.types import (
    AudioClip,
    AudioQualityReport,
    AudioView,
    AudioViewName,
    AudioViews,
    SpeechRegion,
)

__all__ = [
    "AudioClip",
    "AudioDecodeError",
    "AudioQualityReport",
    "AudioView",
    "AudioViewName",
    "AudioViews",
    "GainResult",
    "ModelRef",
    "PipelineProvenance",
    "PipelineStep",
    "SpeechRegion",
    "VoiceEvidence",
    "apply_fixed_gain",
    "compute_audio_quality",
    "compute_voice_evidence",
    "decode_audio",
]

__all__ += [
    "PreInferenceDecision",
    "QualityReason",
    "QualityThresholds",
    "evaluate_pre_inference",
]
