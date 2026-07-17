from __future__ import annotations

from speechtotext.asr.base import VerifiedLocalAsrBackend
from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult
from speechtotext.audio.fingerprint import PipelineProvenance
from speechtotext.audio.types import AudioClip
from speechtotext.confidence.calibration import LogisticCalibrator
from speechtotext.confidence.features import extract_asr_features
from speechtotext.models import VerifiedModelArtifact


class CalibratingLocalAsrBackend:
    def __init__(
        self,
        raw_backend: VerifiedLocalAsrBackend,
        calibrator: LogisticCalibrator,
        *,
        pipeline: PipelineProvenance,
        request: TranscriptionRequest,
        expected_language: str,
        usable_max_wer: float,
    ) -> None:
        if not isinstance(raw_backend, VerifiedLocalAsrBackend) or not isinstance(
            raw_backend.model_artifact, VerifiedModelArtifact
        ):
            raise TypeError("decorator exige VerifiedLocalAsrBackend")
        if not isinstance(calibrator, LogisticCalibrator):
            raise TypeError("decorator exige LogisticCalibrator")
        if not isinstance(pipeline, PipelineProvenance) or not isinstance(
            request, TranscriptionRequest
        ):
            raise TypeError("pipeline/request invalidos")
        raw_backend.model_artifact.require_active()
        self._raw_backend = raw_backend
        self._calibrator = calibrator
        self._pipeline = pipeline
        self._request = request
        self._expected_language = expected_language
        self._usable_max_wer = usable_max_wer
        self._validate_binding()

    def _validate_binding(self) -> None:
        self.model_artifact.require_active()
        self._calibrator.validate_for(
            backend=self._raw_backend,
            pipeline=self._pipeline,
            request=self._request,
            expected_language=self._expected_language,
            usable_max_wer=self._usable_max_wer,
        )

    @property
    def raw_backend(self) -> VerifiedLocalAsrBackend:
        return self._raw_backend

    @property
    def backend_id(self) -> str:
        return self._raw_backend.backend_id

    @property
    def model_id(self) -> str:
        return self._raw_backend.model_id

    @property
    def model_version(self) -> str:
        return self._raw_backend.model_version

    @property
    def model_artifact(self) -> VerifiedModelArtifact:
        return self._raw_backend.model_artifact

    @property
    def backend_artifact_kind(self) -> str:
        return self._raw_backend.backend_artifact_kind

    @property
    def backend_artifact_fingerprint(self) -> str:
        return self._raw_backend.backend_artifact_fingerprint

    @property
    def config_fingerprint(self) -> str:
        return self._raw_backend.config_fingerprint

    @property
    def calibrator_version(self) -> str:
        return self._calibrator.artifact.version

    def warm(self) -> None:
        self._validate_binding()
        self._raw_backend.warm()

    def transcribe(
        self,
        clip: AudioClip,
        request: TranscriptionRequest,
    ) -> TranscriptionResult:
        if not isinstance(clip, AudioClip) or not isinstance(
            request, TranscriptionRequest
        ):
            raise TypeError("clip/request invalidos")
        view = clip.view("asr")
        if (
            request.fingerprint != self._request.fingerprint
            or view.provenance != self._pipeline
        ):
            raise ValueError("pipeline/request no coincide con el binding calibrado")
        self._validate_binding()
        raw = self._raw_backend.transcribe(clip, request)
        if raw.calibrated_confidence is not None or raw.calibrator_version is not None:
            raise ValueError("raw backend no puede devolver confianza ya calibrada")
        features = extract_asr_features(
            raw,
            clip.quality,
            expected_language=self._expected_language,
        )
        return self._calibrator.apply(
            raw,
            features,
            backend=self._raw_backend,
            view=view,
            request=request,
            expected_language=self._expected_language,
            usable_max_wer=self._usable_max_wer,
        )

    def close(self) -> None:
        close = getattr(self._raw_backend, "close", None)
        if callable(close):
            close()
