from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from speechtotext.audio.fingerprint import PipelineProvenance, PipelineStep

AudioViewName = Literal["capture", "analysis", "identity", "spoof", "asr"]
_AUDIO_VIEW_FACTORY_TOKEN = object()


@dataclass(frozen=True, order=True)
class SpeechRegion:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        values = (self.start_s, self.end_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("speech region debe usar tiempos finitos")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("speech region debe cumplir 0 <= start_s < end_s")
        object.__setattr__(self, "start_s", float(self.start_s))
        object.__setattr__(self, "end_s", float(self.end_s))


@dataclass(frozen=True, init=False)
class AudioView:
    samples: np.ndarray
    sample_rate: int
    provenance: PipelineProvenance

    @classmethod
    def _create(
        cls, samples, sample_rate, provenance, *, _factory_token=None
    ) -> "AudioView":
        if not isinstance(provenance, PipelineProvenance):
            raise TypeError("provenance exige PipelineProvenance")
        if _factory_token is not _AUDIO_VIEW_FACTORY_TOKEN:
            raise TypeError("AudioView solo puede crearse mediante su factory publica")
        instance = object.__new__(cls)
        object.__setattr__(instance, "samples", samples)
        object.__setattr__(instance, "sample_rate", sample_rate)
        object.__setattr__(instance, "provenance", provenance)
        instance._validate()
        return instance

    def _validate(self) -> None:
        owned = np.array(self.samples, dtype=np.float32, order="C", copy=True)
        if owned.ndim != 1:
            raise ValueError("audio debe ser mono y unidimensional")
        if not np.isfinite(owned).all():
            raise ValueError("audio contiene muestras no finitas")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate debe ser un entero positivo")
        if self.provenance.sample_rate != self.sample_rate:
            raise ValueError("sample_rate no coincide con provenance")
        # A bytes-backed view cannot be made writeable again by the caller.
        array = np.frombuffer(owned.tobytes(order="C"), dtype=np.float32)
        object.__setattr__(self, "samples", array)

    @classmethod
    def capture(cls, samples, sample_rate, *, step: PipelineStep) -> "AudioView":
        provenance = PipelineProvenance.capture(sample_rate=sample_rate, step=step)
        return cls._create(
            samples,
            sample_rate,
            provenance,
            _factory_token=_AUDIO_VIEW_FACTORY_TOKEN,
        )

    @classmethod
    def derive(
        cls,
        parent: "AudioView",
        samples,
        *,
        sample_rate: int | None = None,
        steps: tuple[PipelineStep, ...],
        models=(),
        thresholds=None,
    ) -> "AudioView":
        if not isinstance(parent, AudioView):
            raise TypeError("parent exige AudioView")
        target_rate = parent.sample_rate if sample_rate is None else sample_rate
        if target_rate <= 0:
            raise ValueError("sample_rate destino debe ser positivo")
        provenance = PipelineProvenance.derive(
            parent.provenance,
            sample_rate=target_rate,
            steps=steps,
            models=models,
            thresholds={} if thresholds is None else thresholds,
        )
        return cls._create(
            samples,
            target_rate,
            provenance,
            _factory_token=_AUDIO_VIEW_FACTORY_TOKEN,
        )

    @property
    def pipeline_fingerprint(self) -> str:
        return self.provenance.fingerprint

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class AudioViews:
    capture: AudioView
    analysis: AudioView
    asr: AudioView
    identity: AudioView | None = None
    spoof: AudioView | None = None

    def __post_init__(self) -> None:
        required = (self.capture, self.analysis, self.asr)
        optional = (self.identity, self.spoof)
        if any(not isinstance(view, AudioView) for view in required) or any(
            view is not None and not isinstance(view, AudioView)
            for view in optional
        ):
            raise TypeError("AudioViews exige instancias AudioView")


@dataclass(frozen=True)
class AudioQualityReport:
    duration_ms: int
    effective_voice_ms: int
    input_rms_dbfs: float
    processed_rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    noise_floor_dbfs: float | None
    snr_db: float | None
    requested_gain_db: float
    applied_gain_db: float
    dropped_frames: int
    discontinuities: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        counts = (
            self.duration_ms,
            self.effective_voice_ms,
            self.dropped_frames,
            self.discontinuities,
        )
        if any(type(value) is not int for value in counts):
            raise ValueError("duraciones y contadores deben ser enteros")
        if self.duration_ms < 0:
            raise ValueError("duration_ms no puede ser negativo")
        if not 0 <= self.effective_voice_ms <= self.duration_ms:
            raise ValueError("effective_voice_ms debe estar dentro de duration_ms")
        required = (
            self.input_rms_dbfs,
            self.processed_rms_dbfs,
            self.peak_dbfs,
            self.clipping_ratio,
            self.requested_gain_db,
            self.applied_gain_db,
        )
        optional = (self.noise_floor_dbfs, self.snr_db)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in required
        ):
            raise ValueError("metricas de calidad requeridas deben ser finitas")
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
            for value in optional
        ):
            raise ValueError("metricas de calidad opcionales deben ser finitas o None")
        if not 0.0 <= self.clipping_ratio <= 1.0:
            raise ValueError("clipping_ratio debe estar entre 0 y 1")
        if self.dropped_frames < 0 or self.discontinuities < 0:
            raise ValueError("contadores de transporte no pueden ser negativos")
        if (
            not isinstance(self.warnings, tuple)
            or any(not isinstance(item, str) or not item for item in self.warnings)
            or len(self.warnings) != len(set(self.warnings))
        ):
            raise ValueError("warnings exige codigos string unicos en tuple")


@dataclass(frozen=True)
class AudioClip:
    started_at: float
    ended_at: float
    source_id: str
    speech_regions: tuple[SpeechRegion, ...]
    quality: AudioQualityReport
    views: AudioViews

    def __post_init__(self) -> None:
        stamps = (self.started_at, self.ended_at)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in stamps
        ):
            raise ValueError("timestamps del clip deben ser finitos")
        if self.ended_at <= self.started_at:
            raise ValueError("ended_at debe ser posterior a started_at")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id no puede estar vacio")
        if not isinstance(self.quality, AudioQualityReport):
            raise TypeError("quality exige AudioQualityReport")
        if not isinstance(self.views, AudioViews):
            raise TypeError("views exige AudioViews")
        if (
            not isinstance(self.speech_regions, tuple)
            or any(not isinstance(region, SpeechRegion) for region in self.speech_regions)
        ):
            raise TypeError("speech_regions exige tuple de SpeechRegion")
        duration = self.ended_at - self.started_at
        regions = tuple(sorted(self.speech_regions))
        previous_end = 0.0
        for region in regions:
            if region.end_s > duration or region.start_s < previous_end:
                raise ValueError("speech region solapada o fuera de la duracion del clip")
            previous_end = region.end_s
        object.__setattr__(self, "speech_regions", regions)
        present_views = (
            self.views.capture,
            self.views.analysis,
            self.views.asr,
            *(view for view in (self.views.identity, self.views.spoof)
              if view is not None),
        )
        if any(
            abs(view.duration_s - duration) > (1.0 / view.sample_rate + 1e-12)
            for view in present_views
        ):
            raise ValueError("duracion de vistas no coincide con AudioClip")
        if abs(self.quality.duration_ms - round(duration * 1000.0)) > 1:
            raise ValueError("quality.duration_ms no coincide con AudioClip")
        object.__setattr__(self, "started_at", float(self.started_at))
        object.__setattr__(self, "ended_at", float(self.ended_at))

    def view(self, name: AudioViewName) -> AudioView:
        if name not in {"capture", "analysis", "identity", "spoof", "asr"}:
            raise KeyError(f"vista desconocida: {name}")
        value = getattr(self.views, name)
        if value is None:
            raise KeyError(f"vista no disponible: {name}")
        return value
