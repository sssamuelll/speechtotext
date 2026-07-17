# src/speechtotext/audio/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from speechtotext.audio.types import AudioView
from speechtotext.audio.fingerprint import PipelineStep


class AudioDecodeError(RuntimeError):
    """PyAV no pudo producir audio PCM canonico."""


def _frames(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def decode_audio(
    stream: BinaryIO,
    *,
    sample_rate: int,
    av_module: object | None = None,
) -> AudioView:
    if sample_rate <= 0:
        raise ValueError("sample_rate debe ser positivo")
    if (
        isinstance(stream, (str, bytes, bytearray, Path))
        or not hasattr(stream, "read")
        or not hasattr(stream, "seek")
        or not callable(getattr(stream, "seekable", None))
        or not stream.seekable()
        or not isinstance(stream.read(0), bytes)
    ):
        raise TypeError("decode_audio exige un stream binario leased y seekable")
    if av_module is None:
        import av as av_module

    chunks: list[np.ndarray] = []
    try:
        with av_module.open(stream) as container:
            if not container.streams.audio:
                raise AudioDecodeError("el archivo no contiene stream de audio")
            stream = container.streams.audio[0]
            resampler = av_module.AudioResampler(
                format="flt",
                layout="mono",
                rate=sample_rate,
            )
            for frame in container.decode(stream):
                for converted in _frames(resampler.resample(frame)):
                    chunks.append(
                        np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                    )
            for converted in _frames(resampler.resample(None)):
                chunks.append(
                    np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                )
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"PyAV no pudo decodificar el asset leased: {exc}") from exc
    nonempty = [chunk for chunk in chunks if chunk.size]
    if not nonempty:
        raise AudioDecodeError("PyAV produjo audio sin muestras")
    return AudioView.capture(
        np.concatenate(nonempty),
        sample_rate,
        step=PipelineStep(
            "pyav-decode",
            "1",
            {"layout": "mono", "dtype": "float32", "sample_rate": sample_rate},
        ),
    )
