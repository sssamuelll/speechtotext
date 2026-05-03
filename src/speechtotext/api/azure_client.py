"""Cliente fino sobre Azure AI Speech Pronunciation Assessment."""
from __future__ import annotations

from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

from speechtotext.api.schemas import (
    PhonemeScore,
    PronunciationScores,
    ScoreResponse,
    WordScore,
)


class AzureSpeechError(RuntimeError):
    """Azure devolvió un resultado no exitoso."""

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


def score_pronunciation(
    wav_path: Path,
    reference_text: str,
    language: str,
    *,
    azure_key: str,
    azure_region: str,
) -> ScoreResponse:
    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.speech_recognition_language = language

    audio_config = speechsdk.audio.AudioConfig(filename=str(wav_path))

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.NoMatch:
        raise AzureSpeechError("No se detectó voz en el audio.", recoverable=True)
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        raise AzureSpeechError(f"Azure canceló la petición: {details.reason} — {details.error_details}")
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise AzureSpeechError(f"Resultado inesperado: {result.reason}")

    pa = speechsdk.PronunciationAssessmentResult(result)

    words = [
        WordScore(
            word=w.word,
            accuracy_score=w.accuracy_score,
            error_type=str(w.error_type),
            phonemes=[
                PhonemeScore(phoneme=p.phoneme, accuracy_score=p.accuracy_score)
                for p in (w.phonemes or [])
            ],
        )
        for w in pa.words
    ]

    return ScoreResponse(
        recognized_text=result.text,
        reference_text=reference_text,
        language=language,
        scores=PronunciationScores(
            accuracy=pa.accuracy_score,
            fluency=pa.fluency_score,
            completeness=pa.completeness_score,
            pronunciation=pa.pronunciation_score,
        ),
        words=words,
    )
