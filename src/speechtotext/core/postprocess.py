"""Light post-processing of transcribed text (numbers/times)."""
from __future__ import annotations

import re

# H.MM -> H:MM when the audio dictates a time ("at 8.33"). Whisper large writes
# the time with a period; we require TWO valid minute digits (00-59) to avoid touching
# single-decimal magnitudes (7.2, 7.5), which remain as digits.
# ponytail: heuristic; "8.30 percent" would also be converted. Accepted ceiling
# for this domain (talks with times). If it becomes a problem: require "at"/"h" context.
_HOUR = re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)\b")


def normalize_hours(text: str) -> str:
    return _HOUR.sub(r"\1:\2", text)
