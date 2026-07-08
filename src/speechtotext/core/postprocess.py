"""Post-proceso ligero del texto transcrito (número/horas)."""
from __future__ import annotations

import re

# H.MM -> H:MM cuando el audio dicta una hora ("a las 8.33"). Whisper large escribe
# la hora con punto; exigimos DOS dígitos de minutos válidos (00-59) para no tocar
# magnitudes de un solo decimal (7.2, 7.5), que se conservan como dígitos.
# ponytail: heurística; "8.30 por ciento" también se convertiría. Ceiling aceptado
# para este dominio (ponencias con horas). Si molesta: exigir contexto "a las"/"h".
_HORA = re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)\b")


def normalize_hours(text: str) -> str:
    return _HORA.sub(r"\1:\2", text)
