"""Dibuja docs/img/benchmark-{light,dark}.svg a partir de lo medido el 2026-09-11.

Sin dependencias: el SVG se escribe a mano para que el gráfico del README se pueda
regenerar con `python scripts/benchmark_chart.py` y quede idéntico. Los números salen
del informe del benchmark (14 min de reunión real, dos voces, 1820 palabras); las
configuraciones troceadas no entran porque no se alinearon palabra a palabra.

    python scripts/benchmark_chart.py
"""
from __future__ import annotations

from pathlib import Path

# (etiqueta, motor, modelo, x = veces tiempo real, y = errores por 1000 palabras)
PUNTOS = [
    ("large-v3",            "fw",   "large", 1.27,  1.6),
    ("large-v3 + hotwords", "fw",   "large", 1.30,  3.8),
    ("large-v3 + VAD",      "fw",   "large", 1.31,  7.7),
    ("small",               "fw",   "small", 6.4,  29.1),
    ("large-v3",            "wcpp", "large", 7.97,  6.0),
    ("small",               "wcpp", "small", 15.7, 27.5),
]

# Paleta de referencia de la skill dataviz, validada con su script en los dos modos.
TEMAS = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", fw="#2a78d6", wcpp="#eb6834"),
    "dark":  dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", fw="#3987e5", wcpp="#d95926"),
}

W, H = 760, 440
L, R, T, B = 56, 736, 96, 360          # caja del plot
X_MAX, Y_MAX = 16.0, 30.0
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _x(v: float) -> float:
    return L + (R - L) * v / X_MAX


def _y(v: float) -> float:
    return B - (B - T) * v / Y_MAX


def _text(x, y, s, *, size=12, fill, anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def _marker(x, y, color, surface, hueco: bool) -> str:
    # 2 px de anillo en color de superficie, y el punto: relleno = large-v3, hueco = small.
    ring = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{surface}"/>'
    if hueco:
        dot = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{surface}" stroke="{color}" stroke-width="2"/>'
    else:
        dot = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
    return ring + dot


def render(tema: str) -> str:
    t = TEMAS[tema]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Velocidad contra errores de seis configuraciones de transcripción">',
           f'<rect width="{W}" height="{H}" rx="6" fill="{t["surface"]}"/>']

    out.append(_text(24, 30, "Velocidad contra errores en 14 minutos de reunión real",
                     size=15, weight="600", fill=t["primary"]))
    out.append(_text(24, 50, "Un punto por configuración. Abajo y a la derecha es mejor.",
                     size=12, fill=t["secondary"]))

    # rejilla horizontal (hairline, sólida) + ticks de y
    for v in (0, 10, 20, 30):
        y = _y(v)
        color = t["axis"] if v == 0 else t["grid"]
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{color}" stroke-width="1"/>')
        out.append(_text(L - 8, y + 4, str(v), size=11, fill=t["muted"], anchor="end"))
    out.append(_text(L, T - 14, "errores por 1000 palabras", size=11, fill=t["muted"]))

    # ticks de x
    for v in (0, 4, 8, 12, 16):
        x = _x(v)
        out.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 5}" stroke="{t["axis"]}" stroke-width="1"/>')
        out.append(_text(x, B + 20, f"{v}×", size=11, fill=t["muted"], anchor="middle"))
    out.append(_text(R, B + 38, "velocidad (× tiempo real)", size=11, fill=t["muted"], anchor="end"))

    # puntos y etiquetas (la etiqueta lleva tinta de texto, nunca el color de la serie)
    for label, motor, modelo, xv, yv in PUNTOS:
        x, y = _x(xv), _y(yv)
        out.append(_marker(x, y, t[motor], t["surface"], hueco=(modelo == "small")))
        derecha = x < R - 120
        out.append(_text(x + 13 if derecha else x - 13, y + 4, label, size=12,
                         fill=t["secondary"], anchor="start" if derecha else "end"))

    # leyenda: motor por color, tamaño de modelo por relleno
    ly = 404
    out.append(_marker(32, ly, t["fw"], t["surface"], hueco=False))
    out.append(_text(46, ly + 4, "faster-whisper · CPU int8", fill=t["secondary"]))
    out.append(_marker(232, ly, t["wcpp"], t["surface"], hueco=False))
    out.append(_text(246, ly + 4, "whisper.cpp · CUDA (GTX 980)", fill=t["secondary"]))
    out.append(_marker(462, ly, t["muted"], t["surface"], hueco=False))
    out.append(_text(476, ly + 4, "large-v3", fill=t["secondary"]))
    out.append(_marker(560, ly, t["muted"], t["surface"], hueco=True))
    out.append(_text(574, ly + 4, "small", fill=t["secondary"]))

    out.append(_text(24, 430, "2026-09-11 · 866 s de audio, dos voces, 1820 palabras · "
                     "errores contados sobre 66 sitios decidibles · reloj de un Ryzen 9 5900X",
                     size=10.5, fill=t["muted"]))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    destino = Path(__file__).resolve().parent.parent / "docs" / "img"
    destino.mkdir(parents=True, exist_ok=True)
    for tema in TEMAS:
        (destino / f"benchmark-{tema}.svg").write_text(render(tema), encoding="utf-8")
        print(f"escrito docs/img/benchmark-{tema}.svg")


if __name__ == "__main__":
    main()
