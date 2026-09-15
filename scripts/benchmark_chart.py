"""Draws docs/img/benchmark-{light,dark}.svg from what was measured on 2026-09-11.

No dependencies: the SVG is written by hand so the README's chart can be
regenerated with `python scripts/benchmark_chart.py` and come out identical.
The numbers come from the benchmark report (14 min of a real meeting, two
voices, 1820 words); the chunked configurations are left out because they
were not aligned word for word.

    python scripts/benchmark_chart.py
"""
from __future__ import annotations

from pathlib import Path

# (label, engine, model, x = times real time, y = errors per 1000 words)
POINTS = [
    ("large-v3",            "fw",   "large", 1.27,  1.6),
    ("large-v3 + hotwords", "fw",   "large", 1.30,  3.8),
    ("large-v3 + VAD",      "fw",   "large", 1.31,  7.7),
    ("small",               "fw",   "small", 6.4,  29.1),
    ("large-v3",            "wcpp", "large", 7.97,  6.0),
    ("small",               "wcpp", "small", 15.7, 27.5),
]

# Reference palette from the dataviz skill, validated with its script in both modes.
THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", fw="#2a78d6", wcpp="#eb6834"),
    "dark":  dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", fw="#3987e5", wcpp="#d95926"),
}

W, H = 760, 440
L, R, T, B = 56, 736, 96, 360          # plot box
X_MAX, Y_MAX = 16.0, 30.0
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _x(v: float) -> float:
    return L + (R - L) * v / X_MAX


def _y(v: float) -> float:
    return B - (B - T) * v / Y_MAX


def _text(x, y, s, *, size=12, fill, anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def _marker(x, y, color, surface, hollow: bool) -> str:
    # 2 px ring in surface color, plus the dot: filled = large-v3, hollow = small.
    ring = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{surface}"/>'
    if hollow:
        dot = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{surface}" stroke="{color}" stroke-width="2"/>'
    else:
        dot = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
    return ring + dot


def render(theme: str) -> str:
    t = THEMES[theme]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Speed against errors for six transcription configurations">',
           f'<rect width="{W}" height="{H}" rx="6" fill="{t["surface"]}"/>']

    out.append(_text(24, 30, "Speed against errors over 14 minutes of a real meeting",
                     size=15, weight="600", fill=t["primary"]))
    out.append(_text(24, 50, "One dot per configuration. Down and to the right is better.",
                     size=12, fill=t["secondary"]))

    # horizontal gridlines (hairline, solid) + y ticks
    for v in (0, 10, 20, 30):
        y = _y(v)
        color = t["axis"] if v == 0 else t["grid"]
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{color}" stroke-width="1"/>')
        out.append(_text(L - 8, y + 4, str(v), size=11, fill=t["muted"], anchor="end"))
    out.append(_text(L, T - 14, "errors per 1000 words", size=11, fill=t["muted"]))

    # x ticks
    for v in (0, 4, 8, 12, 16):
        x = _x(v)
        out.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 5}" stroke="{t["axis"]}" stroke-width="1"/>')
        out.append(_text(x, B + 20, f"{v}×", size=11, fill=t["muted"], anchor="middle"))
    out.append(_text(R, B + 38, "speed (x real time)", size=11, fill=t["muted"], anchor="end"))

    # points and labels (the label always carries text ink, never the series color)
    for label, engine, model, xv, yv in POINTS:
        x, y = _x(xv), _y(yv)
        out.append(_marker(x, y, t[engine], t["surface"], hollow=(model == "small")))
        right = x < R - 120
        out.append(_text(x + 13 if right else x - 13, y + 4, label, size=12,
                         fill=t["secondary"], anchor="start" if right else "end"))

    # legend: engine by color, model size by fill
    ly = 404
    out.append(_marker(32, ly, t["fw"], t["surface"], hollow=False))
    out.append(_text(46, ly + 4, "faster-whisper · CPU int8", fill=t["secondary"]))
    out.append(_marker(232, ly, t["wcpp"], t["surface"], hollow=False))
    out.append(_text(246, ly + 4, "whisper.cpp · CUDA (GTX 980)", fill=t["secondary"]))
    out.append(_marker(462, ly, t["muted"], t["surface"], hollow=False))
    out.append(_text(476, ly + 4, "large-v3", fill=t["secondary"]))
    out.append(_marker(560, ly, t["muted"], t["surface"], hollow=True))
    out.append(_text(574, ly + 4, "small", fill=t["secondary"]))

    out.append(_text(24, 430, "2026-09-11 · 866 s of audio, two voices, 1820 words · "
                     "errors counted over 66 decidable sites · wall clock on a Ryzen 9 5900X",
                     size=10.5, fill=t["muted"]))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    dest = Path(__file__).resolve().parent.parent / "docs" / "img"
    dest.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (dest / f"benchmark-{theme}.svg").write_text(render(theme), encoding="utf-8")
        print(f"wrote docs/img/benchmark-{theme}.svg")


if __name__ == "__main__":
    main()
