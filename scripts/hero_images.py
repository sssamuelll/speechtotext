"""Draws docs/img/hero-{light,dark}.svg, the README's masthead, and the
1280x640 social-preview card next to them.

The wordmark and the tagline are converted to outlines: GitHub serves README
images through a proxy that loads no web fonts, so text has to travel as
paths. Shaping (kerning) is HarfBuzz's, outlines are fontTools', and the
fonts come from Google Fonts on first run into scripts/.fonts/ (Instrument
Serif and Inter, both under the SIL Open Font License).

    pip install fonttools uharfbuzz
    python scripts/hero_images.py

The PNG for the social preview is a screenshot of social-light.svg at
1280x640; any browser does it. Every text/background pair was measured at
4.5:1 or better over the rendered mesh before it shipped.
"""
import pathlib
import sys
import urllib.request

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
FONTS = pathlib.Path(__file__).resolve().parent / ".fonts"
OUT = ROOT / "docs" / "img"

FONT_URLS = {
    "InstrumentSerif-Regular.ttf": "https://fonts.gstatic.com/s/instrumentserif/v5/jizBRFtNs2ka5fXjeivQ4LroWlx-2zI.ttf",
    "Inter-500.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuI6fMZg.ttf",
    "Inter-400.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
}

TAGLINE = "The audio never leaves your machine."
SUBLINE = "Local speech to text for files · Python library · CLI · MCP server"

# The desktop app's tokens: a warm mesh in light, plum and violet in dark, one
# ink for the wordmark and a softer one for the line under it.
THEMES = {
    "light": dict(base="#f8f3ee", b1="#f3d0b5", b2="#dcb7cb", b3="#a892d2",
                  ink="#18181b", ink2="#3f3a47", line="#000000", line_op="0.07"),
    "dark": dict(base="#1b1424", b1="#5a3690", b2="#743461", b3="#352a72",
                 ink="#f6f1f7", ink2="#d6cbe0", line="#ffffff", line_op="0.10"),
}


def _fetch_fonts() -> None:
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        target = FONTS / name
        if not target.exists():
            print(f"fetching {name}")
            urllib.request.urlretrieve(url, target)


class Face:
    def __init__(self, path: pathlib.Path):
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.upem = self.tt["head"].unitsPerEm
        self.order = self.tt.getGlyphOrder()
        self.font = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
        self.font.scale = (self.upem, self.upem)
        hb.ot_font_set_funcs(self.font)

    def shape(self, text: str, size: float, tracking: float = 0.0):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {"kern": True, "liga": True})
        s = size / self.upem
        glyphs, pen_x = [], 0.0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            glyphs.append((self.order[info.codepoint], pen_x + pos.x_offset * s, pos.y_offset * s))
            pen_x += pos.x_advance * s + tracking
        return glyphs, pen_x - tracking

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        return self.shape(text, size, tracking)[1]

    def path(self, text: str, size: float, cx: float, baseline: float, fill: str, tracking: float = 0.0) -> str:
        glyphs, width = self.shape(text, size, tracking)
        x0 = cx - width / 2
        s = size / self.upem
        parts = []
        for name, gx, gy in glyphs:
            pen = SVGPathPen(self.gs, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
            self.gs[name].draw(TransformPen(pen, (s, 0, 0, -s, x0 + gx, baseline - gy)))
            d = pen.getCommands()
            if d:
                parts.append(d)
        return f'<path fill="{fill}" d="{" ".join(parts)}"/>'


def size_for_width(face: Face, text: str, target: float) -> float:
    return 100.0 * target / face.width(text, 100.0)


def card(faces: dict, theme: str, w: int, h: int, *, wordmark_frac: float, wordmark_y: float,
         tagline_size: float, tagline_y: float, subline: tuple[float, float] | None,
         radius: int = 28) -> str:
    t = THEMES[theme]
    blur = round(h * 0.22)
    # Three soft bodies of color, placed so the middle band, where the text
    # sits, is the calmest part of the canvas.
    blobs = [
        (0.14 * w, 0.20 * h, 0.80 * h, t["b1"], 0.90),
        (0.84 * w, 0.92 * h, 0.95 * h, t["b2"], 0.85),
        (0.78 * w, 0.06 * h, 0.70 * h, t["b3"], 0.75),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title">',
        f"<title id=\"title\">speechtotext — {TAGLINE}</title>",
        "<defs>",
        f'<clipPath id="c"><rect width="{w}" height="{h}" rx="{radius}"/></clipPath>',
        f'<filter id="b" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="{blur}"/></filter>',
        "</defs>",
        '<g clip-path="url(#c)">',
        f'<rect width="{w}" height="{h}" fill="{t["base"]}"/>',
        '<g filter="url(#b)">',
    ]
    for cx, cy, r, color, op in blobs:
        parts.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{color}" opacity="{op}"/>')
    parts.append("</g>")
    parts.append(f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{radius}" fill="none" '
                 f'stroke="{t["line"]}" stroke-opacity="{t["line_op"]}"/>')
    parts.append("</g>")
    serif, inter_500, inter_400 = faces["serif"], faces["inter_500"], faces["inter_400"]
    size = size_for_width(serif, "speechtotext", wordmark_frac * w)
    parts.append(serif.path("speechtotext", size, w / 2, wordmark_y, t["ink"], tracking=-size * 0.004))
    parts.append(inter_500.path(TAGLINE, tagline_size, w / 2, tagline_y, t["ink2"]))
    if subline:
        parts.append(inter_400.path(SUBLINE, subline[0], w / 2, subline[1], t["ink2"]))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    _fetch_fonts()
    faces = {
        "serif": Face(FONTS / "InstrumentSerif-Regular.ttf"),
        "inter_500": Face(FONTS / "Inter-500.ttf"),
        "inter_400": Face(FONTS / "Inter-400.ttf"),
    }
    hero = dict(wordmark_frac=0.60, wordmark_y=176, tagline_size=27, tagline_y=236, subline=None)
    social = dict(wordmark_frac=0.62, wordmark_y=318, tagline_size=36, tagline_y=396,
                  subline=(23, 452), radius=0)
    for theme in THEMES:
        (OUT / f"hero-{theme}.svg").write_text(card(faces, theme, 1200, 320, **hero), encoding="utf-8")
    (OUT / "social-preview.svg").write_text(card(faces, "light", 1280, 640, **social), encoding="utf-8")
    for p in sorted(OUT.glob("hero-*.svg")) + [OUT / "social-preview.svg"]:
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
