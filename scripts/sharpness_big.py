"""All-models comparison per photo at 1:1 native pixels — no nearest-zoom.

Two outputs per photo:
  big_<photo>.png          — 4×4 grid of 1:1 crops (288×288 native), 15 panels
  big_full_<photo>.png     — 4×4 grid of full 768×1024, downscaled by PIL Lanczos
                             to 384×512 per panel so the figure stays openable

Run after sharpness_models.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import OUT, WORK  # noqa: E402

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]

# Display order: baseline first, then community standouts, then bundled set.
PANELS = [
    ("JPEG↓ → upscayl-std (baseline)",  "_jpgdown_upscayl-standard-4x.png"),
    ("raw → upscayl-standard",          "_raw_upscayl-standard-4x.png"),
    ("raw → upscayl-lite",              "_raw_upscayl-lite-4x.png"),
    ("raw → high-fidelity",             "_raw_high-fidelity-4x.png"),
    ("raw → ultramix-balanced",         "_raw_ultramix-balanced-4x.png"),
    ("raw → digital-art",               "_raw_digital-art-4x.png"),
    ("raw → remacri",                   "_raw_remacri-4x.png"),
    ("raw → ultrasharp",                "_raw_ultrasharp-4x.png"),
    ("raw → 4xNomos8kSC",               "_raw_4xNomos8kSC.png"),
    ("raw → 4xLSDIRplusC",              "_raw_4xLSDIRplusC.png"),
    ("raw → 4xHFA2k",                   "_raw_4xHFA2k.png"),
    ("raw → NMKD-Siax",                 "_raw_4x_NMKD-Siax_200k.png"),
    ("raw → NMKD-Superscale",           "_raw_4x_NMKD-Superscale-SP_178000_G.png"),
    ("raw → RealESRGAN-v3",             "_raw_RealESRGAN_General_x4_v3.png"),
    ("raw → RealESRGAN-WDN-v3",         "_raw_RealESRGAN_General_WDN_x4_v3.png"),
]

# fractional center on 768×1024, half-side
REGIONS = {
    "1777733165452": ((0.50, 0.78), 144),  # ember-texture
    "1777804010136": ((0.50, 0.36), 144),  # upper-cup-rim
    "1771110170401": ((0.45, 0.78), 144),  # belt-ornament
}

GRID_COLS = 4
GAP = 10
LABEL_H = 50


def make_label(text: str, w: int, h: int = LABEL_H, fontsize: int = 16) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", fontsize)
    except OSError:
        font = ImageFont.load_default()
    bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.multiline_text(((w - tw) / 2, (h - th) / 2), text,
                     fill="black", font=font, align="center", spacing=2)
    return img


def grid(panels: list[tuple[str, Image.Image]], heading: str, out: Path) -> None:
    if not panels:
        return
    pw, ph = panels[0][1].size
    rows = (len(panels) + GRID_COLS - 1) // GRID_COLS

    cell_w = pw + GAP
    cell_h = ph + LABEL_H + GAP
    canvas_w = GRID_COLS * cell_w + GAP
    canvas_h = 36 + rows * cell_h + GAP
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(make_label(heading, canvas_w, h=32, fontsize=18), (0, 2))

    for i, (title, im) in enumerate(panels):
        r, c = divmod(i, GRID_COLS)
        x = GAP + c * cell_w
        y = 36 + r * cell_h
        canvas.paste(make_label(title, pw, h=LABEL_H, fontsize=14), (x, y))
        canvas.paste(im, (x, y + LABEL_H))

    canvas.save(out, format="PNG", optimize=True)
    print(f"wrote {out}  ({canvas_w}×{canvas_h})")


def compose_for_photo(photo_id: str) -> None:
    photo_work = WORK / photo_id
    (cx_f, cy_f), half = REGIONS[photo_id]

    # Read first existing image to learn output dimensions
    crops: list[tuple[str, Image.Image]] = []
    fulls: list[tuple[str, Image.Image]] = []
    for title, fname in PANELS:
        path = photo_work / fname
        if not path.exists():
            print(f"missing: {path}  (skipping panel)")
            continue
        im = Image.open(path).convert("RGB")
        pw, ph = im.size
        cx, cy = int(pw * cx_f), int(ph * cy_f)
        box = (max(0, cx - half), max(0, cy - half),
               min(pw, cx + half), min(ph, cy + half))
        crops.append((title, im.crop(box)))                       # 1:1 crop
        fulls.append((title, im.resize((pw // 2, ph // 2),
                                        Image.Resampling.LANCZOS)))

    grid(crops, f"{photo_id}.jpg — crop {2*half}×{2*half} @ 1:1 (native pixels)",
         OUT / f"big_{photo_id}.png")
    grid(fulls, f"{photo_id}.jpg — full 768×1024 ↓2 (PIL Lanczos)",
         OUT / f"big_full_{photo_id}.png")


def main() -> None:
    for p in PHOTOS:
        compose_for_photo(p)


if __name__ == "__main__":
    main()
