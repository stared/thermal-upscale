"""σ=0.81 fixed; sweep SR models. Visual chart per photo.

Panels (each at 1:1 native, 288×288 crop):
  0: camera baseline   (JPEG↓ → upscayl-standard)
  1: raw → upscayl-lite        (soft reference, no preprocessing)
  2: σ=0.81 → upscayl-standard
  3: σ=0.81 → upscayl-lite     (current recommendation — RealESRGAN-v3)
  4: σ=0.81 → 4xNomos8kSC
  5: σ=0.81 → high-fidelity-4x
  6: σ=0.81 → 4xHFA2k
  7: σ=0.81 → digital-art-4x

Model set follows saved priorities: faithful > clean > sharp. Excludes
ultrasharp / remacri / NMKD-Siax which invent detail (per priorities memory).

Run: uv run scripts/show_optimum_models.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, WORK, UPSCAYL_BIN, UPSCAYL_MODELS,
    parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)
from sharpen_models import wiener_deconvolve  # noqa: E402

CACHE_MODELS = Path.home() / ".cache/thermal-upscale/models"

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]

REGIONS = {
    "1777733165452": ((0.50, 0.78), 144),
    "1777804010136": ((0.50, 0.36), 144),
    "1771110170401": ((0.45, 0.78), 144),
}

SIGMA = 0.81

# (display label, model name, models dir)
MODELS = [
    ("upscayl-standard", "upscayl-standard-4x", UPSCAYL_MODELS),
    ("upscayl-lite (=RealESRGAN-v3)", "upscayl-lite-4x", UPSCAYL_MODELS),
    ("4xNomos8kSC",       "4xNomos8kSC",          CACHE_MODELS),
    ("high-fidelity",     "high-fidelity-4x",     UPSCAYL_MODELS),
    ("4xHFA2k",           "4xHFA2k",              CACHE_MODELS),
    ("digital-art",       "digital-art-4x",       UPSCAYL_MODELS),
]


def render_raw(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    return colormap_rgb(normalize_with_cut(thermal, 1.0), "inferno")


def run_upscayl(in_png: Path, out_png: Path, model: str, mdir: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN), "-i", str(in_png), "-o", str(out_png),
         "-m", str(mdir), "-n", model, "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"{model}: {(proc.stderr or proc.stdout)[-200:]}")
    return time.perf_counter() - t0


def make_label(text: str, w: int, h: int = 56, fontsize: int = 14) -> Image.Image:
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


def grid(panels: list[tuple[str, Image.Image]], heading: str, out: Path,
         cols: int = 4) -> None:
    pw, ph = panels[0][1].size
    rows = (len(panels) + cols - 1) // cols
    gap = 10
    label_h = 56
    cell_w = pw + gap
    cell_h = ph + label_h + gap
    canvas_w = cols * cell_w + gap
    canvas_h = 36 + rows * cell_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(make_label(heading, canvas_w, h=32, fontsize=18), (0, 2))
    for i, (title, im) in enumerate(panels):
        r, c = divmod(i, cols)
        x = gap + c * cell_w
        y = 36 + r * cell_h
        canvas.paste(make_label(title, pw, h=label_h, fontsize=14), (x, y))
        canvas.paste(im, (x, y + label_h))
    canvas.save(out, format="PNG", optimize=True)
    print(f"wrote {out}  ({canvas_w}×{canvas_h})")


def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")

    for photo_id in PHOTOS:
        jpeg = INPUT / f"{photo_id}.jpg"
        photo_work = WORK / photo_id
        photo_work.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {photo_id}.jpg ===")
        raw = render_raw(jpeg)

        # Pre-render the σ=0.81 sharpened input
        sharp_in = photo_work / "_optmodels_sharp081.png"
        if not sharp_in.exists():
            Image.fromarray(wiener_deconvolve(raw, sigma=SIGMA, K=1e-3)).save(sharp_in)
        raw_in = photo_work / "_pre2_raw.png"
        if not raw_in.exists():
            Image.fromarray(raw).save(raw_in)

        outputs: list[tuple[str, Path]] = []

        # 1) Camera baseline
        base = photo_work / "_jpgdown_upscayl-standard-4x.png"
        outputs.append(("camera baseline\n(JPEG↓ → upscayl-std)", base))

        # 2) raw → lite (soft reference)
        raw_lite = photo_work / "_show_raw_lite.png"
        if not raw_lite.exists():
            run_upscayl(raw_in, raw_lite, "upscayl-lite-4x", UPSCAYL_MODELS)
        outputs.append(("raw → upscayl-lite\n(no preproc)", raw_lite))

        # 3+) σ=0.81 → each model
        for label, mname, mdir in MODELS:
            out_png = photo_work / f"_optmodels_s081_{mname}.png"
            if not out_png.exists():
                dt = run_upscayl(sharp_in, out_png, mname, mdir)
                print(f"  σ=0.81 → {mname:30s}  {dt:.2f}s")
            outputs.append((f"σ=0.81 → {label}", out_png))

        # Crop & compose
        (cx_f, cy_f), half = REGIONS[photo_id]
        crops: list[tuple[str, Image.Image]] = []
        for label, p in outputs:
            im = Image.open(p).convert("RGB")
            cx, cy = int(im.width * cx_f), int(im.height * cy_f)
            box = (cx - half, cy - half, cx + half, cy + half)
            crops.append((label, im.crop(box)))

        grid(crops,
             f"{photo_id}.jpg — σ=0.81 × SR model sweep  (crop 288×288 @ 1:1)",
             OUT / f"opt_models_{photo_id}.png", cols=4)


if __name__ == "__main__":
    main()
