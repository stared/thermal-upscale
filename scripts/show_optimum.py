"""Show the σ=0.81 optimum visually, vs σ=1.0 (over-sharp) and JPEG baseline.

Per photo, 5 panels at 1:1 native pixels (288×288 crop):
  0: JPEG↓ → upscayl-standard      (camera baseline)
  1: raw → upscayl-lite             (no preproc; user's preferred clean baseline)
  2: raw → Wiener σ=0.81 → lite     (PyTorch-OPTIMIZED to match JPEG sharpness)
  3: raw → Wiener σ=1.0  → lite     (over-sharp; eyeball pick before optimization)
  4: raw → Wiener σ=1.3  → lite     (clearly over-haloed for reference)
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

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]

REGIONS = {
    "1777733165452": ((0.50, 0.78), 144),
    "1777804010136": ((0.50, 0.36), 144),
    "1771110170401": ((0.45, 0.78), 144),
}

SIGMAS = [None, 0.81, 1.0, 1.3]   # None = no preprocessing


def render_raw(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    return colormap_rgb(normalize_with_cut(thermal, 1.0), "inferno")


def run_upscayl(in_png: Path, out_png: Path, model: str) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN), "-i", str(in_png), "-o", str(out_png),
         "-m", str(UPSCAYL_MODELS), "-n", model, "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"{model}: {(proc.stderr or proc.stdout)[-200:]}")
    return time.perf_counter() - t0


def make_label(text: str, w: int, h: int = 60, fontsize: int = 15) -> Image.Image:
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


def main() -> None:
    for photo_id in PHOTOS:
        jpeg = INPUT / f"{photo_id}.jpg"
        photo_work = WORK / photo_id
        photo_work.mkdir(parents=True, exist_ok=True)
        raw = render_raw(jpeg)

        # Render & SR each variant
        outputs: list[tuple[str, Path]] = []

        # 1) Camera baseline = JPEG↓ → upscayl-standard
        base_path = photo_work / "_jpgdown_upscayl-standard-4x.png"
        outputs.append(("camera baseline\n(JPEG↓ → upscayl-std)", base_path))

        # 2-5) Wiener variants → upscayl-lite
        for sigma in SIGMAS:
            if sigma is None:
                in_path = photo_work / "_pre2_raw.png"
                if not in_path.exists():
                    Image.fromarray(raw).save(in_path)
                tag = "raw"
                label = "raw (no preproc)\n→ upscayl-lite"
            else:
                slug = f"sigma{int(sigma*100):03d}"
                in_path = photo_work / f"_show_{slug}.png"
                if not in_path.exists():
                    Image.fromarray(wiener_deconvolve(raw, sigma=sigma, K=1e-3)).save(in_path)
                tag = slug
                marker = " ← optimum" if abs(sigma - 0.81) < 0.005 else ""
                label = f"Wiener σ={sigma}\n→ upscayl-lite{marker}"

            sr_path = photo_work / f"_show_{tag}_lite.png"
            if not sr_path.exists():
                dt = run_upscayl(in_path, sr_path, "upscayl-lite-4x")
                print(f"  {photo_id}  σ={sigma}  upscayl-lite {dt:.2f}s")
            outputs.append((label, sr_path))

        # Compose chart
        (cx_f, cy_f), half = REGIONS[photo_id]
        crops: list[tuple[str, Image.Image]] = []
        for label, p in outputs:
            im = Image.open(p).convert("RGB")
            cx, cy = int(im.width * cx_f), int(im.height * cy_f)
            box = (cx - half, cy - half, cx + half, cy + half)
            crops.append((label, im.crop(box)))

        pw, ph = crops[0][1].size
        n = len(crops)
        gap = 12
        label_h = 60
        canvas_w = n * (pw + gap) + gap
        canvas_h = 36 + (ph + label_h + gap) + gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        canvas.paste(make_label(f"{photo_id}.jpg — σ_opt=0.81 vs alternatives  "
                                f"(crop {2*half}×{2*half} @ 1:1)",
                                canvas_w, h=32, fontsize=18), (0, 2))
        x = gap
        y = 36
        for label, im in crops:
            canvas.paste(make_label(label, pw, h=label_h, fontsize=14), (x, y))
            canvas.paste(im, (x, y + label_h))
            x += pw + gap

        out = OUT / f"opt_{photo_id}.png"
        canvas.save(out, format="PNG", optimize=True)
        print(f"wrote {out}  ({canvas_w}×{canvas_h})")


if __name__ == "__main__":
    main()
