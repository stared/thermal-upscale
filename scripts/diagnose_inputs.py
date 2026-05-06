"""Compare what's IN the 192×256 inputs (before any SR), at 1:1 native.

For each photo, two crops on regions of interest:
  campfire: upper (twig/branch) region + ember region
  person:   belt-ornament + chains around neck

For each crop, three input variants are shown:
  raw             — pure sensor uint16 → inferno
  raw + Wiener0.81 — the optimized preprocessing
  JPEG↓ (Lanczos)  — what the camera-rendered JPEG looks like at 192×256

If detail visible in JPEG↓ is also visible in raw or raw+Wiener: the SR
needs to do better. If the detail is ONLY in JPEG↓: the camera ISP
fabricated it — raw-mode is fundamentally missing it.

Run: uv run scripts/diagnose_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)
from sharpen_models import wiener_deconvolve  # noqa: E402

# Each photo: list of (crop label, fractional center on 192×256, half-side in px)
INVESTIGATION = {
    "1777733165452": [
        ("upper twigs / branches",     (0.50, 0.20), 50),
        ("ember / wing-edge texture",  (0.50, 0.55), 50),
    ],
    "1771110170401": [
        ("belt-ornament (circular)",    (0.45, 0.78), 50),
        ("chain around neck",           (0.55, 0.27), 50),
    ],
    "1777804010136": [
        ("front cup rim + contents",    (0.55, 0.55), 50),
    ],
}


def render_raw(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    return colormap_rgb(normalize_with_cut(thermal, 1.0), "inferno")


def render_jpeg_down(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    return np.array(img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS))


def render_jpeg_full(jpeg: Path) -> np.ndarray:
    """Full camera JPEG at native resolution, for comparison."""
    return np.array(Image.open(jpeg).convert("RGB"))


def make_label(text: str, w: int, h: int = 50, fontsize: int = 13) -> Image.Image:
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


def make_chart(photo_id: str, raw: np.ndarray, sharp: np.ndarray,
               jpgdown: np.ndarray, jpg_full: np.ndarray,
               regions: list[tuple[str, tuple[float, float], int]]) -> None:
    H, W, _ = raw.shape   # 256×192 typically

    n_rows = len(regions)
    panel_zoom = 4   # nearest-up the 100×100 crops to 400×400 for visibility

    panels_per_row = 4   # raw / raw+σ=0.81 / JPEG↓ / JPEG-full

    cell_pixels = 100 * panel_zoom  # 400 each
    gap = 12
    label_h = 50
    cell_w = cell_pixels + gap
    cell_h = cell_pixels + label_h + gap
    col_titles = ["raw 192×256", "raw + Wiener σ=0.81 192×256",
                  "JPEG↓ 192×256 (Lanczos)", "JPEG full 1120×1494"]

    canvas_w = panels_per_row * cell_w + gap
    canvas_h = 36 + n_rows * cell_h + 30 * n_rows + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(make_label(f"{photo_id}.jpg — does the detail exist in the raw?  "
                            f"(crops 100×100 zoomed {panel_zoom}× nearest)",
                            canvas_w, h=32, fontsize=18), (0, 2))

    y0 = 36
    for row, (region_label, (cx_f, cy_f), half) in enumerate(regions):
        # Title strip with region name
        canvas.paste(make_label(f"crop: {region_label}",
                                canvas_w, h=24, fontsize=14), (0, y0))
        # Compute box on 192×256 (raw / sharp / jpgdown all share size)
        cx, cy = int(W * cx_f), int(H * cy_f)
        box = (max(0, cx - half), max(0, cy - half),
               min(W, cx + half), min(H, cy + half))
        # And matching box on the full JPEG (image is in image-space, not raw-space —
        # use same fractional center)
        Hf, Wf, _ = jpg_full.shape
        cxf, cyf = int(Wf * cx_f), int(Hf * cy_f)
        # Scale half so it covers the same fraction of image as the 100×100 raw crop
        half_full = int(half * Hf / H)
        box_full = (max(0, cxf - half_full), max(0, cyf - half_full),
                    min(Wf, cxf + half_full), min(Hf, cyf + half_full))

        sources = [raw, sharp, jpgdown]
        crops = []
        for src in sources:
            cr = Image.fromarray(src).crop(box).resize(
                (cell_pixels, cell_pixels), Image.Resampling.NEAREST)
            crops.append(cr)
        # JPEG-full crop: scaled with Lanczos to match cell size (it's already a high-res image)
        cr_full = Image.fromarray(jpg_full).crop(box_full).resize(
            (cell_pixels, cell_pixels), Image.Resampling.LANCZOS)
        crops.append(cr_full)

        y_panels = y0 + 24
        x = gap
        for i, im in enumerate(crops):
            canvas.paste(make_label(col_titles[i], cell_pixels, h=label_h, fontsize=13),
                         (x, y_panels))
            canvas.paste(im, (x, y_panels + label_h))
            x += cell_w

        y0 += cell_h + 24

    out = OUT / f"diag_inputs_{photo_id}.png"
    canvas.save(out, format="PNG", optimize=True)
    print(f"wrote {out}  ({canvas_w}×{canvas_h})")


def main() -> None:
    for photo_id, regions in INVESTIGATION.items():
        jpeg = INPUT / f"{photo_id}.jpg"
        raw = render_raw(jpeg)
        sharp = wiener_deconvolve(raw, sigma=0.81, K=1e-3)
        jpgdown = render_jpeg_down(jpeg)
        jpgfull = render_jpeg_full(jpeg)
        make_chart(photo_id, raw, sharp, jpgdown, jpgfull, regions)


if __name__ == "__main__":
    main()
