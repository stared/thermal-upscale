"""Stage 1 only: 192×256 inferno inputs from JPEG-path vs raw-path, same chart.

Apples-to-apples: both panels go through the SAME colormap (inferno). The JPEG
path reverses the camera's ironbow LUT first to recover normalized temps, so
the only remaining difference between left and right is whether the source is
the camera-baked JPEG or the APP3 raw uint16.

Run: uv run scripts/show_colormaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal, colormap_rgb,
    jpeg_to_normalized_temp, normalize_with_cut,
)

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]
COLORMAP = "inferno"
CUT_PERCENTILE = 1.0  # for the raw path; JPEG path doesn't need this (camera already normalized)


def from_jpeg(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    rgb = np.array(img.convert("RGB"))
    norm_full = jpeg_to_normalized_temp(rgb)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    norm_small = np.array(
        Image.fromarray((norm_full * 255).astype(np.uint8))
        .resize((ir_w, ir_h), Image.Resampling.LANCZOS)
    ).astype(np.float64) / 255.0
    return colormap_rgb(norm_small, COLORMAP)


def from_raw(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, CUT_PERCENTILE)
    return colormap_rgb(norm, COLORMAP)


def main() -> None:
    for photo_name in PHOTOS:
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        rgb_j = from_jpeg(jpeg)
        rgb_r = from_raw(jpeg)

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        for ax, (label, rgb) in zip(axes, [
            ("JPEG → inverse-LUT → inferno", rgb_j),
            (f"raw → cut_percentile={CUT_PERCENTILE} → inferno", rgb_r),
        ]):
            big = np.array(Image.fromarray(rgb).resize(
                (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.NEAREST))
            ax.imshow(big)
            ax.set_title(label, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{photo_name} — apples-to-apples 192×256 inputs (no upscale)\n"
                     f"both panels colored with {COLORMAP}", fontsize=12)
        plt.tight_layout()
        out = OUT / f"input_jpeg_vs_raw_{photo_id}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
