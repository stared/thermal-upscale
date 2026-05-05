"""Stage 1 only: 192×256 raw → inferno colormap. Min-max vs percentile, same chart.

One figure per photo, two panels (min-max | percentile), inferno only.

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
)

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]
COLORMAP = "inferno"


def render(jpeg: Path, mode: str) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h).astype(np.float64)
    if mode == "minmax":
        lo, hi = thermal.min(), thermal.max()
    else:  # percentile
        lo, hi = np.percentile(thermal, [1.0, 99.0])
    norm = np.clip((thermal - lo) / max(1.0, hi - lo), 0.0, 1.0)
    return colormap_rgb(norm, COLORMAP)


def main() -> None:
    for photo_name in PHOTOS:
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        for ax, (mode, label) in zip(axes, [
            ("minmax", "min → max (per image)"),
            ("percentile", "1st – 99th percentile clip"),
        ]):
            rgb = render(jpeg, mode)
            big = np.array(Image.fromarray(rgb).resize(
                (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.NEAREST))
            ax.imshow(big)
            ax.set_title(label, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{photo_name} — raw 192×256 → inferno (no upscale)\n"
                     f"normalization: min-max vs percentile", fontsize=12)
        plt.tight_layout()
        out = OUT / f"input_norm_compare_{photo_id}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
