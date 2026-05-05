"""Stage 1 only: render the 192×256 raw colormap inputs. No upscaling.

For each photo, two figures:
  - norm_input_minmax_<id>.png    — old per-image min-max normalization
  - norm_input_percentile_<id>.png — current 1-99 percentile clipping
Each shows 3 colormaps (ironbow / inferno / turbo).

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
COLORMAPS = ["ironbow", "inferno", "turbo"]


def render(jpeg: Path, colormap: str, mode: str) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h).astype(np.float64)
    if mode == "minmax":
        lo, hi = thermal.min(), thermal.max()
    else:  # percentile
        lo, hi = np.percentile(thermal, [1.0, 99.0])
    norm = np.clip((thermal - lo) / max(1.0, hi - lo), 0.0, 1.0)
    return colormap_rgb(norm, colormap)


def main() -> None:
    for photo_name in PHOTOS:
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        for mode in ("minmax", "percentile"):
            fig, axes = plt.subplots(1, 3, figsize=(11, 5))
            for ax, cmap in zip(axes, COLORMAPS):
                rgb = render(jpeg, cmap, mode)
                # Nearest-upsample 4x just for visibility on screen — same pixels.
                big = np.array(Image.fromarray(rgb).resize(
                    (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.NEAREST))
                ax.imshow(big)
                ax.set_title(cmap, fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
            mode_label = ("per-image min→max"
                          if mode == "minmax"
                          else "1st–99th percentile clip")
            fig.suptitle(f"{photo_name} — raw 192×256 colormap input (no upscale)\n"
                         f"normalization: {mode_label}", fontsize=12)
            plt.tight_layout()
            out = OUT / f"input_{mode}_{photo_id}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
