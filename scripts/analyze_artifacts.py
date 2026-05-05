"""Targeted artifact analysis: focused crops around problem regions.

For each photo, render multiple small crops side-by-side across all 14 models,
plus the original JPEG and the 192x256 downscale, so we can see exactly what
each model is doing to specific edges / textures / halos.

Run: uv run scripts/analyze_artifacts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "examples/_work"
OUT = REPO / "examples"
INPUT = REPO / "examples/input"

MODELS = [
    "upscayl-standard-4x",
    "upscayl-lite-4x",
    "high-fidelity-4x",
    "remacri-4x",
    "ultramix-balanced-4x",
    "ultrasharp-4x",
    "digital-art-4x",
    "RealESRGAN_General_x4_v3",
    "RealESRGAN_General_WDN_x4_v3",
    "4x_NMKD-Siax_200k",
    "4x_NMKD-Superscale-SP_178000_G",
    "4xNomos8kSC",
    "4xLSDIRplusC",
    "4xHFA2k",
]

# (photo_id, region label, fractional crop center on the 768x1024 output, half-side in output px)
REGIONS = {
    "1777804010136": [
        ("upper-cup-rim",   (0.50, 0.36), 96),   # back/upper cup with halo
        ("front-cup-rim",   (0.55, 0.55), 96),   # front bowl rim — strong curved edge
        ("straw-edge",      (0.78, 0.50), 96),   # straw poking out, fine vertical line
        ("hand-skin",       (0.30, 0.78), 96),   # smooth gradient on the hand
    ],
    "1777733165452": [
        ("wire-mesh",       (0.50, 0.62), 96),   # the wire/grate inside the fire
        ("ember-texture",   (0.50, 0.78), 96),   # bright glowing embers
        ("branch-twigs",    (0.65, 0.10), 96),   # fine branches in upper-right
        ("smooth-sky",      (0.30, 0.30), 96),   # smooth orange/pink area
    ],
    "1771110170401": [
        ("belt-ornament",   (0.45, 0.78), 96),   # circular metal ornament — fine spokes (hallucination test)
        ("vneck-collar",    (0.50, 0.45), 96),   # sharp diagonal V-neck edge
        ("sunglasses",      (0.55, 0.18), 96),   # dark sunglasses against warm face
        ("skirt-chain",     (0.55, 0.92), 96),   # fine chain pattern on skirt
    ],
}


def crop_to_array(path: Path, cx: int, cy: int, half: int) -> np.ndarray:
    with Image.open(path) as im:
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(im.width, cx + half), min(im.height, cy + half)
        return np.array(im.crop((x0, y0, x1, y1)))


def render_region_grid(photo_id: str, region: str, center: tuple[float, float],
                       half: int, out: Path) -> None:
    """One grid: source-JPEG-crop + downscaled-crop + 14 model outputs at 1:1."""
    photo_work = WORK / photo_id
    sample_path = photo_work / f"{MODELS[0]}.png"
    with Image.open(sample_path) as im:
        target_w, target_h = im.size
    cx, cy = int(target_w * center[0]), int(target_h * center[1])

    # Source JPEG (1120x1494) — crop equivalent region
    src_path = INPUT / f"{photo_id}.jpg"
    with Image.open(src_path) as srcim:
        sw, sh = srcim.size
    sx, sy = int(sw * center[0]), int(sh * center[1])
    src_half = int(half * (sw / target_w))  # ~30% larger since JPG is 1.46x bigger
    src_crop = crop_to_array(src_path, sx, sy, src_half)

    # Downscaled 192x256 — crop tiny region
    ds_path = photo_work / "downscaled.png"
    with Image.open(ds_path) as dsim:
        dsw, dsh = dsim.size
    dsx, dsy = int(dsw * center[0]), int(dsh * center[1])
    ds_half = max(8, half // 4)
    ds_crop = crop_to_array(ds_path, dsx, dsy, ds_half)

    # All model crops
    model_crops = [(m, crop_to_array(photo_work / f"{m}.png", cx, cy, half)) for m in MODELS]

    panels: list[tuple[str, np.ndarray]] = [
        (f"src JPEG\n(1120x1494, x4-equiv crop)", src_crop),
        (f"downscaled\n192x256 (x4-equiv crop)", ds_crop),
        *[(m, arr) for m, arr in model_crops],
    ]

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    panel_in = 3.6
    fig, axes = plt.subplots(rows, cols, figsize=(panel_in * cols, panel_in * rows + 0.7))
    fig.suptitle(f"{photo_id}.jpg — {region} (center={center}, ±{half}px on 768x1024)",
                 fontsize=13, y=0.998)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, (label, arr) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        ax.imshow(arr)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(panels), rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for photo_id, regions in REGIONS.items():
        for region_name, center, half in regions:
            out = OUT / f"artifact_{photo_id}_{region_name}.png"
            render_region_grid(photo_id, region_name, center, half, out)


if __name__ == "__main__":
    main()
