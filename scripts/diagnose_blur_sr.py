"""Sharpness comparison at 768×1024 — does SR close the raw vs JPEG↓ gap?

Reads existing upscayl outputs from examples/_work/<photo>/ and reports the
same gradient-magnitude / Laplacian-variance metrics at the upscaled resolution.
This tells us whether the softer raw input stays soft through SR, or whether
SR adds enough detail to match (or even exceed) the JPEG-path output.

Run: uv run scripts/diagnose_blur_sr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import OUT  # noqa: E402

WORK = OUT.parent / "examples/_work"

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]
# (label, filename relative to <photo> work dir)
PAIRS = [
    ("JPEG↓ → SR (upscayl-std)", "upscayl-standard-4x.png"),
    ("raw  → SR (upscayl-std)", "raw_inferno_upscayl.png"),
]


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float64)


def grad_mag(g: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(g, axis=1, mode="reflect"),
                    sobel(g, axis=0, mode="reflect"))


def lap_var(g: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(convolve2d(g, k, mode="same", boundary="symm").var())


def main() -> None:
    print(f"{'photo':16s} {'path':32s} {'∇mean':>8s} {'∇p95':>8s} {'lapvar':>9s}")
    print("-" * 78)
    for p in PHOTOS:
        for label, fname in PAIRS:
            path = WORK / p / fname
            if not path.exists():
                print(f"{p:16s} {label:32s}  (missing: {fname})")
                continue
            rgb = np.array(Image.open(path).convert("RGB"))
            g = luminance(rgb)
            gm = grad_mag(g)
            print(f"{p:16s} {label:32s} "
                  f"{gm.mean():8.2f} {np.percentile(gm,95):8.2f} {lap_var(g):9.1f}")
        print()


if __name__ == "__main__":
    main()
