"""Find raw-mode normalization params that match the JPEG's color distribution.

Goal: keep the raw path's clean edges (no halos) but make the overall color
histogram look like the camera-baked JPEG, so the *aesthetic* matches even
though the pixels don't (and shouldn't — JPEG has halos we want gone).

Sweeps a grid over (cut_percentile, gamma); scores each by EMD-style distance
between histograms of normalized temp values; picks the lowest-distance combo.

Run: uv run scripts/match_histogram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal,
    jpeg_to_normalized_temp, colormap_rgb, normalize_with_cut,
)

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]
COLORMAP = "inferno"

# Grid. (cut_percentile, gamma).
CUT_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
GAMMA_GRID = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0]
N_BINS = 64


def jpeg_norm_192x256(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    rgb = np.array(img.convert("RGB"))
    norm_full = jpeg_to_normalized_temp(rgb)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    return np.array(
        Image.fromarray((norm_full * 255).astype(np.uint8))
        .resize((ir_w, ir_h), Image.Resampling.LANCZOS)
    ).astype(np.float64) / 255.0


def raw_thermal(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    return extract_raw_thermal(img, ir_w, ir_h)


def transform(thermal: np.ndarray, cut_pct: float, gamma: float) -> np.ndarray:
    norm = normalize_with_cut(thermal, cut_pct)
    if gamma != 1.0:
        norm = norm ** gamma
    return norm


def hist(arr: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(arr, bins=N_BINS, range=(0.0, 1.0), density=True)
    # Make it a proper PMF over the bins
    return h / max(1e-9, h.sum())


def emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Earth-Mover Distance: L1 of cumulative distributions."""
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


def search(target_hist: np.ndarray, thermal: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    best = (np.inf, 1.0, 1.0, None)
    for cp in CUT_GRID:
        for g in GAMMA_GRID:
            h = hist(transform(thermal, cp, g))
            d = emd_1d(h, target_hist)
            if d < best[0]:
                best = (d, cp, g, h)
    return best  # type: ignore


def main() -> None:
    fig, axes = plt.subplots(len(PHOTOS), 4, figsize=(16, 4.4 * len(PHOTOS)))
    bins = np.linspace(0, 1, N_BINS + 1)[:-1]
    summary: list[tuple[str, float, float, float]] = []

    for row, photo_name in enumerate(PHOTOS):
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        target_norm = jpeg_norm_192x256(jpeg)
        target_h = hist(target_norm)

        thermal = raw_thermal(jpeg)
        default_norm = transform(thermal, 1.0, 1.0)
        default_h = hist(default_norm)
        d_default = emd_1d(default_h, target_h)

        d_best, cp, g, best_h = search(target_h, thermal)
        best_norm = transform(thermal, cp, g)
        summary.append((photo_id, cp, g, d_best))

        # Col 0: histograms
        ax = axes[row, 0]
        ax.plot(bins, target_h, lw=2.2, color="black", label="JPEG (target)")
        ax.plot(bins, default_h, lw=1.4, color="tab:blue", alpha=0.7,
                label=f"raw default (cp=1, g=1) EMD={d_default:.3f}")
        ax.plot(bins, best_h, lw=1.4, color="tab:red",
                label=f"raw best (cp={cp}, g={g}) EMD={d_best:.3f}")
        ax.set_title(f"{photo_id}  —  histogram of normalized temp")
        ax.set_xlabel("normalized temp"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Col 1: JPEG-derived inferno
        axes[row, 1].imshow(colormap_rgb(target_norm, COLORMAP))
        axes[row, 1].set_title("JPEG → inverse-LUT → inferno (target)")
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        # Col 2: raw default
        axes[row, 2].imshow(colormap_rgb(default_norm, COLORMAP))
        axes[row, 2].set_title("raw  cp=1.0  g=1.0 (default)")
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])

        # Col 3: raw best
        axes[row, 3].imshow(colormap_rgb(best_norm, COLORMAP))
        axes[row, 3].set_title(f"raw  cp={cp}  g={g}  (best of grid)")
        axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])

    fig.suptitle(
        f"Histogram matching — raw normalization to JPEG-derived target  "
        f"(grid: cut_percentile ∈ {CUT_GRID}, gamma ∈ {GAMMA_GRID})",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUT / "histogram_match.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("\nbest per photo:")
    for pid, cp, g, d in summary:
        print(f"  {pid:18s} cut_percentile={cp:>4} gamma={g:>4}  EMD={d:.4f}")


if __name__ == "__main__":
    main()
