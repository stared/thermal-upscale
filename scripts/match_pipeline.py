"""Histogram-only Ansätze for a T → RGB function fit to the JPEG aesthetic.

NO pixel-by-pixel pairing between raw and JPEG. All fits use histograms or
rank-rank correspondence only — so the camera's halos and pixel-level
sharpening cannot leak into the fit.

Per photo, compares:
  TARGET   JPEG → inverse ironbow LUT → inferno   (the look we want to match)
  A. CDF-MATCH       rank-rank into target CDF, apply inferno     (non-parametric, exact match)
  B. RANK-LUT        sort raw and JPEG independently, pair by rank, bin → RGB LUT
  C. MONO-SPLINE     N-knot monotonic piecewise-linear T→T_norm, scipy-optimized → inferno

Score: EMD between luminance histograms (256 bins) of output vs target.

Run: uv run scripts/match_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal,
    jpeg_to_normalized_temp, colormap_rgb,
)

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]
COLORMAP = "inferno"
N_BINS = 256


# ---------- helpers ----------

def jpeg_norm_192x256(jpeg: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (norm_temp_192x256, jpeg_rgb_192x256). Both Lanczos-downscaled.
    Note: spatial alignment is *not* used by any fitting step — only by the
    inverse-LUT call to get a per-pixel temp for sorting purposes."""
    img = Image.open(jpeg)
    rgb_full = np.array(img.convert("RGB"))
    norm_full = jpeg_to_normalized_temp(rgb_full)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    norm = np.array(
        Image.fromarray((norm_full * 255).astype(np.uint8))
        .resize((ir_w, ir_h), Image.Resampling.LANCZOS)
    ).astype(np.float64) / 255.0
    rgb = np.array(
        img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS)
    )
    return norm, rgb


def raw_thermal(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    return extract_raw_thermal(img, ir_w, ir_h)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


def hist(arr: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(arr, bins=N_BINS, range=(0.0, 1.0), density=True)
    return h / max(1e-9, h.sum())


def emd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))


# ---------- ansätze (histogram-only) ----------

def cdf_match(raw: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    """A. Non-parametric CDF/histogram match → inferno.
    For each raw pixel, find its rank within raw, look up target value at same rank.
    Uses rank statistics only — no spatial pairing."""
    flat = raw.flatten().astype(np.float64)
    ranks = np.argsort(np.argsort(flat))
    target_sorted = np.sort(target_norm.flatten())
    matched = target_sorted[ranks].reshape(raw.shape)
    return colormap_rgb(matched, COLORMAP)


def rank_lut(raw: np.ndarray, jpeg_rgb: np.ndarray, target_norm: np.ndarray,
             n: int = 256) -> np.ndarray:
    """B. Histogram-only empirical T→RGB LUT.
    1. Sort raw values (asc).
    2. Sort JPEG pixels by their derived normalized temp (asc) — independently of raw.
    3. Pair by RANK (not by pixel position). Bin into n quantile bins.
    4. Per bin, average the JPEG RGB values at the matching ranks → 256-entry LUT.
    5. Apply LUT to raw via raw's own ranks.

    No flat_rgb[raw_order] anywhere — the two sortings are independent so the
    JPEG's halos (which sit on edge pixels at specific spatial positions) cannot
    propagate into the LUT."""
    flat_raw = raw.flatten().astype(np.float64)
    flat_target_norm = target_norm.flatten()
    flat_rgb = jpeg_rgb.reshape(-1, 3).astype(np.float64)

    target_order = np.argsort(flat_target_norm)              # by JPEG-derived temp
    sorted_jpeg_rgb_by_target_rank = flat_rgb[target_order]  # ranked JPEG colors

    n_pixels = len(flat_raw)
    edges = np.linspace(0, n_pixels, n + 1, dtype=int)
    lut = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        s, e = edges[i], edges[i + 1]
        if e > s:
            lut[i] = sorted_jpeg_rgb_by_target_rank[s:e].mean(axis=0)
        elif i > 0:
            lut[i] = lut[i - 1]
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    raw_ranks = np.argsort(np.argsort(flat_raw))
    bin_idx = np.clip(raw_ranks * n // n_pixels, 0, n - 1)
    return lut[bin_idx].reshape(*raw.shape, 3)


def mono_spline(raw: np.ndarray, target_norm: np.ndarray, n_knots: int = 8
                ) -> tuple[np.ndarray, np.ndarray]:
    """C. Monotonic piecewise-linear T→T_norm with n_knots, scipy-optimized.
    Loss: EMD between histogram of mapped values and target histogram.

    Parameterize Δy ≥ 0 increments at each knot, normalize so total = 1.
    Spline domain is the raw value's own min..max."""
    flat_raw = raw.flatten().astype(np.float64)
    raw_min, raw_max = float(flat_raw.min()), float(flat_raw.max())
    knot_x = np.linspace(raw_min, raw_max, n_knots)
    target_h = hist(target_norm)

    def to_knot_y(theta: np.ndarray) -> np.ndarray:
        d = np.exp(theta)                       # positive increments
        y = np.concatenate([[0.0], np.cumsum(d)])
        return y / y[-1]                        # normalize to [0, 1]

    def loss(theta: np.ndarray) -> float:
        knot_y = to_knot_y(theta)
        # Map: extra knot at the very start so spline has n_knots+1 ys.
        x = np.linspace(raw_min, raw_max, n_knots + 1)
        mapped = np.interp(flat_raw, x, knot_y)
        return emd(hist(mapped), target_h)

    theta0 = np.zeros(n_knots)                  # uniform increments → identity
    res = minimize(loss, theta0, method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    knot_y = to_knot_y(res.x)
    x = np.linspace(raw_min, raw_max, n_knots + 1)
    mapped = np.interp(flat_raw, x, knot_y).reshape(raw.shape)
    return colormap_rgb(mapped, COLORMAP), knot_y


# ---------- main ----------

def main() -> None:
    fig, axes = plt.subplots(len(PHOTOS), 5, figsize=(20, 4.4 * len(PHOTOS)))
    bins = np.linspace(0, 1, N_BINS + 1)[:-1]
    summary: list[tuple[str, float, float, float]] = []

    for row, photo_name in enumerate(PHOTOS):
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        target_norm, jpeg_rgb = jpeg_norm_192x256(jpeg)
        target_inferno = colormap_rgb(target_norm, COLORMAP)
        target_lum_h = hist(luminance(target_inferno))

        thermal = raw_thermal(jpeg)

        out_a = cdf_match(thermal, target_norm)
        out_b = rank_lut(thermal, jpeg_rgb, target_norm, n=256)
        out_c, _ = mono_spline(thermal, target_norm, n_knots=8)

        d_a = emd(hist(luminance(out_a)), target_lum_h)
        d_b = emd(hist(luminance(out_b)), target_lum_h)
        d_c = emd(hist(luminance(out_c)), target_lum_h)
        summary.append((photo_id, d_a, d_b, d_c))

        # col 0: histograms
        ax = axes[row, 0]
        ax.plot(bins, target_lum_h, lw=2.2, color="black", label="TARGET")
        ax.plot(bins, hist(luminance(out_a)), lw=1.4, color="tab:red",
                label=f"A. CDF-match            EMD={d_a:.3f}")
        ax.plot(bins, hist(luminance(out_b)), lw=1.4, color="tab:green",
                label=f"B. rank-LUT             EMD={d_b:.3f}")
        ax.plot(bins, hist(luminance(out_c)), lw=1.4, color="tab:blue",
                label=f"C. mono-spline (8 knots) EMD={d_c:.3f}")
        ax.set_title(f"{photo_id} — luminance histograms")
        ax.set_xlabel("luminance"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        for col, (label, rgb) in enumerate([
            ("TARGET (JPEG → inferno)", target_inferno),
            ("A. CDF-match → inferno", out_a),
            ("B. rank-LUT (camera-style)", out_b),
            ("C. mono-spline → inferno", out_c),
        ], start=1):
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.suptitle("Histogram-only Ansätze: T → RGB on raw, fit to JPEG distribution",
                 fontsize=13)
    plt.tight_layout()
    out = OUT / "match_pipeline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("\nEMD vs target luminance histogram:")
    print(f"  {'photo':18s}  A.CDF   B.rankLUT  C.spline")
    for pid, da, db, dc in summary:
        print(f"  {pid:18s}  {da:.3f}    {db:.3f}    {dc:.3f}")


if __name__ == "__main__":
    main()
