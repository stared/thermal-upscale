"""Several Ansätze for a T → RGB function that gives JPEG-like aesthetic
on raw data, without halos.

Compares (per photo):
  TARGET   JPEG-RGB → inverse ironbow LUT → inferno    (the look we want to match)
  A. CDF-MATCH    raw temp → CDF-matched to target → inferno      (non-parametric)
  B. EMPIRICAL    raw temp → quantile-bin-averaged JPEG RGB       (direct T → RGB LUT)
  C. PARAM        raw temp → (cut_percentile, gamma) → inferno     (2-param baseline)

Score: EMD between luminance histograms (256 bins) of output vs target.
Each approach has zero hand-tuned params at runtime — A and B are derived
from the data; C grid-searches its 2 params.

Run: uv run scripts/match_pipeline.py
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
N_BINS = 256


# ---------- helpers ----------

def jpeg_norm_192x256(jpeg: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (norm_temp_192x256, jpeg_rgb_192x256). Both Lanczos-downscaled."""
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


# ---------- ansätze ----------

def cdf_match(raw: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    """A. Non-parametric: re-map raw temps so their CDF matches target_norm's,
    then apply inferno. Histograms match exactly by construction."""
    flat = raw.flatten().astype(np.float64)
    ranks = np.argsort(np.argsort(flat))                  # 0..N-1
    target_sorted = np.sort(target_norm.flatten())
    matched = target_sorted[ranks].reshape(raw.shape)
    return colormap_rgb(matched, COLORMAP)


def empirical_lut(raw: np.ndarray, jpeg_rgb: np.ndarray, n: int = 256) -> np.ndarray:
    """B. Direct T→RGB LUT: bin raw values into n quantile bins; for each bin,
    average the JPEG RGB at those same pixels. Result is the camera's effective
    LUT for THIS image (modulo halos, which average out across pixels at the
    same temp). Apply this LUT to the raw temps."""
    flat_raw = raw.flatten().astype(np.float64)
    flat_rgb = jpeg_rgb.reshape(-1, 3).astype(np.float64)
    order = np.argsort(flat_raw)
    sorted_rgb = flat_rgb[order]

    n_pixels = len(flat_raw)
    edges = np.linspace(0, n_pixels, n + 1, dtype=int)
    lut = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        s, e = edges[i], edges[i + 1]
        if e > s:
            lut[i] = sorted_rgb[s:e].mean(axis=0)
        elif i > 0:
            lut[i] = lut[i - 1]
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Map each raw pixel to its quantile bin.
    ranks = np.argsort(np.argsort(flat_raw))               # 0..N-1
    bin_idx = np.clip((ranks * n // n_pixels), 0, n - 1)
    return lut[bin_idx].reshape(*raw.shape, 3)


def param_best(raw: np.ndarray, target_h: np.ndarray) -> tuple[np.ndarray, float, float]:
    """C. 2-param baseline. Grid search (cut_percentile, gamma) → minimize EMD."""
    cuts = [0.0, 0.5, 1.0, 2.0, 5.0]
    gammas = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0]
    best = (np.inf, 1.0, 1.0, None)
    for cp in cuts:
        for g in gammas:
            n = normalize_with_cut(raw, cp) ** g
            d = emd(hist(n), target_h)
            if d < best[0]:
                best = (d, cp, g, n)
    _, cp, g, n = best
    return colormap_rgb(n, COLORMAP), cp, g


# ---------- main ----------

def main() -> None:
    fig, axes = plt.subplots(len(PHOTOS), 5, figsize=(20, 4.4 * len(PHOTOS)))
    bins = np.linspace(0, 1, N_BINS + 1)[:-1]
    summary: list[tuple[str, float, float, float, float, float]] = []

    for row, photo_name in enumerate(PHOTOS):
        photo_id = Path(photo_name).stem
        jpeg = INPUT / photo_name

        target_norm, jpeg_rgb = jpeg_norm_192x256(jpeg)
        target_inferno = colormap_rgb(target_norm, COLORMAP)
        target_lum_h = hist(luminance(target_inferno))

        thermal = raw_thermal(jpeg)

        # A. CDF-match
        out_a = cdf_match(thermal, target_norm)
        d_a = emd(hist(luminance(out_a)), target_lum_h)

        # B. Empirical LUT
        out_b = empirical_lut(thermal, jpeg_rgb)
        d_b = emd(hist(luminance(out_b)), target_lum_h)

        # C. Parametric best
        out_c, cp_c, g_c = param_best(thermal, hist(target_norm))
        d_c = emd(hist(luminance(out_c)), target_lum_h)

        # Default baseline for comparison
        default_norm = normalize_with_cut(thermal, 1.0)
        default_inferno = colormap_rgb(default_norm, COLORMAP)
        d_d = emd(hist(luminance(default_inferno)), target_lum_h)

        summary.append((photo_id, d_a, d_b, d_c, cp_c, g_c))

        # col 0: luminance histograms
        ax = axes[row, 0]
        ax.plot(bins, target_lum_h, lw=2.2, color="black", label="TARGET (JPEG inferno)")
        ax.plot(bins, hist(luminance(default_inferno)), lw=1.2, color="gray", alpha=0.7,
                label=f"raw default cp=1 g=1   EMD={d_d:.2f}")
        ax.plot(bins, hist(luminance(out_a)), lw=1.4, color="tab:red",
                label=f"A. CDF-match              EMD={d_a:.2f}")
        ax.plot(bins, hist(luminance(out_b)), lw=1.4, color="tab:green",
                label=f"B. empirical LUT         EMD={d_b:.2f}")
        ax.plot(bins, hist(luminance(out_c)), lw=1.4, color="tab:blue",
                label=f"C. param cp={cp_c} g={g_c}  EMD={d_c:.2f}")
        ax.set_title(f"{photo_id}  —  luminance histograms")
        ax.set_xlabel("luminance"); ax.set_ylabel("density")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # cols 1-4: visuals
        for col, (label, rgb) in enumerate([
            ("TARGET (JPEG → inferno)", target_inferno),
            ("A. CDF-match → inferno", out_a),
            ("B. empirical T→RGB LUT", out_b),
            (f"C. param cp={cp_c} g={g_c}", out_c),
        ], start=1):
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.suptitle("Ansätze for raw → RGB matching JPEG-derived target distribution"
                 " (luminance EMD lower = closer match)", fontsize=13)
    plt.tight_layout()
    out = OUT / "match_pipeline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("\nEMD vs target (luminance histogram):")
    print(f"  {'photo':18s}  A.CDF   B.LUT   C.param  (cp, g)")
    for pid, d_a, d_b, d_c, cp, g in summary:
        print(f"  {pid:18s}  {d_a:.3f}  {d_b:.3f}  {d_c:.3f}   ({cp}, {g})")


if __name__ == "__main__":
    main()
