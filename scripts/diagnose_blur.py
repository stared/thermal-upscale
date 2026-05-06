"""Why does raw-rendered output look blurrier than the camera-downscaled JPEG?

Hypothesis to test: the camera's ISP applies sharpening / edge enhancement when
producing the rendered JPEG.  Even after Lanczos-downscaling back to 192×256,
that sharpening leaves a residue (halos, exaggerated edges) which the eye reads
as "more detail."  The raw is the un-sharpened sensor signal — softer in
absolute terms.

We measure this from each photo's data, not from a guess:
    1) Native shapes — confirm raw and JPEG-down are both at the IJPEG header
       resolution (no hidden up/downsampling).
    2) Per-pixel sharpness proxies:
         - gradient-magnitude (Sobel) mean & 95th percentile
         - Laplacian variance (classic blur metric)
       computed on the LUMINANCE of each rendering (so we factor out colormap).
    3) Radial 2D-FFT power spectrum on a normalized luminance — shows whether
       JPEG-down has more energy at high spatial frequencies than raw.
    4) Visual chart per photo: raw / JPEG-down at 1:1 nearest-up, plus a small
       crop, plus the radial-PSD plot, plus per-edge profile across a sharp
       feature.

Run: uv run scripts/diagnose_blur.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float64)


def gradient_mag(g: np.ndarray) -> np.ndarray:
    gx = sobel(g, axis=1, mode="reflect")
    gy = sobel(g, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def laplacian_var(g: np.ndarray) -> float:
    # 3×3 Laplacian — blur metric (low variance = blurry)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    L = convolve2d(g, k, mode="same", boundary="symm")
    return float(L.var())


def radial_psd(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2D power spectral density."""
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    P = (F.real ** 2 + F.imag ** 2)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    r_max = min(cy, cx)
    psd = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    psd = psd[:r_max]
    freq = np.arange(r_max) / (2 * r_max)  # 0..0.5 cycles/pixel (Nyquist=0.5)
    return freq, psd


def render_raw_lum(jpeg: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)  # uint16, in JPEG orientation
    norm = normalize_with_cut(thermal, cut_percentile=1.0)
    rgb = colormap_rgb(norm, "inferno")
    # luminance of the colormap rendering, for sharpness metrics
    return rgb, luminance(rgb), thermal.shape


def render_jpeg_down(jpeg: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    rgb = np.array(img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS))
    return rgb, luminance(rgb), rgb.shape[:2]


def diagnose(photo: str, ax_row, summary_lines):
    jpeg = INPUT / photo
    raw_rgb, raw_lum, raw_shape = render_raw_lum(jpeg)
    jpg_rgb, jpg_lum, jpg_shape = render_jpeg_down(jpeg)

    assert raw_lum.shape == jpg_lum.shape, (
        f"shape mismatch {raw_lum.shape} vs {jpg_lum.shape} — fix orientation first"
    )

    # Sharpness metrics — compute on each luminance map
    raw_grad = gradient_mag(raw_lum)
    jpg_grad = gradient_mag(jpg_lum)
    raw_lapvar = laplacian_var(raw_lum)
    jpg_lapvar = laplacian_var(jpg_lum)

    # Radial PSD — log-log
    fr, raw_psd = radial_psd(raw_lum)
    fj, jpg_psd = radial_psd(jpg_lum)

    line = (
        f"{photo:24s}  shape raw={raw_shape}, jpg={jpg_shape}\n"
        f"  gradient-mag  mean: raw={raw_grad.mean():6.2f}  jpg={jpg_grad.mean():6.2f}  "
        f"  ratio jpg/raw = {jpg_grad.mean()/raw_grad.mean():.2f}\n"
        f"  gradient-mag  p95 : raw={np.percentile(raw_grad,95):6.2f}  "
        f"jpg={np.percentile(jpg_grad,95):6.2f}  "
        f"ratio = {np.percentile(jpg_grad,95)/np.percentile(raw_grad,95):.2f}\n"
        f"  Laplacian var: raw={raw_lapvar:7.1f}  jpg={jpg_lapvar:7.1f}  "
        f"  ratio = {jpg_lapvar/max(1e-9,raw_lapvar):.2f}\n"
    )
    summary_lines.append(line)
    print(line)

    ax_row[0].imshow(raw_rgb)
    ax_row[0].set_title(f"raw → inferno  ({raw_shape[0]}×{raw_shape[1]})", fontsize=10)
    ax_row[0].set_xticks([]); ax_row[0].set_yticks([])

    ax_row[1].imshow(jpg_rgb)
    ax_row[1].set_title(f"JPEG ↓ Lanczos  ({jpg_shape[0]}×{jpg_shape[1]})", fontsize=10)
    ax_row[1].set_xticks([]); ax_row[1].set_yticks([])

    # Gradient-mag heatmaps — visualize where the "extra sharpness" lives
    vmax = float(max(raw_grad.max(), jpg_grad.max()))
    ax_row[2].imshow(raw_grad, cmap="magma", vmin=0, vmax=vmax)
    ax_row[2].set_title(f"|∇L| raw  mean={raw_grad.mean():.1f}", fontsize=10)
    ax_row[2].set_xticks([]); ax_row[2].set_yticks([])
    ax_row[3].imshow(jpg_grad, cmap="magma", vmin=0, vmax=vmax)
    ax_row[3].set_title(f"|∇L| JPEG↓  mean={jpg_grad.mean():.1f}", fontsize=10)
    ax_row[3].set_xticks([]); ax_row[3].set_yticks([])

    # Radial PSD
    ax = ax_row[4]
    # Skip DC and the very lowest bin — they dominate.
    ax.loglog(fr[1:], raw_psd[1:], lw=1.6, label="raw")
    ax.loglog(fj[1:], jpg_psd[1:], lw=1.6, label="JPEG↓")
    ax.set_xlabel("cycles / pixel", fontsize=9)
    ax.set_ylabel("radial PSD", fontsize=9)
    ax.set_title("radially-avg PSD", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)


def main() -> None:
    n = len(PHOTOS)
    fig, axes = plt.subplots(n, 5, figsize=(22, 4.4 * n))
    if n == 1:
        axes = axes[None, :]
    summary_lines: list[str] = []
    for r, p in enumerate(PHOTOS):
        diagnose(p, axes[r], summary_lines)

    fig.suptitle(
        "raw vs JPEG↓ at sensor resolution — sharpness diagnostics\n"
        "if the JPEG-down has higher gradient-mag, higher Laplacian variance, "
        "and more high-freq PSD,\n"
        "the camera ISP is sharpening before save (and Lanczos-downscaling back "
        "preserves residue).",
        fontsize=12,
    )
    plt.tight_layout()
    out = OUT / "diagnose_blur.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    Path(OUT / "diagnose_blur.txt").write_text("".join(summary_lines))


if __name__ == "__main__":
    main()
