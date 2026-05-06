"""Thermal Master P3 thermal-image upscaler.

Two pipelines:

  default (JPEG-path):
    JPEG → Lanczos↓ to native sensor (192×256) → upscayl SR ×4
    The camera ISP already did bad-pixel correction, NUC, denoise, sharpening.
    Slight halos from ISP edge enhancement remain.

  --from-raw (experimental):
    JPEG (extract APP3 uint16) → percentile-normalize → colormap → Wiener σ
    deconvolution → upscayl SR ×4
    Bypasses ISP halos but exposes sensor fixed-pattern noise (no NUC).

Usage:
    uv run scripts/thermal_upscale.py INPUT.jpg OUTPUT.png
    uv run scripts/thermal_upscale.py INPUT.jpg OUTPUT.png --from-raw
    uv run scripts/thermal_upscale.py INPUT.jpg OUTPUT.png --model upscayl-standard-4x
    uv run scripts/thermal_upscale.py INPUT.jpg OUTPUT.png --from-raw --sigma 0

Defaults:
    --model upscayl-lite-4x      (= RealESRGAN-General-v3; faithful, soft, fast)
    --sigma                      0   in JPEG mode (no preprocessing)
                                 0.81 with --from-raw (PyTorch-optimized to
                                       match camera-JPEG sharpness; FPN visible)
    --cut-percentile 1.0         (raw mode only)
    --colormap inferno           (raw mode only)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    UPSCAYL_BIN, UPSCAYL_MODELS,
    parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

CACHE_MODELS = Path.home() / ".cache/thermal-upscale/models"


# ---------- Wiener deconvolution (closed form) ----------

def _gaussian_2d(sigma: float, size: int) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def wiener_deconvolve(rgb: np.ndarray, sigma: float, K: float = 1e-3) -> np.ndarray:
    """Closed-form Gaussian-PSF Wiener deconvolution, per channel.
    σ is the assumed PSF width; K is the noise-to-signal ratio (smaller = more
    aggressive deconvolution). σ=0.81 matches the camera-JPEG sharpness on
    Thermal Master P3 frames; lower σ = more conservative."""
    if sigma <= 0:
        return rgb
    H, W, _ = rgb.shape
    size = max(7, int(6 * sigma + 1) | 1)
    psf = _gaussian_2d(sigma, size)
    pad = np.zeros((H, W), dtype=np.float64)
    py, px = (H - size) // 2, (W - size) // 2
    pad[py:py + size, px:px + size] = psf
    pad = np.fft.ifftshift(pad)
    PSF_F = np.fft.fft2(pad)
    out = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        ch = rgb[..., c].astype(np.float64)
        ch_F = np.fft.fft2(ch)
        G = np.conj(PSF_F) / (np.abs(PSF_F) ** 2 + K)
        out[..., c] = np.fft.ifft2(ch_F * G).real
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- pipeline branches ----------

def render_jpeg_down(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    return np.array(img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS))


def render_raw(jpeg: Path, cut_percentile: float, colormap: str) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile)
    return colormap_rgb(norm, colormap)


# ---------- model resolution ----------

def find_model_dir(model: str) -> Path:
    """Look for model in bundled Upscayl dir, then user cache."""
    for d in (UPSCAYL_MODELS, CACHE_MODELS):
        if (d / f"{model}.bin").exists() and (d / f"{model}.param").exists():
            return d
    raise SystemExit(
        f"Model '{model}' not found.\n"
        f"  Looked in: {UPSCAYL_MODELS}\n"
        f"             {CACHE_MODELS}\n"
        f"Either place {model}.bin/.param in one of those, or pick a different "
        f"--model.  Available bundled: "
        f"{', '.join(sorted(p.stem for p in UPSCAYL_MODELS.glob('*.bin')))}"
    )


def run_upscayl(in_png: Path, out_png: Path, model: str, mdir: Path) -> float:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(in_png), "-o", str(out_png),
         "-m", str(mdir), "-n", model,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise SystemExit(f"upscayl-bin {model} failed:\n"
                         f"{(proc.stderr or proc.stdout).strip()[-500:]}")
    return time.perf_counter() - t0


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Upscale Thermal Master P3 thermal images by 4× using Upscayl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=Path, help="Thermal Master P3 JPEG")
    p.add_argument("output", type=Path, help="Output PNG (4× upscaled)")
    p.add_argument(
        "--from-raw", action="store_true", dest="from_raw",
        help="Extract APP3 uint16 thermal data instead of Lanczos-downscaling "
             "the camera JPEG. Bypasses ISP halos but exposes sensor FPN. "
             "Default --sigma changes from 0 to 0.81 when this flag is set.")
    p.add_argument(
        "--model", default="upscayl-lite-4x",
        help="SR model name (default: upscayl-lite-4x = RealESRGAN-General-v3).")
    p.add_argument(
        "--sigma", type=float, default=None,
        help="Wiener-deconvolution σ for sharpening preprocessing. "
             "Defaults: 0 (off) in JPEG mode, 0.81 with --from-raw. "
             "0 disables; 0.81 matches camera-JPEG gradient energy.")
    p.add_argument(
        "--K", type=float, default=1e-3,
        help="Wiener noise-to-signal ratio (raw mode). "
             "Larger = less aggressive (default 1e-3).")
    p.add_argument(
        "--cut-percentile", type=float, default=1.0, dest="cut_percentile",
        help="Raw-mode percentile clip (default 1.0 = clip 1st & 99th percentile).")
    p.add_argument(
        "--colormap", default="inferno",
        help="Raw-mode colormap (default: inferno; matplotlib name or 'ironbow').")
    p.add_argument(
        "--keep-intermediate", action="store_true",
        help="Save the 192×256 intermediate next to the output (for inspection).")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    mdir = find_model_dir(args.model)

    # Resolve default sigma based on pipeline
    sigma = args.sigma if args.sigma is not None else (0.81 if args.from_raw else 0.0)

    # Render the 192×256 intermediate
    if args.from_raw:
        try:
            intermediate = render_raw(args.input, args.cut_percentile, args.colormap)
        except ValueError as e:
            raise SystemExit(f"raw mode error: {e}\n"
                             f"(drop --from-raw if this JPEG has no APP3 raw)")
        suffix = f"raw_s{sigma:.2f}_p{args.cut_percentile:.1f}"
    else:
        intermediate = render_jpeg_down(args.input)
        suffix = "jpeg" if sigma == 0 else f"jpeg_s{sigma:.2f}"

    if sigma > 0:
        intermediate = wiener_deconvolve(intermediate, sigma=sigma, K=args.K)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        intermediate_png = Path(td) / "intermediate.png"
        Image.fromarray(intermediate).save(intermediate_png, format="PNG")

        if args.keep_intermediate:
            keep_path = args.output.with_name(
                f"{args.output.stem}_{suffix}_192x256.png")
            Image.fromarray(intermediate).save(keep_path, format="PNG")
            print(f"intermediate: {keep_path}")

        dt = run_upscayl(intermediate_png, args.output, args.model, mdir)

    h, w = intermediate.shape[:2]
    mode = "raw" if args.from_raw else "jpeg"
    print(f"{args.input.name}  →  {args.output.name}  "
          f"({mode} mode, σ={sigma:.2f}, {args.model}, {dt:.2f}s, "
          f"{w}×{h} → {w*4}×{h*4})")


if __name__ == "__main__":
    main()
