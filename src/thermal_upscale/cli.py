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

Defaults:
    --model    upscayl-standard-4x  in JPEG mode (default)
               upscayl-lite-4x      with --from-raw  (= RealESRGAN-General-v3;
                                                       faithful, soft, fast)
    --sigma    0                    in JPEG mode (no preprocessing)
               0.81                 with --from-raw  (PyTorch-optimized to
                                                       match camera-JPEG sharpness)
    --cut-percentile 1.0            (raw mode only)
    --colormap inferno              (raw mode only)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image

from .pipeline import (
    find_model_dir,
    render_jpeg_down,
    render_raw,
    run_upscayl,
    wiener_deconvolve,
)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="thermal-upscale",
        description="Upscale Thermal Master P3 thermal images by 4× using Upscayl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=Path, help="Thermal Master P3 JPEG")
    p.add_argument(
        "output", type=Path, nargs="?", default=None,
        help="Output PNG (4× upscaled). Default: <input-stem>_improved.png "
             "next to the input.")
    p.add_argument(
        "--from-raw", action="store_true", dest="from_raw",
        help="Extract APP3 uint16 thermal data instead of Lanczos-downscaling "
             "the camera JPEG. Bypasses ISP halos but exposes sensor FPN. "
             "Default --sigma changes from 0 to 0.81 when this flag is set.")
    p.add_argument(
        "--model", default=None,
        help="SR model name. Defaults: upscayl-standard-4x in JPEG mode, "
             "upscayl-lite-4x (= RealESRGAN-General-v3) with --from-raw.")
    p.add_argument(
        "--sigma", type=float, default=None,
        help="Wiener-deconvolution σ for sharpening preprocessing. "
             "Defaults: 0 (off) in JPEG mode, 0.81 with --from-raw. "
             "0 disables; 0.81 matches camera-JPEG gradient energy.")
    p.add_argument(
        "--K", type=float, default=1e-3,
        help="Wiener noise-to-signal ratio. Larger = less aggressive (default 1e-3).")
    p.add_argument(
        "--cut-percentile", type=float, default=1.0, dest="cut_percentile",
        help="Raw-mode percentile clip (default 1.0 = clip 1st & 99th percentile).")
    p.add_argument(
        "--colormap", default="inferno",
        help="Raw-mode colormap (default: inferno; matplotlib name or 'ironbow').")
    p.add_argument(
        "--keep-intermediate", action="store_true", dest="keep_intermediate",
        help="Save the 192×256 intermediate next to the output (for inspection).")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_improved.png")

    model = args.model if args.model is not None else (
        "upscayl-lite-4x" if args.from_raw else "upscayl-standard-4x")
    mdir = find_model_dir(model)
    sigma = args.sigma if args.sigma is not None else (0.81 if args.from_raw else 0.0)

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

        dt = run_upscayl(intermediate_png, args.output, model, mdir)

    h, w = intermediate.shape[:2]
    mode = "raw" if args.from_raw else "jpeg"
    print(f"{args.input.name}  →  {args.output.name}  "
          f"({mode} mode, σ={sigma:.2f}, {model}, {dt:.2f}s, "
          f"{w}×{h} → {w*4}×{h*4})")


if __name__ == "__main__":
    main()
