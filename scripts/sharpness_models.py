"""Run all available SR models on raw-rendered input, measure sharpness.

Compared:
  baseline = JPEG ↓ Lanczos 192×256 → upscayl-standard-4x
  candidates = raw → inferno 192×256 → <each model>

Sharpness is measured on luminance: ∇-mag mean / p95, Laplacian variance.
Faithfulness is the eyeball pass — high sharpness from a model that invents
stripes/streaks (per saved priorities) doesn't help, so the chart matters more
than the metric.

Run: uv run scripts/sharpness_models.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, WORK, UPSCAYL_BIN, UPSCAYL_MODELS,
    parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

CACHE_MODELS = Path.home() / ".cache/thermal-upscale/models"

# Every model on disk. Charts will rank them by sharpness; user judges
# faithfulness by eye.
MODELS = [
    # Bundled in Upscayl.app
    ("upscayl-standard-4x",      UPSCAYL_MODELS),
    ("upscayl-lite-4x",          UPSCAYL_MODELS),
    ("high-fidelity-4x",         UPSCAYL_MODELS),
    ("remacri-4x",               UPSCAYL_MODELS),
    ("ultramix-balanced-4x",     UPSCAYL_MODELS),
    ("ultrasharp-4x",            UPSCAYL_MODELS),
    ("digital-art-4x",           UPSCAYL_MODELS),
    # Cached community models
    ("4x_NMKD-Siax_200k",                CACHE_MODELS),
    ("4x_NMKD-Superscale-SP_178000_G",   CACHE_MODELS),
    ("4xNomos8kSC",                       CACHE_MODELS),
    ("4xLSDIRplusC",                      CACHE_MODELS),
    ("4xHFA2k",                           CACHE_MODELS),
    ("RealESRGAN_General_x4_v3",          CACHE_MODELS),
    ("RealESRGAN_General_WDN_x4_v3",      CACHE_MODELS),
]

PHOTOS = ["1777733165452.jpg", "1777804010136.jpg", "1771110170401.jpg"]

# fractional center on 768×1024 for the discriminating crop, half-side px
REGIONS = {
    "1777733165452": ("ember-texture",  (0.50, 0.78), 144),
    "1777804010136": ("upper-cup-rim",  (0.50, 0.36), 144),
    "1771110170401": ("belt-ornament",  (0.45, 0.78), 144),
}


# ---------- sharpness metrics ----------

def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float64)


def grad_mag(g: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(g, axis=1, mode="reflect"),
                    sobel(g, axis=0, mode="reflect"))


def lap_var(g: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(convolve2d(g, k, mode="same", boundary="symm").var())


def metrics_of(rgb: np.ndarray) -> tuple[float, float, float]:
    g = luminance(rgb)
    gm = grad_mag(g)
    return float(gm.mean()), float(np.percentile(gm, 95)), lap_var(g)


# ---------- rendering ----------

def render_raw_inferno(jpeg: Path, out_png: Path) -> None:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile=1.0)
    Image.fromarray(colormap_rgb(norm, "inferno")).save(out_png, format="PNG")


def render_jpeg_down(jpeg: Path, out_png: Path) -> None:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS).save(out_png)


def run_upscayl(in_png: Path, out_png: Path, model: str, models_dir: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(in_png), "-o", str(out_png),
         "-m", str(models_dir), "-n", model,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"upscayl-bin {model} failed: "
                           f"{(proc.stderr or proc.stdout).strip()[-300:]}")
    return time.perf_counter() - t0


# ---------- per-photo runner ----------

def process_photo(photo_name: str) -> dict:
    photo_id = Path(photo_name).stem
    photo_work = WORK / photo_id
    photo_work.mkdir(parents=True, exist_ok=True)
    jpeg = INPUT / photo_name

    raw_in  = photo_work / "raw_inferno.png"
    jpg_in  = photo_work / "downscaled.png"
    if not raw_in.exists():
        render_raw_inferno(jpeg, raw_in)
    if not jpg_in.exists():
        render_jpeg_down(jpeg, jpg_in)

    results: dict[str, tuple[Path, float, float, float]] = {}

    base_out = photo_work / "_jpgdown_upscayl-standard-4x.png"
    if not base_out.exists():
        dt = run_upscayl(jpg_in, base_out, "upscayl-standard-4x", UPSCAYL_MODELS)
        print(f"  baseline JPEG↓→upscayl-standard  {dt:.2f}s")
    rgb = np.array(Image.open(base_out).convert("RGB"))
    results["JPEG↓ → upscayl-standard (baseline)"] = (base_out, *metrics_of(rgb))

    for model, mdir in MODELS:
        out_png = photo_work / f"_raw_{model}.png"
        if not out_png.exists():
            try:
                dt = run_upscayl(raw_in, out_png, model, mdir)
                print(f"  raw → {model:36s}  {dt:.2f}s")
            except RuntimeError as e:
                print(f"  raw → {model:36s}  FAILED  {e}")
                continue
        rgb = np.array(Image.open(out_png).convert("RGB"))
        results[f"raw → {model}"] = (out_png, *metrics_of(rgb))

    return results


def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")

    rows = [f"{'photo':14s} {'pipeline':45s} "
            f"{'∇mean':>8s} {'∇p95':>8s} {'lapvar':>9s}",
            "-" * 90]

    for photo_name in PHOTOS:
        photo_id = Path(photo_name).stem
        print(f"\n=== {photo_name} ===")
        results = process_photo(photo_name)
        for label, (_, gmean, gp95, lv) in results.items():
            rows.append(f"{photo_id:14s} {label:45s} "
                        f"{gmean:8.2f} {gp95:8.2f} {lv:9.1f}")
        rows.append("")

    print("\n" + "\n".join(rows))
    (OUT / "sharpness_models.txt").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
