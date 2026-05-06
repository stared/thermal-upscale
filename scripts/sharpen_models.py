"""Wiener sharpening preprocessing × multiple SR models.

Cross product:
  sharpener ∈ {none, Wiener σ=0.8, Wiener σ=1.0, Wiener σ=1.3}
  model ∈ {upscayl-standard, upscayl-lite (= RealESRGAN-v3), 4xNomos8kSC}

Plus a JPEG↓ → upscayl-standard baseline reference.

Total panels per photo = 4 × 3 + 1 = 13. Rendered as a grid of 1:1 crops.
We're looking for a (sharpener, model) pair that produces sharp edges
WITHOUT halos — i.e. visually beats the original camera JPEG.

Run: uv run scripts/sharpen_models.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).parent))
from from_raw import (  # noqa: E402
    INPUT, OUT, WORK, UPSCAYL_BIN, UPSCAYL_MODELS,
    parse_ijpeg_header, extract_raw_thermal,
    normalize_with_cut, colormap_rgb,
)

CACHE_MODELS = Path.home() / ".cache/thermal-upscale/models"

PHOTOS = ["1777733165452", "1777804010136", "1771110170401"]

# (label, model name, models dir).  Picks: standard (default), lite (= RealESRGAN-v3,
# faithful/soft/fast — user's preferred), 4xNomos8kSC (clean & sharp without micro-contrast).
MODELS = [
    ("upscayl-standard",  "upscayl-standard-4x",  UPSCAYL_MODELS),
    ("upscayl-lite",      "upscayl-lite-4x",      UPSCAYL_MODELS),
    ("4xNomos8kSC",       "4xNomos8kSC",          CACHE_MODELS),
]

# (label, σ).  σ=None = no preprocessing.
SHARPENERS = [
    ("raw",          None),
    ("Wiener σ=0.8", 0.8),
    ("Wiener σ=1.0", 1.0),
    ("Wiener σ=1.3", 1.3),
]

REGIONS = {  # fractional center on 768×1024, half-side
    "1777733165452": ((0.50, 0.78), 144),  # ember-texture
    "1777804010136": ((0.50, 0.36), 144),  # upper-cup-rim
    "1771110170401": ((0.45, 0.78), 144),  # belt-ornament
}


# ---------- raw → 192×256 inferno RGB ----------

def render_raw_inferno(jpeg: Path) -> np.ndarray:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile=1.0)
    return colormap_rgb(norm, "inferno")


def render_jpeg_down(jpeg: Path, out_png: Path) -> None:
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS).save(out_png)


# ---------- Wiener deconvolution (closed form) ----------

def gaussian_2d(sigma: float, size: int) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g1 = np.exp(-x ** 2 / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return np.outer(g1, g1)


def wiener_deconvolve(rgb: np.ndarray, sigma: float, K: float = 1e-3) -> np.ndarray:
    H, W, _ = rgb.shape
    size = max(7, int(6 * sigma + 1) | 1)
    psf = gaussian_2d(sigma, size)
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


# ---------- metrics ----------

def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float64)


def grad_mag(g: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(g, axis=1, mode="reflect"),
                    sobel(g, axis=0, mode="reflect"))


def lap_var(g: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(convolve2d(g, k, mode="same", boundary="symm").var())


def metrics_of(rgb: np.ndarray) -> dict:
    g = luminance(rgb)
    gm = grad_mag(g)
    return {"gmean": float(gm.mean()),
            "gp95": float(np.percentile(gm, 95)),
            "lapvar": lap_var(g)}


# ---------- pipeline ----------

def run_upscayl(in_png: Path, out_png: Path, model: str, mdir: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(in_png), "-o", str(out_png),
         "-m", str(mdir), "-n", model,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"upscayl-bin {model} failed: "
                           f"{(proc.stderr or proc.stdout).strip()[-300:]}")
    return time.perf_counter() - t0


def slug(label: str) -> str:
    return (label.replace(" ", "_").replace("σ", "s").replace("=", "")
                  .replace(".", "p").replace(",", ""))


# ---------- chart ----------

def make_label(text: str, w: int, h: int = 56, fontsize: int = 13) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", fontsize)
    except OSError:
        font = ImageFont.load_default()
    bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.multiline_text(((w - tw) / 2, (h - th) / 2), text,
                     fill="black", font=font, align="center", spacing=2)
    return img


def grid(panels: list[tuple[str, Image.Image]], heading: str, out: Path,
         cols: int = 4) -> None:
    pw, ph = panels[0][1].size
    rows = (len(panels) + cols - 1) // cols
    gap = 8
    label_h = 56
    cell_w = pw + gap
    cell_h = ph + label_h + gap
    canvas_w = cols * cell_w + gap
    canvas_h = 36 + rows * cell_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(make_label(heading, canvas_w, h=32, fontsize=18), (0, 2))
    for i, (title, im) in enumerate(panels):
        r, c = divmod(i, cols)
        x = gap + c * cell_w
        y = 36 + r * cell_h
        canvas.paste(make_label(title, pw, h=label_h, fontsize=13), (x, y))
        canvas.paste(im, (x, y + label_h))
    canvas.save(out, format="PNG", optimize=True)
    print(f"wrote {out}  ({canvas_w}×{canvas_h})")


# ---------- main ----------

def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")

    rows = [f"{'photo':14s} {'pipeline':56s} "
            f"{'∇mean':>8s} {'∇p95':>8s} {'lapvar':>9s}",
            "-" * 100]

    for photo_id in PHOTOS:
        photo_work = WORK / photo_id
        photo_work.mkdir(parents=True, exist_ok=True)
        jpeg = INPUT / f"{photo_id}.jpg"
        print(f"\n=== {photo_id}.jpg ===")

        raw_rgb = render_raw_inferno(jpeg)

        # Pre-render sharpened inputs once
        sharpener_inputs: dict[str, Path] = {}
        for sname, sigma in SHARPENERS:
            sin = photo_work / f"_pre2_{slug(sname)}.png"
            if not sin.exists():
                if sigma is None:
                    Image.fromarray(raw_rgb).save(sin, format="PNG")
                else:
                    Image.fromarray(wiener_deconvolve(raw_rgb, sigma=sigma, K=1e-3)).save(
                        sin, format="PNG")
            sharpener_inputs[sname] = sin

        (cx_f, cy_f), half = REGIONS[photo_id]

        panels: list[tuple[str, Image.Image]] = []

        # JPEG↓ baseline first
        jpg_in = photo_work / "downscaled.png"
        if not jpg_in.exists():
            render_jpeg_down(jpeg, jpg_in)
        base_out = photo_work / "_jpgdown_upscayl-standard-4x.png"
        if not base_out.exists():
            run_upscayl(jpg_in, base_out, "upscayl-standard-4x", UPSCAYL_MODELS)
        sr = np.array(Image.open(base_out).convert("RGB"))
        m = metrics_of(sr)
        rows.append(f"{photo_id:14s} {'JPEG↓ → standard (baseline)':56s} "
                    f"{m['gmean']:8.2f} {m['gp95']:8.2f} {m['lapvar']:9.1f}")
        cx, cy = int(sr.shape[1] * cx_f), int(sr.shape[0] * cy_f)
        box = (cx - half, cy - half, cx + half, cy + half)
        panels.append(("JPEG↓ → standard\n(baseline)", Image.fromarray(sr).crop(box)))

        # Cross product: model × sharpener
        for mlabel, mname, mdir in MODELS:
            for sname, _ in SHARPENERS:
                in_path = sharpener_inputs[sname]
                out_path = photo_work / f"_pre2_{slug(sname)}_{mname}.png"
                if not out_path.exists():
                    dt = run_upscayl(in_path, out_path, mname, mdir)
                    print(f"  {sname:14s} → {mlabel:18s}  {dt:.2f}s")
                sr = np.array(Image.open(out_path).convert("RGB"))
                m = metrics_of(sr)
                rows.append(f"{photo_id:14s} "
                            f"{('raw → '+sname+' → '+mlabel):56s} "
                            f"{m['gmean']:8.2f} {m['gp95']:8.2f} {m['lapvar']:9.1f}")
                cx, cy = int(sr.shape[1] * cx_f), int(sr.shape[0] * cy_f)
                box = (cx - half, cy - half, cx + half, cy + half)
                panels.append((f"{sname}  →  {mlabel}",
                              Image.fromarray(sr).crop(box)))

        rows.append("")
        grid(panels,
             f"{photo_id}.jpg — sharpener × model  (crop {2*half}×{2*half} @ 1:1)",
             OUT / f"sharpen_models_{photo_id}.png", cols=4)

    print("\n" + "\n".join(rows))
    (OUT / "sharpen_models.txt").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
