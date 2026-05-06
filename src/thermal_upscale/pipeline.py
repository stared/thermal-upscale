"""Runtime pipeline: APP3 raw extraction, Wiener deconvolution, upscayl.

Production code only — no matplotlib, no demo paths. Research scripts in
scripts/ may add their own matplotlib glue on top of this.
"""
from __future__ import annotations

import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

UPSCAYL_BIN = Path("/Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin")
UPSCAYL_MODELS = Path("/Applications/Upscayl.app/Contents/Resources/models")
CACHE_MODELS = Path.home() / ".cache/thermal-upscale/models"


# ---------- raw extraction (APP2 sensor size + APP3 uint16 thermal) ----------

def parse_ijpeg_header(img: Image.Image) -> tuple[int, int]:
    """Parse the APP2 IJPEG header to get the native sensor (width, height)."""
    for marker, payload in img.applist:
        if marker == "APP2" and b"IJPEG" in payload[:10]:
            ir_w = struct.unpack_from("<H", payload, 42)[0]
            ir_h = struct.unpack_from("<H", payload, 44)[0]
            return ir_w, ir_h
    raise ValueError("No APP2 IJPEG header — not a Thermal Master P3 JPEG")


def extract_raw_thermal(img: Image.Image, ir_w: int, ir_h: int) -> np.ndarray:
    """Return uint16 thermal frame in the JPEG's display orientation."""
    app3 = bytearray()
    for marker, payload in img.applist:
        if marker == "APP3":
            app3.extend(payload)
    if not app3:
        raise ValueError("No APP3 segments — JPEG has no raw thermal data")

    frame_bytes = ir_w * ir_h * 2
    if len(app3) < frame_bytes * 2:
        raise ValueError(f"APP3 too small: {len(app3)}B, need {frame_bytes * 2}B")

    first = np.frombuffer(app3[:frame_bytes], dtype="<u2")
    second = np.frombuffer(app3[frame_bytes:frame_bytes * 2], dtype="<u2")

    def paired_ratio(d: np.ndarray) -> float:
        return float(np.mean((d & 0xFF) == ((d >> 8) & 0xFF)))

    thermal = first if paired_ratio(first) < paired_ratio(second) else second
    arr = thermal.reshape(ir_w, ir_h)
    arr_h, arr_w = arr.shape
    img_w, img_h = img.size
    if (arr_h > arr_w) != (img_h > img_w):
        arr = np.rot90(arr, k=-1)
    return arr


def normalize_with_cut(arr: np.ndarray, cut_percentile: float = 1.0) -> np.ndarray:
    """Clip & normalize to [0, 1].
    cut_percentile = 0   → use full min-max range.
    cut_percentile = p   → clip at p-th and (100-p)-th percentile."""
    a = arr.astype(np.float64)
    if cut_percentile <= 0:
        lo, hi = a.min(), a.max()
    else:
        lo, hi = np.percentile(a, [cut_percentile, 100.0 - cut_percentile])
    return np.clip((a - lo) / max(1e-9, hi - lo), 0.0, 1.0)


# ---------- colormaps ----------

def _build_ironbow_lut(n: int = 256) -> np.ndarray:
    control = np.array([
        [0.00,   0,   0,   4], [0.08,   0,   0,  80], [0.15,   0,   0, 132],
        [0.22,  20,   0, 164], [0.30,  64,   0, 196], [0.37, 112,   0, 200],
        [0.42, 152,   0, 184], [0.47, 188,   0, 152], [0.52, 216,   8, 112],
        [0.57, 236,  24,  68], [0.62, 248,  48,  28], [0.67, 255,  80,   0],
        [0.72, 255, 120,   0], [0.77, 255, 160,   0], [0.82, 255, 200,   0],
        [0.87, 255, 232,   0], [0.92, 255, 252,  48], [0.96, 255, 255, 148],
        [1.00, 255, 255, 255],
    ], dtype=np.float64)
    pos, cols = control[:, 0], control[:, 1:]
    x = np.linspace(0, 1, n)
    lut = np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], axis=-1)
    return np.clip(lut, 0, 255).astype(np.uint8)


_IRONBOW = _build_ironbow_lut(256)


def colormap_rgb(norm: np.ndarray, name: str) -> np.ndarray:
    if name == "ironbow":
        idx = np.clip((norm * 255).round().astype(int), 0, 255)
        return _IRONBOW[idx]
    cmap = colormaps[name]
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


# ---------- Wiener deconvolution ----------

def _gaussian_2d(sigma: float, size: int) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def wiener_deconvolve(rgb: np.ndarray, sigma: float, K: float = 1e-3) -> np.ndarray:
    """Closed-form Gaussian-PSF Wiener deconvolution, per channel.
    σ is the assumed PSF width; K is the noise-to-signal ratio (smaller = more
    aggressive). σ=0.81 matches camera-JPEG sharpness on Thermal Master P3."""
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
    """JPEG-path: Lanczos-downscale the camera JPEG to native sensor (192×256)."""
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    return np.array(img.convert("RGB").resize((ir_w, ir_h), Image.Resampling.LANCZOS))


def render_raw(jpeg: Path, cut_percentile: float = 1.0,
               colormap: str = "inferno") -> np.ndarray:
    """Raw-path: extract APP3 uint16 → percentile-normalize → colormap."""
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile)
    return colormap_rgb(norm, colormap)


# ---------- model + upscayl ----------

def find_model_dir(model: str) -> Path:
    """Search bundled Upscayl dir, then user cache, for {model}.bin/.param."""
    for d in (UPSCAYL_MODELS, CACHE_MODELS):
        if (d / f"{model}.bin").exists() and (d / f"{model}.param").exists():
            return d
    bundled = sorted(p.stem for p in UPSCAYL_MODELS.glob("*.bin")) \
        if UPSCAYL_MODELS.exists() else []
    cached = sorted(p.stem for p in CACHE_MODELS.glob("*.bin")) \
        if CACHE_MODELS.exists() else []
    raise SystemExit(
        f"Model '{model}' not found.\n"
        f"  Looked in: {UPSCAYL_MODELS}\n"
        f"             {CACHE_MODELS}\n"
        f"Place {model}.bin/.param in one of those, or pick a different "
        f"--model.\n  Bundled: {', '.join(bundled) or '(dir missing)'}"
        f"\n  Cached:  {', '.join(cached) or '(none)'}"
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
