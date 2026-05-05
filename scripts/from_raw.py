"""Raw-mode pipeline: extract APP3 uint16 → colormap → upscayl-standard-4x.

Compares the JPEG-path (downscale 1120x1494 → 192x256) against feeding the SR
network a clean colormap render of the raw sensor data, with no JPEG halos.
Same single network in both arms; only the input changes.

Hard error if a JPEG has no APP3 raw data (no fallback to the JPEG path).

Run: uv run scripts/from_raw.py
"""
from __future__ import annotations

import struct
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "examples/input"
WORK = REPO / "examples/_work"
OUT = REPO / "examples"

UPSCAYL_BIN = Path("/Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin")
UPSCAYL_MODELS = Path("/Applications/Upscayl.app/Contents/Resources/models")
MODEL = "upscayl-standard-4x"

PHOTOS = ["1777804010136.jpg", "1777733165452.jpg", "1771110170401.jpg"]
# Active colormap. Ironbow and turbo dropped after the colormap-input sweep —
# turbo is "scientific figure" aesthetic, ironbow is camera-default but adds
# nothing diagnostic over inferno. See LAB_NOTEBOOK.md.
COLORMAPS = ["inferno"]

# (photo_id, region label, fractional center on 768x1024, half-side px)
REGIONS = {
    "1777804010136": ("upper-cup-rim", (0.50, 0.36), 96),
    "1777733165452": ("ember-texture", (0.50, 0.78), 96),
    "1771110170401": ("belt-ornament", (0.45, 0.78), 96),
}


# ---------- raw extraction (adapted from vibe-temp-cc/scripts/explore_thermal.py) ----------

def parse_ijpeg_header(img: Image.Image) -> tuple[int, int]:
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
        # Palette half has hi==lo bytes (e.g. 0x4242). Thermal half doesn't.
        return float(np.mean((d & 0xFF) == ((d >> 8) & 0xFF)))

    thermal = first if paired_ratio(first) < paired_ratio(second) else second
    arr = thermal.reshape(ir_w, ir_h)        # numpy shape: (rows=ir_w, cols=ir_h)
    arr_h, arr_w = arr.shape
    img_w, img_h = img.size
    # Match JPEG aspect (portrait/landscape). Only rotate if they differ.
    if (arr_h > arr_w) != (img_h > img_w):
        arr = np.rot90(arr, k=-1)
    return arr


# ---------- colormaps ----------

def build_ironbow_lut(n: int = 256) -> np.ndarray:
    """Reference ironbow palette, (n, 3) uint8. Adapted from vibe-temp-cc."""
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


_IRONBOW = build_ironbow_lut(256)


def colormap_rgb(norm: np.ndarray, name: str) -> np.ndarray:
    if name == "ironbow":
        idx = np.clip((norm * 255).round().astype(int), 0, 255)
        return _IRONBOW[idx]
    cmap = colormaps[name]
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


# ---------- normalization ----------

def normalize_with_cut(arr: np.ndarray, cut_percentile: float = 1.0) -> np.ndarray:
    """Clip & normalize to [0, 1].

    cut_percentile = 0   → no clipping; use full min-max range.
    cut_percentile = p   → clip at the p-th and (100-p)-th percentiles.
                           Default 1.0 → clip at 1st/99th, matching the camera firmware's
                           rough behavior (extreme hot/cold pixels don't dominate).
    """
    a = arr.astype(np.float64)
    if cut_percentile <= 0:
        lo, hi = a.min(), a.max()
    else:
        lo, hi = np.percentile(a, [cut_percentile, 100.0 - cut_percentile])
    return np.clip((a - lo) / max(1e-9, hi - lo), 0.0, 1.0)


# ---------- inverse ironbow LUT (reverse-engineer JPEG palette) ----------
# Adapted from vibe-temp-cc/scripts/explore_thermal.py:158-213. Builds a 64³ 3D
# RGB→ironbow-index LUT in LAB space; gives back the normalized 0-1 temp the
# camera's pseudocolor encoded.

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float64) / 255.0
    lin = np.where(f > 0.04045, ((f + 0.055) / 1.055) ** 2.4, f / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    def fn(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = fn(x), fn(y), fn(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


_BIN = 4
_NB = 256 // _BIN
_LUT3D: np.ndarray | None = None


def _ensure_inverse_lut() -> np.ndarray:
    global _LUT3D
    if _LUT3D is not None:
        return _LUT3D
    lut_lab = _rgb_to_lab(_IRONBOW)
    vals = np.arange(0, 256, _BIN, dtype=np.uint8) + _BIN // 2
    gr, gg, gb = np.meshgrid(vals, vals, vals, indexing="ij")
    grid_lab = _rgb_to_lab(np.stack([gr, gg, gb], axis=-1))
    flat = grid_lab.reshape(-1, 3)
    out = np.zeros(_NB ** 3, dtype=np.uint8)
    for i in range(0, len(flat), 4096):
        chunk = flat[i:i + 4096]
        d = np.sum((chunk[:, None, :] - lut_lab[None, :, :]) ** 2, axis=2)
        out[i:i + 4096] = np.argmin(d, axis=1)
    _LUT3D = out.reshape(_NB, _NB, _NB)
    return _LUT3D


def jpeg_to_normalized_temp(rgb: np.ndarray) -> np.ndarray:
    """Reverse-engineer the camera ironbow LUT: RGB pixels → normalized 0-1 temp."""
    lut3d = _ensure_inverse_lut()
    ri = np.clip(rgb[..., 0].astype(int) // _BIN, 0, _NB - 1)
    gi = np.clip(rgb[..., 1].astype(int) // _BIN, 0, _NB - 1)
    bi = np.clip(rgb[..., 2].astype(int) // _BIN, 0, _NB - 1)
    return lut3d[ri, gi, bi].astype(np.float64) / 255.0


# ---------- pipeline ----------

def render_raw(jpeg: Path, colormap: str, out_png: Path,
               cut_percentile: float = 1.0) -> None:
    """Raw uint16 → normalize_with_cut → colormap → save 192×256 PNG."""
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h)
    norm = normalize_with_cut(thermal, cut_percentile)
    Image.fromarray(colormap_rgb(norm, colormap)).save(out_png, format="PNG")


def render_jpeg(jpeg: Path, colormap: str, out_png: Path,
                ir_size: tuple[int, int] | None = None) -> None:
    """JPEG RGB → inverse ironbow LUT → normalized temp → Lanczos to 192×256
    → re-apply the chosen colormap. Apples-to-apples partner of render_raw().
    No percentile-clip: the camera already normalized when it baked the palette.
    """
    img = Image.open(jpeg)
    if ir_size is None:
        ir_w, ir_h = parse_ijpeg_header(img)
    else:
        ir_w, ir_h = ir_size
    rgb = np.array(img.convert("RGB"))
    norm_full = jpeg_to_normalized_temp(rgb)
    # Match orientation to the JPEG (already portrait), then Lanczos-downscale
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    norm_small = np.array(
        Image.fromarray((norm_full * 255).astype(np.uint8))
        .resize((ir_w, ir_h), Image.Resampling.LANCZOS)
    ).astype(np.float64) / 255.0
    Image.fromarray(colormap_rgb(norm_small, colormap)).save(out_png, format="PNG")


def run_upscayl(in_png: Path, out_png: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(in_png), "-o", str(out_png),
         "-m", str(UPSCAYL_MODELS), "-n", MODEL,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"upscayl-bin failed: {(proc.stderr or proc.stdout).strip()[-300:]}")
    return time.perf_counter() - t0


# ---------- comparison grid ----------

def build_grid(photo_id: str, region: str, center: tuple[float, float], half: int,
               out: Path) -> None:
    """One figure: JPEG-path vs raw-inferno path, on the same chart.
    Top row 192×256 inputs (nearest-up), bottom row 768×1024 outputs cropped @ 1:1."""
    photo_work = WORK / photo_id
    cmap = COLORMAPS[0]  # inferno

    inputs = [
        ("JPEG ↓ Lanczos", photo_work / "downscaled.png"),
        (f"raw → {cmap}", photo_work / f"raw_{cmap}.png"),
    ]
    outputs = [
        ("JPEG → upscayl-std", photo_work / f"{MODEL}.png"),
        (f"raw → {cmap} → upscayl-std", photo_work / f"raw_{cmap}_upscayl.png"),
    ]

    with Image.open(outputs[0][1]) as im:
        tw, th = im.size
    cx, cy = int(tw * center[0]), int(th * center[1])
    crop_box = (max(0, cx - half), max(0, cy - half),
                min(tw, cx + half), min(th, cy + half))

    fig, axes = plt.subplots(2, 2, figsize=(9, 10))
    fig.suptitle(f"{photo_id}.jpg — JPEG-path vs raw-mode (single network: {MODEL})\n"
                 f"top: 192×256 input (nearest-up); bottom: {region} crop "
                 f"{crop_box[2]-crop_box[0]}×{crop_box[3]-crop_box[1]} @ 1:1",
                 fontsize=12)

    for col, (label, path) in enumerate(inputs):
        with Image.open(path) as im:
            up = im.resize((im.width * 4, im.height * 4), Image.Resampling.NEAREST)
        axes[0, col].imshow(np.array(up))
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

    for col, (label, path) in enumerate(outputs):
        with Image.open(path) as im:
            axes[1, col].imshow(np.array(im.crop(crop_box)))
        axes[1, col].set_title(label, fontsize=11)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")
    WORK.mkdir(parents=True, exist_ok=True)

    for photo_name in PHOTOS:
        photo_id = Path(photo_name).stem
        photo_work = WORK / photo_id
        photo_work.mkdir(parents=True, exist_ok=True)
        jpeg = INPUT / photo_name
        print(f"\n=== {photo_name} ===")

        # Render raw inputs + run upscayl on each
        for cmap in COLORMAPS:
            inp = photo_work / f"raw_{cmap}.png"
            out_up = photo_work / f"raw_{cmap}_upscayl.png"
            render_raw(jpeg, cmap, inp, cut_percentile=1.0)
            dt = run_upscayl(inp, out_up)
            print(f"  raw-{cmap:8s}  upscayl {dt:.2f}s")

        # Build comparison grid
        region, center, half = REGIONS[photo_id]
        build_grid(photo_id, region, center, half,
                   OUT / f"raw_vs_jpeg_{photo_id}.png")


if __name__ == "__main__":
    main()
