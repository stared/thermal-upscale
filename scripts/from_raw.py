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
COLORMAPS = ["ironbow", "inferno", "turbo"]

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


# ---------- pipeline ----------

def render_raw(jpeg: Path, colormap: str, out_png: Path,
               clip_pct: tuple[float, float] = (1.0, 99.0)) -> None:
    """Extract raw → percentile-clipped normalize → colormap → save.

    Percentile clipping (default 1–99) avoids extreme hot/cold pixels (fire core,
    dead pixels) from compressing the rest of the dynamic range. This is what
    the camera firmware effectively does.
    """
    img = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(img)
    thermal = extract_raw_thermal(img, ir_w, ir_h).astype(np.float64)
    lo, hi = np.percentile(thermal, clip_pct)
    norm = np.clip((thermal - lo) / max(1.0, hi - lo), 0.0, 1.0)
    rgb = colormap_rgb(norm, colormap)
    Image.fromarray(rgb).save(out_png, format="PNG")


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
    """Two-row grid: 192x256 inputs (top), 768x1024 outputs cropped @1:1 (bottom).
    Columns: jpeg-path, raw-ironbow, raw-inferno, raw-turbo."""
    photo_work = WORK / photo_id

    inputs = [
        ("JPEG ↓ Lanczos", photo_work / "downscaled.png"),
        *[(f"raw-{c}", photo_work / f"raw_{c}.png") for c in COLORMAPS],
    ]
    outputs = [
        ("JPEG → upscayl-std", photo_work / f"{MODEL}.png"),
        *[(f"raw-{c} → upscayl-std", photo_work / f"raw_{c}_upscayl.png") for c in COLORMAPS],
    ]

    with Image.open(outputs[0][1]) as im:
        tw, th = im.size
    cx, cy = int(tw * center[0]), int(th * center[1])
    crop_box = (max(0, cx - half), max(0, cy - half),
                min(tw, cx + half), min(th, cy + half))

    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(4.0 * cols, 8.5))
    fig.suptitle(f"{photo_id}.jpg — JPEG-path vs raw-mode (single network: {MODEL})\n"
                 f"top: 192×256 input (nearest-up); bottom: {region} crop {crop_box[2]-crop_box[0]}×{crop_box[3]-crop_box[1]} @ 1:1",
                 fontsize=12)

    for col, (label, path) in enumerate(inputs):
        ax = axes[0, col]
        with Image.open(path) as im:
            up = im.resize((im.width * 4, im.height * 4), Image.Resampling.NEAREST)
        ax.imshow(np.array(up))
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for col, (label, path) in enumerate(outputs):
        ax = axes[1, col]
        with Image.open(path) as im:
            ax.imshow(np.array(im.crop(crop_box)))
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

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
            render_raw(jpeg, cmap, inp)
            dt = run_upscayl(inp, out_up)
            print(f"  raw-{cmap:8s}  upscayl {dt:.2f}s")

        # Build comparison grid
        region, center, half = REGIONS[photo_id]
        build_grid(photo_id, region, center, half,
                   OUT / f"raw_vs_jpeg_{photo_id}.png")


if __name__ == "__main__":
    main()
