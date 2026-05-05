"""Compare AI super-resolution models on Thermal Master P3 frames.

Pipeline per photo: input JPEG -> Lanczos-downscale to native sensor (192x256)
                  -> run each model -> 768x1024 outputs
                  -> two grid PNGs (full @ 1:1 pixels, and a tight crop @ 1:1).

Run: uv run scripts/compare_models.py
"""
from __future__ import annotations

import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO / "examples/input"
WORK = REPO / "examples/_work"
OUT_DIR = REPO / "examples"

UPSCAYL_APP = Path("/Applications/Upscayl.app/Contents/Resources")
UPSCAYL_BIN = UPSCAYL_APP / "bin/upscayl-bin"
UPSCAYL_MODELS = UPSCAYL_APP / "models"
CUSTOM_MODELS = Path.home() / ".cache/thermal-upscale/models"

# Active models. Dropped after the 2026-05-05 sweep: ultrasharp-4x, remacri-4x,
# ultramix-balanced-4x, 4x_NMKD-Siax_200k, 4x_NMKD-Superscale-SP_178000_G,
# 4xLSDIRplusC — see LAB_NOTEBOOK.md.
MODELS: list[tuple[str, Path]] = [
    ("upscayl-standard-4x", UPSCAYL_MODELS),
    ("upscayl-lite-4x", UPSCAYL_MODELS),
    ("high-fidelity-4x", UPSCAYL_MODELS),
    ("digital-art-4x", UPSCAYL_MODELS),
    ("RealESRGAN_General_x4_v3", CUSTOM_MODELS),
    ("RealESRGAN_General_WDN_x4_v3", CUSTOM_MODELS),
    ("4xNomos8kSC", CUSTOM_MODELS),
    ("4xHFA2k", CUSTOM_MODELS),
]

# (photo filename, fractional crop center (cx_frac, cy_frac), crop side in output px)
PHOTOS: list[tuple[str, tuple[float, float], int]] = [
    ("1777804010136.jpg", (0.50, 0.55), 384),  # cups/shisha — bowl rim & arm edges
    ("1777733165452.jpg", (0.50, 0.72), 384),  # campfire — wire/embers
    ("1771110170401.jpg", (0.50, 0.78), 384),  # person/dance — belt ornament & skirt detail
]


@dataclass
class Result:
    key: str
    path: Path
    seconds: float
    error: str | None = None


def parse_ijpeg_header(img: Image.Image) -> tuple[int, int]:
    """Read (ir_width, ir_height) from APP2 IJPEG segment. Adapted from vibe-temp-cc."""
    for marker, payload in img.applist:
        if marker == "APP2" and b"IJPEG" in payload[:10]:
            ir_w = struct.unpack_from("<H", payload, 42)[0]
            ir_h = struct.unpack_from("<H", payload, 44)[0]
            return ir_w, ir_h
    raise ValueError("No APP2 IJPEG header found — is this a Thermal Master P3 JPEG?")


def downscale_native(jpeg: Path, out: Path) -> tuple[int, int]:
    raw = Image.open(jpeg)
    ir_w, ir_h = parse_ijpeg_header(raw)
    img = raw.convert("RGB")
    if (img.height > img.width) != (ir_h > ir_w):
        ir_w, ir_h = ir_h, ir_w
    img.resize((ir_w, ir_h), Image.Resampling.LANCZOS).save(out, format="PNG")
    return ir_w, ir_h


def run_upscayl(input_png: Path, output_png: Path, model: str, models_dir: Path) -> Result:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(UPSCAYL_BIN),
         "-i", str(input_png), "-o", str(output_png),
         "-m", str(models_dir), "-n", model,
         "-s", "4", "-f", "png"],
        capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not output_png.exists():
        return Result(model, output_png, dt, error=(proc.stderr or proc.stdout).strip()[-300:])
    return Result(model, output_png, dt)


# ---------- grid renderers ----------

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def grid_full_pil(results: list[Result], cols: int, out: Path, title: str) -> None:
    """Compose 1:1-pixel panels with PIL (no matplotlib resampling). Each panel = full output."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return
    with Image.open(ok[0].path) as im:
        pw, ph = im.size
    rows = (len(ok) + cols - 1) // cols

    label_h = 36
    pad = 8
    title_h = 50
    cell_w = pw + pad
    cell_h = ph + label_h + pad
    canvas_w = cell_w * cols + pad
    canvas_h = title_h + cell_h * rows + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad + 6), title, font=_font(28), fill=(230, 230, 230))

    label_font = _font(20)
    for i, r in enumerate(ok):
        col, row = i % cols, i // cols
        x = pad + col * cell_w
        y = title_h + row * cell_h
        with Image.open(r.path) as im:
            canvas.paste(im, (x, y))
        draw.rectangle([x, y + ph, x + pw, y + ph + label_h], fill=(40, 40, 40))
        draw.text((x + 8, y + ph + 6), f"{r.key}  ({r.seconds:.2f}s)",
                  font=label_font, fill=(255, 255, 255))

    canvas.save(out, format="PNG", optimize=False)
    print(f"wrote {out} ({canvas_w}x{canvas_h})")


def grid_crop_mpl(results: list[Result], crop: tuple[int, int, int, int], cols: int,
                  out: Path, title: str, panel_h_in: float = 4.5) -> None:
    """Crop grid via matplotlib — small enough that mpl rendering is fine."""
    ok = [r for r in results if r.error is None]
    n = len(ok)
    rows = (n + cols - 1) // cols
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    aspect = cw / ch

    fig, axes = plt.subplots(rows, cols, figsize=(panel_h_in * aspect * cols,
                                                  panel_h_in * rows + 0.6))
    fig.suptitle(title, fontsize=14, y=0.995)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, r in enumerate(ok):
        ax = axes[i // cols, i % cols]
        with Image.open(r.path) as im:
            ax.imshow(np.array(im.crop(crop)))
        ax.set_title(f"{r.key}\n{r.seconds:.2f}s", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------- per-photo pipeline ----------

def process_photo(jpeg: Path, crop_center: tuple[float, float], crop_side: int) -> None:
    photo_id = jpeg.stem
    photo_work = WORK / photo_id
    photo_work.mkdir(parents=True, exist_ok=True)

    downscaled = photo_work / "downscaled.png"
    ir_w, ir_h = downscale_native(jpeg, downscaled)
    target_w, target_h = ir_w * 4, ir_h * 4
    print(f"\n=== {jpeg.name} ===")
    print(f"downscaled {ir_w}x{ir_h}, upscaled target {target_w}x{target_h}")

    results: list[Result] = []
    for key, models_dir in MODELS:
        out = photo_work / f"{key}.png"
        r = run_upscayl(downscaled, out, key, models_dir)
        msg = f"ERR {r.error}" if r.error else f"{r.seconds:6.2f}s"
        print(f"  {key:34s} {msg}")
        results.append(r)

    failed = [r for r in results if r.error]
    print(f"  {len(results) - len(failed)}/{len(results)} ok, {len(failed)} failed")

    cx_f, cy_f = crop_center
    cx, cy = int(target_w * cx_f), int(target_h * cy_f)
    half = crop_side // 2
    crop = (max(0, cx - half), max(0, cy - half),
            min(target_w, cx + half), min(target_h, cy + half))

    grid_full_pil(results, cols=4, out=OUT_DIR / f"comparison_full_{photo_id}.png",
                  title=f"{jpeg.name} — full {target_w}x{target_h} per panel (1:1 pixels)")
    grid_crop_mpl(results, crop=crop, cols=3,
                  out=OUT_DIR / f"comparison_crop_{photo_id}.png",
                  title=f"{jpeg.name} — {crop[2] - crop[0]}x{crop[3] - crop[1]} crop @ 1:1")


def main() -> None:
    if not UPSCAYL_BIN.exists():
        raise SystemExit(f"upscayl-bin not found at {UPSCAYL_BIN}")
    WORK.mkdir(parents=True, exist_ok=True)

    for photo_name, crop_center, crop_side in PHOTOS:
        photo = INPUT_DIR / photo_name
        if not photo.exists():
            print(f"skip missing {photo}")
            continue
        process_photo(photo, crop_center, crop_side)


if __name__ == "__main__":
    main()
