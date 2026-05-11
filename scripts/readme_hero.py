"""Build README figures.

  readme_hero.png   — full view, 2 photos × 4 variants.
  readme_crops.png  — 1:1 crop of the canonical discriminating region per photo
                      (cups: upper-cup-rim; person: belt-ornament), same 4 variants.

Columns: original camera JPEG, upscayl-standard-4x, RealESRGAN_General_x4_v3 (GAN),
digital-art-4x. Rows: cups, person.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
INPUT = EXAMPLES / "input"
WORK = EXAMPLES / "_work"

PHOTOS = [
    # (id, row label, region label, fractional center on 768×1024, half-side px on SR canvas)
    ("1777804010136", "cups",   "upper-cup-rim", (0.50, 0.36), 144),
    ("1771110170401", "person", "belt-ornament", (0.45, 0.78), 144),
]

COLS = [
    ("Camera JPEG",           lambda pid: INPUT / f"{pid}.jpg"),
    ("upscayl-standard-4x",   lambda pid: WORK / pid / "upscayl-standard-4x.png"),
    ("RealESRGAN-General-v3", lambda pid: WORK / pid / "RealESRGAN_General_x4_v3.png"),
    ("digital-art-4x",        lambda pid: WORK / pid / "digital-art-4x.png"),
]

SR_W, SR_H = 768, 1024


def render(rows, get_image, out_path, *, dpi=110, panel_w=3.0, panel_h=4.0):
    fig, axes = plt.subplots(
        len(rows), len(COLS),
        figsize=(panel_w * len(COLS), panel_h * len(rows)),
        constrained_layout=True,
    )
    if len(rows) == 1:
        axes = [axes]

    for r, row in enumerate(rows):
        pid = row[0]
        row_label = row[1]
        for c, (title, path_fn) in enumerate(COLS):
            ax = axes[r][c]
            ax.imshow(get_image(row, c, path_fn(pid)))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def crop_box(cx_frac, cy_frac, half, w, h):
    cx, cy = cx_frac * w, cy_frac * h
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(w, int(cx + half)), min(h, int(cy + half)))


def main():
    # Full view.
    def full_get(_row, _c, path):
        return Image.open(path)
    render(PHOTOS, full_get, EXAMPLES / "readme_hero.png", panel_w=3.0, panel_h=4.0)

    # 1:1 crops on the SR canvas. JPEG is first resized to SR_W×SR_H so the
    # fractional crop covers the same physical region across all four columns.
    def crop_get(row, _c, path):
        _, _, _, center, half = row
        im = Image.open(path)
        if im.size != (SR_W, SR_H):
            im = im.resize((SR_W, SR_H), Image.LANCZOS)
        return im.crop(crop_box(center[0], center[1], half, SR_W, SR_H))
    render(PHOTOS, crop_get, EXAMPLES / "readme_crops.png", panel_w=3.0, panel_h=3.0)


if __name__ == "__main__":
    main()
