"""Build README before/after hero: 2 photos x 4 variants.

Columns: original camera JPEG, upscayl-standard-4x, RealESRGAN_General_x4_v3 (GAN),
digital-art-4x. Rows: cups, person.

All variants come from the existing JPEG-path outputs in examples/_work/<photo_id>/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
INPUT = EXAMPLES / "input"
WORK = EXAMPLES / "_work"

PHOTOS = [
    ("1777804010136", "cups"),
    ("1771110170401", "person"),
]

COLS = [
    ("Camera JPEG (1120×1494)", lambda pid: INPUT / f"{pid}.jpg"),
    ("upscayl-standard-4x",     lambda pid: WORK / pid / "upscayl-standard-4x.png"),
    ("RealESRGAN-General-v3",   lambda pid: WORK / pid / "RealESRGAN_General_x4_v3.png"),
    ("digital-art-4x",          lambda pid: WORK / pid / "digital-art-4x.png"),
]


def main():
    fig, axes = plt.subplots(
        len(PHOTOS), len(COLS),
        figsize=(3 * len(COLS), 4 * len(PHOTOS)),
        constrained_layout=True,
    )
    if len(PHOTOS) == 1:
        axes = [axes]

    for r, (pid, label) in enumerate(PHOTOS):
        for c, (title, path_fn) in enumerate(COLS):
            ax = axes[r][c]
            img = Image.open(path_fn(pid))
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)

    out = EXAMPLES / "readme_hero.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
