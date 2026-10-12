"""Draw procedural (code-generated) western blots and a preview sheet.

python experiments/level1/generate_procedural.py --n 100 --out data/raw/claude_procedural
"""

import argparse
from pathlib import Path

from PIL import Image

from fake_blot_detection.synth import draw_blot


def contact_sheet(images: list[Image.Image], cols: int = 4) -> Image.Image:
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("L", (cols * w, rows * h), 255)
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/raw/claude_procedural"))
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    previews = []
    for i in range(args.n):
        seed = args.seed_offset + i
        img = draw_blot(seed)
        img.save(args.out / f"blot_{seed:05d}.png")
        if len(previews) < 12:
            previews.append(img)
    contact_sheet(previews).save(args.out.parent / f"{args.out.name}_preview.png")
    print(f"wrote {args.n} blots to {args.out} and a preview sheet next to it")


if __name__ == "__main__":
    main()
