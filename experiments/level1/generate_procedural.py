"""Draw procedural (code-generated) western blots and a preview sheet.

    python experiments/level1/generate_procedural.py --n 150 \
        --like data/raw/biofors_blots --out data/raw/claude_procedural

With --like, each generated blot copies the size and the gray-level histogram of a
randomly chosen real blot from that folder. Real and generated images then match on
size, shape, brightness and contrast, so a detector has to judge the blot itself.
"""

import argparse
import random
from pathlib import Path

from PIL import Image

from fake_blot_detection.images import list_images
from fake_blot_detection.synth import draw_blot, match_histogram


def contact_sheet(images: list[Image.Image], cols: int = 4, cell=(400, 120)) -> Image.Image:
    """Grid of images, each scaled to fit an equal-sized cell."""
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("L", (cols * cell[0], rows * cell[1]), 255)
    for i, im in enumerate(images):
        im = im.copy()
        im.thumbnail((cell[0] - 10, cell[1] - 10))
        sheet.paste(im, ((i % cols) * cell[0] + 5, (i // cols) * cell[1] + 5))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/raw/claude_procedural"))
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--like", type=Path, help="folder of real blots to match")
    args = ap.parse_args()

    references = list_images(args.like) if args.like else []

    args.out.mkdir(parents=True, exist_ok=True)
    previews = []
    for i in range(args.n):
        seed = args.seed_offset + i
        if references:
            with Image.open(random.Random(seed).choice(references)) as ref:
                img = match_histogram(draw_blot(seed, ref.width, ref.height), ref)
        else:
            img = draw_blot(seed)
        img.save(args.out / f"blot_{seed:05d}.png")
        if len(previews) < 24:
            previews.append(img)
    contact_sheet(previews).save(args.out.parent / f"{args.out.name}_preview.png")
    print(f"wrote {args.n} blots to {args.out} and a preview sheet next to it")


if __name__ == "__main__":
    main()
