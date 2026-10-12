"""Download BioFors and pull out blot/gel images from papers with nothing flagged.

BioFors (Sabir et al., ICCV 2021): https://github.com/vimal-isi-edu/BioFors
Its annotation files label each image's type (Blot/Gel, Microscopy, ...) and list the
images annotated as manipulated (duplication, cut/sharp transition). A whole paper is
skipped if any of its images is flagged, which is stricter than skipping single images.
"Not flagged" is not proof of being genuine, but it is the best label BioFors offers.

    python experiments/level1/get_biofors_blots.py --n 150 --out data/raw/biofors_blots

The 2 GB archive is saved to data/biofors/ and reused on later runs.
"""

import argparse
import csv
import json
import random
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ANNOTATION_URL = "https://raw.githubusercontent.com/vimal-isi-edu/BioFors/main/annotation_files/"
ANNOTATION_FILES = ["classification.json", "idd_gt.json", "cstd_gt.json", "edd_gt.json"]
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1UVSJ6h7r8pmOWYZkqWeAZ_YvwbFr1wV3&export=download&confirm=t"
)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    tmp.rename(dest)


def flagged_papers(annotation_dir: Path) -> set[str]:
    papers = set()
    for name in ["idd_gt.json", "cstd_gt.json", "edd_gt.json"]:
        papers |= set(json.loads((annotation_dir / name).read_text()))
    return papers


def candidate_blots(annotation_dir: Path) -> dict[str, list[str]]:
    """Paper id -> blot/gel image names, for papers with no flagged images."""
    classification = json.loads((annotation_dir / "classification.json").read_text())
    flagged = flagged_papers(annotation_dir)
    out = {}
    for paper, images in classification.items():
        if paper in flagged:
            continue
        blots = sorted(name for name, kind in images.items() if kind == "Blot/Gel")
        if blots:
            out[paper] = blots
    return out


def choose(candidates: dict[str, list[str]], n: int, per_paper: int, seed: int) -> set[str]:
    """Pick up to `per_paper` images from each paper (so no paper dominates), then n total."""
    rng = random.Random(seed)
    pool = []
    for paper in sorted(candidates):
        names = candidates[paper]
        pool += [f"{paper}/{name}" for name in rng.sample(names, min(per_paper, len(names)))]
    rng.shuffle(pool)
    return set(pool[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="how many blots to extract")
    ap.add_argument("--per-paper", type=int, default=2, help="max blots taken from one paper")
    ap.add_argument("--min-side", type=int, default=40, help="skip images smaller than this")
    ap.add_argument(
        "--max-dark",
        type=float,
        default=0.5,
        help="skip images where more than this fraction of pixels is dark: these are "
        "mostly DNA gels (white bands on black), not western blots",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("data/biofors"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/biofors_blots"))
    args = ap.parse_args()

    annotation_dir = args.cache / "annotation_files"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATION_FILES:
        download(ANNOTATION_URL + name, annotation_dir / name)
    archive = args.cache / "biofors_images.tar"
    download(ARCHIVE_URL, archive)

    candidates = candidate_blots(annotation_dir)
    total = sum(len(v) for v in candidates.values())
    print(f"{total} blot/gel images in {len(candidates)} papers with nothing flagged")
    # Ask for extra so images below --min-side can be dropped and n still reached.
    wanted = choose(candidates, args.n * 2, args.per_paper, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    kept = []
    with tarfile.open(archive) as tar:
        for member in tar:
            key = "/".join(Path(member.name).parts[-2:])
            if not member.isfile() or key not in wanted:
                continue
            with Image.open(tar.extractfile(member)) as img:
                img.load()
                if min(img.size) < args.min_side:
                    continue
                gray = np.asarray(img.convert("L"))
                if (gray < 128).mean() > args.max_dark:
                    continue
                paper, name = key.split("/")
                img.save(args.out / f"{paper}_{name}")
                kept.append(
                    {"paper": paper, "image": name, "width": img.width, "height": img.height}
                )

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    for row in kept[args.n :]:
        (args.out / f"{row['paper']}_{row['image']}").unlink()
    kept = kept[: args.n]

    with open(args.out.parent / f"{args.out.name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper", "image", "width", "height"])
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda r: (r["paper"], r["image"])))
    papers = defaultdict(int)
    for row in kept:
        papers[row["paper"]] += 1
    print(f"extracted {len(kept)} blots from {len(papers)} papers to {args.out}")


if __name__ == "__main__":
    main()
