"""Standardize real and AI-generated blots into one dataset with a manifest.

Each --source is NAME=FOLDER:LABEL, where LABEL is "real" or "ai". For example:

    python experiments/level1/prepare.py \
        --source biofors=data/raw/biofors_blots:real \
        --source claude_procedural=data/raw/claude_procedural:ai \
        --max-per-source 100 --out data/level1
"""

import argparse
from collections import Counter
from pathlib import Path

from fake_blot_detection.images import StandardizeConfig, build_dataset


def parse_source(text: str) -> tuple[str, tuple[Path, str]]:
    name, rest = text.split("=", 1)
    folder, label = rest.rsplit(":", 1)
    return name, (Path(folder), label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, type=parse_source)
    ap.add_argument("--out", type=Path, default=Path("data/level1"))
    ap.add_argument("--max-per-source", type=int)
    ap.add_argument("--long-side", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args()

    rows = build_dataset(
        dict(args.source),
        args.out,
        StandardizeConfig(args.long_side, args.jpeg_quality),
        max_per_source=args.max_per_source,
    )
    counts = Counter((r["source"], r["label"]) for r in rows)
    for (source, label), n in sorted(counts.items()):
        print(f"{source:>24} ({label}): {n}")
    print(f"manifest: {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
