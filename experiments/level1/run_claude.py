"""Ask Claude to judge every image in a prepared dataset. Needs BLOT_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.

    python experiments/level1/run_claude.py --data data/level1 --limit 10   # try a few first
    python experiments/level1/run_claude.py --data data/level1             # then the rest

Results go to results/level1/<model>_<effort>.jsonl; re-running skips images already done.
"""

import argparse
from pathlib import Path

from fake_blot_detection.detect import ClaudeDetector, run_detector
from fake_blot_detection.images import read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/level1"))
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--limit", type=int, help="only judge the first N images in the manifest")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    manifest = read_manifest(args.data / "manifest.csv")[: args.limit]
    out = args.out or Path("results/level1") / f"{args.model}_{args.effort}.jsonl"
    detector = ClaudeDetector(model=args.model, effort=args.effort)
    run_detector(detector, manifest, args.data, out)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
