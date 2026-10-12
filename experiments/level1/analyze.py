"""Score detector results and check for shortcut cues.

python experiments/level1/analyze.py --data data/level1 results/level1/*.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from fake_blot_detection.images import read_manifest
from fake_blot_detection.metrics import bootstrap_ci, image_stats, roc_auc, tpr_at_fpr


def report_detector(path: Path, manifest: list[dict]) -> None:
    results = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        results[rec["image_id"]] = rec
    print(f"\n=== {path.name} ({len(results)} of {len(manifest)} images judged)")

    outcomes = Counter(
        "answered" if "probability_ai" in r else r.get("stop_reason") or "error"
        for r in results.values()
    )
    print("outcomes:", dict(outcomes))
    refusals = [r for r in results.values() if r.get("stop_reason") == "refusal"]
    if refusals:
        print("refusal categories:", dict(Counter(r.get("refusal_category") for r in refusals)))

    scored = [
        (row, results[row["image_id"]])
        for row in manifest
        if "probability_ai" in results.get(row["image_id"], {})
    ]
    if not scored:
        return
    labels = [row["label"] == "ai" for row, _ in scored]
    probs = [res["probability_ai"] for _, res in scored]
    verdicts = [res["verdict"] == "ai" for _, res in scored]

    acc = sum(v == y for v, y in zip(verdicts, labels)) / len(labels)
    auc = roc_auc(labels, probs)
    lo, hi = bootstrap_ci(labels, probs)
    print(f"accuracy (verdict): {acc:.3f}")
    print(f"ROC-AUC (probability): {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"TPR @ 5% FPR: {tpr_at_fpr(labels, probs, 0.05):.3f}")

    print("by source (fraction judged AI, mean probability_ai):")
    for source in sorted({row["source"] for row, _ in scored}):
        sub = [(row, res) for row, res in scored if row["source"] == source]
        frac = sum(res["verdict"] == "ai" for _, res in sub) / len(sub)
        mean_p = sum(res["probability_ai"] for _, res in sub) / len(sub)
        print(
            f"  {source:>24} ({sub[0][0]['label']}, n={len(sub)}): "
            f"judged AI {frac:.2f}, mean p {mean_p:.2f}"
        )


def shortcut_check(manifest: list[dict], data_dir: Path) -> None:
    """AUC of single pixel statistics. Values far from 0.5 (say > 0.8 or < 0.2) mean real and
    AI images differ in a trivial way a detector could use without understanding blots."""
    stats = [image_stats(data_dir / row["path"]) for row in manifest]
    labels = [row["label"] == "ai" for row in manifest]
    print("\n=== shortcut check: AUC of each simple pixel statistic (0.5 = no signal)")
    for key in stats[0]:
        auc = roc_auc(labels, [s[key] for s in stats])
        flag = "  <-- separates the classes on its own" if abs(auc - 0.5) > 0.3 else ""
        print(f"  {key:>14}: {auc:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/level1"))
    args = ap.parse_args()

    manifest = read_manifest(args.data / "manifest.csv")
    print(f"dataset: {len(manifest)} images", dict(Counter(r["label"] for r in manifest)))
    shortcut_check(manifest, args.data)
    for path in args.results:
        report_detector(path, manifest)


if __name__ == "__main__":
    main()
