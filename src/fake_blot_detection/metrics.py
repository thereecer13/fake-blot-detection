"""Detection metrics and a shortcut check, using only numpy."""

import numpy as np
from PIL import Image


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve. labels: 1 = positive (AI), 0 = negative (real).

    Equals the probability that a random positive scores higher than a random negative,
    counting ties as half.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def tpr_at_fpr(labels, scores, max_fpr: float) -> float:
    """Highest true-positive rate reachable while keeping false-positive rate <= max_fpr."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    best = 0.0
    for t in np.unique(scores):
        if (neg >= t).mean() <= max_fpr:
            best = max(best, (pos >= t).mean())
    return float(best)


def bootstrap_ci(
    labels, scores, metric=roc_auc, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap confidence interval, resampling images."""
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        value = metric(labels[idx], scores[idx])
        if not np.isnan(value):
            stats.append(value)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def image_stats(path) -> dict[str, float]:
    """Simple pixel statistics. If any of these separates real from AI on its own,
    the synthetic images differ in some trivial way and the experiment needs fixing."""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
    hist = np.histogram(a, bins=64, range=(0, 1))[0] / a.size
    nonzero = hist[hist > 0]
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "aspect_ratio": a.shape[1] / a.shape[0],
        "noise": float(np.abs(np.diff(a, axis=1)).mean()),
        "entropy": float(-(nonzero * np.log2(nonzero)).sum()),
        "dark_fraction": float((a < 0.5).mean()),
    }
