import math

from fake_blot_detection.metrics import roc_auc, tpr_at_fpr


def test_auc_perfect_and_inverted():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_ties_count_half():
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_tpr_at_fpr():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.95, 0.99]
    assert tpr_at_fpr(labels, scores, 0.0) == 0.5  # threshold 0.95 catches 2 of 4
    assert tpr_at_fpr(labels, scores, 0.25) == 1.0  # threshold 0.4 allows 1 false positive
