"""Anti-spoofing metrics: EER, AUC, accuracy, confusion-matrix counts.

The Equal Error Rate computation follows the same convention as ``ASVspoof``:
the FAR / FRR curves are tabulated by sweeping the spoof score threshold over
the score support, and EER is the point where both errors coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class ClassificationReport:
    """Single-pass classification report."""

    eer: float
    eer_threshold: float
    auc: float
    accuracy: float
    confusion: np.ndarray   # shape (2, 2): [[TN, FP], [FN, TP]]

    def as_dict(self) -> dict:
        return {
            "eer": float(self.eer),
            "eer_threshold": float(self.eer_threshold),
            "auc": float(self.auc),
            "accuracy": float(self.accuracy),
            "confusion": self.confusion.tolist(),
        }


def equal_error_rate(scores: Sequence[float],
                     labels: Sequence[int]) -> Tuple[float, float]:
    """Return ``(eer, threshold)``.

    ``scores`` is the model's "spoof"-class score (higher = more spoof-like);
    ``labels`` is 0 for bona-fide and 1 for spoof.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return float("nan"), float("nan")
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order]
    pos = (y_sorted == 1).astype(np.float64)
    neg = (y_sorted == 0).astype(np.float64)
    # Cumulative counts of spoof samples *below* threshold s_sorted[i] (rejected)
    # and bona-fide samples *above* it (accepted as spoof = false alarm).
    cum_pos_below = np.cumsum(pos)
    cum_neg_above = neg.sum() - np.cumsum(neg)
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos == 0 or total_neg == 0:
        return float("nan"), float("nan")
    frr = cum_pos_below / total_pos                # spoof rejected as bona-fide
    far = cum_neg_above / total_neg                # bona-fide accepted as spoof
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(s_sorted[idx])


def evaluate_classifier(scores: Sequence[float],
                        labels: Sequence[int],
                        threshold: float | None = None) -> ClassificationReport:
    """One-shot classification metric pack."""
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    eer, eer_th = equal_error_rate(scores_np, labels_np)
    th = eer_th if threshold is None else float(threshold)

    try:
        auc = float(roc_auc_score(labels_np, scores_np))
    except ValueError:
        auc = float("nan")

    preds = (scores_np >= th).astype(np.int64)
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    accuracy = (tp + tn) / max(1, len(labels_np))

    return ClassificationReport(
        eer=eer, eer_threshold=th, auc=auc, accuracy=accuracy,
        confusion=np.asarray([[tn, fp], [fn, tp]], dtype=np.int64),
    )


def det_curve(scores: Sequence[float],
              labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(far, frr)`` arrays for a DET-curve plot."""
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores_np)
    y_sorted = labels_np[order]
    pos = (y_sorted == 1).astype(np.float64)
    neg = (y_sorted == 0).astype(np.float64)
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos == 0 or total_neg == 0:
        return np.array([]), np.array([])
    far = (neg.sum() - np.cumsum(neg)) / total_neg
    frr = np.cumsum(pos) / total_pos
    return far, frr
