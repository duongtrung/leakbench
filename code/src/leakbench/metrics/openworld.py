"""Open-world scoring (paper §Design "Closed- and open-world protocols", §Eval
"Open-world"). A fraction of traces belong to an unmonitored "other" class; leakage
on the *monitored* classes is reported as precision/recall/F1, not accuracy.

Base rates decide open-world outcomes (WF lesson): precision stays high but recall
collapses as monitored tasks become rare.
"""

from __future__ import annotations

import numpy as np

# Convention: the unmonitored "other" class is labeled with OTHER_LABEL.
OTHER_LABEL = -1


def open_world_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 over the *monitored* classes.

    A prediction is a true positive if the true label is monitored (!= OTHER) and
    the prediction equals it; a false positive if the model predicts a monitored
    class but the true label is OTHER or a different monitored class; a false
    negative if a monitored true label is predicted as OTHER or a wrong monitored
    class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    monitored_true = y_true != OTHER_LABEL
    pred_monitored = y_pred != OTHER_LABEL

    tp = int(np.sum(monitored_true & pred_monitored & (y_true == y_pred)))
    fp = int(np.sum(pred_monitored & ~(monitored_true & (y_true == y_pred))))
    fn = int(np.sum(monitored_true & ~(pred_monitored & (y_true == y_pred))))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
