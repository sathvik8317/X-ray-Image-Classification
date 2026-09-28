import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

DEFAULT_THRESHOLD = 0.5


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Cutoff that maximizes sensitivity + specificity (Youden's J) on the given labels.

    Returns the midpoint between the optimal score and the next lower one: it classifies the
    given samples identically but leaves the widest margin on both sides.
    """
    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    # thresholds[0] is +inf (predict nothing positive), never a useful cutoff.
    best = 1 + int(np.argmax(tpr[1:] - fpr[1:]))
    if best + 1 < len(thresholds):
        return float((thresholds[best] + thresholds[best + 1]) / 2)
    return float(thresholds[best])


def threshold_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".json")


def save_threshold(model_path: str | Path, threshold: float) -> None:
    threshold_path(model_path).write_text(json.dumps({"threshold": threshold}, indent=2) + "\n")


def load_threshold(model_path: str | Path) -> float:
    """Threshold saved next to a model by training, or 0.5 if there is none."""
    path = threshold_path(model_path)
    if not path.exists():
        return DEFAULT_THRESHOLD
    return float(json.loads(path.read_text())["threshold"])
