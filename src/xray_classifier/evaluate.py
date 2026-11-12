import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .config import CATEGORIES, DATA_DIR
from .data import list_images, make_dataset, patient_id, patient_overlap
from .plots import plot_confusion_matrix, plot_reliability_diagram, plot_roc_curve
from .thresholds import DEFAULT_THRESHOLD, load_threshold, threshold_path

CALIBRATION_BINS = 10
BOOTSTRAP_METRICS = ("accuracy", "sensitivity", "specificity", "roc_auc")


@dataclass
class Reliability:
    """Per-bin calibration: mean predicted probability vs. observed pneumonia rate."""

    mean_predicted: np.ndarray
    observed_rate: np.ndarray
    counts: np.ndarray


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    sensitivity: float  # PNEUMONIA recall: share of pneumonia cases caught
    specificity: float  # NORMAL recall: share of healthy patients correctly cleared
    roc_auc: float
    brier: float  # mean squared error of the probabilities; 0 is perfect
    ece: float  # expected calibration error: average |predicted - observed| over bins
    reliability: Reliability
    confusion_matrix: np.ndarray
    report: str
    fpr: np.ndarray
    tpr: np.ndarray

    def summary(self) -> str:
        return (
            f"Accuracy {self.accuracy:.4f} | Sensitivity {self.sensitivity:.4f} | "
            f"Specificity {self.specificity:.4f} | ROC AUC {self.roc_auc:.4f} | "
            f"ECE {self.ece:.4f} | threshold {self.threshold:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "ece": self.ece,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def predict_files(
    model: keras.Model, paths: list[str], labels: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    """Pneumonia probability for each file."""
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    ds = make_dataset(paths, labels, img_size, channels, batch_size)
    return model.predict(ds, verbose=0).ravel()


def predict_split(
    model: keras.Model, split_dir: str | Path, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, pneumonia probabilities) for every image in a split folder."""
    paths, labels = list_images(split_dir)
    return labels, predict_files(model, paths, labels, batch_size)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def reliability(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = CALIBRATION_BINS
) -> Reliability:
    """Equal-width probability bins; empty bins are dropped."""
    bins = np.minimum((np.asarray(y_prob) * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    filled = counts > 0
    prob_sums = np.bincount(bins, weights=y_prob, minlength=n_bins)
    label_sums = np.bincount(bins, weights=y_true, minlength=n_bins)
    return Reliability(
        mean_predicted=prob_sums[filled] / counts[filled],
        observed_rate=label_sums[filled] / counts[filled],
        counts=counts[filled],
    )


def expected_calibration_error(rel: Reliability) -> float:
    gaps = np.abs(rel.mean_predicted - rel.observed_rate)
    return float((gaps * rel.counts).sum() / rel.counts.sum())


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """(accuracy, sensitivity, specificity) for boolean predictions, without sklearn overhead."""
    positive = y_true == 1
    tp = int((y_pred & positive).sum())
    tn = int((~y_pred & ~positive).sum())
    n_pos = int(positive.sum())
    return (tp + tn) / len(y_true), _ratio(tp, n_pos), _ratio(tn, len(y_true) - n_pos)


def bootstrap_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    groups: list[str] | np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for accuracy, sensitivity, specificity, AUC.

    With `groups` (patient IDs), whole patients are resampled, so several images of one
    patient do not count as independent evidence and the intervals are not too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    _, group_index = np.unique(groups, return_inverse=True)
    n_groups = group_index.max() + 1
    members = [np.flatnonzero(group_index == g) for g in range(n_groups)]
    y_pred = y_prob >= threshold
    samples: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n_resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_groups, n_groups)])
        t = y_true[idx]
        if t.min() == t.max():  # AUC needs both classes
            continue
        accuracy, sensitivity, specificity = _threshold_metrics(t, y_pred[idx])
        samples["accuracy"].append(accuracy)
        samples["sensitivity"].append(sensitivity)
        samples["specificity"].append(specificity)
        samples["roc_auc"].append(roc_auc_score(t, y_prob[idx]))
    tail = 100 * (1 - confidence) / 2
    return {
        name: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
        for name, v in samples.items()
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = CALIBRATION_BINS,
) -> Metrics:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    rel = reliability(y_true, y_prob, n_bins)
    return Metrics(
        threshold=threshold,
        accuracy=accuracy_score(y_true, y_pred),
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        roc_auc=auc(fpr, tpr),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(rel),
        reliability=rel,
        confusion_matrix=cm,
        report=classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CATEGORIES, zero_division=0
        ),
        fpr=fpr,
        tpr=tpr,
    )


def evaluate(
    model: keras.Model,
    split_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 32,
) -> Metrics:
    y_true, y_prob = predict_split(model, split_dir, batch_size)
    return compute_metrics(y_true, y_prob, threshold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained chest X-ray classifier.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold (default: the one saved with the model by training, else 0.5)",
    )
    parser.add_argument(
        "--plots-dir", type=Path, help="save confusion matrix, ROC, and reliability plots here"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help="patient-level bootstrap resamples for 95%% confidence intervals (0 to skip)",
    )
    parser.add_argument("--json", type=Path, help="also write the metrics to this JSON file")
    args = parser.parse_args(argv)

    split_dir = args.data_dir / args.split
    if args.split != "train" and (args.data_dir / "train").is_dir():
        shared = patient_overlap(list_images(args.data_dir / "train")[0], list_images(split_dir)[0])
        if shared:
            print(
                f"Warning: {len(shared)} patient(s) appear in both train/ and {args.split}/, "
                "so these metrics may be optimistic"
            )

    threshold = args.threshold
    if threshold is None:
        threshold = load_threshold(args.model_path)
        saved = threshold_path(args.model_path)
        origin = f"saved in {saved}" if saved.exists() else "default"
        print(f"Using threshold {threshold:.3f} ({origin})")

    model = keras.models.load_model(args.model_path)
    paths, y_true = list_images(split_dir)
    y_prob = predict_files(model, paths, y_true)
    metrics = compute_metrics(y_true, y_prob, threshold)
    intervals = {}
    if args.bootstrap:
        groups = [patient_id(p) for p in paths]
        intervals = bootstrap_intervals(y_true, y_prob, threshold, groups, args.bootstrap)

    def line(label: str, name: str, note: str = "") -> str:
        text = f"{label:<13}{getattr(metrics, name):.4f}"
        if name in intervals:
            low, high = intervals[name]
            text += f"  [95% CI {low:.4f}-{high:.4f}]"
        return text + (f"  ({note})" if note else "")

    print(line("Accuracy:", "accuracy"))
    print(line("Sensitivity:", "sensitivity", "pneumonia cases caught"))
    print(line("Specificity:", "specificity", "normal cases cleared"))
    print(line("ROC AUC:", "roc_auc"))
    print(line("Brier score:", "brier", "lower is better"))
    print(line("ECE:", "ece", "calibration error, lower is better"))
    print(metrics.report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": args.model_path.stem, "split": args.split, **metrics.to_dict()}
        record["ci95"] = {name: list(bounds) for name, bounds in intervals.items()}
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Saved {args.json}")

    if args.plots_dir:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        name = args.model_path.stem
        figures = {
            "confusion_matrix": plot_confusion_matrix(
                metrics.confusion_matrix, f"{name} - Confusion Matrix"
            ),
            "roc_curve": plot_roc_curve(
                metrics.fpr, metrics.tpr, metrics.roc_auc, f"{name} - ROC Curve"
            ),
            "reliability": plot_reliability_diagram(
                metrics.reliability, metrics.ece, f"{name} - Reliability"
            ),
        }
        for kind, fig in figures.items():
            path = args.plots_dir / f"{name}_{kind}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
