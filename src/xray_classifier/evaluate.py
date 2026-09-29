import argparse
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve

from .config import CATEGORIES, DATA_DIR
from .data import list_images, make_dataset, patient_overlap
from .plots import plot_confusion_matrix, plot_roc_curve
from .thresholds import DEFAULT_THRESHOLD, load_threshold, threshold_path


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    sensitivity: float  # PNEUMONIA recall: share of pneumonia cases caught
    specificity: float  # NORMAL recall: share of healthy patients correctly cleared
    roc_auc: float
    confusion_matrix: np.ndarray
    report: str
    fpr: np.ndarray
    tpr: np.ndarray

    def summary(self) -> str:
        return (
            f"Accuracy {self.accuracy:.4f} | Sensitivity {self.sensitivity:.4f} | "
            f"Specificity {self.specificity:.4f} | ROC AUC {self.roc_auc:.4f} | "
            f"threshold {self.threshold:.3f}"
        )


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


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> Metrics:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return Metrics(
        threshold=threshold,
        accuracy=accuracy_score(y_true, y_pred),
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        roc_auc=auc(fpr, tpr),
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
    parser.add_argument("--plots-dir", type=Path, help="save confusion matrix and ROC plots here")
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
    metrics = evaluate(model, split_dir, threshold)
    print(f"Accuracy:    {metrics.accuracy:.4f}")
    print(f"Sensitivity: {metrics.sensitivity:.4f}  (pneumonia cases caught)")
    print(f"Specificity: {metrics.specificity:.4f}  (normal cases cleared)")
    print(f"ROC AUC:     {metrics.roc_auc:.4f}")
    print(metrics.report)

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
        }
        for kind, fig in figures.items():
            path = args.plots_dir / f"{name}_{kind}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
