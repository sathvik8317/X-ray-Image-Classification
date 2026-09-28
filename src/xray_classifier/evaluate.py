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


@dataclass
class Metrics:
    accuracy: float
    roc_auc: float
    confusion_matrix: np.ndarray
    report: str
    fpr: np.ndarray
    tpr: np.ndarray


def predict_split(
    model: keras.Model, split_dir: str | Path, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, pneumonia probabilities) for every image in a split folder."""
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    paths, labels = list_images(split_dir)
    ds = make_dataset(paths, labels, img_size, channels, batch_size)
    return labels, model.predict(ds, verbose=0).ravel()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        roc_auc=auc(fpr, tpr),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        report=classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CATEGORIES, zero_division=0
        ),
        fpr=fpr,
        tpr=tpr,
    )


def evaluate(
    model: keras.Model, split_dir: str | Path, threshold: float = 0.5, batch_size: int = 32
) -> Metrics:
    y_true, y_prob = predict_split(model, split_dir, batch_size)
    return compute_metrics(y_true, y_prob, threshold)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained chest X-ray classifier.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
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

    model = keras.models.load_model(args.model_path)
    metrics = evaluate(model, split_dir, args.threshold)
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"ROC AUC:  {metrics.roc_auc:.4f}")
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
