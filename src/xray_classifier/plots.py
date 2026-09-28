import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from .config import CATEGORIES


def plot_history(history: dict[str, list[float]], title: str = "Training") -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss"), strict=True):
        epochs = np.arange(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CATEGORIES).plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, title: str = "ROC Curve"
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_gradcam(image: np.ndarray, heatmap: np.ndarray, title: str = "Grad-CAM") -> Figure:
    """The X-ray next to the X-ray overlaid with its Grad-CAM heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    gray = image[..., 0]
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("X-ray")
    axes[1].imshow(gray, cmap="gray")
    overlay = axes[1].imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    fig.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
