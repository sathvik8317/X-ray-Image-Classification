"""Compare trained models on one split: a Markdown results table, ROC and reliability
overlays, and a Grad-CAM grid of the best model's hits and worst misses (xray-report)."""

import argparse
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .config import DATA_DIR
from .data import list_images, load_image, patient_id
from .evaluate import Metrics, bootstrap_intervals, compute_metrics, predict_files
from .gradcam import gradcam
from .thresholds import load_threshold


def _cell(row: dict, name: str, percent: bool = True) -> str:
    """'94.1 (91.8-96.2)': the value with its 95% CI when the row has one."""
    scale, digits = (100, 1) if percent else (1, 3)
    text = f"{scale * row[name]:.{digits}f}"
    if interval := row.get("ci95", {}).get(name):
        text += f" ({scale * interval[0]:.{digits}f}-{scale * interval[1]:.{digits}f})"
    return text


def results_table(rows: list[dict]) -> str:
    """Markdown table; rows are the JSON records written by this module."""
    lines = [
        "| Model | Sensitivity % | Specificity % | Accuracy % | ROC AUC | ECE | Threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cells = [
            r["model"],
            _cell(r, "sensitivity"),
            _cell(r, "specificity"),
            _cell(r, "accuracy"),
            _cell(r, "roc_auc", percent=False),
            f"{r['ece']:.3f}",
            f"{r['threshold']:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_roc_comparison(results: dict[str, Metrics]) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, m in results.items():
        ax.plot(m.fpr, m.tpr, label=f"{name} (AUC {m.roc_auc:.3f})")
        ax.plot(1 - m.specificity, m.sensitivity, "o", color=ax.lines[-1].get_color())
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC on the test set (dots: chosen thresholds)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_reliability_comparison(results: dict[str, Metrics]) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    for name, m in results.items():
        rel = m.reliability
        ax.plot(
            rel.mean_predicted, rel.observed_rate, marker="o", label=f"{name} (ECE {m.ece:.3f})"
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted pneumonia probability")
    ax.set_ylabel("Observed pneumonia rate")
    ax.set_title("Calibration on the test set")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def pick_gradcam_cases(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> list[tuple[str, int]]:
    """(caption, index): the most confident correct call per class and the worst miss of each kind."""
    pred = y_prob >= threshold
    cases = []
    for caption, mask, score in (
        ("Correct: NORMAL", (y_true == 0) & ~pred, -y_prob),
        ("Correct: PNEUMONIA", (y_true == 1) & pred, y_prob),
        ("False positive", (y_true == 0) & pred, y_prob),
        ("False negative", (y_true == 1) & ~pred, -y_prob),
    ):
        if mask.any():
            candidates = np.flatnonzero(mask)
            cases.append((caption, int(candidates[np.argmax(score[candidates])])))
    return cases


def plot_gradcam_grid(
    model: keras.Model,
    paths: list[str],
    y_prob: np.ndarray,
    cases: list[tuple[str, int]],
    threshold: float,
    title: str,
) -> Figure:
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    fig, axes = plt.subplots(2, len(cases), figsize=(3 * len(cases), 6.4), squeeze=False)
    for col, (caption, i) in enumerate(cases):
        image = load_image(paths[i], img_size, channels)
        heatmap = gradcam(model, image, threshold)
        gray = image[..., 0]
        axes[0, col].imshow(gray, cmap="gray")
        axes[0, col].set_title(f"{caption}\np(pneumonia) = {y_prob[i]:.2f}", fontsize=9)
        axes[1, col].imshow(gray, cmap="gray")
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare trained models for the README.")
    parser.add_argument("model_paths", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--bootstrap", type=int, default=1000, metavar="N")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths, y_true = list_images(args.data_dir / args.split)
    groups = [patient_id(p) for p in paths]
    results, rows, probs, models = {}, [], {}, {}
    for model_path in args.model_paths:
        name = model_path.stem
        model = keras.models.load_model(model_path)
        threshold = load_threshold(model_path)
        y_prob = predict_files(model, paths, y_true)
        metrics = compute_metrics(y_true, y_prob, threshold)
        intervals = (
            bootstrap_intervals(y_true, y_prob, threshold, groups, args.bootstrap)
            if args.bootstrap
            else {}
        )
        results[name], probs[name], models[name] = metrics, y_prob, model
        row = {"model": name, "split": args.split, **metrics.to_dict()}
        row["ci95"] = {k: list(v) for k, v in intervals.items()}
        rows.append(row)
        print(f"{name}: {metrics.summary()}")

    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    table = results_table(rows)
    (args.out_dir / "results.md").write_text(table + "\n")
    print(table)

    best = max(results, key=lambda n: results[n].roc_auc)
    figures = {
        "roc_comparison": plot_roc_comparison(results),
        "reliability_comparison": plot_reliability_comparison(results),
        "gradcam_examples": plot_gradcam_grid(
            models[best],
            paths,
            probs[best],
            pick_gradcam_cases(y_true, probs[best], results[best].threshold),
            results[best].threshold,
            f"Grad-CAM, {best} (top: X-ray, bottom: heatmap)",
        ),
    }
    for kind, fig in figures.items():
        out = args.out_dir / f"{kind}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out}")
    print(f"Grad-CAM grid uses the best model by ROC AUC: {best}")


if __name__ == "__main__":
    main()
