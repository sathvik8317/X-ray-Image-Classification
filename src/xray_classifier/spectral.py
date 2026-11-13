"""Spectral ablation: which spatial frequencies does a model rely on?

Each test image is filtered in the Fourier domain, keeping only low frequencies (overall
shapes and brightness) or only high frequencies (edges, fine texture, noise), and the
model's ROC AUC is measured again. A model that reads lung opacities should survive a
low-pass filter well. One that still scores well on a high-pass image with the anatomy
gone is reading texture, scanner noise or JPEG artefacts: a shortcut.

AUC is used because it does not depend on the decision threshold, which a filter shifts.
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR
from .data import list_images, load_image, make_dataset

# Gaussian filter widths as a fraction of the Nyquist frequency (0.5 cycles per pixel).
DEFAULT_CUTOFFS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
MODES = ("lowpass", "highpass")


def _radius(height: int, width: int) -> np.ndarray:
    """Distance of each FFT bin from zero frequency, as a fraction of Nyquist."""
    fy = np.fft.fftfreq(height)[:, np.newaxis]
    fx = np.fft.fftfreq(width)[np.newaxis, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def frequency_filter(images: np.ndarray, cutoff: float, mode: str) -> np.ndarray:
    """Gaussian low- or high-pass filter of a batch (n, h, w, c) of images in [0, 1].

    The high-pass keeps each image's mean brightness so it stays in the model's input
    range; only the structure is removed. Output is clipped to [0, 1].
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    _, height, width, _ = images.shape
    lowpass = np.exp(-0.5 * (_radius(height, width) / cutoff) ** 2)
    response = lowpass if mode == "lowpass" else 1.0 - lowpass
    spectrum = np.fft.fft2(images, axes=(1, 2)) * response[np.newaxis, :, :, np.newaxis]
    filtered = np.fft.ifft2(spectrum, axes=(1, 2)).real
    if mode == "highpass":
        filtered += images.mean(axis=(1, 2), keepdims=True)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def spectral_ablation(
    model: keras.Model,
    paths: list[str],
    labels: np.ndarray,
    cutoffs: tuple[float, ...] = DEFAULT_CUTOFFS,
    batch_size: int = 32,
) -> dict:
    """ROC AUC on unfiltered images and on every (mode, cutoff) filtered copy."""
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    ds = make_dataset(paths, labels, img_size, channels, batch_size)
    variants = [("none", 0.0)] + [(m, c) for m in MODES for c in cutoffs]
    probs: dict[tuple[str, float], list[np.ndarray]] = {v: [] for v in variants}
    for images, _ in ds:
        images = images.numpy()
        for mode, cutoff in variants:
            batch = images if mode == "none" else frequency_filter(images, cutoff, mode)
            probs[(mode, cutoff)].append(model.predict_on_batch(batch).ravel())
    auc = {v: float(roc_auc_score(labels, np.concatenate(p))) for v, p in probs.items()}
    return {
        "baseline_auc": auc[("none", 0.0)],
        "cutoffs": list(cutoffs),
        **{mode: [auc[(mode, c)] for c in cutoffs] for mode in MODES},
    }


def plot_spectral(result: dict, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    cutoffs = result["cutoffs"]
    ax.plot(cutoffs, result["lowpass"], "o-", color="#333333", label="low frequencies only")
    ax.plot(cutoffs, result["highpass"], "s--", color="#2563eb", label="high frequencies only")
    ax.axhline(result["baseline_auc"], color="#999999", lw=1, label="unfiltered")
    ax.axhline(0.5, color="#cccccc", lw=1, ls=":", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("filter cutoff (fraction of Nyquist)")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_spectral_comparison(results: list[dict]) -> plt.Figure:
    """Low-pass and high-pass AUC curves of several models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    grays = plt.get_cmap("Greys")(np.linspace(0.45, 0.85, max(len(results) - 1, 1)))
    best = max(range(len(results)), key=lambda i: results[i]["baseline_auc"])
    shades = iter(grays)
    for i, result in enumerate(results):
        color, width = ("#2563eb", 2.2) if i == best else (next(shades), 1.2)
        for ax, mode in zip(axes, MODES, strict=True):
            ax.plot(
                result["cutoffs"],
                result[mode],
                "o-",
                color=color,
                lw=width,
                ms=3,
                label=f"{result['model']} (unfiltered {result['baseline_auc']:.3f})",
            )
    titles = ("Low frequencies only (blurred)", "High frequencies only (fine detail)")
    for ax, title in zip(axes, titles, strict=True):
        ax.axhline(0.5, color="#cccccc", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("filter cutoff (fraction of Nyquist)")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("ROC AUC on the test set")
    axes[0].set_ylim(0.3, 1.0)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def save_examples(path: str, img_size: int, cutoffs: tuple[float, ...], out: Path) -> None:
    """One image under every filter, so the cutoffs are easy to picture."""
    image = load_image(path, img_size, 1)[np.newaxis]
    fig, axes = plt.subplots(len(MODES), len(cutoffs), figsize=(2 * len(cutoffs), 4.4))
    for row, mode in enumerate(MODES):
        for col, cutoff in enumerate(cutoffs):
            ax = axes[row, col]
            ax.imshow(frequency_filter(image, cutoff, mode)[0, ..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{mode} {cutoff}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Spectral ablation: ROC AUC with only low or only high frequencies kept."
    )
    parser.add_argument("model_path", type=Path, nargs="?")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--cutoffs", type=float, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        metavar="JSON",
        help="instead of running a model, plot saved *_spectral.json results side by side",
    )
    args = parser.parse_args(argv)

    if args.compare:
        results = [json.loads(path.read_text()) for path in args.compare]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_spectral_comparison(results)
        fig.savefig(args.out_dir / "spectral_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved {args.out_dir / 'spectral_comparison.png'}")
        return
    if args.model_path is None:
        parser.error("model_path is required unless --compare is given")

    model = keras.models.load_model(args.model_path)
    paths, labels = list_images(args.data_dir / args.split)
    cutoffs = tuple(args.cutoffs)
    result = spectral_ablation(model, paths, labels, cutoffs)
    name = args.model_path.stem
    result = {"model": name, "split": args.split, **result}

    print(f"Unfiltered AUC: {result['baseline_auc']:.4f}")
    print(f"{'cutoff':>8}  {'lowpass':>8}  {'highpass':>8}")
    for c, lo, hi in zip(cutoffs, result["lowpass"], result["highpass"], strict=True):
        print(f"{c:>8.3f}  {lo:>8.4f}  {hi:>8.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}_spectral.json").write_text(json.dumps(result, indent=2) + "\n")
    fig = plot_spectral(result, f"{name} - spectral ablation ({args.split})")
    fig.savefig(args.out_dir / f"{name}_spectral.png", dpi=150)
    plt.close(fig)
    save_examples(paths[0], model.input_shape[1], cutoffs, args.out_dir / "spectral_examples.png")
    print(f"Saved results to {args.out_dir}")


if __name__ == "__main__":
    main()
