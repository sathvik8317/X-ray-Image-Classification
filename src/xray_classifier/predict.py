import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from .config import CATEGORIES
from .data import load_image
from .gradcam import gradcam
from .plots import plot_gradcam
from .thresholds import DEFAULT_THRESHOLD, load_threshold


def predict_image(
    model: keras.Model, path: str | Path, threshold: float = DEFAULT_THRESHOLD
) -> tuple[str, float]:
    """Return (predicted class, pneumonia probability) for one image."""
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    image = load_image(path, img_size, channels)[np.newaxis]
    prob = float(model.predict(image, verbose=0)[0, 0])
    return CATEGORIES[int(prob >= threshold)], prob


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify chest X-ray images.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold (default: the one saved with the model by training, else 0.5)",
    )
    parser.add_argument(
        "--gradcam", type=Path, metavar="DIR", help="save a Grad-CAM heatmap per image here"
    )
    args = parser.parse_args(argv)

    model = keras.models.load_model(args.model_path)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.model_path)
    if args.gradcam:
        args.gradcam.mkdir(parents=True, exist_ok=True)
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    for path in args.images:
        label, prob = predict_image(model, path, threshold)
        print(f"{path}: {label} (pneumonia probability {prob:.3f})")
        if args.gradcam:
            image = load_image(path, img_size, channels)
            title = f"{path.name}: {label} (pneumonia probability {prob:.2f})"
            fig = plot_gradcam(image, gradcam(model, image, threshold), title)
            out = args.gradcam / f"{path.stem}_gradcam.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
