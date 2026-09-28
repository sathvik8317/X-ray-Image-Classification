import argparse
from pathlib import Path

import keras
import numpy as np

from .config import CATEGORIES
from .data import load_image
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
    args = parser.parse_args(argv)

    model = keras.models.load_model(args.model_path)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.model_path)
    for path in args.images:
        label, prob = predict_image(model, path, threshold)
        print(f"{path}: {label} (pneumonia probability {prob:.3f})")


if __name__ == "__main__":
    main()
