import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .config import DATA_DIR, IMG_SIZE, MODELS_DIR
from .data import make_dataset, train_val_files
from .evaluate import Metrics, compute_metrics, predict_files
from .models import MODEL_BUILDERS
from .thresholds import choose_threshold, save_threshold

DEFAULT_BATCH_SIZE = {"cnn": 4, "vgg16": 32}


def make_callbacks(checkpoint_path: str | Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


@dataclass
class TrainingResult:
    model: keras.Model  # best checkpoint, reloaded from disk
    history: dict[str, list[float]]
    threshold: float  # chosen on the validation split, saved next to the checkpoint
    val_metrics: Metrics


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Weights that make each class contribute equally to the loss."""
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    return {0: float(weights[0]), 1: float(weights[1])}


def train(
    model: keras.Model,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    augment: bool = False,
    val_fraction: float = 0.15,
    class_weight: bool = True,
    seed: int | None = None,
) -> TrainingResult:
    """Fit with a patient-grouped validation split; save the best model and its threshold.

    val_fraction=0 validates on the provided val/ folder (16 images) instead.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    (train_paths, train_labels), (val_paths, val_labels) = train_val_files(data_dir, val_fraction)
    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)}")
    train_ds = make_dataset(
        train_paths, train_labels, img_size, channels, batch_size, augment, shuffle=True, seed=seed
    )
    val_ds = make_dataset(val_paths, val_labels, img_size, channels, batch_size)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=balanced_class_weights(train_labels) if class_weight else None,
        callbacks=make_callbacks(checkpoint_path),
    )

    best = keras.models.load_model(checkpoint_path)
    val_prob = predict_files(best, val_paths, val_labels, batch_size)
    threshold = choose_threshold(val_labels, val_prob)
    save_threshold(checkpoint_path, threshold)
    return TrainingResult(
        model=best,
        history=history.history,
        threshold=threshold,
        val_metrics=compute_metrics(val_labels, val_prob, threshold),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a chest X-ray pneumonia classifier.")
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="cnn")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output", type=Path, help="where to save the best model (default: models/<model>.keras)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, help="default: 4 for cnn, 32 for vgg16")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        help="random zoom augmentation (default: on for vgg16, off for cnn)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="share of train/ + val/ held out for validation, split by patient "
        "(default: 0.15; 0 uses the provided 16-image val/ folder)",
    )
    parser.add_argument(
        "--class-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight the loss to balance NORMAL and PNEUMONIA (default: on)",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    output = args.output or MODELS_DIR / f"{args.model}.keras"
    augment = args.augment if args.augment is not None else args.model == "vgg16"

    model = MODEL_BUILDERS[args.model](args.img_size)
    start = time.perf_counter()
    result = train(
        model,
        args.data_dir,
        output,
        epochs=args.epochs,
        batch_size=args.batch_size or DEFAULT_BATCH_SIZE[args.model],
        augment=augment,
        val_fraction=args.val_fraction,
        class_weight=args.class_weight,
        seed=args.seed,
    )
    print(f"Training time: {time.perf_counter() - start:.1f}s")
    print(f"Best model saved to {output}")
    print(f"Validation: {result.val_metrics.summary()}")


if __name__ == "__main__":
    main()
