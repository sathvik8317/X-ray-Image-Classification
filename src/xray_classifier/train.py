import argparse
import time
from pathlib import Path

import keras

from .config import DATA_DIR, IMG_SIZE, MODELS_DIR
from .data import make_dataset
from .models import MODEL_BUILDERS

DEFAULT_BATCH_SIZE = {"cnn": 4, "vgg16": 32}


def make_callbacks(checkpoint_path: str | Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def train(
    model: keras.Model,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    augment: bool = False,
    seed: int | None = None,
) -> keras.callbacks.History:
    """Fit on data_dir/train, validate on data_dir/val, and save the best model."""
    data_dir, checkpoint_path = Path(data_dir), Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    img_size, channels = model.input_shape[1], model.input_shape[-1]
    train_ds = make_dataset(
        data_dir / "train", img_size, channels, batch_size, augment=augment, shuffle=True, seed=seed
    )
    val_ds = make_dataset(data_dir / "val", img_size, channels, batch_size)
    return model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=make_callbacks(checkpoint_path),
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
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    output = args.output or MODELS_DIR / f"{args.model}.keras"
    augment = args.augment if args.augment is not None else args.model == "vgg16"

    model = MODEL_BUILDERS[args.model](args.img_size)
    start = time.perf_counter()
    train(
        model,
        args.data_dir,
        output,
        epochs=args.epochs,
        batch_size=args.batch_size or DEFAULT_BATCH_SIZE[args.model],
        augment=augment,
        seed=args.seed,
    )
    print(f"Training time: {time.perf_counter() - start:.1f}s")
    print(f"Best model saved to {output}")


if __name__ == "__main__":
    main()
