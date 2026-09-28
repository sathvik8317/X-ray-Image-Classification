from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import CATEGORIES, IMG_SIZE

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def class_counts(split_dir: str | Path) -> dict[str, int]:
    split_dir = Path(split_dir)
    return {
        c: sum(1 for p in (split_dir / c).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for c in CATEGORIES
    }


def make_dataset(
    split_dir: str | Path,
    img_size: int = IMG_SIZE,
    channels: int = 1,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Batches of (images in [0, 1], labels) from a split folder; label 1 = PNEUMONIA."""
    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        label_mode="binary",
        class_names=list(CATEGORIES),
        color_mode="grayscale" if channels == 1 else "rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        # The notebook's ImageDataGenerator shear_range=0.2 meant 0.2 degrees, effectively a
        # no-op, so only the zoom augmentation is carried over.
        zoom = keras.layers.RandomZoom(0.2, seed=seed)
        ds = ds.map(lambda x, y: (zoom(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def dataset_labels(ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y.numpy() for _, y in ds]).ravel().astype(int)


def load_image(path: str | Path, img_size: int = IMG_SIZE, channels: int = 1) -> np.ndarray:
    """One image as (img_size, img_size, channels) in [0, 1], preprocessed like make_dataset."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=channels, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return img.numpy() / 255.0
