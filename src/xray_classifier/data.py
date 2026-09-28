from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import CATEGORIES, IMG_SIZE

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def list_images(split_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Image paths and labels (1 = PNEUMONIA) in a split folder, in a stable order."""
    paths, labels = [], []
    for label, category in enumerate(CATEGORIES):
        for path in sorted((Path(split_dir) / category).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(path))
                labels.append(label)
    if not paths:
        raise ValueError(f"No images found in {split_dir}")
    return paths, np.array(labels)


def class_counts(split_dir: str | Path) -> dict[str, int]:
    _, labels = list_images(split_dir)
    return {category: int((labels == i).sum()) for i, category in enumerate(CATEGORIES)}


def _decode(path: tf.Tensor | str, img_size: int, channels: int) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=channels, expand_animations=False)
    return tf.image.resize(img, (img_size, img_size)) / 255.0


def make_dataset(
    split_dir: str | Path,
    img_size: int = IMG_SIZE,
    channels: int = 1,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Batches of (images in [0, 1], labels) from a split folder; label 1 = PNEUMONIA.

    With shuffle=True the whole split is reshuffled every epoch.
    """
    paths, labels = list_images(split_dir)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype("float32")[:, np.newaxis]))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: (_decode(path, img_size, channels), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    if augment:
        # The notebook's ImageDataGenerator shear_range=0.2 meant 0.2 degrees, effectively a
        # no-op, so only the zoom augmentation is carried over.
        zoom = keras.layers.RandomZoom(0.2, seed=seed)
        ds = ds.map(lambda x, y: (zoom(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_image(path: str | Path, img_size: int = IMG_SIZE, channels: int = 1) -> np.ndarray:
    """One image as (img_size, img_size, channels) in [0, 1], preprocessed like make_dataset."""
    return _decode(str(path), img_size, channels).numpy()
