import re
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from .config import CATEGORIES, IMG_SIZE

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

# Kaggle filenames: person<id>_<bacteria|virus>_<n>.jpeg, IM-<id>-<n>.jpeg, NORMAL2-IM-<id>-<n>.jpeg
_PATIENT_PATTERNS = (re.compile(r"^(person\d+)_"), re.compile(r"^(?:NORMAL2-)?(IM-\d+)-"))

Files = tuple[list[str], np.ndarray]


def list_images(split_dir: str | Path) -> Files:
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


def patient_id(path: str | Path) -> str:
    """Patient ID from a Kaggle filename; unrecognized names count as their own patient."""
    name = Path(path).name
    for pattern in _PATIENT_PATTERNS:
        if match := pattern.match(name):
            return match.group(1)
    return Path(path).stem


def split_by_patient(
    paths: list[str], labels: np.ndarray, val_fraction: float = 0.15, seed: int = 0
) -> tuple[Files, Files]:
    """Stratified train/validation split in which no patient appears on both sides."""
    if not 0 < val_fraction <= 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5], got {val_fraction}")
    splitter = StratifiedGroupKFold(
        n_splits=round(1 / val_fraction), shuffle=True, random_state=seed
    )
    groups = [patient_id(p) for p in paths]
    train_idx, val_idx = next(splitter.split(paths, labels, groups))
    paths_array = np.asarray(paths)
    return (
        (paths_array[train_idx].tolist(), labels[train_idx]),
        (paths_array[val_idx].tolist(), labels[val_idx]),
    )


def train_val_files(
    data_dir: str | Path, val_fraction: float = 0.15, seed: int = 0
) -> tuple[Files, Files]:
    """Train/validation files: val/ as provided if val_fraction=0, else train/ + val/ re-split.

    Re-splitting is the default because the provided val/ folder has only 16 images.
    """
    data_dir = Path(data_dir)
    if not val_fraction:
        return list_images(data_dir / "train"), list_images(data_dir / "val")
    paths, labels = list_images(data_dir / "train")
    if (data_dir / "val").is_dir():
        val_paths, val_labels = list_images(data_dir / "val")
        paths, labels = paths + val_paths, np.concatenate([labels, val_labels])
    return split_by_patient(paths, labels, val_fraction, seed)


def patient_overlap(paths_a: list[str], paths_b: list[str]) -> set[str]:
    return {patient_id(p) for p in paths_a} & {patient_id(p) for p in paths_b}


def _decode(path: tf.Tensor | str, img_size: int, channels: int) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=channels, expand_animations=False)
    return tf.image.resize(img, (img_size, img_size)) / 255.0


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    img_size: int = IMG_SIZE,
    channels: int = 1,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Batches of (images in [0, 1], labels); with shuffle=True, reshuffled every epoch."""
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
