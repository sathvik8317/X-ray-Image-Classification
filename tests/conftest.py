import numpy as np
import pytest
import tensorflow as tf

from xray_classifier.config import CATEGORIES

IMAGES_PER_CLASS = 4
IMG_SIZE = 32  # smallest input VGG16 accepts


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A tiny chest_xray-style dataset: {train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg."""
    root = tmp_path_factory.mktemp("chest_xray")
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for label, category in enumerate(CATEGORIES):
            folder = root / split / category
            folder.mkdir(parents=True)
            for i in range(IMAGES_PER_CLASS):
                base = 60 if label == 0 else 190
                img = np.clip(rng.normal(base, 25, (48, 64, 1)), 0, 255).astype(np.uint8)
                tf.io.write_file(str(folder / f"img_{i}.jpeg"), tf.io.encode_jpeg(img))
    (root / "train" / "NORMAL" / ".DS_Store").write_bytes(b"not an image")
    return root


@pytest.fixture(scope="session")
def trained_cnn_path(data_dir, tmp_path_factory):
    from xray_classifier.models import build_custom_cnn
    from xray_classifier.train import train

    path = tmp_path_factory.mktemp("models") / "cnn.keras"
    train(build_custom_cnn(IMG_SIZE), data_dir, path, epochs=1, batch_size=4)
    return path
