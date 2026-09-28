import numpy as np
import pytest
from conftest import IMAGES_PER_CLASS, IMG_SIZE

from xray_classifier.data import class_counts, list_images, load_image, make_dataset

N_IMAGES = 2 * IMAGES_PER_CLASS
LABELS = [0] * IMAGES_PER_CLASS + [1] * IMAGES_PER_CLASS


def test_list_images_ignores_non_images(data_dir):
    paths, labels = list_images(data_dir / "train")
    assert len(paths) == N_IMAGES
    assert all(p.endswith(".jpeg") for p in paths)
    assert labels.tolist() == LABELS


def test_list_images_empty_folder_raises(tmp_path):
    for category in ("NORMAL", "PNEUMONIA"):
        (tmp_path / category).mkdir()
    with pytest.raises(ValueError, match="No images found"):
        list_images(tmp_path)


def test_class_counts(data_dir):
    assert class_counts(data_dir / "train") == {
        "NORMAL": IMAGES_PER_CLASS,
        "PNEUMONIA": IMAGES_PER_CLASS,
    }


@pytest.mark.parametrize("channels", [1, 3])
def test_make_dataset_shapes_labels_and_range(data_dir, channels):
    ds = make_dataset(data_dir / "train", IMG_SIZE, channels=channels, batch_size=4)
    images, labels = next(iter(ds))
    assert images.shape == (4, IMG_SIZE, IMG_SIZE, channels)
    assert 0.0 <= float(images.numpy().min()) and float(images.numpy().max()) <= 1.0
    assert labels.shape == (4, 1)
    all_labels = np.concatenate([y.numpy() for _, y in ds]).ravel()
    assert all_labels.tolist() == LABELS


def test_shuffled_dataset_reshuffles_every_epoch(data_dir):
    ds = make_dataset(data_dir / "train", IMG_SIZE, batch_size=N_IMAGES, shuffle=True, seed=0)
    epoch_1 = next(iter(ds))[0].numpy()
    epoch_2 = next(iter(ds))[0].numpy()
    assert not np.array_equal(epoch_1, epoch_2)
    # Same images, different order.
    np.testing.assert_allclose(np.sort(epoch_1.ravel()), np.sort(epoch_2.ravel()))


def test_augmented_dataset_keeps_shape(data_dir):
    ds = make_dataset(
        data_dir / "train", IMG_SIZE, batch_size=4, augment=True, shuffle=True, seed=0
    )
    images, _ = next(iter(ds))
    assert images.shape == (4, IMG_SIZE, IMG_SIZE, 1)


@pytest.mark.parametrize("channels", [1, 3])
def test_load_image_matches_dataset_preprocessing(data_dir, channels):
    path = data_dir / "test" / "NORMAL" / "img_0.jpeg"
    image = load_image(path, IMG_SIZE, channels=channels)
    assert image.shape == (IMG_SIZE, IMG_SIZE, channels)
    first_batch, _ = next(iter(make_dataset(data_dir / "test", IMG_SIZE, channels, batch_size=1)))
    np.testing.assert_allclose(image, first_batch[0].numpy(), atol=1e-6)
