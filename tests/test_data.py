import numpy as np
import pytest
from conftest import IMAGES_PER_CLASS, IMG_SIZE, first_image

from xray_classifier.data import (
    class_counts,
    list_images,
    load_image,
    make_dataset,
    patient_id,
    patient_overlap,
    split_by_patient,
    train_val_files,
)

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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("person1_bacteria_1.jpeg", "person1"),
        ("person1_virus_6.jpeg", "person1"),
        ("person1946_bacteria_4874.jpeg", "person1946"),
        ("IM-0115-0001.jpeg", "IM-0115"),
        ("NORMAL2-IM-1427-0001.jpeg", "IM-1427"),
        ("scan_42.png", "scan_42"),
    ],
)
def test_patient_id(filename, expected):
    assert patient_id(f"/data/train/X/{filename}") == expected


def _patients(n_normal, n_pneumonia, images_per_patient=3):
    paths, labels = [], []
    for p in range(n_normal):
        for i in range(images_per_patient):
            paths.append(f"NORMAL/IM-{p:04d}-{i:04d}.jpeg")
            labels.append(0)
    for p in range(n_pneumonia):
        for i in range(images_per_patient):
            paths.append(f"PNEUMONIA/person{p}_virus_{i}.jpeg")
            labels.append(1)
    return paths, np.array(labels)


def test_split_by_patient_keeps_patients_together_and_stratifies():
    paths, labels = _patients(n_normal=30, n_pneumonia=90)
    (train_paths, train_labels), (val_paths, val_labels) = split_by_patient(paths, labels, 0.2)
    assert not patient_overlap(train_paths, val_paths)
    assert sorted(train_paths + val_paths) == sorted(paths)
    assert len(val_paths) == pytest.approx(0.2 * len(paths), rel=0.15)
    assert val_labels.mean() == pytest.approx(labels.mean(), abs=0.05)
    assert len(train_labels) == len(train_paths)


def test_split_by_patient_is_deterministic():
    paths, labels = _patients(n_normal=10, n_pneumonia=20)
    assert (
        split_by_patient(paths, labels, seed=3)[1][0]
        == split_by_patient(paths, labels, seed=3)[1][0]
    )


@pytest.mark.parametrize("val_fraction", [0, 0.6, -0.1])
def test_split_by_patient_rejects_bad_fraction(val_fraction):
    paths, labels = _patients(n_normal=4, n_pneumonia=4)
    with pytest.raises(ValueError, match="val_fraction"):
        split_by_patient(paths, labels, val_fraction)


def test_train_val_files_uses_provided_val_folder_when_fraction_is_zero(data_dir):
    (train_paths, _), (val_paths, _) = train_val_files(data_dir, val_fraction=0)
    assert train_paths == list_images(data_dir / "train")[0]
    assert val_paths == list_images(data_dir / "val")[0]


def test_train_val_files_pools_and_resplits_by_patient(data_dir):
    (train_paths, train_labels), (val_paths, val_labels) = train_val_files(data_dir, 0.5)
    assert len(train_paths) + len(val_paths) == 2 * N_IMAGES
    assert not patient_overlap(train_paths, val_paths)
    assert set(val_labels) == {0, 1} and set(train_labels) == {0, 1}


def _dataset(data_dir, split, **kwargs):
    paths, labels = list_images(data_dir / split)
    return make_dataset(paths, labels, IMG_SIZE, **kwargs)


@pytest.mark.parametrize("channels", [1, 3])
def test_make_dataset_shapes_labels_and_range(data_dir, channels):
    ds = _dataset(data_dir, "train", channels=channels, batch_size=4)
    images, labels = next(iter(ds))
    assert images.shape == (4, IMG_SIZE, IMG_SIZE, channels)
    assert 0.0 <= float(images.numpy().min()) and float(images.numpy().max()) <= 1.0
    assert labels.shape == (4, 1)
    all_labels = np.concatenate([y.numpy() for _, y in ds]).ravel()
    assert all_labels.tolist() == LABELS


def test_shuffled_dataset_reshuffles_every_epoch(data_dir):
    ds = _dataset(data_dir, "train", batch_size=N_IMAGES, shuffle=True, seed=0)
    epoch_1 = next(iter(ds))[0].numpy()
    epoch_2 = next(iter(ds))[0].numpy()
    assert not np.array_equal(epoch_1, epoch_2)
    # Same images, different order.
    np.testing.assert_allclose(np.sort(epoch_1.ravel()), np.sort(epoch_2.ravel()))


def test_augmented_dataset_keeps_shape(data_dir):
    ds = _dataset(data_dir, "train", batch_size=4, augment=True, shuffle=True, seed=0)
    images, _ = next(iter(ds))
    assert images.shape == (4, IMG_SIZE, IMG_SIZE, 1)


@pytest.mark.parametrize("channels", [1, 3])
def test_load_image_matches_dataset_preprocessing(data_dir, channels):
    image = load_image(first_image(data_dir, "test", "NORMAL"), IMG_SIZE, channels=channels)
    assert image.shape == (IMG_SIZE, IMG_SIZE, channels)
    first_batch, _ = next(iter(_dataset(data_dir, "test", channels=channels, batch_size=1)))
    np.testing.assert_allclose(image, first_batch[0].numpy(), atol=1e-6)
