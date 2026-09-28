import shutil

import keras
import numpy as np
import pytest
from conftest import IMG_SIZE

from xray_classifier.models import build_custom_cnn, build_vgg16
from xray_classifier.train import balanced_class_weights, main, train


def test_train_custom_cnn_saves_loadable_checkpoint(data_dir, tmp_path):
    checkpoint = tmp_path / "cnn.keras"
    history = train(
        build_custom_cnn(IMG_SIZE), data_dir, checkpoint, epochs=2, batch_size=4, val_fraction=0.5
    )
    assert set(history.history) >= {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert keras.models.load_model(checkpoint).input_shape == (None, IMG_SIZE, IMG_SIZE, 1)


def test_train_vgg16_with_augmentation(data_dir, tmp_path):
    checkpoint = tmp_path / "vgg16.keras"
    model = build_vgg16(IMG_SIZE, weights=None)
    train(model, data_dir, checkpoint, epochs=1, batch_size=4, augment=True, val_fraction=0, seed=0)
    assert keras.models.load_model(checkpoint).input_shape == (None, IMG_SIZE, IMG_SIZE, 3)


def test_cli_trains_and_writes_output(data_dir, tmp_path, capsys):
    output = tmp_path / "models" / "cnn.keras"
    main(
        [
            "--model=cnn",
            f"--data-dir={data_dir}",
            f"--output={output}",
            "--epochs=1",
            f"--img-size={IMG_SIZE}",
            "--seed=0",
            "--val-fraction=0.5",
        ]
    )
    assert output.exists()
    out = capsys.readouterr().out
    assert "Training on 8 images, validating on 8" in out
    assert f"Best model saved to {output}" in out


def test_balanced_class_weights():
    weights = balanced_class_weights(np.array([0] * 25 + [1] * 75))
    assert weights == {0: 2.0, 1: pytest.approx(2 / 3)}
    # Each class's total weight is equal.
    assert 25 * weights[0] == pytest.approx(75 * weights[1])


def test_class_weight_is_applied_on_imbalanced_data(data_dir, tmp_path):
    imbalanced = tmp_path / "imbalanced"
    shutil.copytree(data_dir, imbalanced)
    for path in sorted((imbalanced / "train" / "NORMAL").glob("*.jpeg"))[2:]:
        path.unlink()  # 2 NORMAL vs 4 PNEUMONIA

    def first_epoch_loss(class_weight):
        keras.utils.set_random_seed(0)
        history = train(
            build_custom_cnn(IMG_SIZE),
            imbalanced,
            tmp_path / f"cw_{class_weight}.keras",
            epochs=1,
            batch_size=6,
            val_fraction=0,
            class_weight=class_weight,
            seed=0,
        )
        return history.history["loss"][0]

    assert first_epoch_loss(True) != pytest.approx(first_epoch_loss(False), rel=1e-3)
