import shutil

import keras
import numpy as np
import pytest
from conftest import IMG_SIZE

from xray_classifier.models import (
    build_custom_cnn,
    build_efficientnet_v2_b0,
    build_vgg16,
    pretrained_base,
)
from xray_classifier.thresholds import load_threshold
from xray_classifier.train import balanced_class_weights, main, make_callbacks, train


def test_train_custom_cnn_saves_loadable_checkpoint(data_dir, tmp_path):
    checkpoint = tmp_path / "cnn.keras"
    result = train(
        build_custom_cnn(IMG_SIZE), data_dir, checkpoint, epochs=2, batch_size=4, val_fraction=0.5
    )
    assert set(result.history) >= {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert len(result.history["loss"]) == 2
    assert keras.models.load_model(checkpoint).input_shape == (None, IMG_SIZE, IMG_SIZE, 1)
    assert result.model.input_shape == (None, IMG_SIZE, IMG_SIZE, 1)


def test_train_saves_threshold_chosen_on_validation(data_dir, tmp_path):
    checkpoint = tmp_path / "cnn.keras"
    result = train(
        build_custom_cnn(IMG_SIZE), data_dir, checkpoint, epochs=1, batch_size=4, val_fraction=0.5
    )
    assert 0.0 < result.threshold <= 1.0
    assert load_threshold(checkpoint) == pytest.approx(result.threshold)
    assert result.val_metrics.threshold == result.threshold
    assert result.val_metrics.confusion_matrix.sum() == 8


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
    assert "Validation: Accuracy" in out
    assert output.with_suffix(".json").exists()


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
        result = train(
            build_custom_cnn(IMG_SIZE),
            imbalanced,
            tmp_path / f"cw_{class_weight}.keras",
            epochs=1,
            batch_size=6,
            val_fraction=0,
            class_weight=class_weight,
            seed=0,
        )
        return result.history["loss"][0]

    assert first_epoch_loss(True) != pytest.approx(first_epoch_loss(False), rel=1e-3)


def test_train_pooled_pretrained_model(data_dir, tmp_path):
    checkpoint = tmp_path / "effnet.keras"
    model = build_efficientnet_v2_b0(IMG_SIZE, weights=None)
    result = train(
        model, data_dir, checkpoint, epochs=1, batch_size=4, augment=True, val_fraction=0.5
    )
    assert result.model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    assert checkpoint.with_suffix(".json").exists()


def test_fine_tuning_adds_a_second_phase(data_dir, tmp_path):
    checkpoint = tmp_path / "vgg16.keras"
    model = build_vgg16(IMG_SIZE, weights=None)
    result = train(
        model, data_dir, checkpoint, epochs=1, batch_size=4, val_fraction=0.5, fine_tune_epochs=2
    )
    assert len(result.history["loss"]) == 3
    assert checkpoint.exists() and checkpoint.with_suffix(".json").exists()


def test_later_phase_checkpoint_only_saves_improvements(tmp_path):
    checkpoint_cb = next(
        cb
        for cb in make_callbacks(tmp_path / "m.keras", best_val_loss=0.25)
        if isinstance(cb, keras.callbacks.ModelCheckpoint)
    )
    assert checkpoint_cb.best == 0.25


def test_cli_rejects_fine_tuning_the_custom_cnn(data_dir, capsys):
    with pytest.raises(SystemExit):
        main(["--model=cnn", f"--data-dir={data_dir}", "--fine-tune-epochs=2"])
    assert "needs a pretrained model" in capsys.readouterr().err


def test_fine_tuning_updates_last_stage_weights_only(data_dir, tmp_path):
    model = build_vgg16(IMG_SIZE, weights=None)
    base = pretrained_base(model)
    before = {
        name: base.get_layer(name).get_weights()[0].copy()
        for name in ("block1_conv1", "block5_conv3")
    }
    train(
        model,
        data_dir,
        tmp_path / "vgg16.keras",
        epochs=1,
        batch_size=4,
        val_fraction=0.5,
        fine_tune_epochs=1,
        fine_tune_learning_rate=1e-3,
    )
    # `model` is the in-memory model that went through both phases.
    assert np.array_equal(base.get_layer("block1_conv1").get_weights()[0], before["block1_conv1"])
    assert not np.array_equal(
        base.get_layer("block5_conv3").get_weights()[0], before["block5_conv3"]
    )
