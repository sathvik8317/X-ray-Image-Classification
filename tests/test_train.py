import keras
from conftest import IMG_SIZE

from xray_classifier.models import build_custom_cnn, build_vgg16
from xray_classifier.train import main, train


def test_train_custom_cnn_saves_loadable_checkpoint(data_dir, tmp_path):
    checkpoint = tmp_path / "cnn.keras"
    history = train(build_custom_cnn(IMG_SIZE), data_dir, checkpoint, epochs=2, batch_size=4)
    assert set(history.history) >= {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert keras.models.load_model(checkpoint).input_shape == (None, IMG_SIZE, IMG_SIZE, 1)


def test_train_vgg16_with_augmentation(data_dir, tmp_path):
    checkpoint = tmp_path / "vgg16.keras"
    model = build_vgg16(IMG_SIZE, weights=None)
    train(model, data_dir, checkpoint, epochs=1, batch_size=4, augment=True, seed=0)
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
        ]
    )
    assert output.exists()
    assert f"Best model saved to {output}" in capsys.readouterr().out
