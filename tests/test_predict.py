import keras
from conftest import first_image

from xray_classifier.predict import main, predict_image


def test_predict_image_returns_class_and_probability(data_dir, trained_cnn_path):
    model = keras.models.load_model(trained_cnn_path)
    label, prob = predict_image(model, first_image(data_dir, "test", "PNEUMONIA"))
    assert label in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= prob <= 1.0
    assert label == ("PNEUMONIA" if prob >= 0.5 else "NORMAL")


def test_threshold_controls_label(data_dir, trained_cnn_path):
    model = keras.models.load_model(trained_cnn_path)
    image = first_image(data_dir, "test", "NORMAL")
    assert predict_image(model, image, threshold=0.0)[0] == "PNEUMONIA"
    assert predict_image(model, image, threshold=1.01)[0] == "NORMAL"


def test_cli_prints_one_line_per_image(data_dir, trained_cnn_path, capsys):
    images = sorted((data_dir / "test" / "NORMAL").glob("*.jpeg"))[:2]
    main([str(trained_cnn_path), *map(str, images)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all("pneumonia probability" in line for line in lines)
