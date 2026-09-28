import shutil

import keras
import numpy as np
import pytest
from conftest import IMAGES_PER_CLASS

from xray_classifier.evaluate import compute_metrics, main, predict_split
from xray_classifier.plots import plot_confusion_matrix, plot_history, plot_roc_curve


def test_compute_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m.accuracy == 0.5
    assert m.roc_auc == pytest.approx(0.75)
    assert m.confusion_matrix.tolist() == [[1, 1], [1, 1]]
    assert "NORMAL" in m.report and "PNEUMONIA" in m.report


def test_compute_metrics_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    assert compute_metrics(y_true, y_prob, threshold=0.35).confusion_matrix.tolist() == [
        [1, 1],
        [0, 2],
    ]


def test_predict_split_returns_label_and_probability_per_image(data_dir, trained_cnn_path):
    model = keras.models.load_model(trained_cnn_path)
    y_true, y_prob = predict_split(model, data_dir / "test")
    assert y_true.tolist() == [0] * IMAGES_PER_CLASS + [1] * IMAGES_PER_CLASS
    assert y_prob.shape == (2 * IMAGES_PER_CLASS,)
    assert ((0 <= y_prob) & (y_prob <= 1)).all()


def test_cli_prints_metrics_and_saves_plots(data_dir, trained_cnn_path, tmp_path, capsys):
    main([str(trained_cnn_path), f"--data-dir={data_dir}", f"--plots-dir={tmp_path}"])
    out = capsys.readouterr().out
    assert "Accuracy:" in out and "ROC AUC:" in out
    assert (tmp_path / "cnn_confusion_matrix.png").exists()
    assert (tmp_path / "cnn_roc_curve.png").exists()


def test_plots_return_figures():
    history = keras.callbacks.History()
    history.history = {"accuracy": [0.5, 0.7], "val_accuracy": [0.4, 0.6], "loss": [1, 0.5],
                       "val_loss": [1.2, 0.8]}  # fmt: skip
    assert len(plot_history(history).axes) == 2
    assert plot_confusion_matrix(np.array([[3, 1], [0, 4]])).axes[0].get_title()
    assert plot_roc_curve(np.array([0, 1]), np.array([0, 1]), 0.5).axes[0].get_title()


def test_cli_warns_when_test_patients_appear_in_train(data_dir, trained_cnn_path, tmp_path, capsys):
    leaky = tmp_path / "leaky"
    shutil.copytree(data_dir, leaky)
    train_image = sorted((leaky / "train" / "PNEUMONIA").glob("*.jpeg"))[0]
    shutil.copy(train_image, leaky / "test" / "PNEUMONIA" / train_image.name)

    main([str(trained_cnn_path), f"--data-dir={leaky}"])
    assert "1 patient(s) appear in both train/ and test/" in capsys.readouterr().out

    main([str(trained_cnn_path), f"--data-dir={data_dir}"])
    assert "appear in both" not in capsys.readouterr().out
