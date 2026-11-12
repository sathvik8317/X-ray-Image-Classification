import json
import shutil

import keras
import numpy as np
import pytest
from conftest import IMAGES_PER_CLASS

from xray_classifier.evaluate import (
    bootstrap_intervals,
    compute_metrics,
    expected_calibration_error,
    main,
    predict_split,
    reliability,
)
from xray_classifier.plots import (
    plot_confusion_matrix,
    plot_history,
    plot_reliability_diagram,
    plot_roc_curve,
)


def test_compute_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m.accuracy == 0.5
    assert m.roc_auc == pytest.approx(0.75)
    assert m.confusion_matrix.tolist() == [[1, 1], [1, 1]]
    assert m.sensitivity == 0.5 and m.specificity == 0.5
    assert "NORMAL" in m.report and "PNEUMONIA" in m.report


def test_compute_metrics_sensitivity_and_specificity():
    # 3 of 4 pneumonia cases caught, 1 of 2 normal cases cleared.
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.2, 0.7, 0.9, 0.8, 0.6, 0.3])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m.sensitivity == 0.75
    assert m.specificity == 0.5
    assert "Sensitivity 0.7500" in m.summary()


def test_compute_metrics_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.35)
    assert m.confusion_matrix.tolist() == [[1, 1], [0, 2]]
    assert m.threshold == 0.35


def test_predict_split_returns_label_and_probability_per_image(data_dir, trained_cnn_path):
    model = keras.models.load_model(trained_cnn_path)
    y_true, y_prob = predict_split(model, data_dir / "test")
    assert y_true.tolist() == [0] * IMAGES_PER_CLASS + [1] * IMAGES_PER_CLASS
    assert y_prob.shape == (2 * IMAGES_PER_CLASS,)
    assert ((0 <= y_prob) & (y_prob <= 1)).all()


def test_cli_prints_metrics_and_saves_plots(data_dir, trained_cnn_path, tmp_path, capsys):
    main([str(trained_cnn_path), f"--data-dir={data_dir}", f"--plots-dir={tmp_path}"])
    out = capsys.readouterr().out
    assert "Using threshold" in out and "cnn.json" in out
    assert all(k in out for k in ("Accuracy:", "Sensitivity:", "Specificity:", "ROC AUC:"))
    assert (tmp_path / "cnn_confusion_matrix.png").exists()
    assert (tmp_path / "cnn_roc_curve.png").exists()
    assert (tmp_path / "cnn_reliability.png").exists()
    assert "95% CI" in out and "ECE:" in out


def test_cli_writes_json(data_dir, trained_cnn_path, tmp_path):
    out = tmp_path / "metrics" / "cnn.json"
    main([str(trained_cnn_path), f"--data-dir={data_dir}", "--bootstrap=50", f"--json={out}"])
    record = json.loads(out.read_text())
    assert record["model"] == "cnn" and record["split"] == "test"
    assert set(record["ci95"]) == {"accuracy", "sensitivity", "specificity", "roc_auc"}
    low, high = record["ci95"]["accuracy"]
    assert low <= record["accuracy"] <= high


def test_reliability_bins_and_ece():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    rel = reliability(y_true, y_prob, n_bins=10)
    np.testing.assert_allclose(rel.mean_predicted, [0.05, 0.15, 0.975])
    np.testing.assert_allclose(rel.observed_rate, [0.0, 0.5, 1.0])
    assert rel.counts.tolist() == [1, 2, 2]
    # |0.05-0| * 1 + |0.15-0.5| * 2 + |0.975-1| * 2, over 5 images
    assert expected_calibration_error(rel) == pytest.approx((0.05 + 0.7 + 0.05) / 5)


def test_perfect_calibration_has_zero_ece():
    y_true = np.array([0] * 5 + [1] * 5)
    y_prob = np.full(10, 0.5)
    assert compute_metrics(y_true, y_prob).ece == pytest.approx(0.0)


def test_bootstrap_intervals_contain_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    point = compute_metrics(y_true, y_prob)
    intervals = bootstrap_intervals(y_true, y_prob, n_resamples=200)
    for name, (low, high) in intervals.items():
        assert low <= getattr(point, name) <= high
        assert high - low < 0.25


def test_patient_bootstrap_is_wider_than_image_bootstrap():
    # 20 patients with 10 identical images each: really only 20 independent samples.
    rng = np.random.default_rng(1)
    patient_true = rng.integers(0, 2, 20)
    # A weak signal, so some patients are misclassified and the intervals have width.
    patient_prob = np.clip(patient_true * 0.2 + rng.uniform(0.2, 0.7, 20), 0, 1)
    y_true, y_prob = np.repeat(patient_true, 10), np.repeat(patient_prob, 10)
    groups = np.repeat(np.arange(20), 10)
    by_image = bootstrap_intervals(y_true, y_prob, n_resamples=300)
    by_patient = bootstrap_intervals(y_true, y_prob, groups=groups, n_resamples=300)

    def width(intervals):
        return intervals["accuracy"][1] - intervals["accuracy"][0]

    assert width(by_patient) > 1.5 * width(by_image)


def test_plots_return_figures():
    history = {"accuracy": [0.5, 0.7], "val_accuracy": [0.4, 0.6], "loss": [1, 0.5],
               "val_loss": [1.2, 0.8]}  # fmt: skip
    assert len(plot_history(history).axes) == 2
    assert plot_confusion_matrix(np.array([[3, 1], [0, 4]])).axes[0].get_title()
    assert plot_roc_curve(np.array([0, 1]), np.array([0, 1]), 0.5).axes[0].get_title()
    rel = reliability(np.array([0, 1, 1]), np.array([0.2, 0.7, 0.9]))
    assert plot_reliability_diagram(rel, 0.1).axes[0].get_title()


def test_cli_warns_when_test_patients_appear_in_train(data_dir, trained_cnn_path, tmp_path, capsys):
    leaky = tmp_path / "leaky"
    shutil.copytree(data_dir, leaky)
    train_image = sorted((leaky / "train" / "PNEUMONIA").glob("*.jpeg"))[0]
    shutil.copy(train_image, leaky / "test" / "PNEUMONIA" / train_image.name)

    main([str(trained_cnn_path), f"--data-dir={leaky}"])
    assert "1 patient(s) appear in both train/ and test/" in capsys.readouterr().out

    main([str(trained_cnn_path), f"--data-dir={data_dir}"])
    assert "appear in both" not in capsys.readouterr().out
