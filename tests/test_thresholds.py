import numpy as np
import pytest

from xray_classifier.thresholds import (
    choose_threshold,
    load_threshold,
    save_threshold,
    threshold_path,
)


def test_choose_threshold_separates_classes():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.85, 0.9])
    # Any cutoff in (0.7, 0.8] separates perfectly; the midpoint of the gap is chosen.
    assert choose_threshold(y_true, y_prob) == pytest.approx(0.75)


def test_choose_threshold_stays_off_saturated_scores():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.01, 0.2, 1.0, 1.0, 1.0])
    threshold = choose_threshold(y_true, y_prob)
    assert threshold == pytest.approx(0.6)


def test_choose_threshold_moves_below_half_when_scores_run_low():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.3, 0.35, 0.4])
    threshold = choose_threshold(y_true, y_prob)
    assert threshold < 0.5
    assert ((y_prob >= threshold) == y_true.astype(bool)).all()


def test_choose_threshold_single_class_falls_back_to_default():
    assert choose_threshold(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])) == 0.5


def test_save_and_load_threshold(tmp_path):
    model_path = tmp_path / "cnn.keras"
    assert load_threshold(model_path) == 0.5
    save_threshold(model_path, 0.37)
    assert threshold_path(model_path) == tmp_path / "cnn.json"
    assert load_threshold(model_path) == pytest.approx(0.37)
