import numpy as np
import pytest
from conftest import first_image

from xray_classifier.app import DISPLAY_SIZE, build_demo, make_classifier
from xray_classifier.plots import gradcam_overlay


def test_classifier_returns_prediction_text_and_overlay(data_dir, trained_cnn_path):
    classify = make_classifier(trained_cnn_path)
    text, overlay = classify(str(first_image(data_dir, "test", "PNEUMONIA")))
    assert text.split()[0] in {"NORMAL", "PNEUMONIA"}
    assert "pneumonia probability" in text and "threshold" in text
    assert overlay.shape == (DISPLAY_SIZE, DISPLAY_SIZE, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_overlay_blends_image_and_heatmap():
    image = np.full((4, 4, 1), 0.5, dtype="float32")
    heatmap = np.zeros((4, 4))
    heatmap[0, 0] = 1.0
    overlay = gradcam_overlay(image, heatmap)
    assert overlay.shape == (4, 4, 3) and overlay.dtype == np.uint8
    assert not np.array_equal(overlay[0, 0], overlay[3, 3])


def test_build_demo(trained_cnn_path):
    gr = pytest.importorskip("gradio")
    assert isinstance(build_demo(trained_cnn_path), gr.Interface)
