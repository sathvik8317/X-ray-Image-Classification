import keras
import numpy as np
import pytest
from conftest import IMG_SIZE, first_image
from keras import layers

from xray_classifier.data import load_image
from xray_classifier.gradcam import gradcam
from xray_classifier.models import build_custom_cnn, build_efficientnet_v2_b0, build_vgg16
from xray_classifier.plots import plot_gradcam


@pytest.mark.parametrize(
    ("build", "channels"),
    [
        (lambda: build_custom_cnn(IMG_SIZE), 1),
        (lambda: build_vgg16(IMG_SIZE, weights=None), 3),
        (lambda: build_efficientnet_v2_b0(IMG_SIZE, weights=None), 3),
    ],
)
def test_heatmap_shape_and_range(data_dir, build, channels):
    image = load_image(first_image(data_dir, "test", "PNEUMONIA"), IMG_SIZE, channels)
    heatmap = gradcam(build(), image)
    assert heatmap.shape == (IMG_SIZE, IMG_SIZE)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


def _identity_model(output_weight):
    """feature map = image; output = sigmoid(output_weight * mean(image))."""
    inputs = keras.Input((16, 16, 1))
    features = layers.Conv2D(1, 1, use_bias=False, kernel_initializer="ones")(inputs)
    pooled = layers.GlobalAveragePooling2D()(features)
    output = layers.Dense(
        1, activation="sigmoid", kernel_initializer=keras.initializers.Constant(output_weight)
    )(pooled)
    return keras.Model(inputs, output)


def test_heatmap_highlights_the_region_driving_a_pneumonia_prediction():
    image = np.zeros((16, 16, 1), dtype="float32")
    image[2:6, 10:14] = 1.0
    heatmap = gradcam(_identity_model(output_weight=50.0), image)
    assert np.unravel_index(heatmap.argmax(), heatmap.shape) in {
        (r, c) for r in range(2, 6) for c in range(10, 14)
    }
    assert heatmap[12:, :4].max() < 0.1


def test_heatmap_survives_saturated_predictions():
    image = np.ones((16, 16, 1), dtype="float32")
    model = _identity_model(output_weight=1000.0)
    assert float(model(image[np.newaxis])[0, 0]) == 1.0
    assert gradcam(model, image).max() == pytest.approx(1.0)


def test_plot_gradcam():
    fig = plot_gradcam(np.zeros((8, 8, 1)), np.zeros((8, 8)), "title")
    assert len(fig.axes) == 3  # two panels and the colorbar


@pytest.mark.parametrize(
    ("build", "channels"),
    [
        (lambda: build_custom_cnn(IMG_SIZE), 1),
        (lambda: build_vgg16(IMG_SIZE, weights=None), 3),
    ],
)
def test_heatmap_matches_after_save_and_load(build, channels, tmp_path):
    model = build()
    model.save(tmp_path / "m.keras")
    loaded = keras.models.load_model(tmp_path / "m.keras")
    image = np.random.default_rng(0).random((IMG_SIZE, IMG_SIZE, channels)).astype("float32")
    np.testing.assert_allclose(gradcam(loaded, image), gradcam(model, image), atol=1e-5)
