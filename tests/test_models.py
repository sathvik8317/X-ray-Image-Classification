import keras
import numpy as np
from conftest import IMG_SIZE
from keras import layers

from xray_classifier.models import build_custom_cnn, build_vgg16


def test_custom_cnn_takes_grayscale_and_outputs_probability():
    model = build_custom_cnn(IMG_SIZE)
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 1)
    assert model.output_shape == (None, 1)


def test_vgg16_base_is_frozen():
    model = build_vgg16(IMG_SIZE, weights=None)
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    assert model.output_shape == (None, 1)
    # Only the four Dense layers of the head train (kernel + bias each).
    assert len(model.trainable_weights) == 8


def test_vgg16_applies_imagenet_preprocessing():
    model = build_vgg16(IMG_SIZE, weights=None)
    base = next(layer for layer in model.layers if layer.name.startswith("vgg16"))
    flatten = next(layer for layer in model.layers if isinstance(layer, layers.Flatten))
    features = keras.Model(model.inputs, flatten.input)

    x = np.random.default_rng(0).random((2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    expected = base(keras.applications.vgg16.preprocess_input(x * 255.0))
    np.testing.assert_allclose(features(x), expected, rtol=1e-5, atol=1e-5)


def test_vgg16_survives_save_and_load(tmp_path):
    model = build_vgg16(IMG_SIZE, weights=None)
    model.save(tmp_path / "vgg16.keras")
    loaded = keras.models.load_model(tmp_path / "vgg16.keras")
    x = np.random.default_rng(1).random((2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0))
