import keras
import numpy as np
import pytest
from conftest import IMG_SIZE
from keras import layers

from xray_classifier.models import (
    LAST_STAGE_PREFIXES,
    MODEL_BUILDERS,
    build_convnext_tiny,
    build_custom_cnn,
    build_efficientnet_v2_b0,
    build_vgg16,
    pretrained_base,
    unfreeze_last_stage,
)

PRETRAINED = [build_vgg16, build_efficientnet_v2_b0, build_convnext_tiny]


def test_custom_cnn_takes_grayscale_and_outputs_probability():
    model = build_custom_cnn(IMG_SIZE)
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 1)
    assert model.output_shape == (None, 1)
    assert pretrained_base(model) is None


@pytest.mark.parametrize("build", PRETRAINED)
def test_pretrained_models_take_rgb_with_frozen_base(build):
    model = build(IMG_SIZE, weights=None)
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
    assert model.output_shape == (None, 1)
    base = pretrained_base(model)
    assert base is not None and not base.trainable
    assert all(not w.trainable for w in base.weights)


def test_vgg16_trains_only_its_dense_head():
    # Four Dense layers, kernel + bias each.
    assert len(build_vgg16(IMG_SIZE, weights=None).trainable_weights) == 8


@pytest.mark.parametrize("build", [build_efficientnet_v2_b0, build_convnext_tiny])
def test_pooled_models_train_only_the_output_layer(build):
    assert len(build(IMG_SIZE, weights=None).trainable_weights) == 2


def test_vgg16_applies_imagenet_preprocessing():
    model = build_vgg16(IMG_SIZE, weights=None)
    base = pretrained_base(model)
    flatten = next(layer for layer in model.layers if isinstance(layer, layers.Flatten))
    features = keras.Model(model.inputs, flatten.input)

    x = np.random.default_rng(0).random((2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    expected = base(keras.applications.vgg16.preprocess_input(x * 255.0))
    np.testing.assert_allclose(features(x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("build", [build_efficientnet_v2_b0, build_convnext_tiny])
def test_pooled_models_feed_0_255_pixels_to_the_backbone(build):
    model = build(IMG_SIZE, weights=None)
    base = pretrained_base(model)
    pool = next(layer for layer in model.layers if isinstance(layer, layers.GlobalAveragePooling2D))
    features = keras.Model(model.inputs, pool.input)

    x = np.random.default_rng(0).random((2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    np.testing.assert_allclose(features(x), base(x * 255.0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("build", PRETRAINED)
def test_pretrained_models_survive_save_and_load(build, tmp_path):
    model = build(IMG_SIZE, weights=None)
    model.save(tmp_path / "model.keras")
    loaded = keras.models.load_model(tmp_path / "model.keras")
    x = np.random.default_rng(1).random((2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    np.testing.assert_allclose(
        loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6
    )


def test_model_builders_cover_cli_names():
    assert set(MODEL_BUILDERS) == {"cnn", "vgg16", "efficientnetv2-b0", "convnext-tiny"}


@pytest.mark.parametrize("build", PRETRAINED)
def test_unfreeze_last_stage_trains_only_the_last_stage(build):
    model = build(IMG_SIZE, weights=None)
    base = pretrained_base(model)
    frozen_head_weights = len(model.trainable_weights)
    unfreeze_last_stage(model)

    prefixes = LAST_STAGE_PREFIXES[base.name]
    trainable = [layer for layer in base.layers if layer.trainable and layer.weights]
    assert trainable, "nothing was unfrozen"
    assert all(layer.name.startswith(prefixes) for layer in trainable)
    assert not any(isinstance(layer, layers.BatchNormalization) for layer in trainable)
    assert not base.layers[1].trainable  # the first stage stays frozen
    assert len(model.trainable_weights) > frozen_head_weights
    assert float(model.optimizer.learning_rate) == pytest.approx(1e-5)


def test_last_stage_prefixes_match_backbone_names():
    names = {pretrained_base(build(IMG_SIZE, weights=None)).name for build in PRETRAINED}
    assert names == set(LAST_STAGE_PREFIXES)


def test_unfreeze_last_stage_rejects_custom_cnn():
    with pytest.raises(ValueError, match="no pretrained base"):
        unfreeze_last_stage(build_custom_cnn(IMG_SIZE))
