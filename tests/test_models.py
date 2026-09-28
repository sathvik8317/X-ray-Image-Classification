from conftest import IMG_SIZE

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
