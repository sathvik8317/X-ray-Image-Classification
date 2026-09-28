from collections.abc import Callable

import keras
from keras import layers

from .config import IMG_SIZE

PRETRAINED_IMG_SIZE = 224


def _compile(model: keras.Model) -> keras.Model:
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_custom_cnn(img_size: int = IMG_SIZE) -> keras.Model:
    """Three conv blocks on grayscale input, inputs scaled to [0, 1]."""
    model = keras.Sequential(
        [
            keras.Input(shape=(img_size, img_size, 1)),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            layers.Conv2D(256, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="custom_cnn",
    )
    return _compile(model)


def build_vgg16(img_size: int = IMG_SIZE, weights: str | None = "imagenet") -> keras.Model:
    """Frozen VGG16 base with a dense head on RGB input, inputs scaled to [0, 1]."""
    base = keras.applications.VGG16(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    # The ImageNet weights expect VGG16's own preprocessing (0-255 BGR, mean-subtracted).
    x = keras.applications.vgg16.preprocess_input(inputs * 255.0)
    x = layers.Flatten()(base(x, training=False))
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, output, name="vgg16_transfer"))


def _build_pooled_transfer(
    backbone: Callable[..., keras.Model], img_size: int, weights: str | None, name: str
) -> keras.Model:
    base = backbone(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
        include_preprocessing=True,
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    # The backbone's built-in preprocessing expects 0-255 pixels. training=False keeps its
    # BatchNorm statistics frozen, including when the base is unfrozen for fine-tuning.
    x = base(inputs * 255.0, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, output, name=name))


def build_efficientnet_v2_b0(
    img_size: int = PRETRAINED_IMG_SIZE, weights: str | None = "imagenet"
) -> keras.Model:
    """Frozen EfficientNetV2-B0 base with a pooled linear head, inputs scaled to [0, 1]."""
    return _build_pooled_transfer(
        keras.applications.EfficientNetV2B0, img_size, weights, "efficientnetv2b0_transfer"
    )


def build_convnext_tiny(
    img_size: int = PRETRAINED_IMG_SIZE, weights: str | None = "imagenet"
) -> keras.Model:
    """Frozen ConvNeXt-Tiny base with a pooled linear head, inputs scaled to [0, 1]."""
    return _build_pooled_transfer(
        keras.applications.ConvNeXtTiny, img_size, weights, "convnext_tiny_transfer"
    )


MODEL_BUILDERS = {
    "cnn": build_custom_cnn,
    "vgg16": build_vgg16,
    "efficientnetv2-b0": build_efficientnet_v2_b0,
    "convnext-tiny": build_convnext_tiny,
}


def pretrained_base(model: keras.Model) -> keras.Model | None:
    """The nested pretrained backbone of a transfer model, or None for the custom CNN."""
    return next((layer for layer in model.layers if isinstance(layer, keras.Model)), None)
