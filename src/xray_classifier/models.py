import keras
from keras import layers

from .config import IMG_SIZE


def build_custom_cnn(img_size: int = IMG_SIZE) -> keras.Model:
    """Three conv blocks on grayscale input, as in the original notebook."""
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
            layers.Dense(64),  # no activation, as in the original notebook
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="custom_cnn",
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_vgg16(img_size: int = IMG_SIZE, weights: str | None = "imagenet") -> keras.Model:
    """Frozen VGG16 base with a dense head on RGB input."""
    base = keras.applications.VGG16(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(base.input, output, name="vgg16_transfer")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {"cnn": build_custom_cnn, "vgg16": build_vgg16}
