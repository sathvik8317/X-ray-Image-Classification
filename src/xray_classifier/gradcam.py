import keras
import numpy as np
import tensorflow as tf
from keras import layers

from .thresholds import DEFAULT_THRESHOLD


def gradcam(
    model: keras.Model, image: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Grad-CAM heatmap in [0, 1] (same height and width as `image`) for the predicted class.

    `image` is one preprocessed image as returned by `load_image`, shape (h, w, channels).
    """
    output_layer = model.layers[-1]
    if not isinstance(output_layer, layers.Dense):
        raise ValueError("expected the model to end in a Dense output layer")
    grad_model = _features_and_head(model, output_layer)
    kernel, bias = output_layer.kernel, output_layer.bias

    with tf.GradientTape() as tape:
        features, head = grad_model(tf.convert_to_tensor(image[np.newaxis]), training=False)
        # Use the logit: the sigmoid saturates on confident predictions and kills the gradient.
        logit = (tf.matmul(head, kernel) + bias)[0, 0]
        predicted_pneumonia = tf.sigmoid(logit) >= threshold
        score = tf.where(predicted_pneumonia, logit, -logit)
    grads = tape.gradient(score, features)[0]

    channel_weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(features[0] * channel_weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + keras.config.epsilon())
    cam = tf.image.resize(cam[..., tf.newaxis], image.shape[:2])[..., 0]
    return cam.numpy()


def _features_and_head(model: keras.Model, output_layer: layers.Layer) -> keras.Model:
    """Model mapping the input to (last spatial feature map, input of the output layer)."""
    # The last spatial feature map is whatever the model flattens or pools.
    reducer = next(
        layer
        for layer in model.layers
        if isinstance(layer, layers.Flatten | layers.GlobalAveragePooling2D)
    )
    if not isinstance(model, keras.Sequential):
        return keras.Model(model.inputs, [reducer.input, output_layer.input])
    # A reloaded Sequential model's layer tensors are not connected to model.inputs, so
    # rebuild the graph from its layers (same weights).
    inputs = keras.Input(model.input_shape[1:])
    x, outputs = inputs, {}
    for layer in model.layers:
        if layer is reducer or layer is output_layer:
            outputs[layer.name] = x
        x = layer(x)
    return keras.Model(inputs, [outputs[reducer.name], outputs[output_layer.name]])
