"""Export a pooled transfer model to ONNX for the browser demo, with a CAM output.

The pooled models end in GlobalAveragePooling2D -> Dropout -> Dense(1). For that head,
Grad-CAM equals the class activation map: the gradient of the logit with respect to each
feature channel is that channel's Dense weight divided by the map's area, a constant. So
the map is just the feature map projected onto the Dense weights, which the exported model
returns as a second output. The browser then needs no gradients: it flips the map's sign
for a NORMAL prediction, applies ReLU, normalizes and upsamples it.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
from keras import layers

from .thresholds import load_threshold


def with_cam_output(model: keras.Model) -> keras.Model:
    """Model mapping an image to (pneumonia probability, raw CAM of shape (h', w', 1))."""
    head = model.layers[-1]
    pool = next((ly for ly in model.layers if isinstance(ly, layers.GlobalAveragePooling2D)), None)
    if pool is None or not isinstance(head, layers.Dense):
        raise ValueError(
            f"{model.name} needs a GlobalAveragePooling2D -> Dense head for CAM export"
        )
    features_model = keras.Model(model.inputs, [pool.input, model.output])
    inputs = keras.Input(model.input_shape[1:], name="image")
    features, prob = features_model(inputs)
    project = layers.Dense(1, use_bias=False, name="cam")
    cam = project(features)
    project.set_weights([head.get_weights()[0]])
    prob = layers.Identity(name="prob")(prob)
    return keras.Model(inputs, [prob, cam], name=f"{model.name}_cam")


def export_onnx(model_path: Path, out_path: Path) -> dict:
    """Write <out>.onnx and a metadata JSON next to it; returns the metadata."""
    model = keras.models.load_model(model_path)
    cam_model = with_cam_output(model)
    # Keras only exports a model that has been called, which fixes its input signature.
    cam_model(np.zeros((1, *model.input_shape[1:]), dtype="float32"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Go through a SavedModel: Keras' direct ONNX export leaves the backbone's
    # Normalization constants as extra graph inputs; tf2onnx on a SavedModel freezes them.
    with tempfile.TemporaryDirectory() as saved:
        cam_model.export(saved, format="tf_saved_model", verbose=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                saved,
                "--output",
                str(out_path),
                "--opset",
                "17",
            ],
            check=True,
        )
    meta = {
        "source_model": model_path.name,
        "input_size": int(model.input_shape[1]),
        "channels": int(model.input_shape[-1]),
        "input_range": [0.0, 1.0],
        "layout": "NHWC",
        "threshold": load_threshold(model_path),
        "cam_size": int(cam_model.output_shape[1][1]),
        "outputs": ["prob", "cam"],
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a model to ONNX with a CAM output.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--out", type=Path, help="output .onnx path (default: next to the model)")
    args = parser.parse_args(argv)
    out = args.out or args.model_path.with_suffix(".onnx")
    meta = export_onnx(args.model_path, out)
    print(f"Saved {out} ({meta['input_size']}px input, {meta['cam_size']}px CAM)")


if __name__ == "__main__":
    main()
