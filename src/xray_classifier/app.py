import argparse
from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .data import load_image
from .gradcam import gradcam
from .plots import gradcam_overlay
from .predict import predict_image
from .thresholds import load_threshold

DISPLAY_SIZE = 512
DISCLAIMER = "Research demo only. This is not a medical device and must not be used for diagnosis."


def make_classifier(model_path: str | Path) -> Callable[[str], tuple[str, np.ndarray]]:
    """Function mapping an image path to (prediction text, Grad-CAM overlay)."""
    model = keras.models.load_model(model_path)
    threshold = load_threshold(model_path)
    img_size, channels = model.input_shape[1], model.input_shape[-1]

    def classify(image_path: str) -> tuple[str, np.ndarray]:
        label, prob = predict_image(model, image_path, threshold)
        heatmap = gradcam(model, load_image(image_path, img_size, channels), threshold)
        heatmap = tf.image.resize(heatmap[..., np.newaxis], (DISPLAY_SIZE, DISPLAY_SIZE))
        display = load_image(image_path, DISPLAY_SIZE, channels=1)
        text = f"{label} (pneumonia probability {prob:.2f}, threshold {threshold:.2f})"
        return text, gradcam_overlay(display, heatmap.numpy()[..., 0])

    return classify


def build_demo(model_path: str | Path):
    import gradio as gr  # optional dependency: pip install -e ".[app]"

    return gr.Interface(
        fn=make_classifier(model_path),
        inputs=gr.Image(type="filepath", label="Chest X-ray"),
        outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Grad-CAM")],
        title="Chest X-ray Pneumonia Classifier",
        description=f"Model: {Path(model_path).name}. {DISCLAIMER}",
        flagging_mode="never",
        api_name="predict",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a web demo for a trained model.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    try:
        demo = build_demo(args.model_path)
    except ImportError:
        parser.exit(1, 'The demo app needs Gradio: pip install -e ".[app]"\n')
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
