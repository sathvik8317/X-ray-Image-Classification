"""Hugging Face Space entry point: the xray_classifier Gradio demo with a bundled model.

The Space holds model.keras, its threshold in model.json, and a few test-set images in
examples/. See space/README.md in the GitHub repo for how to publish it.
"""

from pathlib import Path

from xray_classifier.app import build_demo

HERE = Path(__file__).parent

demo = build_demo(HERE / "model.keras", sorted((HERE / "examples").glob("*.jpeg")))

if __name__ == "__main__":
    demo.launch()
