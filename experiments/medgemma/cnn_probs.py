"""Write the CNN's test-set pneumonia probabilities to JSON for `disagreement.py`.

Run in its own process (TensorFlow) before `disagreement.py` (PyTorch), so the two
frameworks never hold GPU memory at the same time.
"""

import argparse
import json
from pathlib import Path

import keras

from xray_classifier.data import list_images
from xray_classifier.evaluate import predict_files
from xray_classifier.thresholds import load_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_path", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("cnn_probs.json"))
    args = parser.parse_args()

    paths, labels = list_images(args.split_dir)
    probs = predict_files(keras.models.load_model(args.model_path), paths, labels)
    record = {
        "paths": paths,
        "labels": labels.tolist(),
        "probs": probs.tolist(),
        "threshold": load_threshold(args.model_path),
    }
    args.out.write_text(json.dumps(record))
    print(f"Saved {len(paths)} probabilities to {args.out}")


if __name__ == "__main__":
    main()
