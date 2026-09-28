import os
from pathlib import Path

CATEGORIES = ("NORMAL", "PNEUMONIA")
IMG_SIZE = 100

DATA_DIR = Path(os.environ.get("XRAY_DATA_DIR", "data/chest_xray"))
MODELS_DIR = Path(os.environ.get("XRAY_MODELS_DIR", "models"))
