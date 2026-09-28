# Chest X‑Ray Pneumonia Classification

Convolutional neural networks (CNNs) that classify chest X‑ray images as **Normal** (healthy) or **Pneumonia**. The code is a small Python package, `xray_classifier`, with command-line tools for training, evaluation, and prediction, plus a notebook that walks through the whole workflow.

---

## 📖 Table of Contents

1. [About](#about)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Project Setup](#project-setup)

   * [Clone Repository](#clone-repository)
   * [Install Dependencies](#install-dependencies)
   * [Download Dataset](#download-dataset)
   * [Development Setup](#development-setup)
5. [Usage](#usage)

   * [Command Line](#command-line)
   * [Notebook](#notebook)
   * [Configuration](#configuration)
6. [Results](#results)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

---

## About

The project trains and compares two models on the public [Chest X‑Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):

* **Custom CNN**: three Conv2D + MaxPooling + Dropout blocks on 100x100 grayscale images.
* **VGG16 transfer learning**: an ImageNet-pretrained VGG16 base (frozen) with a dense head, on 100x100 RGB images with random zoom augmentation.

Features:

* A `tf.data` input pipeline that streams images from the `train/`, `val/`, and `test/` folders.
* Training with early stopping, learning-rate reduction on plateau, and checkpointing of the best model (`.keras` format).
* Evaluation with accuracy, precision/recall/F1, a confusion matrix, and an ROC curve with AUC.
* Plots of training curves, confusion matrix, and ROC curve.
* Single-image prediction using the same preprocessing as training.

## Project Structure

```
.
├── X_ray_Image_Classification.ipynb   # walkthrough notebook (Colab-ready)
├── src/xray_classifier/
│   ├── config.py      # class names, image size, data and model paths
│   ├── data.py        # tf.data pipeline and single-image loading
│   ├── models.py      # custom CNN and VGG16 model builders
│   ├── train.py       # training loop with callbacks (xray-train)
│   ├── evaluate.py    # metrics on a data split (xray-evaluate)
│   ├── predict.py     # single-image prediction (xray-predict)
│   └── plots.py       # training curves, confusion matrix, ROC curve
├── tests/             # pytest suite, runs on a small synthetic dataset
└── pyproject.toml     # package metadata, dependencies, tool config
```

## Prerequisites

* **Python 3.10+**
* A **Kaggle account** and API token, to download the dataset

## Project Setup

### Clone Repository

```bash
git clone https://github.com/sathvik8317/X-ray-Image-Classification.git
cd X-ray-Image-Classification
```

### Install Dependencies

Create a virtual environment (recommended) and install the package:

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

This installs `xray_classifier` in editable mode along with its dependencies (TensorFlow, NumPy, Matplotlib, scikit-learn, Kaggle), which are listed in `pyproject.toml`.

### Download Dataset

1. Place your Kaggle API token (`kaggle.json`) in `~/.kaggle/`.
2. Run:

   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data --unzip
   ```
3. After extraction, the folder structure should be:

   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

You can also download the zip manually from the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and unzip it into `data/`.

### Development Setup

For contributors, install the dev tools and enable the git hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

The hooks run [Ruff](https://docs.astral.sh/ruff/) (lint and format) on the code and notebook, strip execution counts and transient metadata from the notebook with `nbstripout` (outputs are kept), and block accidentally committed large files. Run them manually with `pre-commit run --all-files`.

Never commit `kaggle.json`, the dataset, or trained model files. They are listed in `.gitignore`.

## Usage

### Command Line

Train a model. The best checkpoint (lowest validation loss) is saved to `models/<model>.keras`:

```bash
xray-train --model cnn
xray-train --model vgg16
```

Useful options: `--epochs` (default 10), `--batch-size` (default 4 for `cnn`, 32 for `vgg16`), `--augment/--no-augment` (default on for `vgg16` only), `--seed`, `--output`. Run `xray-train --help` for the full list.

Evaluate a trained model on the test set, optionally saving the plots:

```bash
xray-evaluate models/vgg16.keras --plots-dir reports/
```

Classify individual images:

```bash
xray-predict models/vgg16.keras path/to/image1.jpeg path/to/image2.jpeg
```

Each command is also available as `python -m xray_classifier.<train|evaluate|predict>`.

### Notebook

Open `X_ray_Image_Classification.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) and run the cells in order. It explores the data, then trains, evaluates, and compares both models. In Colab, the first cell clones this repository and installs the package; upload `kaggle.json` to the Colab working directory so the download cell can fetch the dataset.

### Configuration

Paths default to `data/chest_xray` for the dataset and `models/` for checkpoints. Override them with environment variables, for example to point at a mounted Google Drive folder:

```bash
export XRAY_DATA_DIR=/content/drive/MyDrive/Datasets/chest_xray
export XRAY_MODELS_DIR=/content/drive/MyDrive/models
```

## Results

Recorded from the original Colab run of the notebook (10 epochs, before the callbacks and the `tf.data` pipeline were added):

| Model | Validation accuracy (16 images) | Test accuracy (624 images) |
| --- | --- | --- |
| Custom CNN | 81.3% | 74.7% |
| VGG16 transfer learning | 68.8% | 85.9% |

The provided validation split has only 16 images, so validation accuracy swings widely between epochs; the test set is the more reliable measure. Rerun training to get current numbers, including precision, recall, and ROC AUC from `xray-evaluate`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests build a small synthetic dataset, so they run in seconds and do not need the Kaggle data.

## Contributing

Feel free to fork this repository and submit pull requests for:

* Additional model architectures (ResNet, EfficientNet).
* Hyperparameter tuning scripts.
* Deployment examples (Flask, FastAPI).

## License

This project is released under the **Apache License**. See [LICENSE](LICENSE) for details.
