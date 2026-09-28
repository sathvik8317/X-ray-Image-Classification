# Chest X‑Ray Pneumonia Classification

Convolutional neural networks (CNNs) that classify chest X‑ray images as **Normal** (healthy) or **Pneumonia**. The code is a small Python package, `xray_classifier`, with command-line tools for training, evaluation, and prediction, a notebook that walks through the whole workflow, and a web demo.

> **Research and education only.** This project is not a medical device and must not be used to diagnose patients.

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
   * [Demo App](#demo-app)
   * [Configuration](#configuration)
6. [Results](#results)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

---

## About

The project trains and compares models on the public [Chest X‑Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):

| Model (`--model`) | Input | Description |
| --- | --- | --- |
| `cnn` | 100x100 grayscale | Three Conv2D + MaxPooling + Dropout blocks, trained from scratch |
| `vgg16` | 100x100 RGB | Frozen ImageNet VGG16 base with a dense head |
| `efficientnetv2-b0` | 224x224 RGB | Frozen ImageNet EfficientNetV2-B0 base with a pooled linear head |
| `convnext-tiny` | 224x224 RGB | Frozen ImageNet ConvNeXt-Tiny base with a pooled linear head |

Each pretrained model has its ImageNet input preprocessing built in, so every model takes images scaled to [0, 1].

How training works:

* **Validation split by patient.** The dataset's `val/` folder has only 16 images, too few to steer training. By default `train/` and `val/` are pooled and 15% is held out for validation, stratified by class and grouped by the patient ID in the filenames (`person123_...`, `IM-0115-...`), so no patient appears in both training and validation.
* **Class weights.** The training set has about three PNEUMONIA images for every NORMAL one. The loss is weighted so both classes count equally.
* **Callbacks.** Early stopping, learning-rate reduction on plateau, and a checkpoint of the best model (lowest validation loss) in `.keras` format.
* **Calibrated threshold.** After training, the decision threshold that maximizes sensitivity + specificity on the validation split is saved next to the model as `<model>.json`. Evaluation and prediction use it automatically.
* **Optional fine-tuning.** For pretrained models, a second phase can unfreeze the backbone's last stage and train it at a low learning rate. The checkpoint is only replaced if fine-tuning improves the validation loss.

Evaluation reports accuracy, **sensitivity** (share of pneumonia cases caught), **specificity** (share of normal cases cleared), ROC AUC, a classification report, and a confusion matrix. **Grad-CAM** heatmaps show which image regions drive a prediction, to check that a model looks at the lungs rather than at shortcuts such as text markers or image borders.

## Project Structure

```
.
├── X_ray_Image_Classification.ipynb   # walkthrough notebook (Colab-ready)
├── src/xray_classifier/
│   ├── config.py      # class names, image size, data and model paths
│   ├── data.py        # file listing, patient-grouped split, tf.data pipeline
│   ├── models.py      # model builders and fine-tuning (unfreeze last stage)
│   ├── train.py       # training with callbacks and threshold calibration (xray-train)
│   ├── thresholds.py  # choosing, saving, and loading decision thresholds
│   ├── evaluate.py    # metrics on a data split (xray-evaluate)
│   ├── predict.py     # single-image prediction (xray-predict)
│   ├── gradcam.py     # Grad-CAM heatmaps
│   ├── plots.py       # training curves, confusion matrix, ROC curve, Grad-CAM figures
│   └── app.py         # Gradio web demo (xray-app)
├── tests/             # pytest suite, runs on small synthetic datasets
├── .github/workflows/ # CI: pre-commit and pytest
└── pyproject.toml     # package metadata, dependencies, tool config
```

## Prerequisites

* **Python 3.10+**
* A **Kaggle account** and API token, to download the dataset
* A GPU is recommended for the pretrained models, especially at 224x224

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

This installs `xray_classifier` in editable mode along with its dependencies (TensorFlow, NumPy, Matplotlib, scikit-learn, Kaggle), which are listed in `pyproject.toml`. For the web demo, also run `pip install -e ".[app]"`.

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

Train a model. The best checkpoint is saved to `models/<model>.keras` and its threshold to `models/<model>.json`:

```bash
xray-train --model cnn
xray-train --model vgg16
xray-train --model efficientnetv2-b0 --fine-tune-epochs 5
```

| Option | Default | Description |
| --- | --- | --- |
| `--epochs` | 10 | Epochs of the main (frozen-base) phase |
| `--batch-size` | 4 for `cnn`, 32 otherwise | |
| `--img-size` | 100 for `cnn`/`vgg16`, 224 otherwise | |
| `--augment/--no-augment` | off for `cnn`, on otherwise | Random zoom augmentation |
| `--val-fraction` | 0.15 | Share held out for validation, split by patient; `0` uses the provided 16-image `val/` folder |
| `--class-weight/--no-class-weight` | on | Balance NORMAL and PNEUMONIA in the loss |
| `--fine-tune-epochs` | 0 | Extra epochs training the backbone's last stage (pretrained models only) |
| `--fine-tune-lr` | 1e-5 | Learning rate for fine-tuning |
| `--seed` | none | Seed for reproducible runs |

Evaluate a trained model on the test set, optionally saving the plots. It warns if patients in the evaluated split also appear in `train/`:

```bash
xray-evaluate models/vgg16.keras --plots-dir reports/
```

Classify individual images, optionally saving Grad-CAM heatmaps:

```bash
xray-predict models/vgg16.keras path/to/image1.jpeg path/to/image2.jpeg --gradcam reports/gradcam/
```

Both use the threshold saved with the model unless `--threshold` is given. Each command is also available as `python -m xray_classifier.<train|evaluate|predict>`.

### Notebook

Open `X_ray_Image_Classification.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) and run the cells in order. It explores the data, trains the custom CNN, VGG16, and EfficientNetV2-B0 (with fine-tuning), compares them on the test set, and shows Grad-CAM heatmaps. In Colab, the first cell clones this repository and installs the package; upload `kaggle.json` to the Colab working directory so the download cell can fetch the dataset.

### Demo App

A small web page to upload an X-ray and see the prediction and its Grad-CAM heatmap:

```bash
pip install -e ".[app]"
xray-app models/vgg16.keras          # then open http://127.0.0.1:7860
```

### Configuration

Paths default to `data/chest_xray` for the dataset and `models/` for checkpoints. Override them with environment variables, for example to point at a mounted Google Drive folder:

```bash
export XRAY_DATA_DIR=/content/drive/MyDrive/Datasets/chest_xray
export XRAY_MODELS_DIR=/content/drive/MyDrive/models
```

## Results

Recorded from the original Colab run of the notebook (10 epochs):

| Model | Validation accuracy (16 images) | Test accuracy (624 images) |
| --- | --- | --- |
| Custom CNN | 81.3% | 74.7% |
| VGG16 transfer learning | 68.8% | 85.9% |

These numbers predate the current training setup: the patient-grouped validation split, class weights, threshold calibration, callbacks, and two model fixes (a missing ReLU in the custom CNN, and missing ImageNet preprocessing for VGG16). They also report accuracy only, which hides the trade-off between catching pneumonia and clearing healthy patients on this imbalanced test set. Rerun training to get current numbers, including sensitivity, specificity, and ROC AUC from `xray-evaluate`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests build small synthetic datasets, so they need neither the Kaggle data nor a GPU (about a minute and a half on a laptop CPU). GitHub Actions runs the pre-commit hooks and the test suite on Python 3.10 and 3.12 for every pull request.

## Contributing

Feel free to fork this repository and submit pull requests for:

* Hyperparameter tuning and comparisons of the models on the real data.
* Evaluation on an external chest X-ray dataset, to check how well the models generalize beyond this one.
* Additional architectures or test-time augmentation.

## License

This project is released under the **Apache License**. See [LICENSE](LICENSE) for details.
