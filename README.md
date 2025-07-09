# Chest X‑Ray Pneumonia Classification

A Jupyter Notebook project that implements convolutional neural networks (CNNs) to classify chest X‑ray images into **Normal** (healthy) or **Pneumonia** categories. The entire workflow—from data acquisition and preprocessing to model training, evaluation, and visualization—is contained in a single notebook for clarity and reproducibility.

---

## 📖 Table of Contents

1. [About](#about)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Project Setup](#project-setup)

   * [Clone Repository](#clone-repository)
   * [Install Dependencies](#install-dependencies)
   * [Download Dataset](#download-dataset)
5. [Usage](#usage)
6. [Notebook Outline](#notebook-outline)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## About

The notebook **X\_ray\_Image\_Classification.ipynb** guides you through:

* Downloading and extracting the public [Chest X‑Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
* Loading images from `train/`, `val/`, and `test/` directories.
* Preprocessing (resizing, grayscale conversion, normalization).
* Data augmentation using `ImageDataGenerator`.
* Building two CNNs:

  * A custom Sequential model (Conv2D + MaxPooling + Dropout).
  * A transfer learning model using VGG16 as the base.
* Training with real-time visualization of loss & accuracy.
* Evaluating performance (accuracy, confusion matrix, ROC curve).
* Saving the best model for future inference.

## Features

* Automated dataset download via Kaggle API or manual instructions.
* Grayscale & RGB pipelines for custom and transfer-learning models.
* Data augmentation to reduce overfitting.
* Clear visualizations: training curves, confusion matrix, ROC.
* Model export (`.h5`) for deployment.

## Prerequisites

* **Python 3.8+**
* **Jupyter Notebook**
* **Kaggle CLI** (optional, for dataset download)

## Project Setup

### Clone Repository

```bash
git clone https://github.com/<your-username>/xray-image-classification.git
cd xray-image-classification
```

### Install Dependencies

Create a virtual environment (recommended) and install required packages:

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**requirements.txt** should include:

```
opencv-python
tensorflow
keras
numpy
matplotlib
kaggle
```

### Download Dataset

#### Option 1: Kaggle CLI (recommended)

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

#### Option 2: Manual Download

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Download and unzip into a `data/chest_xray/` directory.

## Usage

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
2. Open `X_ray_Image_Classification.ipynb`.
3. Update the `data_dir` variable at the top of the notebook to point to `./data/chest_xray/train`.
4. Run cells sequentially.

> **Tip:** To switch between the custom CNN and the VGG16-based model, comment/uncomment the respective training sections.

## Notebook Outline

1. **Setup & Imports**: Libraries, paths, global variables.
2. **Data Exploration**: Sample images, class distribution.
3. **Preprocessing Functions**: Resizing, normalization.
4. **Dataset Generators**: `ImageDataGenerator` for train/val/test.
5. **Model Definitions**:

   * Custom Sequential CNN.
   * VGG16 transfer-learning model.
6. **Training Loops**: Fit models, measure training time.
7. **Evaluation**: Accuracy, confusion matrix, ROC curve.
8. **Inference & Saving**: Predict on new images and export `.h5` model.

## Results

Typical performance metrics:

* **Custom CNN**: \~92% validation accuracy
* **VGG16 Transfer Learning**: \~96% validation accuracy

Visualization outputs include:

* Training vs. validation accuracy & loss plots
* Confusion matrix heatmap
* ROC curve with AUC score

## Contributing

Feel free to fork this repository and submit pull requests for:

* Additional model architectures (ResNet, EfficientNet).
* Hyperparameter tuning scripts.
* Deployment examples (Flask, FastAPI).

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
