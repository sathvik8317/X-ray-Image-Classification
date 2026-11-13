---
title: Chest X-ray Pneumonia Classifier
emoji: 🫁
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: EfficientNetV2-B0 pneumonia classifier with Grad-CAM
---

# Chest X-ray Pneumonia Classifier

Upload a chest X-ray to get a NORMAL / PNEUMONIA prediction and a Grad-CAM heatmap of the regions behind it. The model is EfficientNetV2-B0 fine-tuned on the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). On its 624-image test set it scores 97.9% sensitivity, 76.9% specificity and 0.974 ROC AUC.

> **Research demo only.** This is not a medical device and must not be used for diagnosis. The model was trained on pediatric X-rays from one hospital and has known weaknesses, including signs of shortcut learning.

Code, training details and full results: https://github.com/sathvik8317/X-ray-Image-Classification
