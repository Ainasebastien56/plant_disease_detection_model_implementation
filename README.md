# 🌿 Plant Disease Detection — Mauritius Crops

A deep learning project for automated detection of plant diseases in crops common to Mauritius, using transfer learning and model comparison across multiple CNN architectures.

## 📋 Overview

This notebook trains and compares five image classification models on the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), targeting four crops relevant to Mauritian agriculture. The goal is to identify the best-performing architecture for plant disease detection.

**Target crops:** Corn · Potato · Pepper · Tomato

## 🧠 Models Compared

| # | Model | Strategy |
|---|-------|----------|
| 1 | Baseline CNN | Custom 3-block CNN trained from scratch |
| 2 | VGG16 | Transfer learning + 2-phase fine-tuning (blocks 3–5) |
| 3 | VGG19 | Transfer learning + 2-phase fine-tuning (blocks 3–5) |
| 4 | EfficientNetB0 | Transfer learning + 2-phase fine-tuning (last 30 layers) |
| 5 | EfficientNetB3 | Transfer learning + 2-phase fine-tuning (last 30 layers) |

All transfer learning models follow a **2-phase training strategy**:
- **Phase 1** — Freeze the base, train only the classification head
- **Phase 2** — Unfreeze the last conv blocks/layers and fine-tune at a lower learning rate

## 📁 Project Structure

```
.
├── PlantVillage/
│   ├── train/
│   ├── val/
│   └── test/               # Auto-created from 50% of val set
├── models/
│   ├── cnn_best.keras
│   ├── vgg16_best/         # SavedModel format
│   ├── vgg19_best/         # SavedModel format
│   ├── effb0_best/         # SavedModel format
│   └── effb3_best/         # SavedModel format
└── Disease_Detection_Mauritius.ipynb
```

## ⚙️ Setup

### Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

### Dataset

Download the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and place it under `../PlantVillage/` with `train/` and `val/` subdirectories organised by class. The notebook will automatically split 50% of the validation set into a dedicated `test/` folder on first run.

## 🔧 Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Phase 1 LR | 1e-4 |
| Phase 2 LR | 1e-5 |
| Max epochs (Phase 1) | 20 |
| Max epochs (Phase 2) | 50 |
| Early stopping patience | 5 (P1) / 7 (P2) |

**Data augmentation (training only):** random rotation (±20°), zoom (10%), horizontal flip.

## 📊 Evaluation

Each model is evaluated on the held-out test set using:
- Classification report (precision, recall, F1-score per class)
- Confusion matrix heatmap

A visual prediction grid is also generated for EfficientNetB3, showing predicted vs. actual labels with confidence scores (green = correct, red = incorrect).

## 📦 Saved Models

Trained models are exported in TensorFlow **SavedModel** format (`.keras` for CNN, `tf` SavedModel for the rest) under `../models/`, ready for deployment or further evaluation.

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- scikit-learn
- Matplotlib & Seaborn
- NumPy
