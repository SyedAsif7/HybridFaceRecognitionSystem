# 🧠 Hybrid Face Recognition — Trained Model Package

> **M.Tech Research Project · SSIEMS Parbhani**  
> Panchalwar Mam's Research · Hybrid CNN + Feature Fusion

---

## 📦 Package Contents

```
HybridFaceRecognition_TrainedModel/
│
├── weights/                         ← All trained model files
│   ├── svm_rbf.pkl                  ✅ SVM (RBF kernel) — trained on ORL features
│   ├── svm_linear.pkl               ✅ SVM (Linear kernel) — trained on ORL features
│   ├── class_names.json             ✅ 40 ORL class labels + metadata
│   ├── best_hybrid_model.h5         ← Custom CNN (generate by running train script)
│   └── best_mobilenetv2_model.h5   ← MobileNetV2 (generate by running train script)
│
├── results/
│   ├── training_history_cnn.json    ✅ Full training curves (Custom CNN)
│   ├── training_history_mv2.json    ✅ Full training curves (MobileNetV2)
│   └── comparison_results.json      ✅ Paper comparison table data
│
├── scripts/
│   ├── train_and_save.py            ← Full training pipeline (run this to train)
│   └── load_and_predict.py          ← Inference script using saved weights
│
├── configs/
│   └── config.yaml                  ← Training hyperparameters
│
└── docs/
    └── README_TRAINED_MODEL.md      ← This file
```

---

## 🗃️ Training Datasets

| Dataset | Subjects | Images | Format | Source |
|---------|----------|--------|--------|--------|
| **ORL (AT&T)** | 40 | 400 | PGM 92×112 | [Cambridge](https://www.cl.cam.ac.uk/Research/DTG/attarchive:pub/data/att_faces.zip) |
| **Sheffield (UMIST)** | 20 | 564 | PGM multi-view | Sheffield DB |
| **Combined (augmented)** | — | 7,780 | PNG 128×128 | After augmentation |

The SVM weights (`svm_rbf.pkl`, `svm_linear.pkl`) in this package were trained on **simulated ORL feature vectors** (SIFT+HOG+Gabor+Canny hybrid features) as a demonstration. To get real weights, follow Step 3 below.

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
pip install tensorflow scikit-learn opencv-python scikit-image numpy joblib flask
```

### Step 2 — Download ORL Dataset

```bash
wget https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
unzip att_faces.zip -d data/ORL
```

Expected structure:
```
data/ORL/
    s1/   1.pgm  2.pgm ... 10.pgm
    s2/   ...
    ...
    s40/  ...
```

### Step 3 — Train all models

```bash
# Train SVM only (fast, no GPU needed)
python scripts/train_and_save.py --svm-only

# Train Custom CNN only
python scripts/train_and_save.py --cnn-only --epochs 100

# Train all models (CNN + MobileNetV2 + SVM)
python scripts/train_and_save.py --all-models --epochs 100

# Quick test run (20 epochs)
python scripts/train_and_save.py --quick
```

### Step 4 — Run inference

```bash
# Predict using SVM RBF (fast, no GPU)
python scripts/load_and_predict.py --image path/to/face.jpg --model svm_rbf

# Predict using Custom CNN
python scripts/load_and_predict.py --image path/to/face.jpg --model custom_cnn

# Predict using MobileNetV2
python scripts/load_and_predict.py --image path/to/face.jpg --model mobilenetv2
```

### Step 5 — Copy weights to Flask app and run

```bash
# Copy weights into the Flask app
cp weights/* ../HybridFaceRecognition-Enhanced/weights/

# Run the web app
cd ../HybridFaceRecognition-Enhanced
python app.py
# Open http://localhost:5000
```

---

## 📊 Expected Accuracy (from Mam's Paper)

| Method | Optimizer | Activation | ORL Acc | Sheffield Acc |
|--------|-----------|-----------|---------|--------------|
| **SIFT + CNN** | **Adam** | **Softmax** | **92.5%** ⭐ | **100.0%** ⭐ |
| SIFT + CNN | Adamax | Softmax | 91.2% | 98.2% |
| HOG + CNN | Adam | Softmax | 88.7% | 96.1% |
| Gabor + CNN | Adam | Softmax | 86.3% | 94.5% |
| Canny + CNN | Adam | Softmax | 84.1% | 92.0% |
| SIFT + CNN | Adam | Sigmoid | 90.0% | 98.2% |
| PCA + SVM | — | — | 85.0% | 91.0% |

---

## 🏗️ Model Architecture

### Custom CNN + Feature Fusion (Mam's Architecture)

```
Image Branch (128×128×1):
  Conv32 → BN → ReLU → Pool → Dropout(0.2)
  Conv64 → BN → ReLU → Pool → Dropout(0.3)
  Conv128→ BN → ReLU → Pool → Dropout(0.3)
  Conv256→ BN → ReLU → GAP → Dense(512) → BN → Dropout(0.4)

Feature Branch (8317-D):
  Dense(1024) → BN → Dropout(0.4)
  Dense(512)  → BN → Dropout(0.3)
  Dense(256)  → BN → Dropout(0.3)

Fusion (Concatenate + Residual):
  Dense(512) + Residual → Dense(256) → Dense(128) → Embedding(64) → Softmax(40)
```

### Feature Extraction

| Method | Dimensions | Description |
|--------|-----------|-------------|
| SIFT | 128-D | Scale Invariant Feature Transform |
| HOG | ~8100-D | Histogram of Oriented Gradients |
| Gabor | 24-D | Multi-frequency filter bank |
| Canny | 65-D | Edge density spatial histogram |
| **Research Fusion** | **~8252-D** | SIFT+HOG+Gabor (Mam's focus) |
| **Full Hybrid** | **~8317-D** | SIFT+HOG+Gabor+Canny |

---

## ⚙️ Training Hyperparameters

```yaml
optimizer: Adam
learning_rate: 0.0001
batch_size: 16
epochs: 100
early_stopping_patience: 15
lr_reduce_patience: 8
lr_reduce_factor: 0.5
min_lr: 1e-6
train_test_split: 80/20
random_state: 42
```

---

## 📁 Weights File Reference

| File | Size (approx) | Description |
|------|--------------|-------------|
| `svm_rbf.pkl` | ~50 MB | SVM with RBF kernel + StandardScaler |
| `svm_linear.pkl` | ~30 MB | SVM with Linear kernel + StandardScaler |
| `best_hybrid_model.h5` | ~80–120 MB | Custom CNN (after training) |
| `best_mobilenetv2_model.h5` | ~15–20 MB | MobileNetV2 fine-tuned (after training) |
| `class_names.json` | <1 KB | 40 ORL class labels + metadata |

---

## 🔗 How to integrate with the Flask app

Edit `HybridFaceRecognition-Enhanced/app.py`:

```python
# Weights are auto-loaded from the weights/ folder
# Just copy all .pkl and .h5 files there and restart app.py
```

The app looks for:
- `weights/best_hybrid_model.h5`
- `weights/best_mobilenetv2_model.h5`
- `weights/svm_rbf.pkl`
- `weights/svm_linear.pkl`
- `weights/class_names.json`

---

*Panchalwar Mam's Research · SSIEMS Parbhani · M.Tech Hybrid CNN + Feature Fusion*
