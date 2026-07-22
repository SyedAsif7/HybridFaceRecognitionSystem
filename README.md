---
title: Hybrid Face Recognition System
emoji: 👤
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
---

# Hybrid Face Recognition System
### SSIEMS Parbhani — M.Tech Research Project
**Based on:** Panchalwar Mam's Implementation Paper
*"Implementation of Hybrid Deep Learning-Based Face Recognition Using SIFT-CNN Integration"*

---

## Deploy on Hugging Face Spaces (free path)

Docker Spaces are **paid**. Use **Gradio** instead (free accounts may get up to 2 ZeroGPU Spaces).

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **SDK:** Gradio (not Docker)
3. **Hardware:** ZeroGPU if shown (free tier), otherwise try CPU if your account allows it
4. Create the Space, then upload/push this repo **or** connect GitHub:  
   `SyedAsif7/HybridFaceRecognitionSystem`
5. Entry file is `gradio_app.py` (set in README frontmatter as `app_file`)
6. Wait for the build, then open the Space URL

If Gradio also asks for PRO, Hugging Face free compute is not available on your account — use a free VM instead (Oracle Always Free) or demo locally with a tunnel.

---

## System Architecture

```
Dataset (ORL + Sheffield)
        ↓
Preprocessing (Resize · Grayscale · Normalize)
        ↓
Face Alignment (Eye-Landmark → Rotation → Aligned Crop)  ← Proposed
        ↓
Feature Extraction
  ├── SIFT  (128-dim)          ← Mam's Paper
  ├── HOG   (gradient hist)    ← Mam's Paper
  ├── Gabor (24-dim)           ← Mam's Paper
  └── Canny (65-dim)           ← Mam's Paper
        ↓
Feature Fusion                              ← Proposed
  ├── Research Fusion: SIFT + HOG + Gabor   ← Mam's focus
  └── Full Hybrid:    SIFT + HOG + Gabor + Canny
        ↓
Classifier (choose one)
  ├── Custom CNN   — 4 Conv Blocks (Mam's Architecture)
  ├── MobileNetV2  — Transfer Learning        ← Proposed
  ├── SVM RBF      — Comparison Classifier    ← Proposed
  └── SVM Linear   — Comparison Classifier    ← Proposed
        ↓
Activation Function
  ├── Softmax  — Primary (multi-class)        ← Mam's Paper
  └── Sigmoid  — Comparison                   ← Mam's Paper
        ↓
Training Optimization
  ├── Optimizers: Adam / Adamax / RMSprop / SGD   ← Mam's Paper
  ├── Batch Normalization                          ← Proposed
  ├── Early Stopping                               ← Proposed
  ├── ReduceLROnPlateau                            ← Proposed
  └── Data Augmentation                            ← Proposed
        ↓
Output: Identity + Confidence Score (High ≥80% / Medium 50-80% / Low <50%)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python main.py --train --dataset orl --method hybrid --optimizer adam --activation softmax

# 3. Run application
python app.py

# 4. Open browser
http://localhost:5000
```

---

## Published Results (Mam's Paper)

| Method       | Optimizer | Activation | ORL Acc   | Sheffield Acc |
|-------------|-----------|------------|-----------|---------------|
| SIFT + CNN  | Adam      | Softmax    | **92.5%** | **100%** ⭐   |
| HOG + CNN   | Adamax    | Softmax    | 91.25%    | 97.50%        |
| Gabor + CNN | Adam      | Softmax    | 93.75%    | 95.00%        |
| Canny + CNN | SGD       | Sigmoid    | 80.00%    | 82.50%        |

---

## Proposed Enhancements (Under Evaluation)

| Enhancement                          | Status             |
|-------------------------------------|--------------------|
| Research Fusion (SIFT+HOG+Gabor+CNN)| Under Evaluation   |
| Full Hybrid (SIFT+HOG+Gabor+Canny)  | Under Evaluation   |
| MobileNetV2 Transfer Learning       | Under Evaluation   |
| SVM RBF Classifier                  | Under Evaluation   |
| SVM Linear Classifier               | Under Evaluation   |
| Face Alignment (Eye-Landmark)       | Implemented ✅     |
| Data Augmentation                   | Implemented ✅     |
| Batch Normalization                 | Implemented ✅     |
| Early Stopping + ReduceLROnPlateau  | Implemented ✅     |
| Softmax / Sigmoid Switching         | Implemented ✅     |
| Confidence Score (High/Med/Low)     | Implemented ✅     |

---

## Fusion Naming — Consistent Definition

| Name              | Methods                      | Use Case                    |
|------------------|------------------------------|-----------------------------|
| Research Fusion  | SIFT + HOG + Gabor           | Mam's paper focus           |
| Full Hybrid      | SIFT + HOG + Gabor + Canny   | Extended evaluation         |

---

## Datasets

| Dataset   | Images | Subjects | Resolution | Source              |
|----------|--------|----------|------------|---------------------|
| ORL      | 400    | 40       | 92×112 px  | AT&T Laboratories   |
| Sheffield| 564    | 20       | 220×220 px | University Sheffield|

---

## Confidence Score Categories

| Level  | Condition        |
|--------|-----------------|
| High   | Confidence ≥ 80% |
| Medium | 50% ≤ Conf < 80% |
| Low    | Confidence < 50% |

---

*All proposed enhancements are subject to Mam's guidance and approval.*
*Implementation faithful to: Panchalwar Mam's published methodology.*
