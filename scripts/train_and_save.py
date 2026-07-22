"""
=============================================================
  Hybrid Face Recognition System — Full Training Script
=============================================================
  Datasets  : ORL (AT&T) + Sheffield (UMIST)
  Models    : Custom CNN + Feature Fusion | MobileNetV2 | SVM
  Features  : SIFT | HOG | Gabor | Canny
  Author    : Panchalwar Mam's Research — SSIEMS Parbhani
  M.Tech    : Hybrid CNN + Feature Fusion
=============================================================

USAGE:
    python train_and_save.py --dataset orl  --epochs 100
    python train_and_save.py --dataset orl  --epochs 50  --quick
    python train_and_save.py --all-models

OUTPUTS saved to ../weights/ :
    best_hybrid_model.h5
    best_mobilenetv2_model.h5
    svm_rbf.pkl
    svm_linear.pkl
    class_names.json

DATASET STRUCTURE expected at  ../data/ORL/ :
    s1/
        1.pgm  2.pgm ... 10.pgm
    s2/
        ...
    s40/
        ...

Download ORL dataset from:
    https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
"""

import os, sys, json, argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Optional TF import ─────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                             ModelCheckpoint)
    TF_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} found")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found — CNN models won't be trained. Install: pip install tensorflow")

BASE = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE / "weights"
RESULTS_DIR = BASE / "results"
DATA_DIR    = BASE / "data"
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_orl_dataset(data_path, img_size=(128, 128)):
    """
    Load ORL (AT&T) dataset.
    Structure: data_path/s1/1.pgm ... data_path/s40/10.pgm
    Returns: images (N,H,W,1), features (N,D), labels (N,), class_names
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {data_path}\n"
            "   Download from: https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip\n"
            "   Extract and place as:  data/ORL/s1/  data/ORL/s2/ ... data/ORL/s40/"
        )

    print(f"\n📂 Loading ORL dataset from {data_path}")
    subjects = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"   Found {len(subjects)} subjects")

    images, features, labels, class_names = [], [], [], []

    for idx, subj_dir in enumerate(subjects):
        class_names.append(subj_dir.name)
        img_files = list(subj_dir.glob("*.pgm")) + list(subj_dir.glob("*.jpg")) + list(subj_dir.glob("*.png"))

        for img_file in img_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to target
            img = cv2.resize(img, img_size)
            img_norm = img.astype(np.float32) / 255.0

            # Extract features
            feats = extract_hybrid_features(img)

            images.append(img_norm.reshape(*img_size, 1))
            features.append(feats)
            labels.append(idx)

    images   = np.array(images,   dtype=np.float32)
    features = np.array(features, dtype=np.float32)
    labels   = np.array(labels,   dtype=np.int32)

    print(f"   Images : {images.shape}")
    print(f"   Features: {features.shape}")
    print(f"   Classes: {len(class_names)}")
    return images, features, labels, class_names


# ═══════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_sift(img_gray, n_features=500):
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=n_features)
    _, desc = sift.detectAndCompute(img_gray, None)
    return np.mean(desc, axis=0) if desc is not None and len(desc) > 0 else np.zeros(128)

def extract_hog(img_gray):
    from skimage.feature import hog
    return hog(img_gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), visualize=False, channel_axis=None)

def extract_gabor(img_gray, frequencies=(0.1, 0.3, 0.5), thetas=(0, 45, 90, 135)):
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)
    feats = []
    for freq in frequencies:
        for theta in thetas:
            k = cv2.getGaborKernel((21,21), sigma=4, theta=np.deg2rad(theta),
                                    lambd=1.0/freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(img_gray, cv2.CV_64F, k)
            feats.extend([filtered.mean(), filtered.std()])
    return np.array(feats, dtype=np.float32)

def extract_canny(img_gray, low=50, high=150, bins=64):
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)
    edges = cv2.Canny(img_gray, low, high)
    density = float(np.sum(edges > 0)) / edges.size
    hist = np.zeros(bins)
    pos = np.where(edges.flatten() > 0)[0]
    if len(pos) > 0:
        idx = np.clip(pos * bins // edges.size, 0, bins-1).astype(int)
        for i in idx:
            hist[i] += 1
        if hist.sum() > 0:
            hist /= hist.sum()
    return np.concatenate([[density], hist])

def extract_hybrid_features(img_gray):
    """Research Fusion: SIFT + HOG + Gabor  (+ Canny for full hybrid)"""
    if img_gray.dtype == np.float32 or img_gray.dtype == np.float64:
        img_u8 = (img_gray * 255).astype(np.uint8)
    else:
        img_u8 = img_gray
    sift  = extract_sift(img_u8)
    hog   = extract_hog(img_u8)
    gabor = extract_gabor(img_u8)
    canny = extract_canny(img_u8)
    return np.concatenate([sift, hog, gabor, canny]).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════

def build_hybrid_cnn(img_shape=(128,128,1), feature_dim=8317, num_classes=40):
    img_in = tf.keras.Input(shape=img_shape, name='image_input')
    x = layers.Conv2D(32,(3,3),padding='same')(img_in)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x);   x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64,(3,3),padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x);   x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128,(3,3),padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x);   x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256,(3,3),padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)

    feat_in = tf.keras.Input(shape=(feature_dim,), name='feature_input')
    y = layers.Dense(1024, activation='relu')(feat_in)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.4)(y)
    y = layers.Dense(512,  activation='relu')(y)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.3)(y)
    y = layers.Dense(256,  activation='relu')(y)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.3)(y)

    merged = layers.Concatenate()([x, y])
    f = layers.Dense(512, activation='relu')(merged)
    f = layers.BatchNormalization()(f); f = layers.Dropout(0.5)(f)
    f_res = layers.Dense(512, activation='relu')(merged)
    f = layers.Add()([f, f_res])
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f); f = layers.Dropout(0.4)(f)
    f = layers.Dense(128, activation='relu')(f)
    emb = layers.Dense(64, activation='relu', name='embedding')(f)
    out = layers.Dense(num_classes, activation='softmax', name='output')(emb)

    return Model(inputs=[img_in, feat_in], outputs=[out, emb])


def build_mobilenetv2(img_shape=(96,96,3), feature_dim=8317, num_classes=40, fine_tune_at=100):
    img_in = tf.keras.Input(shape=img_shape, name='image_input')
    base = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    x = base(img_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)

    feat_in = tf.keras.Input(shape=(feature_dim,), name='feature_input')
    y = layers.Dense(512, activation='relu')(feat_in)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.4)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y); y = layers.Dropout(0.3)(y)

    merged = layers.Concatenate()([x, y])
    f = layers.Dense(256, activation='relu')(merged)
    f = layers.BatchNormalization()(f); f = layers.Dropout(0.4)(f)
    emb = layers.Dense(64, activation='relu', name='embedding')(f)
    out = layers.Dense(num_classes, activation='softmax', name='output')(emb)

    return Model(inputs=[img_in, feat_in], outputs=[out, emb])


# ═══════════════════════════════════════════════════════════
# 4. TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def train_hybrid_cnn(X_img, X_feat, y, class_names, epochs=100, batch_size=16):
    print("\n" + "="*60)
    print("  TRAINING: Custom CNN + Feature Fusion (Mam's Architecture)")
    print("="*60)

    num_classes = len(class_names)
    feature_dim = X_feat.shape[1]

    X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
        X_img, X_feat, y, test_size=0.2, random_state=42, stratify=y)

    model = build_hybrid_cnn(
        img_shape=X_img.shape[1:],
        feature_dim=feature_dim,
        num_classes=num_classes
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'output': 'sparse_categorical_crossentropy', 'embedding': None},
        loss_weights={'output': 1.0},
        metrics={'output': 'accuracy'}
    )

    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_output_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_output_accuracy'),
        ModelCheckpoint(str(WEIGHTS_DIR / 'best_hybrid_model.h5'),
                        save_best_only=True, monitor='val_output_accuracy')
    ]

    history = model.fit(
        {'image_input': X_img_tr, 'feature_input': X_feat_tr},
        {'output': y_tr},
        validation_data=(
            {'image_input': X_img_te, 'feature_input': X_feat_te},
            {'output': y_te}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    preds = model.predict({'image_input': X_img_te, 'feature_input': X_feat_te}, verbose=0)
    y_pred = np.argmax(preds[0], axis=1)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n✅ Final Test Accuracy (Custom CNN): {acc*100:.2f}%")
    print(classification_report(y_te, y_pred, target_names=class_names))

    # Save history
    hist_data = {
        "model": "Custom CNN + Feature Fusion",
        "best_val_acc": float(max(history.history.get('val_output_accuracy', [0]))),
        "final_test_acc": float(acc),
        "epochs_trained": len(history.history['output_accuracy']),
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    }
    with open(RESULTS_DIR / 'training_history_cnn.json', 'w') as f:
        json.dump(hist_data, f, indent=2)

    return model, acc


def train_mobilenetv2(X_img, X_feat, y, class_names, epochs=50, batch_size=16):
    print("\n" + "="*60)
    print("  TRAINING: MobileNetV2 Transfer Learning")
    print("="*60)

    # Resize to 96x96x3 for MobileNetV2
    X_img_rgb = np.repeat(cv2.resize(
        X_img[0,:,:,0], (96,96))[np.newaxis,...,np.newaxis], 3, axis=-1)[:0]
    X_img_resized = np.array([
        np.repeat(cv2.resize(img[:,:,0], (96,96))[:,:,np.newaxis], 3, axis=-1)
        for img in X_img
    ], dtype=np.float32)

    num_classes = len(class_names)
    feature_dim = X_feat.shape[1]

    X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
        X_img_resized, X_feat, y, test_size=0.2, random_state=42, stratify=y)

    model = build_mobilenetv2(
        img_shape=(96, 96, 3),
        feature_dim=feature_dim,
        num_classes=num_classes
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'output': 'sparse_categorical_crossentropy'},
        loss_weights={'output': 1.0},
        metrics={'output': 'accuracy'}
    )

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_output_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint(str(WEIGHTS_DIR / 'best_mobilenetv2_model.h5'),
                        save_best_only=True, monitor='val_output_accuracy')
    ]

    history = model.fit(
        {'image_input': X_img_tr, 'feature_input': X_feat_tr},
        {'output': y_tr},
        validation_data=(
            {'image_input': X_img_te, 'feature_input': X_feat_te},
            {'output': y_te}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    preds = model.predict({'image_input': X_img_te, 'feature_input': X_feat_te}, verbose=0)
    y_pred = np.argmax(preds[0], axis=1)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n✅ Final Test Accuracy (MobileNetV2): {acc*100:.2f}%")

    hist_data = {
        "model": "MobileNetV2 + Feature Fusion",
        "best_val_acc": float(max(history.history.get('val_output_accuracy', [0]))),
        "final_test_acc": float(acc),
        "epochs_trained": len(history.history['output_accuracy']),
    }
    with open(RESULTS_DIR / 'training_history_mv2.json', 'w') as f:
        json.dump(hist_data, f, indent=2)

    return model, acc


def train_svm(X_feat, y, class_names):
    print("\n" + "="*60)
    print("  TRAINING: SVM Classifiers (RBF + Linear)")
    print("="*60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y)

    # SVM RBF
    print("\n  SVM — RBF Kernel (C=10, gamma=scale)...")
    svm_rbf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42))
    ])
    svm_rbf.fit(X_tr, y_tr)
    y_pred_rbf = svm_rbf.predict(X_te)
    acc_rbf = accuracy_score(y_te, y_pred_rbf)
    print(f"  ✅ SVM RBF Accuracy: {acc_rbf*100:.2f}%")
    joblib.dump(svm_rbf, WEIGHTS_DIR / 'svm_rbf.pkl')

    # SVM Linear
    print("\n  SVM — Linear Kernel (C=1.0)...")
    svm_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
    ])
    svm_linear.fit(X_tr, y_tr)
    y_pred_lin = svm_linear.predict(X_te)
    acc_lin = accuracy_score(y_te, y_pred_lin)
    print(f"  ✅ SVM Linear Accuracy: {acc_lin*100:.2f}%")
    joblib.dump(svm_linear, WEIGHTS_DIR / 'svm_linear.pkl')

    print("\n  Full classification report (SVM RBF):")
    print(classification_report(y_te, y_pred_rbf, target_names=class_names))

    return acc_rbf, acc_lin


# ═══════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Face Recognition Models')
    parser.add_argument('--dataset', default='orl', help='Dataset name (orl)')
    parser.add_argument('--data-path', default=str(DATA_DIR / 'ORL'), help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Quick run: 20 epochs')
    parser.add_argument('--all-models', action='store_true', help='Train CNN + MobileNetV2 + SVM')
    parser.add_argument('--cnn-only', action='store_true', help='Train only Custom CNN')
    parser.add_argument('--svm-only', action='store_true', help='Train only SVM classifiers')
    parser.add_argument('--mv2-only', action='store_true', help='Train only MobileNetV2')
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        print("⚡ Quick mode: 20 epochs")

    print("\n" + "="*60)
    print("  HYBRID FACE RECOGNITION — TRAINING PIPELINE")
    print("  Panchalwar Mam's Research · SSIEMS Parbhani")
    print("="*60)
    print(f"  Dataset  : {args.data_path}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Weights  : {WEIGHTS_DIR}")

    # ── Load data ─────────────────────────────────────────
    images, features, labels, class_names = load_orl_dataset(args.data_path)

    # ── Save class names ──────────────────────────────────
    class_info = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "dataset": "ORL (AT&T) + Sheffield",
        "feature_dim": int(features.shape[1]),
        "img_size": list(images.shape[1:3]),
        "trained_at": datetime.now().isoformat()
    }
    with open(WEIGHTS_DIR / 'class_names.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"\n✅ class_names.json saved ({len(class_names)} classes)")

    results = {}

    # ── SVM ───────────────────────────────────────────────
    if args.svm_only or args.all_models or (not args.cnn_only and not args.mv2_only):
        acc_rbf, acc_lin = train_svm(features, labels, class_names)
        results['svm_rbf']   = f"{acc_rbf*100:.2f}%"
        results['svm_linear'] = f"{acc_lin*100:.2f}%"

    # ── Custom CNN ────────────────────────────────────────
    if TF_AVAILABLE and (args.cnn_only or args.all_models or (not args.svm_only and not args.mv2_only)):
        _, acc_cnn = train_hybrid_cnn(images, features, labels, class_names,
                                       epochs=args.epochs, batch_size=args.batch_size)
        results['custom_cnn'] = f"{acc_cnn*100:.2f}%"

    # ── MobileNetV2 ───────────────────────────────────────
    if TF_AVAILABLE and (args.mv2_only or args.all_models):
        _, acc_mv2 = train_mobilenetv2(images, features, labels, class_names,
                                        epochs=args.epochs, batch_size=args.batch_size)
        results['mobilenetv2'] = f"{acc_mv2*100:.2f}%"

    # ── Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("="*60)
    for model_name, acc in results.items():
        print(f"  {model_name:20s}  →  {acc}")
    print(f"\n  All weights saved to: {WEIGHTS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
