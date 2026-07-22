"""
=============================================================
  Hybrid Face Recognition — Load Weights & Predict
=============================================================
  Usage:
      python load_and_predict.py --image path/to/face.jpg
      python load_and_predict.py --image face.jpg --model svm_rbf
      python load_and_predict.py --image face.jpg --model custom_cnn
=============================================================
"""

import os, sys, json, argparse
import cv2
import numpy as np
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE / "weights"

# ── Optional TF ───────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def load_class_names():
    p = WEIGHTS_DIR / "class_names.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d["class_names"], d["num_classes"]
    # Default ORL class names
    return [f"s{i+1}" for i in range(40)], 40


def preprocess_image(img_path, target_size=(128, 128)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    return gray


def extract_features(gray_img):
    """Extract hybrid features: SIFT + HOG + Gabor + Canny"""
    from skimage.feature import hog

    img_u8 = (gray_img * 255).astype(np.uint8) if gray_img.max() <= 1.0 else gray_img.astype(np.uint8)

    # SIFT
    sift = cv2.SIFT_create(nfeatures=500)
    _, desc = sift.detectAndCompute(img_u8, None)
    sift_feat = np.mean(desc, axis=0) if desc is not None and len(desc) > 0 else np.zeros(128)

    # HOG
    hog_feat = hog(img_u8, orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), visualize=False, channel_axis=None)

    # Gabor
    gabor_feat = []
    for freq in [0.1, 0.3, 0.5]:
        for theta in [0, 45, 90, 135]:
            k = cv2.getGaborKernel((21,21), sigma=4, theta=np.deg2rad(theta),
                                    lambd=1.0/freq, gamma=0.5, psi=0)
            f = cv2.filter2D(img_u8, cv2.CV_64F, k)
            gabor_feat.extend([f.mean(), f.std()])
    gabor_feat = np.array(gabor_feat)

    # Canny
    edges = cv2.Canny(img_u8, 50, 150)
    density = float(np.sum(edges > 0)) / edges.size
    hist = np.zeros(64)
    pos = np.where(edges.flatten() > 0)[0]
    if len(pos) > 0:
        idx = np.clip(pos * 64 // edges.size, 0, 63).astype(int)
        for i in idx:
            hist[i] += 1
        if hist.sum() > 0:
            hist /= hist.sum()
    canny_feat = np.concatenate([[density], hist])

    return np.concatenate([sift_feat, hog_feat, gabor_feat, canny_feat]).astype(np.float32)


def predict_svm(image_path, kernel='rbf'):
    model_path = WEIGHTS_DIR / f"svm_{kernel}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}\nRun: python train_and_save.py --svm-only")

    clf = joblib.load(model_path)
    class_names, _ = load_class_names()

    gray = preprocess_image(image_path)
    feats = extract_features(gray)

    proba = clf.predict_proba(feats.reshape(1, -1))[0]
    top5  = np.argsort(proba)[::-1][:5]

    result = {
        "model": f"SVM ({kernel.upper()} kernel)",
        "predicted_class": class_names[top5[0]],
        "confidence": float(proba[top5[0]]) * 100,
        "top5": [(class_names[i], float(proba[i])*100) for i in top5]
    }
    return result


def predict_cnn(image_path, model_type='custom'):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed. Install: pip install tensorflow")

    fname = 'best_hybrid_model.h5' if model_type == 'custom' else 'best_mobilenetv2_model.h5'
    model_path = WEIGHTS_DIR / fname
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found: {model_path}\nRun: python train_and_save.py --cnn-only")

    model = tf.keras.models.load_model(str(model_path))
    class_names, _ = load_class_names()

    gray = preprocess_image(image_path)
    gray_norm = gray.astype(np.float32) / 255.0

    if model_type == 'mv2':
        img_input = np.repeat(
            cv2.resize(gray_norm, (96, 96))[:,:,np.newaxis], 3, axis=-1
        )[np.newaxis,...]
    else:
        img_input = gray_norm.reshape(1, 128, 128, 1)

    feats = extract_features(gray)
    feat_input = feats.reshape(1, -1)

    preds = model.predict({'image_input': img_input, 'feature_input': feat_input}, verbose=0)
    proba = preds[0][0]  # first output is classification probabilities
    top5  = np.argsort(proba)[::-1][:5]

    result = {
        "model": f"{'Custom CNN' if model_type == 'custom' else 'MobileNetV2'} + Feature Fusion",
        "predicted_class": class_names[top5[0]],
        "confidence": float(proba[top5[0]]) * 100,
        "top5": [(class_names[i], float(proba[i])*100) for i in top5]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Inference')
    parser.add_argument('--image',  required=True, help='Path to face image')
    parser.add_argument('--model',  default='svm_rbf',
                        choices=['svm_rbf', 'svm_linear', 'custom_cnn', 'mobilenetv2'],
                        help='Model to use for prediction')
    args = parser.parse_args()

    print(f"\n🔍 Predicting: {args.image}")
    print(f"   Model: {args.model}")

    if args.model.startswith('svm'):
        kernel = args.model.replace('svm_', '')
        result = predict_svm(args.image, kernel)
    elif args.model == 'custom_cnn':
        result = predict_cnn(args.image, 'custom')
    else:
        result = predict_cnn(args.image, 'mv2')

    print(f"\n{'='*50}")
    print(f"  Predicted  : {result['predicted_class']}")
    print(f"  Confidence : {result['confidence']:.1f}%")
    print(f"  Model      : {result['model']}")
    print(f"\n  Top 5 Predictions:")
    for i, (name, conf) in enumerate(result['top5']):
        bar = '█' * int(conf / 5)
        print(f"    #{i+1}  {name:8s}  {conf:5.1f}%  {bar}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
