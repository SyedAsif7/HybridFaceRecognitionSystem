"""
=============================================================
  load_weights_into_model.py
  Hybrid Face Recognition — Weight Loader & H5 Generator
=============================================================
  Run this script ONCE after installing TensorFlow to:
  1. Build the Keras model architectures
  2. Load the saved .npz weights into each model
  3. Save the final .h5 files that the Flask app uses

  Usage:
      pip install tensorflow scikit-learn opencv-python scikit-image
      python load_weights_into_model.py

  Output:
      weights/best_hybrid_model.h5
      weights/best_mobilenetv2_model.h5

  After this, run:
      python app.py
=============================================================
"""

import numpy as np
import json
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def load_npz(name):
    path = WEIGHTS_DIR / name
    data = np.load(str(path))
    return dict(data)


def build_and_save_hybrid_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    print("\n[1/2] Building Custom CNN + Feature Fusion...")

    FEAT_DIM  = 8317
    NUM_CLASS = 40
    IMG_SHAPE = (128, 128, 1)

    # ── Image branch ──────────────────────────────────────
    img_in = tf.keras.Input(shape=IMG_SHAPE, name='image_input')
    x = layers.Conv2D(32, (3,3), padding='same', name='conv2d_1')(img_in)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), padding='same', name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3,3), padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='dense_img')(x)
    x = layers.BatchNormalization(name='bn_img')(x)
    x = layers.Dropout(0.4)(x)

    # ── Feature branch ────────────────────────────────────
    feat_in = tf.keras.Input(shape=(FEAT_DIM,), name='feature_input')
    y = layers.Dense(1024, activation='relu', name='dense_feat_5')(feat_in)
    y = layers.BatchNormalization(name='bn_5')(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(512, activation='relu', name='dense_feat_6')(y)
    y = layers.BatchNormalization(name='bn_6')(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu', name='dense_feat_7')(y)
    y = layers.BatchNormalization(name='bn_7')(y)
    y = layers.Dropout(0.3)(y)

    # ── Fusion ────────────────────────────────────────────
    merged = layers.Concatenate()([x, y])
    f      = layers.Dense(512, activation='relu', name='dense_fuse1')(merged)
    f      = layers.BatchNormalization(name='bn_fuse1')(f)
    f      = layers.Dropout(0.5)(f)
    f_res  = layers.Dense(512, activation='relu', name='dense_res')(merged)
    f      = layers.Add()([f, f_res])
    f      = layers.Dense(256, activation='relu', name='dense_fuse2')(f)
    f      = layers.BatchNormalization(name='bn_fuse2')(f)
    f      = layers.Dropout(0.4)(f)
    f      = layers.Dense(128, activation='relu', name='dense_fuse3')(f)
    emb    = layers.Dense(64,  activation='relu', name='embedding')(f)
    out    = layers.Dense(NUM_CLASS, activation='softmax', name='output')(emb)

    model = Model(inputs=[img_in, feat_in], outputs=[out, emb])
    print(f"   Model built — {model.count_params():,} parameters")

    # ── Load weights from .npz ────────────────────────────
    print("   Loading weights from best_hybrid_model_weights.npz ...")
    npz = load_npz('best_hybrid_model_weights.npz')

    layer_map = {
        'conv2d_1':   ['conv2d_1/kernel',  'conv2d_1/bias'],
        'bn_1':       ['bn_1/gamma', 'bn_1/beta', 'bn_1/moving_mean', 'bn_1/moving_var'],
        'conv2d_2':   ['conv2d_2/kernel',  'conv2d_2/bias'],
        'bn_2':       ['bn_2/gamma', 'bn_2/beta', 'bn_2/moving_mean', 'bn_2/moving_var'],
        'conv2d_3':   ['conv2d_3/kernel',  'conv2d_3/bias'],
        'bn_3':       ['bn_3/gamma', 'bn_3/beta', 'bn_3/moving_mean', 'bn_3/moving_var'],
        'conv2d_4':   ['conv2d_4/kernel',  'conv2d_4/bias'],
        'bn_4':       ['bn_4/gamma', 'bn_4/beta', 'bn_4/moving_mean', 'bn_4/moving_var'],
        'dense_img':  ['dense_img/kernel', 'dense_img/bias'],
        'bn_img':     ['bn_img/gamma', 'bn_img/beta', 'bn_img/moving_mean', 'bn_img/moving_var'],
        'dense_feat_5': ['dense_feat_5/kernel', 'dense_feat_5/bias'],
        'bn_5':       ['bn_5/gamma', 'bn_5/beta', 'bn_5/moving_mean', 'bn_5/moving_var'],
        'dense_feat_6': ['dense_feat_6/kernel', 'dense_feat_6/bias'],
        'bn_6':       ['bn_6/gamma', 'bn_6/beta', 'bn_6/moving_mean', 'bn_6/moving_var'],
        'dense_feat_7': ['dense_feat_7/kernel', 'dense_feat_7/bias'],
        'bn_7':       ['bn_7/gamma', 'bn_7/beta', 'bn_7/moving_mean', 'bn_7/moving_var'],
        'dense_fuse1':['dense_fuse1/kernel', 'dense_fuse1/bias'],
        'bn_fuse1':   ['bn_fuse1/gamma', 'bn_fuse1/beta', 'bn_fuse1/moving_mean', 'bn_fuse1/moving_var'],
        'dense_res':  ['dense_res/kernel', 'dense_res/bias'],
        'dense_fuse2':['dense_fuse2/kernel', 'dense_fuse2/bias'],
        'bn_fuse2':   ['bn_fuse2/gamma', 'bn_fuse2/beta', 'bn_fuse2/moving_mean', 'bn_fuse2/moving_var'],
        'dense_fuse3':['dense_fuse3/kernel', 'dense_fuse3/bias'],
        'embedding':  ['embedding/kernel', 'embedding/bias'],
        'output':     ['output/kernel', 'output/bias'],
    }

    for layer in model.layers:
        if layer.name in layer_map:
            keys = layer_map[layer.name]
            weights_list = [npz[k] for k in keys if k in npz]
            if weights_list:
                try:
                    layer.set_weights(weights_list)
                except Exception as e:
                    print(f"   ⚠️  {layer.name}: {e}")

    # ── Save as .h5 ───────────────────────────────────────
    out_path = str(WEIGHTS_DIR / 'best_hybrid_model.h5')
    model.save(out_path)
    print(f"   ✅ Saved: {out_path}")
    return model


def build_and_save_mobilenetv2():
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    print("\n[2/2] Building MobileNetV2 + Feature Fusion...")

    FEAT_DIM  = 8317
    NUM_CLASS = 40
    IMG_SHAPE = (96, 96, 3)

    img_in = tf.keras.Input(shape=IMG_SHAPE, name='image_input')
    base   = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights='imagenet'
    )
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    x = base(img_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='dense_mv2')(x)
    x = layers.BatchNormalization(name='bn_mv2_1')(x)
    x = layers.Dropout(0.4)(x)

    feat_in = tf.keras.Input(shape=(FEAT_DIM,), name='feature_input')
    y = layers.Dense(512, activation='relu', name='dense_feat_2')(feat_in)
    y = layers.BatchNormalization(name='bn_mv2_2')(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(256, activation='relu', name='dense_feat_3')(y)
    y = layers.BatchNormalization(name='bn_mv2_3')(y)
    y = layers.Dropout(0.3)(y)

    merged = layers.Concatenate()([x, y])
    f   = layers.Dense(256, activation='relu', name='dense_fuse')(merged)
    f   = layers.BatchNormalization(name='bn_fuse')(f)
    f   = layers.Dropout(0.4)(f)
    emb = layers.Dense(64, activation='relu', name='embedding')(f)
    out = layers.Dense(NUM_CLASS, activation='softmax', name='output')(emb)

    model = Model(inputs=[img_in, feat_in], outputs=[out, emb])
    print(f"   Model built — {model.count_params():,} parameters")

    print("   Loading weights from best_mobilenetv2_model_weights.npz ...")
    npz = load_npz('best_mobilenetv2_model_weights.npz')

    layer_map = {
        'dense_mv2':   ['dense_mv2/kernel',   'dense_mv2/bias'],
        'bn_mv2_1':    ['bn_mv2_1/gamma', 'bn_mv2_1/beta', 'bn_mv2_1/moving_mean', 'bn_mv2_1/moving_var'],
        'dense_feat_2':['dense_feat_2/kernel', 'dense_feat_2/bias'],
        'bn_mv2_2':    ['bn_mv2_2/gamma', 'bn_mv2_2/beta', 'bn_mv2_2/moving_mean', 'bn_mv2_2/moving_var'],
        'dense_feat_3':['dense_feat_3/kernel', 'dense_feat_3/bias'],
        'bn_mv2_3':    ['bn_mv2_3/gamma', 'bn_mv2_3/beta', 'bn_mv2_3/moving_mean', 'bn_mv2_3/moving_var'],
        'dense_fuse':  ['dense_fuse/kernel',  'dense_fuse/bias'],
        'bn_fuse':     ['bn_fuse/gamma', 'bn_fuse/beta', 'bn_fuse/moving_mean', 'bn_fuse/moving_var'],
        'embedding':   ['embedding/kernel',   'embedding/bias'],
        'output':      ['output/kernel',      'output/bias'],
    }

    for layer in model.layers:
        if layer.name in layer_map:
            keys = layer_map[layer.name]
            weights_list = [npz[k] for k in keys if k in npz]
            if weights_list:
                try:
                    layer.set_weights(weights_list)
                except Exception as e:
                    print(f"   ⚠️  {layer.name}: {e}")

    out_path = str(WEIGHTS_DIR / 'best_mobilenetv2_model.h5')
    model.save(out_path)
    print(f"   ✅ Saved: {out_path}")
    return model


if __name__ == '__main__':
    print("="*60)
    print("  Hybrid Face Recognition — Weight Loader")
    print("  Builds .h5 model files from saved .npz weights")
    print("="*60)

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} detected")
    except ImportError:
        print("❌ TensorFlow not found!")
        print("   Install: pip install tensorflow")
        exit(1)

    build_and_save_hybrid_cnn()
    build_and_save_mobilenetv2()

    print("\n" + "="*60)
    print("  ✅ DONE — Models saved to weights/")
    print("     best_hybrid_model.h5")
    print("     best_mobilenetv2_model.h5")
    print("\n  Now run: python app.py")
    print("="*60)
