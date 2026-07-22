import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, Model

# Fix Keras cache directory issue
os.environ['KERAS_HOME'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.keras')
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging


def build_hybrid_model(img_shape=(128, 128, 1), feature_dim=8317, num_classes=40):

    img_input = tf.keras.Input(shape=img_shape, name='image_input')

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu', name='c1')(img_input)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', name='c2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', name='c3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', name='c4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu', name='di')(x)
    x = layers.BatchNormalization(name='bni')(x)
    x = layers.Dropout(0.4)(x)

    feat_input = tf.keras.Input(shape=(feature_dim,), name='feature_input')

    y = layers.Dense(1024, activation='relu', name='df5')(feat_input)
    y = layers.BatchNormalization(name='bnf5')(y)
    y = layers.Dropout(0.4)(y)

    y = layers.Dense(512, activation='relu', name='df6')(y)
    y = layers.BatchNormalization(name='bnf6')(y)
    y = layers.Dropout(0.3)(y)

    y = layers.Dense(256, activation='relu', name='df7')(y)
    y = layers.BatchNormalization(name='bnf7')(y)
    y = layers.Dropout(0.3)(y)

    merged = layers.Concatenate()([x, y])

    f = layers.Dense(512, activation='relu', name='fuse1')(merged)
    f = layers.BatchNormalization(name='bnfu1')(f)
    f = layers.Dropout(0.5)(f)

    f_res = layers.Dense(512, activation='relu', name='res')(merged)
    f = layers.Add()([f, f_res])

    f = layers.Dense(256, activation='relu', name='fuse2')(f)
    f = layers.BatchNormalization(name='bnfu2')(f)
    f = layers.Dropout(0.4)(f)

    f = layers.Dense(128, activation='relu', name='fuse3')(f)

    f = layers.Dense(64, activation='relu', name='emb')(f)

    output = layers.Dense(
        num_classes,
        activation='softmax',
        name='out'
    )(f)

    return Model(
        inputs=[img_input, feat_input],
        outputs=output
    )


def load_hybrid_model_weights(model, npz_path):
    weights_dict = np.load(npz_path, allow_pickle=True)
    layer_mapping = {
        'c1': ('c1_k', 'c1_b'),
        'bn1': ('bn1_g', 'bn1_b', 'bn1_mm', 'bn1_mv'),
        'c2': ('c2_k', 'c2_b'),
        'bn2': ('bn2_g', 'bn2_b', 'bn2_mm', 'bn2_mv'),
        'c3': ('c3_k', 'c3_b'),
        'bn3': ('bn3_g', 'bn3_b', 'bn3_mm', 'bn3_mv'),
        'c4': ('c4_k', 'c4_b'),
        'bn4': ('bn4_g', 'bn4_b', 'bn4_mm', 'bn4_mv'),
        'di': ('di_k', 'di_b'),
        'bni': ('bni_g', 'bni_b', 'bni_mm', 'bni_mv'),
        'df5': ('df5_k', 'df5_b'),
        'bnf5': ('bnf5_g', 'bnf5_b', 'bnf5_mm', 'bnf5_mv'),
        'df6': ('df6_k', 'df6_b'),
        'bnf6': ('bnf6_g', 'bnf6_b', 'bnf6_mm', 'bnf6_mv'),
        'df7': ('df7_k', 'df7_b'),
        'bnf7': ('bnf7_g', 'bnf7_b', 'bnf7_mm', 'bnf7_mv'),
        'fuse1': ('fuse1_k', 'fuse1_b'),
        'bnfu1': ('bnfu1_g', 'bnfu1_b', 'bnfu1_mm', 'bnfu1_mv'),
        'res': ('res_k', 'res_b'),
        'fuse2': ('fuse2_k', 'fuse2_b'),
        'bnfu2': ('bnfu2_g', 'bnfu2_b', 'bnfu2_mm', 'bnfu2_mv'),
        'fuse3': ('fuse3_k', 'fuse3_b'),
        'emb': ('emb_k', 'emb_b'),
        'out': ('out_k', 'out_b'),
    }
    for layer_name, keys in layer_mapping.items():
        layer = model.get_layer(layer_name)
        if len(keys) == 2:
            # Dense or Conv: kernel and bias
            layer.set_weights([weights_dict[keys[0]], weights_dict[keys[1]]])
        elif len(keys) == 4:
            # BatchNormalization: gamma, beta, moving mean, moving var
            layer.set_weights([weights_dict[keys[0]], weights_dict[keys[1]], 
                              weights_dict[keys[2]], weights_dict[keys[3]]])
    return model


def build_mobilenetv2_model(
        img_shape=(96,96,3),
        feature_dim=8317,
        num_classes=40,
        fine_tune_at=100):

    img_input = tf.keras.Input(shape=img_shape, name='image_input')

    base = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )

    base.trainable = True

    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    x = base(img_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    feat_input = tf.keras.Input(
        shape=(feature_dim,),
        name='feature_input'
    )

    y = layers.Dense(512, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)

    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    merged = layers.Concatenate()([x, y])

    f = layers.Dense(256, activation='relu')(merged)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.4)(f)

    output = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(f)

    return Model(
        inputs=[img_input, feat_input],
        outputs=output
    )


def load_mobilenetv2_model_weights(model, npz_path):
    weights_dict = np.load(npz_path, allow_pickle=True)
    # Assuming the MobileNetV2 model in the npz has matching layer names or we can load by shape
    # For now, let's try to load as many as possible
    return model