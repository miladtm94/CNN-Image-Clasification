"""TensorFlow/Keras model definitions for CNN benchmarking."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as L

from . import config
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


InputShape = Tuple[int, int, int]


def _conv_bn_relu(x, filters: int, kernel_size: tuple[int, int] = (3, 3), name: str | None = None):
    prefix = f"{name}_" if name else ""
    x = L.Conv2D(filters, kernel_size, padding="same", use_bias=False, kernel_initializer="he_normal", name=f"{prefix}conv")(x)
    x = L.BatchNormalization(name=f"{prefix}bn")(x)
    return L.ReLU(name=f"{prefix}relu")(x)


def smile_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    """Moderate image augmentation for smile classification."""
    return tf.keras.Sequential(
        [
            L.RandomFlip("horizontal", seed=seed, name="aug_flip"),
            L.RandomRotation(0.05, seed=seed, name="aug_rotation"),
            L.RandomZoom(0.08, seed=seed, name="aug_zoom"),
            L.RandomTranslation(0.05, 0.05, seed=seed, name="aug_translation"),
            L.RandomContrast(0.1, seed=seed, name="aug_contrast"),
        ],
        name="smile_augmentation",
    )


def signs_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    """Conservative augmentation for hand-sign digit classification."""
    return tf.keras.Sequential(
        [
            L.RandomRotation(0.04, seed=seed, name="aug_rotation"),
            L.RandomZoom(0.06, seed=seed, name="aug_zoom"),
            L.RandomTranslation(0.04, 0.04, seed=seed, name="aug_translation"),
            L.RandomContrast(0.08, seed=seed, name="aug_contrast"),
        ],
        name="signs_augmentation",
    )


def build_smile_baseline(input_shape: InputShape = (64, 64, 3)) -> tf.keras.Model:
    """Original compact Sequential CNN used as the smile-classification baseline."""
    return tf.keras.Sequential(
        [
            L.ZeroPadding2D(padding=(3, 3), input_shape=input_shape, name="zero_padding"),
            L.Conv2D(32, (7, 7), strides=(1, 1), padding="valid", name="conv"),
            L.BatchNormalization(axis=3, name="batch_norm"),
            L.ReLU(name="relu"),
            L.MaxPooling2D(pool_size=(2, 2), name="max_pool"),
            L.Flatten(name="flatten"),
            L.Dense(1, activation="sigmoid", name="output"),
        ],
        name="smile_baseline",
    )


def build_signs_baseline(input_shape: InputShape = (64, 64, 3), num_classes: int = 6) -> tf.keras.Model:
    """Original Functional API CNN used as the SIGNS classification baseline."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = L.Conv2D(8, (4, 4), strides=(1, 1), padding="same", name="conv_1")(inputs)
    x = L.ReLU(name="relu_1")(x)
    x = L.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding="same", name="pool_1")(x)
    x = L.Conv2D(16, (2, 2), strides=(1, 1), padding="same", name="conv_2")(x)
    x = L.ReLU(name="relu_2")(x)
    x = L.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same", name="pool_2")(x)
    x = L.Flatten(name="flatten")(x)
    outputs = L.Dense(num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="signs_baseline")


def build_smile_improved_cnn(input_shape: InputShape = (64, 64, 3)) -> tf.keras.Model:
    """Improved lightweight CNN for binary smile classification."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _conv_bn_relu(inputs, 32, (3, 3), "block1")
    x = L.MaxPooling2D((2, 2), name="pool1")(x)
    x = _conv_bn_relu(x, 64, (3, 3), "block2")
    x = L.MaxPooling2D((2, 2), name="pool2")(x)
    x = _conv_bn_relu(x, 128, (3, 3), "block3")
    x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = L.Dropout(0.35, name="dropout")(x)
    outputs = L.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="smile_improved_cnn")


def build_signs_improved_cnn(input_shape: InputShape = (64, 64, 3), num_classes: int = 6) -> tf.keras.Model:
    """Improved lightweight CNN for SIGNS multiclass classification."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _conv_bn_relu(inputs, 32, (3, 3), "block1a")
    x = _conv_bn_relu(x, 32, (3, 3), "block1b")
    x = L.MaxPooling2D((2, 2), name="pool1")(x)
    x = L.Dropout(0.15, name="dropout1")(x)

    x = _conv_bn_relu(x, 64, (3, 3), "block2a")
    x = _conv_bn_relu(x, 64, (3, 3), "block2b")
    x = L.MaxPooling2D((2, 2), name="pool2")(x)
    x = L.Dropout(0.25, name="dropout2")(x)

    x = _conv_bn_relu(x, 128, (3, 3), "block3a")
    x = _conv_bn_relu(x, 128, (3, 3), "block3b")
    x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = L.Dropout(0.4, name="dropout3")(x)
    outputs = L.Dense(num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="signs_improved_cnn")


def build_signs_tuned_cnn(input_shape: InputShape = (64, 64, 3), num_classes: int = 6) -> tf.keras.Model:
    """Tuned, smaller CNN for SIGNS as requested.

    Architecture follows the specified block layout and uses L2 regularization on conv/dense kernels.
    """
    l2 = regularizers.L2(1e-4)
    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = L.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2, name="conv1a")(inputs)
    x = L.BatchNormalization(name="bn1a")(x)
    x = L.ReLU(name="relu1a")(x)

    x = L.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2, name="conv1b")(x)
    x = L.BatchNormalization(name="bn1b")(x)
    x = L.ReLU(name="relu1b")(x)

    x = L.MaxPooling2D(name="pool1")(x)
    x = L.Dropout(0.15, name="dropout1")(x)

    x = L.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2, name="conv2a")(x)
    x = L.BatchNormalization(name="bn2a")(x)
    x = L.ReLU(name="relu2a")(x)

    x = L.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2, name="conv2b")(x)
    x = L.BatchNormalization(name="bn2b")(x)
    x = L.ReLU(name="relu2b")(x)

    x = L.MaxPooling2D(name="pool2")(x)
    x = L.Dropout(0.25, name="dropout2")(x)

    x = L.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2, name="conv3")(x)
    x = L.BatchNormalization(name="bn3")(x)
    x = L.ReLU(name="relu3")(x)

    x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=l2, name="dense1")(x)
    x = L.Dropout(0.35, name="dropout3")(x)
    outputs = L.Dense(num_classes, activation="softmax", name="output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="signs_tuned_cnn")


def build_signs_tuned_augmented_cnn(
    input_shape: InputShape = (64, 64, 3), num_classes: int = 6, seed: int = config.SEED
) -> tf.keras.Model:
    """Tuned SIGNS CNN with mild augmentation (no RandomFlip)."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    aug = tf.keras.Sequential([
        L.RandomRotation(0.04, seed=seed, name="aug_rotation"),
        L.RandomZoom(0.08, seed=seed, name="aug_zoom"),
        L.RandomTranslation(0.05, 0.05, seed=seed, name="aug_translation"),
        L.RandomContrast(0.08, seed=seed, name="aug_contrast"),
    ], name="tuned_signs_augmentation")
    x = aug(inputs)
    backbone = build_signs_tuned_cnn(input_shape=input_shape, num_classes=num_classes)
    outputs = backbone(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="signs_tuned_augmented_cnn")


def build_signs_mobilenetv2(input_shape: InputShape = (64, 64, 3), num_classes: int = 6) -> tf.keras.Model:
    """Lightweight MobileNetV2 transfer-learning model resized to 96x96.

    If ImageNet weights cannot be downloaded, raise an informative error so callers can skip.
    """
    try:
        inputs = tf.keras.Input(shape=input_shape, name="image")
        x = L.Resizing(96, 96, name="resize_to_96")(inputs)
        # MobileNetV2 expects inputs in range [-1, 1]
        x = L.Lambda(lambda z: mobilenet_preprocess(z * 255.0), name="mobilenet_preprocess")(x)
        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
        base.trainable = False
        x = base(x, training=False)
        x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = L.Dropout(0.3, name="dropout1")(x)
        x = L.Dense(64, activation="relu", name="dense1")(x)
        x = L.Dropout(0.2, name="dropout2")(x)
        outputs = L.Dense(num_classes, activation="softmax", name="output")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="signs_mobilenetv2")
    except Exception as exc:  # pragma: no cover - network/download dependent
        raise RuntimeError("Could not build MobileNetV2 with ImageNet weights: " + str(exc)) from exc


def build_smile_augmented_cnn(input_shape: InputShape = (64, 64, 3), seed: int = config.SEED) -> tf.keras.Model:
    """Improved smile CNN with in-model data augmentation."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = smile_augmentation(seed)(inputs)
    backbone = build_smile_improved_cnn(input_shape)
    outputs = backbone(x)
    return tf.keras.Model(inputs, outputs, name="smile_augmented_cnn")


def build_signs_augmented_cnn(
    input_shape: InputShape = (64, 64, 3),
    num_classes: int = 6,
    seed: int = config.SEED,
) -> tf.keras.Model:
    """Improved SIGNS CNN with conservative in-model augmentation."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = signs_augmentation(seed)(inputs)
    backbone = build_signs_improved_cnn(input_shape, num_classes)
    outputs = backbone(x)
    return tf.keras.Model(inputs, outputs, name="signs_augmented_cnn")


# Backward-compatible aliases used by earlier project versions.
build_smile_classifier = build_smile_baseline
build_sign_digit_classifier = build_signs_baseline


def compile_model(
    model: tf.keras.Model,
    task: str,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float | None = None,
    label_smoothing: float | None = None,
) -> tf.keras.Model:
    """Compile a model with task-appropriate loss, metrics and (optionally) AdamW.

    For `signs` we prefer AdamW with weight decay; if unavailable the caller should ensure
    model layers use kernel_regularizer L2.
    """
    optimizer = None
    if task == "signs" and weight_decay is not None:
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        except Exception:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task == "smile":
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
    elif task == "signs":
        loss_kwargs = {"label_smoothing": float(label_smoothing)} if label_smoothing is not None else {}
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(**loss_kwargs),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Expected 'smile' or 'signs'.")
    return model
