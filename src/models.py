"""TensorFlow/Keras model definitions for CNN benchmarking."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from . import config


InputShape = Tuple[int, int, int]


def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    name: str | None = None,
) -> tf.Tensor:
    prefix = f"{name}_" if name else ""
    x = L.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{prefix}conv",
    )(x)
    x = L.BatchNormalization(name=f"{prefix}bn")(x)
    return L.ReLU(name=f"{prefix}relu")(x)


def smile_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    """Moderate augmentation for smile classification."""
    return tf.keras.Sequential(
        [
            L.RandomFlip("horizontal", seed=seed, name="aug_flip"),
            L.RandomRotation(0.05, seed=seed, name="aug_rotation"),
            L.RandomZoom(0.08, seed=seed, name="aug_zoom"),
            L.RandomTranslation(0.05, 0.05, seed=seed, name="aug_translation"),
            L.RandomContrast(0.10, seed=seed, name="aug_contrast"),
        ],
        name="smile_augmentation",
    )


def signs_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    """Conservative augmentation for hand-sign digit classification."""
    return tf.keras.Sequential(
        [
            L.RandomRotation(0.03, seed=seed, name="aug_rotation"),
            L.RandomZoom(0.05, seed=seed, name="aug_zoom"),
            L.RandomTranslation(0.04, 0.04, seed=seed, name="aug_translation"),
            L.RandomContrast(0.08, seed=seed, name="aug_contrast"),
        ],
        name="signs_augmentation",
    )


def build_smile_baseline(input_shape: InputShape = (64, 64, 3)) -> tf.keras.Model:
    """Baseline architecture for smile classification."""
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
    """Baseline architecture for SIGNS classification."""
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
    """Improved lightweight CNN for smile classification."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = _conv_bn_relu(inputs, 32, name="block1")
    x = L.MaxPooling2D((2, 2), name="pool1")(x)
    x = _conv_bn_relu(x, 64, name="block2")
    x = L.MaxPooling2D((2, 2), name="pool2")(x)
    x = _conv_bn_relu(x, 128, name="block3")
    x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = L.Dropout(0.35, name="dropout")(x)
    outputs = L.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="smile_improved_cnn")


def build_signs_improved_cnn(input_shape: InputShape = (64, 64, 3), num_classes: int = 6) -> tf.keras.Model:
    """Improved lightweight CNN for SIGNS classification."""
    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = _conv_bn_relu(inputs, 32, name="block1a")
    x = _conv_bn_relu(x, 32, name="block1b")
    x = L.MaxPooling2D((2, 2), name="pool1")(x)
    x = L.Dropout(0.15, name="dropout1")(x)

    x = _conv_bn_relu(x, 64, name="block2a")
    x = _conv_bn_relu(x, 64, name="block2b")
    x = L.MaxPooling2D((2, 2), name="pool2")(x)
    x = L.Dropout(0.25, name="dropout2")(x)

    x = _conv_bn_relu(x, 128, name="block3a")
    x = _conv_bn_relu(x, 128, name="block3b")
    x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = L.Dropout(0.40, name="dropout3")(x)
    outputs = L.Dense(num_classes, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="signs_improved_cnn")


def build_smile_augmented_cnn(input_shape: InputShape = (64, 64, 3), seed: int = config.SEED) -> tf.keras.Model:
    """Improved smile CNN with in-model data augmentation."""
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = smile_augmentation(seed)(inputs)
    backbone = build_smile_improved_cnn(input_shape=input_shape)
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
    backbone = build_signs_improved_cnn(input_shape=input_shape, num_classes=num_classes)
    outputs = backbone(x)
    return tf.keras.Model(inputs, outputs, name="signs_augmented_cnn")


def build_signs_mobilenetv2_transfer(
    input_shape: InputShape = (64, 64, 3),
    num_classes: int = 6,
) -> tf.keras.Model:
    """Optional MobileNetV2 transfer baseline (frozen feature extractor)."""
    try:
        inputs = tf.keras.Input(shape=input_shape, name="image")
        x = L.Resizing(96, 96, name="resize_to_96")(inputs)
        x = L.Lambda(lambda z: mobilenet_preprocess(z * 255.0), name="mobilenet_preprocess")(x)
        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
        base.trainable = False
        x = base(x, training=False)
        x = L.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = L.Dropout(0.3, name="dropout1")(x)
        outputs = L.Dense(num_classes, activation="softmax", name="output")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="signs_mobilenetv2_transfer")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Could not initialize MobileNetV2/ImageNet weights") from exc


def compile_model(
    model: tf.keras.Model,
    task: str,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """Compile model with task-specific losses and metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if task == "smile":
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
    elif task == "signs":
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Expected 'smile' or 'signs'.")

    return model
