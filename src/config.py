"""Central configuration for the CNN benchmarking project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

TRAIN_HAPPY_PATH = DATA_DIR / "train_happy.h5"
TEST_HAPPY_PATH = DATA_DIR / "test_happy.h5"
TRAIN_SIGNS_PATH = DATA_DIR / "train_signs.h5"
TEST_SIGNS_PATH = DATA_DIR / "test_signs.h5"

SEED = 42
DEFAULT_BATCH_SIZE = 64
SMILE_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
SMILE_EPOCHS = 30
SIGNS_EPOCHS = 80
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15

REQUIRED_DATASET_FILES = (
    "train_happy.h5",
    "test_happy.h5",
    "train_signs.h5",
    "test_signs.h5",
)


def ensure_output_dirs() -> None:
    """Create output directories used by training, evaluation, and plotting."""
    for directory in (OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_determinism(seed: int = SEED) -> None:
    """Set common random seeds for reproducible experiments.

    Exact reproducibility can still vary slightly across TensorFlow versions,
    hardware backends, GPU kernels, and operating systems.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some TensorFlow builds do not expose deterministic op control.
        pass
