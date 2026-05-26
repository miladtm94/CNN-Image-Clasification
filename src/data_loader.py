"""Dataset loading and preprocessing utilities for HDF5 image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

from . import config


Array = np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    """Container for prepared train/test image data."""

    x_train: Array
    y_train: Array
    x_test: Array
    y_test: Array
    classes: Array
    raw_y_train: Array
    raw_y_test: Array


class MissingDatasetError(FileNotFoundError):
    """Raised when one or more required local dataset files are missing."""


def _missing_file_message(missing_paths: list[Path]) -> str:
    formatted_paths: list[str] = []
    for path in missing_paths:
        try:
            display = path.resolve().relative_to(config.PROJECT_ROOT.resolve())
        except Exception:
            display = path
        formatted_paths.append(f"  - {display}")
    missing = "\n".join(formatted_paths)
    expected = "\n".join(f"  - datasets/{name}" for name in config.REQUIRED_DATASET_FILES)
    return (
        "Required dataset file(s) were not found.\n\n"
        f"Missing:\n{missing}\n\n"
        "Place the HDF5 files under the repository's datasets/ folder.\n"
        f"Expected files:\n{expected}\n\n"
        "The repository intentionally excludes datasets from version control."
    )


def check_required_files(paths: tuple[Path, ...]) -> None:
    """Raise a clear error if any required dataset files are missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise MissingDatasetError(_missing_file_message(missing))


def _load_hdf5_pair(train_path: Path, test_path: Path) -> Tuple[Array, Array, Array, Array, Array]:
    """Load train/test arrays and class labels from paired HDF5 files."""
    check_required_files((train_path, test_path))
    with h5py.File(train_path, "r") as train_dataset, h5py.File(test_path, "r") as test_dataset:
        x_train = np.asarray(train_dataset["train_set_x"][:])
        y_train = np.asarray(train_dataset["train_set_y"][:]).reshape(1, -1)
        x_test = np.asarray(test_dataset["test_set_x"][:])
        y_test = np.asarray(test_dataset["test_set_y"][:]).reshape(1, -1)
        classes = np.asarray(test_dataset["list_classes"][:])
    return x_train, y_train, x_test, y_test, classes


def normalize_images(x: Array) -> Array:
    """Normalize image pixels to floating-point values in [0, 1]."""
    return x.astype("float32") / 255.0


def prepare_binary_labels(y: Array) -> Array:
    """Return binary labels with shape (num_samples, 1)."""
    return y.reshape(-1, 1).astype("float32")


def one_hot_encode(y: Array, num_classes: int) -> Array:
    """Convert integer labels to one-hot encoded labels."""
    return np.eye(num_classes, dtype="float32")[y.reshape(-1)]


def class_distribution(labels: Array) -> Dict[int, int]:
    """Return a class-count dictionary from flat or one-hot labels."""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    else:
        labels = labels.reshape(-1)
    values, counts = np.unique(labels.astype(int), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def dataset_summary(bundle: DatasetBundle, task_name: str) -> Dict[str, object]:
    """Return a compact dictionary describing a prepared dataset."""
    return {
        "task": task_name,
        "train_samples": int(bundle.x_train.shape[0]),
        "test_samples": int(bundle.x_test.shape[0]),
        "image_shape": tuple(int(v) for v in bundle.x_train.shape[1:]),
        "num_classes": int(len(bundle.classes)),
        "train_class_distribution": class_distribution(bundle.raw_y_train),
        "test_class_distribution": class_distribution(bundle.raw_y_test),
    }


def load_happy_dataset(data_dir: Path | str | None = None) -> DatasetBundle:
    """Load and preprocess the Happy House smile-classification dataset."""
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = _load_hdf5_pair(
        data_dir / "train_happy.h5",
        data_dir / "test_happy.h5",
    )
    return DatasetBundle(
        x_train=normalize_images(x_train_orig),
        y_train=prepare_binary_labels(y_train_orig),
        x_test=normalize_images(x_test_orig),
        y_test=prepare_binary_labels(y_test_orig),
        classes=classes,
        raw_y_train=y_train_orig.reshape(-1),
        raw_y_test=y_test_orig.reshape(-1),
    )


def load_signs_dataset(data_dir: Path | str | None = None, num_classes: int = 6) -> DatasetBundle:
    """Load and preprocess the SIGNS hand-sign digit dataset."""
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = _load_hdf5_pair(
        data_dir / "train_signs.h5",
        data_dir / "test_signs.h5",
    )
    return DatasetBundle(
        x_train=normalize_images(x_train_orig),
        y_train=one_hot_encode(y_train_orig, num_classes=num_classes),
        x_test=normalize_images(x_test_orig),
        y_test=one_hot_encode(y_test_orig, num_classes=num_classes),
        classes=classes,
        raw_y_train=y_train_orig.reshape(-1),
        raw_y_test=y_test_orig.reshape(-1),
    )


def print_dataset_summary(bundle: DatasetBundle, task_name: str) -> None:
    """Print a readable dataset summary."""
    summary = dataset_summary(bundle, task_name)
    for key, value in summary.items():
        print(f"{key}: {value}")
