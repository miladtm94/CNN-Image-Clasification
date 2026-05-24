"""Evaluation utilities for model benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


Array = np.ndarray


def evaluate_keras_model(model: tf.keras.Model, x_test: Array, y_test: Array, batch_size: int = 64) -> Dict[str, float]:
    """Run Keras evaluation and return metric values by name.

    ``return_dict=True`` avoids version-dependent ambiguity in ``model.metrics_names``
    and preserves names such as ``accuracy``, ``auc``, and ``top_2_accuracy``.
    """
    values = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
    return {name: float(value) for name, value in values.items()}


def predict_probabilities(model: tf.keras.Model, x: Array, batch_size: int = 64) -> Array:
    """Return model output probabilities."""
    return model.predict(x, batch_size=batch_size, verbose=0)


def predicted_labels(probabilities: Array, task: str) -> Array:
    """Convert model probabilities to predicted label indices."""
    if task == "smile":
        return (probabilities.reshape(-1) >= 0.5).astype(int)
    if task == "signs":
        return np.argmax(probabilities, axis=1).astype(int)
    raise ValueError(f"Unsupported task: {task}")


def true_labels(y: Array, task: str) -> Array:
    """Convert binary or one-hot targets to flat integer labels."""
    if task == "smile":
        return y.reshape(-1).astype(int)
    if task == "signs":
        return np.argmax(y, axis=1).astype(int)
    raise ValueError(f"Unsupported task: {task}")


def detailed_evaluation(
    model: tf.keras.Model,
    x_test: Array,
    y_test: Array,
    task: str,
    batch_size: int = 64,
    labels: list[int] | None = None,
) -> Dict[str, Any]:
    """Compute classification metrics, confusion matrix, and optional ROC-AUC."""
    keras_metrics = evaluate_keras_model(model, x_test, y_test, batch_size=batch_size)
    probabilities = predict_probabilities(model, x_test, batch_size=batch_size)
    y_true = true_labels(y_test, task)
    y_pred = predicted_labels(probabilities, task)

    if labels is None:
        labels = sorted(np.unique(y_true).astype(int).tolist())

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    result: Dict[str, Any] = {
        "keras_metrics": keras_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
    }

    if task == "smile":
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities.reshape(-1)))
        except ValueError:
            result["roc_auc"] = None

    return result


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save nested metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
