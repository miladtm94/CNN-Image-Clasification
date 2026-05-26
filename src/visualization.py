"""Visualization utilities for training curves, confusion matrices, and benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


Array = np.ndarray


def show_image_samples(images: Array, labels: Array | None = None, indices: Sequence[int] | None = None, max_images: int = 6) -> None:
    """Display a small image grid for interactive inspection."""
    if indices is None:
        indices = list(range(min(max_images, len(images))))
    else:
        indices = list(indices)[:max_images]

    if not indices:
        raise ValueError("No images available to display.")

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(images[idx])
        ax.axis("off")
        if labels is not None:
            label = labels[idx]
            if np.ndim(label) > 0 and len(np.atleast_1d(label)) > 1:
                label = int(np.argmax(label))
            else:
                label = int(np.squeeze(label))
            ax.set_title(f"Label: {label}")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title: str, output_path: Path | None = None) -> pd.DataFrame:
    """Plot loss and accuracy curves from a Keras History object."""
    history_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_df.index + 1, history_df["loss"], label="train_loss")
    if "val_loss" in history_df:
        axes[0].plot(history_df.index + 1, history_df["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title}: Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "accuracy" in history_df:
        axes[1].plot(history_df.index + 1, history_df["accuracy"], label="train_accuracy")
    if "val_accuracy" in history_df:
        axes[1].plot(history_df.index + 1, history_df["val_accuracy"], label="val_accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return history_df


def plot_confusion_matrix(
    matrix: Array,
    class_names: Iterable[str] | None,
    title: str,
    output_path: Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    matrix = np.asarray(matrix)
    if class_names is None:
        class_names = [str(i) for i in range(matrix.shape[0])]
    class_names = [str(name.decode() if isinstance(name, bytes) else name) for name in class_names]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_benchmark_comparison(results_df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Create a bar chart comparing test accuracy across benchmarked models."""
    if results_df.empty or "test_accuracy" not in results_df.columns:
        return

    df = (
        results_df.groupby(["task", "model"], as_index=False)["test_accuracy"]
        .mean()
        .sort_values(["task", "model"])
    )
    df["run"] = df["task"] + " / " + df["model"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["run"], df["test_accuracy"])
    ax.set_ylabel("Test accuracy")
    ax.set_title("Benchmark Test Accuracy Comparison")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(df["test_accuracy"]):
        ax.text(idx, min(float(value) + 0.02, 1.03), f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
