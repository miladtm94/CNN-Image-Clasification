"""Training utilities and command-line entry point."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, Tuple

import pandas as pd
import tensorflow as tf

from . import config
from .data_loader import DatasetBundle, load_happy_dataset, load_signs_dataset
from .evaluate import detailed_evaluation, save_json
from .models import (
    build_signs_augmented_cnn,
    build_signs_baseline,
    build_signs_improved_cnn,
    build_signs_mobilenetv2_transfer,
    build_smile_augmented_cnn,
    build_smile_baseline,
    build_smile_improved_cnn,
    compile_model,
)
from .visualization import plot_confusion_matrix, plot_training_history


ModelBuilder = Callable[..., tf.keras.Model]


def _metric_from_history(history: tf.keras.callbacks.History, metric_name: str, mode: str) -> float | None:
    values = history.history.get(metric_name)
    if not values:
        return None
    return float(max(values) if mode == "max" else min(values))


def make_callbacks(run_name: str, save_model: bool = False) -> list[tf.keras.callbacks.Callback]:
    """Create standard callbacks for model training."""
    config.ensure_output_dirs()
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.METRICS_DIR / f"{run_name}_training_log.csv")),
    ]

    if save_model:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.MODELS_DIR / f"{run_name}.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    return callbacks


def train_and_evaluate(
    task: str,
    model_name: str,
    model_builder: ModelBuilder,
    data: DatasetBundle,
    epochs: int,
    batch_size: int,
    learning_rate: float = config.LEARNING_RATE,
    save_model: bool = False,
    seed: int = config.SEED,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, object]]:
    """Compile, train, evaluate, and persist diagnostics for one model run."""
    config.ensure_output_dirs()
    config.set_global_determinism(seed)

    input_shape = data.x_train.shape[1:]
    try:
        if task == "signs":
            model = model_builder(input_shape=input_shape, num_classes=data.y_train.shape[1])
        else:
            model = model_builder(input_shape=input_shape)
    except Exception as exc:
        raise RuntimeError(f"Model builder failed for {task}/{model_name}: {exc}") from exc

    compile_model(model=model, task=task, learning_rate=learning_rate)

    run_name = f"{task}_{model_name}_seed{seed}"
    start_time = time.perf_counter()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_split=config.VALIDATION_SPLIT,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=make_callbacks(run_name, save_model=save_model),
        verbose=2,
    )
    training_time = time.perf_counter() - start_time

    evaluation = detailed_evaluation(
        model=model,
        x_test=data.x_test,
        y_test=data.y_test,
        task=task,
        batch_size=batch_size,
        labels=list(range(data.y_train.shape[1])) if task == "signs" else [0, 1],
    )

    keras_metrics = evaluation["keras_metrics"]
    summary = {
        "task": task,
        "model": model_name,
        "parameters": int(model.count_params()),
        "epochs_requested": int(epochs),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "best_validation_accuracy": _metric_from_history(history, "val_accuracy", "max"),
        "best_validation_loss": _metric_from_history(history, "val_loss", "min"),
        "test_loss": float(keras_metrics.get("loss", float("nan"))),
        "test_accuracy": float(keras_metrics.get("accuracy", float("nan"))),
        "training_time_seconds": float(training_time),
    }

    if task == "smile":
        summary["roc_auc"] = evaluation.get("roc_auc")

    save_json(evaluation, config.METRICS_DIR / f"{run_name}_detailed_metrics.json")
    plot_training_history(history, title=run_name, output_path=config.FIGURES_DIR / f"{run_name}_training_curves.png")
    plot_confusion_matrix(
        evaluation["confusion_matrix"],
        class_names=data.classes,
        title=f"{run_name}: Confusion Matrix",
        output_path=config.FIGURES_DIR / f"{run_name}_confusion_matrix.png",
    )

    return model, history, summary


def available_models(task: str, include_transfer: bool = False) -> Dict[str, ModelBuilder]:
    """Return benchmark model builders for a task."""
    if task == "smile":
        return {
            "baseline": build_smile_baseline,
            "improved_cnn": build_smile_improved_cnn,
            "augmented_cnn": build_smile_augmented_cnn,
        }

    if task == "signs":
        models: Dict[str, ModelBuilder] = {
            "baseline": build_signs_baseline,
            "improved_cnn": build_signs_improved_cnn,
            "augmented_cnn": build_signs_augmented_cnn,
        }
        if include_transfer:
            models["mobilenetv2_transfer"] = build_signs_mobilenetv2_transfer
        return models

    raise ValueError("task must be 'smile' or 'signs'")


def load_task_data(task: str) -> DatasetBundle:
    """Load preprocessed dataset bundle for a task."""
    if task == "smile":
        return load_happy_dataset()
    if task == "signs":
        return load_signs_dataset()
    raise ValueError("task must be 'smile' or 'signs'")


def run_single_task(
    task: str,
    model_name: str = "improved_cnn",
    epochs: int | None = None,
    batch_size: int | None = None,
    save_model: bool = False,
    seed: int = config.SEED,
    include_transfer: bool = False,
) -> Dict[str, object]:
    """Train and evaluate one selected model for one task."""
    data = load_task_data(task)
    models = available_models(task, include_transfer=include_transfer)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(models)}")

    default_epochs = config.SMILE_EPOCHS if task == "smile" else config.SIGNS_EPOCHS
    epochs = epochs if epochs is not None else default_epochs
    default_batch_size = config.SMILE_BATCH_SIZE if task == "smile" else config.DEFAULT_BATCH_SIZE
    batch_size = batch_size if batch_size is not None else default_batch_size

    _, _, summary = train_and_evaluate(
        task=task,
        model_name=model_name,
        model_builder=models[model_name],
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        save_model=save_model,
        seed=seed,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate one model for one task.")
    parser.add_argument("--task", choices=["smile", "signs"], default="smile")
    parser.add_argument("--model", default="improved_cnn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--include-transfer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_single_task(
        task=args.task,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model=args.save_model,
        seed=args.seed,
        include_transfer=args.include_transfer,
    )
    print(pd.DataFrame([result]).to_string(index=False))
