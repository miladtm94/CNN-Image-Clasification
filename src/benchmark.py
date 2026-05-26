"""Benchmark baseline, improved, and augmented CNN models."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import pandas as pd

from . import config
from .data_loader import MissingDatasetError, dataset_summary, load_happy_dataset, load_signs_dataset, print_dataset_summary
from .train import available_models, train_and_evaluate
from .visualization import plot_benchmark_comparison


def run_benchmark(
    tasks: list[str] | None = None,
    epochs: int | None = None,
    save_models: bool = False,
    seeds: list[int] | None = None,
    include_transfer: bool = False,
) -> pd.DataFrame:
    """Run benchmark experiments and save metrics tables."""
    config.ensure_output_dirs()
    config.set_global_determinism(config.SEED)

    if tasks is None:
        tasks = ["smile", "signs"]
    seeds = seeds or [config.SEED]

    datasets = {}
    if "smile" in tasks:
        datasets["smile"] = load_happy_dataset()
        print_dataset_summary(datasets["smile"], "smile")
    if "signs" in tasks:
        datasets["signs"] = load_signs_dataset()
        print_dataset_summary(datasets["signs"], "signs")

    results: List[Dict[str, object]] = []
    dataset_summaries = {task: dataset_summary(data, task) for task, data in datasets.items()}

    for task in tasks:
        data = datasets[task]
        task_epochs = epochs if epochs is not None else (config.SMILE_EPOCHS if task == "smile" else config.SIGNS_EPOCHS)
        batch_size = config.SMILE_BATCH_SIZE if task == "smile" else config.DEFAULT_BATCH_SIZE

        for model_name, model_builder in available_models(task, include_transfer=include_transfer).items():
            for seed in seeds:
                print(f"\n=== Running {task}/{model_name} (seed={seed}) ===")
                config.set_global_determinism(seed)
                try:
                    _, _, summary = train_and_evaluate(
                        task=task,
                        model_name=model_name,
                        model_builder=model_builder,
                        data=data,
                        epochs=task_epochs,
                        batch_size=batch_size,
                        save_model=save_models,
                        seed=seed,
                    )
                except RuntimeError as exc:
                    print(f"Skipping {task}/{model_name} (seed={seed}): {exc}")
                    continue

                summary["seed"] = int(seed)
                results.append(summary)

    if not results:
        raise RuntimeError("No benchmark runs completed. Check dataset files and model initialization.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["task", "model", "seed"]).reset_index(drop=True)

    csv_path = config.METRICS_DIR / "benchmark_results.csv"
    json_path = config.METRICS_DIR / "benchmark_results.json"
    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_summaries": dataset_summaries,
                "benchmark_results": results,
                "reproducibility_note": (
                    "Seeds are fixed in Python, NumPy, and TensorFlow. Small differences may still occur "
                    "across hardware, TensorFlow versions, and GPU execution."
                ),
            },
            f,
            indent=2,
        )

    plot_benchmark_comparison(results_df, output_path=config.FIGURES_DIR / "benchmark_test_accuracy.png")

    print("\nBenchmark results")
    print(results_df.to_string(index=False))
    print(f"\nSaved per-run metrics to: {csv_path}")
    print(f"Saved JSON metrics to: {json_path}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN benchmark experiments.")
    parser.add_argument("--task", choices=["all", "smile", "signs"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--include-transfer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_tasks = None if args.task == "all" else [args.task]
    try:
        run_benchmark(
            tasks=selected_tasks,
            epochs=args.epochs,
            save_models=args.save_models,
            seeds=args.seeds,
            include_transfer=args.include_transfer,
        )
    except MissingDatasetError as exc:
        print(str(exc))
        raise SystemExit(1)
