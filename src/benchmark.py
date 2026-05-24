"""Benchmark baseline, improved, and augmented CNN models."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import pandas as pd

from . import config
from .data_loader import dataset_summary, load_happy_dataset, load_signs_dataset, print_dataset_summary
from .train import available_models, train_and_evaluate
from .visualization import plot_benchmark_comparison


def run_benchmark(
    tasks: list[str] | None = None,
    epochs: int | None = None,
    save_models: bool = False,
) -> pd.DataFrame:
    """Run the full benchmark and save aggregate results."""
    config.ensure_output_dirs()
    config.set_global_determinism(config.SEED)

    if tasks is None:
        tasks = ["smile", "signs"]

    datasets = {}
    if "smile" in tasks:
        datasets["smile"] = load_happy_dataset()
        print_dataset_summary(datasets["smile"], "smile")
    if "signs" in tasks:
        datasets["signs"] = load_signs_dataset()
        print_dataset_summary(datasets["signs"], "signs")

    results: List[Dict[str, object]] = []
    dataset_summaries = {task: dataset_summary(data, task) for task, data in datasets.items()}

    # Seeds: allow repeated runs for each seed
    seeds = getattr(config, "BENCHMARK_SEEDS", [config.SEED])

    for task in tasks:
        data = datasets[task]
        task_epochs = epochs if epochs is not None else (config.SMILE_EPOCHS if task == "smile" else config.SIGNS_EPOCHS)
        batch_size = config.SMILE_BATCH_SIZE if task == "smile" else config.DEFAULT_BATCH_SIZE
        for model_name, builder in available_models(task).items():
            for seed in seeds:
                print(f"\n=== Running {task} / {model_name} (seed={seed}) ===")
                # Set determinism for each run
                config.set_global_determinism(seed)
                try:
                    _, _, summary = train_and_evaluate(
                        task=task,
                        model_name=model_name,
                        model_builder=builder,
                        data=data,
                        epochs=task_epochs,
                        batch_size=batch_size,
                        save_model=save_models,
                        seed=seed,
                    )
                except RuntimeError as exc:
                    # Skip models that cannot be built (e.g., MobileNet weight download failures)
                    print(f"Skipping {task}/{model_name} for seed {seed}: {exc}")
                    continue
                summary["seed"] = int(seed)
                results.append(summary)

    results_df = pd.DataFrame(results)

    # Save per-run results
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
                    "across hardware, TensorFlow versions, and GPU kernels."
                ),
            },
            f,
            indent=2,
        )

    # Aggregate: compute mean and std for numeric metrics grouped by task+model
    numeric_cols = [c for c in results_df.columns if results_df[c].dtype in ("float64", "int64")]
    agg = results_df.groupby(["task", "model"]).agg(["mean", "std"])
    # Flatten multiindex columns
    agg.columns = ["_" .join([col[0], col[1]]) for col in agg.columns]
    summary_df = agg.reset_index()

    summary_csv = config.METRICS_DIR / "benchmark_summary.csv"
    summary_json = config.METRICS_DIR / "benchmark_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json.loads(summary_df.to_json(orient="records")), f, indent=2)

    try:
        plot_benchmark_comparison(results_df, output_path=config.FIGURES_DIR / "benchmark_test_accuracy.png")
    except Exception:
        pass

    print("\nBenchmark results")
    print(results_df.to_string(index=False))
    print(f"\nSaved per-run metrics to: {csv_path}")
    print(f"Saved per-run JSON to: {json_path}")
    print(f"Saved aggregated summary to: {summary_csv} and {summary_json}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN benchmark experiments.")
    parser.add_argument("--task", choices=["all", "smile", "signs"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count for all selected runs.")
    parser.add_argument("--save-models", action="store_true", help="Save best checkpoints under outputs/models/.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional list of integer seeds to run (e.g. --seeds 1 2 3).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_tasks = None if args.task == "all" else [args.task]
    if getattr(args, "seeds", None):
        config.BENCHMARK_SEEDS = args.seeds
    run_benchmark(tasks=selected_tasks, epochs=args.epochs, save_models=args.save_models)
