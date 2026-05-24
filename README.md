# Convolutional Neural Networks for Image Classification with TensorFlow/Keras

This project implements and benchmarks convolutional neural networks for two image-classification tasks using TensorFlow/Keras. The primary objective is improving SIGNS multiclass recognition while keeping the smile baseline as a stable reference model.

## Project Overview

The project focuses on reproducible CNN training, evaluation, and benchmark reporting. The training pipeline loads local HDF5 datasets, applies task-specific preprocessing, trains multiple model variants, evaluates test performance, and saves metrics and figures under `outputs/`.

## Tasks

1. **Smile classification**
   - Binary classification: smile vs non-smile.
   - Dataset files expected under `datasets/`:
     - `train_happy.h5`
     - `test_happy.h5`

2. **Sign-language digit classification**
   - Multiclass classification: hand-sign digits from 0 to 5.
   - Dataset files expected under `datasets/`:
     - `train_signs.h5`
     - `test_signs.h5`

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── datasets/
│   └── .gitkeep
├── images/
│   └── .gitkeep
├── notebooks/
│   └── cnn_image_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── visualization.py
└── outputs/
    ├── figures/
    │   └── .gitkeep
    ├── metrics/
    │   └── .gitkeep
    └── models/
        └── .gitkeep
```

## Dataset Setup

Dataset files are not included in the repository. Place the required HDF5 files manually under `datasets/`:

```text
datasets/
├── train_happy.h5
├── test_happy.h5
├── train_signs.h5
└── test_signs.h5
```

The project uses relative paths only. The data loader raises a clear error if any required file is missing.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook:

```bash
jupyter notebook notebooks/cnn_image_classification.ipynb
```

Run the full benchmark (example):

```bash
python -m src.benchmark --task signs --epochs 100
python -m src.benchmark --task signs --epochs 100 --seeds 1 2 3
```

Run a single task and model (example):

```bash
python -m src.train --task signs --model tuned_cnn --epochs 100
python -m src.train --task signs --model augmented_cnn --epochs 100
```

Optional arguments:

```bash
python -m src.benchmark --task smile --epochs 20
python -m src.benchmark --save-models
python -m src.train --task signs --model baseline --epochs 50 --batch-size 64
```

## Model Architectures

### Smile Classification

The benchmark compares:

1. **Baseline CNN**
   ```text
   ZeroPadding2D -> Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D -> Flatten -> Dense(sigmoid)
   ```

2. **Improved CNN**
   ```text
   Conv-BN-ReLU -> MaxPool -> Conv-BN-ReLU -> MaxPool -> Conv-BN-ReLU -> GlobalAveragePooling -> Dropout -> Dense(sigmoid)
   ```

3. **Augmented CNN**
   - Uses the improved CNN backbone.
   - Adds moderate in-model augmentation with horizontal flipping, rotation, zoom, translation, and contrast adjustment.

### Sign-Language Digit Classification

The benchmark compares:

1. **Baseline CNN**
   ```text
   Input -> Conv2D -> ReLU -> MaxPooling2D -> Conv2D -> ReLU -> MaxPooling2D -> Flatten -> Dense(softmax)
   ```

2. **Improved CNN**
   ```text
   Conv-BN-ReLU blocks with 32, 64, and 128 filters -> MaxPooling -> Dropout -> GlobalAveragePooling -> Dense(softmax)
   ```

3. **Augmented CNN**
   - Uses the improved CNN backbone.
   - Adds conservative in-model augmentation with small rotation, zoom, translation, and contrast adjustment.
   - Avoids aggressive transformations that could change the hand-sign semantics.

## Benchmark Design

The benchmark script trains and evaluates the following models:

| Task | Models |
|---|---|
| Smile classification | baseline, augmented CNN (reference baseline kept unchanged) |
| Sign-language digit classification | baseline, current augmented CNN, tuned CNN, tuned augmented CNN, optional MobileNetV2 |

For each run, the pipeline records:

- task name
- model name
- trainable/non-trainable parameter count
- requested epochs
- actual epochs trained
- best validation accuracy
- best validation loss
- test accuracy
- test loss
- ROC-AUC for binary smile classification when computable
- training time

The training pipeline uses:

- fixed random seeds
- Adam/AdamW optimizer (AdamW preferred for SIGNS)
- EarlyStopping with restored best weights
- ReduceLROnPlateau
- CSVLogger
- optional ModelCheckpoint

## Results

Run the benchmark to generate per-run and aggregated results. The benchmark writes both per-run metrics and aggregate summaries (mean/std across seeds) to `outputs/metrics/`.

Per-run and aggregate files (CSV + JSON):

```text
outputs/metrics/benchmark_results.csv
outputs/metrics/benchmark_results.json
outputs/metrics/benchmark_summary.csv
outputs/metrics/benchmark_summary.json
```

Per-model detailed metrics (classification reports, confusion matrices) are saved under `outputs/metrics/` for each run.

## Output Artifacts

Generated artifacts are written locally under `outputs/`:

```text
outputs/
├── figures/
│   ├── *_training_curves.png
│   ├── *_confusion_matrix.png
│   └── benchmark_test_accuracy.png
├── metrics/
│   ├── *_training_log.csv
│   ├── *_detailed_metrics.json
│   ├── benchmark_results.csv
│   └── benchmark_results.json
└── models/
    └── *.keras        # only when --save-models is used
```

These outputs are excluded from version control by `.gitignore`.

## Reproducibility Notes

The project fixes random seeds for Python, NumPy, and TensorFlow. Small numerical differences may still occur across TensorFlow versions, operating systems, CPU/GPU execution, and low-level backend kernels.

## License / Data Note

Dataset files are not redistributed in this repository. Users should ensure they have appropriate rights to any local datasets placed under `datasets/`.

Ignored files

The repository intentionally ignores large or private artifacts so the remote remains lightweight. The following are excluded via `.gitignore`:

- dataset HDF5 files under `datasets/`
- raw image assets under `images/`
- model checkpoints under `outputs/models/`
- training logs and temporary files

Benchmark results (figures, tables, and JSON/CSV summaries) are written under `outputs/figures/` and `outputs/metrics/` and are safe to share once you decide which artifacts to publish.
