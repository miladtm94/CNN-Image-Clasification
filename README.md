# TensorFlow/Keras CNN Image Classification Benchmark

## Project Overview
This repository runs reproducible CNN experiments for two local HDF5 image-classification tasks. It includes baseline models, improved custom CNNs, augmentation variants, optional transfer learning, and a benchmark pipeline that saves metrics and figures under `outputs/`.

## Tasks
1. Smile classification (`train_happy.h5`, `test_happy.h5`)
   - Binary target: smile vs non-smile
2. Sign-language digit classification (`train_signs.h5`, `test_signs.h5`)
   - Multiclass target: digits `0-5`

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
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── visualization.py
│   └── config.py
└── outputs/
    ├── figures/
    │   └── .gitkeep
    ├── metrics/
    │   └── .gitkeep
    └── models/
        └── .gitkeep
```

## Dataset Setup
Place local HDF5 files in `datasets/`:
```text
datasets/
├── train_happy.h5
├── test_happy.h5
├── train_signs.h5
└── test_signs.h5
```
If files are missing, loaders and scripts stop with a clear error listing the expected paths.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run notebook:
```bash
jupyter notebook notebooks/cnn_image_classification.ipynb
```

Run full benchmark:
```bash
python -m src.benchmark
```

Run benchmark for one task:
```bash
python -m src.benchmark --task smile
python -m src.benchmark --task signs
```

Optional transfer baseline (SIGNS only):
```bash
python -m src.benchmark --task signs --include-transfer
```

Run one model/task:
```bash
python -m src.train --task smile --model baseline
python -m src.train --task smile --model improved_cnn
python -m src.train --task smile --model augmented_cnn
python -m src.train --task signs --model baseline
python -m src.train --task signs --model improved_cnn
python -m src.train --task signs --model augmented_cnn
```

Useful options:
```bash
python -m src.benchmark --epochs 40 --seeds 42 7 99
python -m src.benchmark --save-models
python -m src.train --task signs --model improved_cnn --epochs 80 --batch-size 64
```

## Model Architectures
### Smile
- `build_smile_baseline()`
  - `ZeroPadding2D -> Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D -> Flatten -> Dense(sigmoid)`
- `build_smile_improved_cnn()`
  - `Conv-BN-ReLU -> MaxPool -> Conv-BN-ReLU -> MaxPool -> Conv-BN-ReLU -> GlobalAveragePooling -> Dropout -> Dense(sigmoid)`
- `build_smile_augmented_cnn()`
  - Uses in-model augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomContrast`) before the improved backbone.

### SIGNS
- `build_signs_baseline()`
  - `Input -> Conv2D -> ReLU -> MaxPooling2D -> Conv2D -> ReLU -> MaxPooling2D -> Flatten -> Dense(softmax)`
- `build_signs_improved_cnn()`
  - Stacked conv blocks (32/64/128 filters) with BN/ReLU, pooling, dropout, global average pooling, and dense softmax head.
- `build_signs_augmented_cnn()`
  - Conservative in-model augmentation (small rotation/zoom/translation/contrast) before improved backbone.
- Optional `build_signs_mobilenetv2_transfer()`
  - Frozen ImageNet MobileNetV2 feature extractor with lightweight classification head.

## Benchmark Design
`python -m src.benchmark` compares:
- Smile: `baseline`, `improved_cnn`, `augmented_cnn`
- SIGNS: `baseline`, `improved_cnn`, `augmented_cnn` (plus optional transfer model)

For each run it stores:
- `task`, `model`, `parameters`
- `epochs_requested`, `epochs_trained`
- `best_validation_accuracy`, `best_validation_loss`
- `test_accuracy`, `test_loss`
- `training_time_seconds`
- `roc_auc` for smile when available

Training callbacks:
- `EarlyStopping(restore_best_weights=True)`
- `ReduceLROnPlateau`
- `CSVLogger` to `outputs/metrics/`
- optional `ModelCheckpoint` to `outputs/models/`

## Results
Latest local benchmark snapshot (seed `42`, generated from `outputs/metrics/benchmark_results.csv`):

### Smile task
| Model | Epochs trained | Best val accuracy | Test accuracy | Test loss | ROC-AUC | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 30 | 0.9667 | 0.9600 | 0.0819 | 0.9960 | 18.40 |
| `augmented_cnn` | 30 | 0.9000 | 0.8133 | 0.3667 | 0.9284 | 35.47 |
| `improved_cnn` | 12 | 0.5222 | 0.5400 | 0.7451 | 0.6124 | 12.66 |

### SIGNS task
| Model | Epochs trained | Best val accuracy | Test accuracy | Test loss | Training time (s) |
|---|---:|---:|---:|---:|---:|
| `augmented_cnn` | 80 | 0.9630 | 0.9333 | 0.1483 | 254.71 |
| `baseline` | 80 | 0.7593 | 0.7750 | 0.8070 | 15.54 |
| `improved_cnn` | 16 | 0.2284 | 0.2167 | 3.5316 | 52.28 |

### Quick discussion
- In this run, `augmented_cnn` is strongest on SIGNS (`0.9333` test accuracy), improving over the SIGNS baseline (`0.7750`).
- For smile classification, the baseline model performs best (`0.9600` test accuracy, `0.9960` ROC-AUC).
- `improved_cnn` underperforms in both tasks in this seed-42 run, indicating the current optimization/regularization settings are not yet stable for this architecture.
- These numbers are from one seed (`42`); use `--seeds` for multi-seed comparison before drawing final conclusions.

### Figures produced by the run
- Benchmark comparison: `outputs/figures/benchmark_test_accuracy.png`
- Smile curves/confusion matrices: `outputs/figures/smile_*_training_curves.png`, `outputs/figures/smile_*_confusion_matrix.png`
- SIGNS curves/confusion matrices: `outputs/figures/signs_*_training_curves.png`, `outputs/figures/signs_*_confusion_matrix.png`

## Output Artifacts
- Training curves: `outputs/figures/*_training_curves.png`
- Confusion matrices: `outputs/figures/*_confusion_matrix.png`
- Benchmark comparison chart: `outputs/figures/benchmark_test_accuracy.png`
- Detailed per-run metrics: `outputs/metrics/*_detailed_metrics.json`
- Per-run logs: `outputs/metrics/*_training_log.csv`
- Optional model checkpoints: `outputs/models/*.keras`

## Reproducibility Notes
- Global seeds are set for Python `random`, NumPy, and TensorFlow in `src/config.py`.
- TensorFlow op determinism is enabled when available.
- Minor metric variation can still occur across hardware, CUDA/cuDNN, and TensorFlow versions.

## License / Data Note
Datasets are not distributed in this repository. Ensure you have the right to use local dataset files placed under `datasets/`.
