# IMU Activity Classification

Deep learning model that classifies postures and movement types from raw 3-sensor IMU data.

## Overview

Two classification tasks trained separately:
- **Experiment 1 — Posture (7 classes):** Supine, Standing, Sitting, Side Right, Side Left, Prone, Crawl
- **Experiment 2 — Movement (5 classes):** Locomotion Periodic, Locomotion Aperiodic, Still, Wobbling, Transition

## Data

13 subjects, ~4 hours of labeled recordings (`annotated_IMU_data/*.mat`). Each subject wears 3 IMU sensors providing accelerometer + gyroscope readings (18 channels total, 26 Hz). Leave-one-subject-out (LOSO) cross-validation is used for evaluation.

## Model

A two-stage architecture (~489K parameters):

1. **CNN Encoder** — processes accelerometer and gyroscope streams through 3D/2D convolutions, fuses all sensors into a per-frame feature vector
2. **WaveNet** — dilated 1D CNN with gated residual connections captures temporal context across frames

## Results (Experiment 1 — Posture)

| Metric | Value |
|---|---|
| Mean LOSO F1 | 0.72 |
| Overall accuracy | 86% |
| Best class | Standing (F1 0.95) |
| Hardest class | CrawlPosture (F1 0.47, rare class) |

Confusion matrix and per-fold/per-class CSVs are saved in `results1/`.

## Usage

Configure experiment and hyperparameters in `conf_models.py`, then:

```bash
python train_model.py
```

Set `experiment_number = 1` for posture, `2` for movement.

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). Install and run:

```bash
uv sync
uv run python train_model.py
```

Key dependencies: PyTorch, scikit-learn, scipy, pandas, matplotlib.
