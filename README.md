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
| Mean LOSO F1 | 0.7228 |
| Overall accuracy | 85.5% |
| Mean precision | 0.7266 |
| Mean recall | 0.7472 |
| Best class | Standing (F1 0.9468) |
| Hardest class | CrawlPosture (F1 0.4731, rare class) |

Per-class F1 scores:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Standing | 0.9160 | 0.9798 | 0.9468 |
| Prone | 0.9405 | 0.8547 | 0.8956 |
| Sitting | 0.8512 | 0.7123 | 0.7756 |
| SideRight | 0.6076 | 0.8889 | 0.7218 |
| Supine | 0.6637 | 0.6608 | 0.6623 |
| SideLeft | 0.6869 | 0.5927 | 0.6364 |
| CrawlPosture | 0.4204 | 0.5410 | 0.4731 |

Confusion matrix and per-fold/per-class CSVs are saved in `results1/`.

## Results (Experiment 2 — Movement)

| Metric | Value |
|---|---|
| Mean LOSO F1 | 0.6089 |
| Overall accuracy | 74.8% |
| Mean precision | 0.7133 |
| Mean recall | 0.7002 |
| Best class | Locomotion_Periodic (F1 0.8881) |
| Hardest class | Locomotion_Aperiodic (F1 0.5075) |

Per-class F1 scores:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Locomotion_Periodic | 0.8632 | 0.9145 | 0.8881 |
| NonLocomotion_Still | 0.8514 | 0.8019 | 0.8259 |
| Transition | 0.6876 | 0.6301 | 0.6576 |
| NonLocomotion_Wobbling | 0.5951 | 0.6968 | 0.6419 |
| Locomotion_Aperiodic | 0.5693 | 0.4578 | 0.5075 |

Confusion matrix and per-fold/per-class CSVs are saved in `results2/`.

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
