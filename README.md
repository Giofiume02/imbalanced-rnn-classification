# AN2DL Challenge — Time-Series Pain Classification (Pirate Pain)

End-to-end pipeline for **multiclass time-series classification** from joint-angle and pain-related signals.
The goal is to predict the pain level (**no_pain / low_pain / high_pain**) for each `sample_index`.
Evaluation metric: **Macro F1**.

This work was developed for the **AN2DL course challenge**; a detailed write-up is available in `reports/AN2DL_Challenges_Report.pdf`.

## Dataset

Expected files (NOT included in this repository):
- `pirate_pain_train.csv` — **(105760, 40)**
- `pirate_pain_train_labels.csv` — **(661, 2)**
- `pirate_pain_test.csv` — **(211840, 40)**

Each `sample_index` contains **160 timesteps**. Labels are provided at sample level (one label per `sample_index`).

### Class distribution (strongly imbalanced)
- no_pain: **77.13%**
- low_pain: **14.22%**
- high_pain: **8.47%**

## Key challenges
- Redundant / non-informative signals (e.g., constant joint features)
- Outliers (e.g., missing limbs / abnormal body-part counters)
- Severe class imbalance
- Avoiding data leakage: splits must be performed **by `sample_index`** (never by row/timestep)

## Method

### Preprocessing & feature engineering
- One-hot encoding for categorical variables and min–max normalization for numerical features.
- Removal of low-variance features and correlated features (threshold-based).
- Temporal features:
  - joint **velocity** and **acceleration**
  - rolling mean / std (window size 10)

### Handling imbalance
Oversampling (Gaussian noise / SMOTE) was tested but increased overfitting risk and could generate unrealistic temporal patterns.
Final approach: **Focal Loss with inverse-frequency weights** + **label smoothing**.

### Model
**Conv1D → (Uni/Bi) RNN (LSTM/GRU) → Attention pooling → MLP classifier**

Training details:
- AdamW optimizer
- Learning-rate scheduling
- Gradient clipping
- Label smoothing

### Ensembling
Final predictions are obtained via an **ensemble** of the best-performing models to improve stability and overall Macro F1.

## Experiments & results (reported)

> Results below are taken from the project report. The validation protocol is a **single 70/30 split by `sample_index`**.

### Impact of main techniques (Macro F1)
| Configuration | Macro F1 |
|---|---:|
| Baseline UniLSTM | 0.9368 |
| + Class balancing (SMOTE + Focal Loss) | 0.9401 |
| + Feature engineering | 0.9470 |
| + Conv1D layer | 0.9479 |
| + Attention | 0.9484 |

### Per-class metrics (Validation split)
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No Pain | 0.9935 | 1.0000 | 0.9968 |
| Low Pain | 0.9655 | 1.0000 | 0.9825 |
| High Pain | 1.0000 | 0.8824 | 0.9375 |
| **Macro Avg** | 0.9864 | 0.9608 | **0.9722** |

## How to run
1. Create an environment (Python 3.10+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
