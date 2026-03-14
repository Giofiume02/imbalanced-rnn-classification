# Pain Level Classification from Time-Series Data with Recurrent Neural Networks

This repository presents a deep learning project for **multiclass time-series classification**, developed for the **AN2DL course**.  
The goal is to predict the **pain level** associated with each temporal sequence using recurrent neural networks.

The project focuses on building an end-to-end pipeline for:

- sequential data preprocessing
- feature selection and feature engineering
- class imbalance mitigation
- recurrent deep learning model design
- hyperparameter tuning
- ensemble learning for robust prediction

The final solution combines engineered temporal features, recurrent architectures, convolutional sequence encoding, and attention mechanisms to improve classification performance on a strongly imbalanced dataset.

---

## Project Overview

The task consists of classifying each sequence into one of three pain-related categories:

- **No Pain**
- **Low Pain**
- **High Pain**

Each sample is a multivariate time series composed of **160 time steps**, with features describing pain-related measurements, body-part counters, and joint-angle signals. The challenge required not only building an accurate classifier, but also addressing several practical machine learning issues, including:

- severe class imbalance
- redundant and low-information features
- temporal consistency of the split
- risk of overfitting
- the need to capture both local and long-range temporal dependencies

The main evaluation metric used throughout the project is the **macro F1-score**, chosen because of the unbalanced class distribution.

---

## Dataset

The original dataset used in the project includes:

- `pirate_pain_train.csv`: **105,760 rows × 40 columns**
- `pirate_pain_train_labels.csv`: **661 labeled sequences**
- `pirate_pain_test.csv`: **211,840 rows × 40 columns**

### Data Structure

Each sequence is identified by a `sample_index` and contains **160 time steps**.  
The feature space includes:

- sample index and time
- four pain-related measurements
- three categorical body-part counters
- thirty joint-angle measurements

### Class Distribution

The dataset is strongly imbalanced, with approximately:

- **77.13%** No Pain
- **14.22%** Low Pain
- **8.47%** High Pain

This imbalance had a major impact on both model training and evaluation strategy.

---

## Main Challenges

The report highlights several key challenges that shaped the modeling strategy:

- identifying correlated or non-informative variables
- detecting outliers and anomalous motion patterns
- preventing data leakage through a correct **sample-wise split**
- handling severe class imbalance
- designing architectures able to model long-term dependencies
- preventing overfitting through regularization and robust validation
- tuning sensitive hyperparameters such as learning rate, hidden size, stride, and window size.

---

## Methodology

The project followed an incremental modeling approach: starting from a simple recurrent baseline and progressively improving the pipeline through better preprocessing, richer temporal features, architectural refinements, and ensemble methods.

### 1. Data Preprocessing

The preprocessing pipeline included:

- one-hot encoding for categorical variables
- min-max normalization for numerical features
- removal of low-variance features
- removal of highly correlated features

This step was important to reduce noise and simplify the representation before sequence modeling.

### 2. Feature Engineering

To enrich the temporal information available to the model, several derived features were added:

- **joint-angle velocity**
- **joint-angle acceleration**
- **rolling mean**
- **rolling standard deviation** 

These engineered features helped capture motion dynamics more effectively than raw inputs alone.

### 3. Class Imbalance Handling

Different strategies were tested to mitigate class imbalance:

- Gaussian-noise oversampling
- SMOTE
- focal loss with inverse-frequency weights

The final preferred solution was **focal loss with inverse-frequency weighting**, since oversampling-based methods tended to increase overfitting or generate unrealistic synthetic temporal patterns.

### 4. Model Architectures

Several recurrent architectures were explored, including:

- vanilla RNN variants
- **LSTM**
- **GRU**
- bidirectional and non-bidirectional configurations

The **bidirectional LSTM and GRU models** often performed best during experimentation, although some non-bidirectional models became highly competitive after feature engineering.

### 5. Architectural Improvements

Two major modeling improvements were introduced:

- a **Conv1D layer** before the recurrent block, to capture local temporal patterns
- an **attention mechanism**, to assign different importance weights to different time steps and produce a context-aware representation for classification

### 6. Optimization and Regularization

The training pipeline also included several techniques to improve robustness:

- **label smoothing**
- **AdamW** optimizer
- **learning rate scheduler**
- **gradient clipping**
- grid search for model and training hyperparameters

### 7. Ensemble Strategy and Results

In the final part I used an **ensemble of the three best-performing models**, which produced the most stable and accurate final predictions. 

## Impact of the Main Techniques

Now I summarizes the incremental improvement in F1-score as follows:

| Configuration | F1 Score |
|---|---:|
| Baseline UniLSTM | 0.9368 |
| + Class balancing (SMOTE + Focal Loss) | 0.9401 |
| + Feature engineering | 0.9470 |
| + Conv1D layer | 0.9479 |
| + Attention | 0.9484 |

### Validation Metrics by Class

On a representative validation split (20% of training data), the reported per-class performance is:

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| No Pain | 0.9935 | 1.0000 | 0.9968 | 154 |
| Low Pain | 0.9655 | 1.0000 | 0.9825 | 28 |
| High Pain | 1.0000 | 0.8824 | 0.9375 | 17 |
| Macro Avg | 0.9864 | 0.9608 | 0.9722 | 199 |

These results show very strong overall performance, while also confirming that the **High Pain** class remains the most difficult one to identify consistently.

---

## Aggregated Validation Evaluation

The figures below summarize out-of-fold validation performance aggregated across cross-validation folds.

![Aggregated confusion matrix](images/confusion_matrix_cv.png)

![Aggregated ROC-AUC curves](images/roc_auc_cv.png)

---

## Best Model

The best single-model configuration was a:

**CNN + UniLSTM + Attention** architecture, trained with:

- AdamW
- learning rate scheduling
- gradient clipping
- label smoothing

This configuration offered the best trade-off between local pattern extraction, sequence memory, and timestep relevance weighting.

---

## Key Findings

Below are the most significant technical takeaways that drove the model's performance:

- feature cleaning and temporal feature engineering substantially improved minority-class performance
- focal loss with inverse-frequency weighting was more effective than oversampling-based methods
- Conv1D layers helped extract short-range motion patterns before recurrent modeling
- attention improved the model’s ability to focus on the most informative time steps
- ensemble learning increased both performance stability and final macro F1-score

---

## Unexpected Outcomes

Some experimental outcomes were especially informative:

- Gaussian-noise oversampling increased overfitting
- SMOTE generated unrealistic temporal sequences
- some non-bidirectional RNNs performed surprisingly well once stronger feature engineering was introduced

These findings reinforce an important practical lesson: methods that work well on tabular data do not always transfer well to sequential data.

---

## Repository Structure

This project is structured as follows:

```bash
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── rnn_pain_classification.ipynb
├── src/
│   ├── models.py                  # CNN + RNN + Attention definitions
│   ├── preprocessing.py           # Data cleaning and scaling
│   ├── feature_engineering.py     # Temporal feature generation
│   ├── train.py                   # Training loops with OneCycleLR scheduling
│   ├── evaluation.py              # Confusion matrices & F1-Score tracking
│   ├── cross_validation.py        # Stratified K-Fold logic at sample level
│   ├── hyperparameter_tuning.py   # Grid search implementation for RNNs
│   └── ensemble.py                # Weighted-voting utilities
├── images/
│   ├── class_distribution.png
│   ├── training_curves.png
│   └── confusion_matrix.png
└── results/
    └── metrics_summary.csv
```
---

## Data Availability and Notes

The original challenge dataset is not included in this repository.

This repository is intended to showcase the project methodology, modeling pipeline, feature engineering strategy, and experimental results.  
Due to challenge-specific data restrictions and to keep the repository lightweight, only the codebase, documentation, and selected outputs are shared.
