# Fairness-Aware Income Prediction with Constraint Optimization

A machine learning system that predicts income levels (above or below $50K/year) using the UCI Adult Census dataset while incorporating fairness constraints to mitigate demographic bias. The project uses LightGBM as the base classifier with Optuna-driven hyperparameter optimization, where the optimization objective balances predictive accuracy against fairness violations measured by demographic parity and equalized odds.

## Architecture Overview

```
UCI Adult Dataset
       |
       v
  Data Loading & Cleaning (loader.py)
       |
       v
  Preprocessing Pipeline (preprocessing.py)
  - SimpleImputer (median for numerical, mode for categorical)
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
  - Protected attribute (sex) preserved separately
       |
       v
  Fairness-Constrained Optimization (model.py)
  - Optuna study with 50 trials
  - Each trial trains a LightGBM classifier
  - Objective = AUC - fairness_penalty - constraint_penalty
  - Fairness penalty: weighted sum of demographic parity violation + equalized odds violation
  - Constraint penalty: 10x multiplier when unfairness exceeds tolerance threshold (0.15)
       |
       v
  Evaluation (metrics.py)
  - Standard metrics: accuracy, AUC-ROC, precision, recall, F1
  - Fairness metrics: demographic parity ratio, equalized odds difference, equal opportunity difference
  - Calibration analysis per protected group
  - Composite score: 50% accuracy + 50% fairness
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Data Loader | `src/.../data/loader.py` | Downloads and cleans UCI Adult dataset |
| Preprocessor | `src/.../data/preprocessing.py` | Feature engineering with sklearn pipelines |
| Model | `src/.../models/model.py` | LightGBM with fairness-constrained Optuna optimization |
| Trainer | `src/.../training/trainer.py` | End-to-end training pipeline orchestration |
| Metrics | `src/.../evaluation/metrics.py` | Performance and fairness evaluation |
| Config | `src/.../utils/config.py` | YAML configuration management |

## Training Results

Results from training on the UCI Adult Census dataset with 50 Optuna optimization trials:

### Dataset Statistics

| Split | Samples |
|-------|---------|
| Training | 27,132 |
| Validation | 9,045 |
| Test | 9,045 |
| Features | 102 (after one-hot encoding) |

### Test Set Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9220 |
| Accuracy | 0.8636 |
| Precision | 0.7707 |
| Recall | 0.6401 |
| F1-Score | 0.6993 |
| Average Precision | 0.8158 |
| Brier Score | 0.0942 |

### Fairness Metrics

| Metric | Value | Target | Achieved |
|--------|-------|--------|----------|
| Demographic Parity Ratio | 0.3341 | 0.85 | No |
| Demographic Parity Difference | 0.1757 | -- | -- |
| Equalized Odds Difference | 0.0725 | 0.08 | No (gap: 0.0075) |
| Equal Opportunity Difference | 0.0805 | -- | -- |
| Composite Score | 0.7764 | -- | -- |

### Optimization Summary

| Metric | Value |
|--------|-------|
| Total Trials | 50 |
| Best Trial Composite Score | -5.0225 |
| Best Trial AUC | 0.9265 |
| Best Trial Fairness Ratio | 0.3515 |

## Analysis of Results

The model achieves strong predictive performance with an AUC-ROC of 0.9220 and accuracy of 0.8636, which is competitive with standard approaches on the Adult dataset.

However, the fairness results reveal a significant and honest limitation: the demographic parity ratio of 0.3341 falls well short of the 0.85 target. This means the model's positive prediction rate for the disadvantaged group (Female) is only about one-third of the rate for the advantaged group (Male). The equalized odds difference of 0.0725 is much closer to its target of 0.08 (gap of only 0.0075), indicating that conditional on the true label, the model's error rates are relatively similar across groups.

The large gap in demographic parity is partially explained by the base rate difference in the underlying data: a substantially higher proportion of males earn above $50K in the training data, and the model largely reproduces this disparity. The constrained optimization formulation penalizes unfairness in the objective function, but the penalty was not strong enough to overcome the learned correlation between the protected attribute and the target.

The negative composite optimization scores (best: -5.02) indicate that the fairness constraint penalty dominated the objective. All trials exceeded the `max_unfairness_tolerance` threshold of 0.15, triggering the 10x constraint penalty, which means the optimizer could not find a region of hyperparameter space where fairness constraints were satisfied while maintaining reasonable accuracy.

Possible improvements include:
- Increasing the fairness constraint weight beyond 1.0
- Using post-processing calibration (e.g., threshold adjustment per group)
- Applying in-processing techniques such as adversarial debiasing
- Removing or decorrelating proxy features that encode protected attribute information
- Reducing the unfairness tolerance threshold more aggressively during optimization

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install

```bash
cd fairness-aware-income-prediction-with-constraint-optimization
pip install -e .
```

## Usage

### Training

```bash
# Create required directories
mkdir -p logs checkpoints

# Run with default settings (100 trials, 1 hour timeout)
python scripts/train.py --config configs/default.yaml

# Run with custom settings and no MLflow
python scripts/train.py --config configs/default.yaml \
    --n-trials 50 \
    --timeout 600 \
    --disable-mlflow

# Run with cross-validation
python scripts/train.py --config configs/default.yaml \
    --cross-validation \
    --disable-mlflow
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model-path checkpoints/fairness_aware_model.pkl \
    --config configs/default.yaml \
    --generate-plots \
    --save-predictions
```

### Configuration

All settings are controlled via `configs/default.yaml`:

- **data**: dataset name, split ratios, protected attribute, target column
- **preprocessing**: missing value handling, encoding, scaling
- **model**: LightGBM base parameters, fairness constraint weights and tolerance
- **optimization**: Optuna settings (trials, timeout, search space)
- **training**: cross-validation folds, early stopping, MLflow toggle
- **evaluation**: which metrics to compute, target metric thresholds

## Project Structure

```
fairness-aware-income-prediction-with-constraint-optimization/
├── configs/
│   └── default.yaml                 # Training configuration
├── scripts/
│   ├── train.py                     # Training entry point
│   └── evaluate.py                  # Evaluation entry point
├── src/
│   └── fairness_aware_income_.../
│       ├── data/
│       │   ├── loader.py            # UCI Adult dataset downloader
│       │   └── preprocessing.py     # Feature engineering pipeline
│       ├── models/
│       │   └── model.py             # Fairness-constrained LightGBM
│       ├── training/
│       │   └── trainer.py           # Training orchestration
│       ├── evaluation/
│       │   └── metrics.py           # Performance & fairness metrics
│       └── utils/
│           └── config.py            # YAML config management
├── tests/                           # Unit tests
├── notebooks/                       # Jupyter notebooks
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Dependencies
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.
