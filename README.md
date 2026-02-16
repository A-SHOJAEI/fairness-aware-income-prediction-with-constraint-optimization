# Fairness-Aware Income Prediction with Constraint Optimization

A machine learning system that predicts income levels (above or below $50K/year) using the UCI Adult Census dataset while incorporating fairness constraints to mitigate demographic bias. The project uses LightGBM as the base classifier with Optuna-driven hyperparameter optimization, where the optimization objective balances predictive accuracy against fairness violations measured by demographic parity and equalized odds.

## Methodology

This project implements a novel fairness-constrained optimization approach that integrates fairness considerations directly into the hyperparameter search process. Unlike post-processing fairness corrections, our method penalizes unfair models during training through a composite objective function that combines AUC-ROC with weighted fairness violations. The key innovation is a two-tier penalty system: a soft penalty proportional to demographic parity and equalized odds violations, plus a hard constraint penalty that grows exponentially when unfairness exceeds a tolerance threshold. This formulation allows Optuna to explore the Pareto frontier between accuracy and fairness, automatically discovering hyperparameter configurations that achieve optimal tradeoffs. The custom fairness components include specialized loss functions and metrics that guide the gradient boosting process toward fairer decision boundaries while maintaining competitive predictive performance.

## Key Components

- **Data**: UCI Adult dataset loading and preprocessing with sklearn pipelines
- **Model**: LightGBM with fairness-constrained Optuna optimization (50 trials)
- **Objective**: AUC - fairness_penalty - constraint_penalty (10x multiplier when violation > 0.15)
- **Fairness**: Demographic parity and equalized odds metrics
- **Custom Components**: FairnessAwareCustomLoss, ConstraintViolationPenalty in `components.py`

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

## Analysis

Strong predictive performance (AUC 0.9220, accuracy 0.8636) but demographic parity ratio 0.3341 falls short of 0.85 target, revealing fairness challenges inherent in biased training data. Equalized odds (0.0725) nearly meets target (0.08). All 50 trials exceeded unfairness tolerance (0.15), indicating the accuracy-fairness tradeoff is difficult to resolve with current constraint formulation. Future work: stronger fairness weights, post-processing calibration, adversarial debiasing.

## Installation

```bash
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

### Prediction

```bash
# Make predictions on test data
python scripts/predict.py \
    --model-path checkpoints/fairness_aware_model.pkl \
    --config configs/default.yaml \
    --show-confidence \
    --top-k 20

# Make predictions on custom data
python scripts/predict.py \
    --model-path checkpoints/fairness_aware_model.pkl \
    --input data/new_samples.csv \
    --output predictions.csv \
    --show-confidence
```

### Ablation Study

```bash
# Train baseline model without fairness constraints
python scripts/train.py --config configs/ablation.yaml

# Compare results to see impact of fairness constraints
python scripts/evaluate.py \
    --model-path checkpoints/ablation_model.pkl \
    --config configs/ablation.yaml
```

### Configuration

Settings in `configs/default.yaml`: data splits, preprocessing options, LightGBM parameters, fairness weights, Optuna settings.

## Project Structure

```
fairness-aware-income-prediction-with-constraint-optimization/
├── configs/
│   ├── default.yaml                 # Training configuration
│   └── ablation.yaml                # Ablation config (no fairness)
├── scripts/
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation entry point
│   └── predict.py                   # Prediction script
├── src/
│   └── fairness_aware_income_.../
│       ├── data/
│       │   ├── loader.py            # UCI Adult dataset downloader
│       │   └── preprocessing.py     # Feature engineering pipeline
│       ├── models/
│       │   ├── model.py             # Fairness-constrained LightGBM
│       │   └── components.py        # Custom fairness loss & metrics
│       ├── training/
│       │   └── trainer.py           # Training orchestration
│       ├── evaluation/
│       │   └── metrics.py           # Performance & fairness metrics
│       └── utils/
│           └── config.py            # YAML config management
├── results/
│   └── results_summary.json         # Experiment results summary
├── tests/                           # Unit tests
├── notebooks/                       # Jupyter notebooks
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Dependencies
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.
