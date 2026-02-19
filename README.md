# Fairness-Aware Income Prediction with Constraint Optimization

A machine learning system that predicts income levels (above or below $50K/year) using the UCI Adult Census dataset while incorporating fairness constraints to mitigate demographic bias. The project uses LightGBM as the base classifier with Optuna-driven hyperparameter optimization, where the optimization objective balances predictive accuracy against fairness violations measured by demographic parity and equalized odds.

## Methodology

This project implements a fairness-constrained optimization approach that integrates fairness considerations directly into the hyperparameter search process. Unlike post-processing fairness corrections, this method penalizes unfair models during training through a composite objective function that combines AUC-ROC with weighted fairness violations.

The key innovation is a two-tier penalty system:

1. **Soft Penalty** -- proportional to demographic parity and equalized odds violations, applied continuously during optimization.
2. **Hard Constraint Penalty** -- grows exponentially when unfairness exceeds a configurable tolerance threshold (default: 0.15), with a 10x multiplier that steers the optimizer away from severely biased configurations.

This formulation allows Optuna to explore the Pareto frontier between accuracy and fairness, automatically discovering hyperparameter configurations that achieve optimal tradeoffs. Custom fairness components include specialized loss functions and metrics that guide the gradient boosting process toward fairer decision boundaries while maintaining competitive predictive performance.

## Key Components

- **Data Pipeline**: UCI Adult dataset loading and preprocessing with scikit-learn pipelines (one-hot encoding, scaling, missing value handling)
- **Model**: LightGBM gradient boosting classifier with fairness-constrained Optuna hyperparameter optimization
- **Objective Function**: `AUC - fairness_penalty - constraint_penalty` (10x multiplier when violation exceeds 0.15)
- **Fairness Metrics**: Demographic parity ratio, equalized odds difference, equal opportunity difference
- **Custom Components**: `FairnessAwareCustomLoss` and `ConstraintViolationPenalty` in `src/.../models/components.py`

## Training Results

Results from training on the UCI Adult Census dataset with Optuna hyperparameter optimization (2 trials completed within a 1-hour timeout budget out of 100 requested).

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
| AUC-ROC | 0.9219 |
| Accuracy | 0.8653 |
| Precision | 0.7732 |
| Recall | 0.6463 |
| F1-Score | 0.7041 |
| Average Precision | 0.8158 |
| Brier Score | 0.0941 |

### Fairness Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Demographic Parity Ratio | 0.3112 | 0.85 | Not achieved |
| Demographic Parity Difference | 0.1847 | -- | -- |
| Equalized Odds Difference | 0.0858 | 0.08 | Near target (gap: 0.0058) |
| Equal Opportunity Difference | 0.1013 | -- | -- |
| Composite Score | 0.7673 | -- | -- |

### Optimization Summary

| Parameter | Value |
|-----------|-------|
| Trials Completed | 2 / 100 (timeout at 1 hour) |
| Best Trial | Trial 0 |
| Best Trial Composite Score | -5.1941 |
| Best Trial AUC | 0.9274 |
| Best Trial Fairness Ratio | 0.3390 |

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_leaves | 155 |
| learning_rate | 0.2250 |
| feature_fraction | 0.8297 |
| bagging_fraction | 0.7993 |
| max_depth | 4 |
| min_child_samples | 84 |
| lambda_l1 | 7.4775 |
| lambda_l2 | 3.2820 |
| n_estimators | 710 |

## Analysis

The model achieves strong predictive performance with an AUC-ROC of 0.9219 and accuracy of 0.8653 on the held-out test set. However, the demographic parity ratio of 0.3112 falls substantially short of the 0.85 target, revealing the inherent difficulty of achieving demographic fairness on the UCI Adult dataset where significant base rate disparities exist across protected groups.

The equalized odds difference of 0.0858 comes close to the 0.08 target (gap of only 0.0058), indicating that the model's true positive and false positive rates are relatively balanced across groups even though the overall selection rates differ.

Both completed Optuna trials produced models that exceeded the unfairness tolerance threshold of 0.15, which triggers the hard constraint penalty. This suggests that the accuracy-fairness tradeoff on this dataset requires either stronger fairness weights, alternative debiasing strategies, or removal of proxy features that encode the protected attribute.

### Future Directions

- Increase the fairness constraint weight beyond 1.0 to more aggressively penalize unfair configurations
- Apply post-processing calibration per protected group to equalize selection rates
- Experiment with in-processing adversarial debiasing techniques
- Identify and remove proxy features that encode the protected attribute (sex)
- Extend the optimization budget to allow more Optuna trials to converge

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

All training settings are managed through `configs/default.yaml`, including data splits, preprocessing options, LightGBM parameters, fairness constraint weights, and Optuna optimization settings.

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
│   ├── results.json                 # Full experiment results
│   └── results_summary.json         # Experiment results summary
├── tests/                           # Unit tests
├── notebooks/                       # Jupyter notebooks
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Dependencies
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.
