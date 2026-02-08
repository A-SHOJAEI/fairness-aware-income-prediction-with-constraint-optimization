"""Test fixtures and configuration for pytest."""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator, Tuple

from src.fairness_aware_income_prediction_with_constraint_optimization.utils.config import Config
from src.fairness_aware_income_prediction_with_constraint_optimization.data.loader import AdultIncomeLoader
from src.fairness_aware_income_prediction_with_constraint_optimization.data.preprocessing import FairnessAwarePreprocessor


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration for testing.

    Returns:
        Config object with test settings.
    """
    config_data = {
        'data': {
            'dataset_name': 'adult',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'protected_attribute': 'sex',
            'target_column': 'income'
        },
        'model': {
            'algorithm': 'lightgbm',
            'fairness_constraint': {
                'enabled': True,
                'demographic_parity_weight': 1.0,
                'equalized_odds_weight': 0.5,
                'max_unfairness_tolerance': 0.15
            },
            'base_params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 10,  # Smaller for faster testing
                'learning_rate': 0.1,
                'verbose': -1,
                'random_state': 42
            }
        },
        'optimization': {
            'n_trials': 5,  # Small number for testing
            'timeout': 30
        },
        'training': {
            'cv_folds': 3,  # Smaller for testing
            'use_mlflow': False  # Disable for testing
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = f.name

    config = Config(config_path)

    # Clean up
    Path(config_path).unlink()

    return config


@pytest.fixture
def sample_adult_data() -> pd.DataFrame:
    """Create sample Adult dataset for testing.

    Returns:
        DataFrame with synthetic Adult dataset structure.
    """
    np.random.seed(42)
    n_samples = 200

    # Create synthetic data matching Adult dataset structure
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Government'], n_samples),
        'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters', 'Doctorate'], n_samples),
        'education_num': np.random.randint(1, 16, n_samples),
        'marital_status': np.random.choice(['Married', 'Never-married', 'Divorced'], n_samples),
        'occupation': np.random.choice(['Tech-support', 'Sales', 'Exec-managerial'], n_samples),
        'relationship': np.random.choice(['Husband', 'Wife', 'Own-child', 'Unmarried'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
        'sex': np.random.choice([0, 1], n_samples),  # Already encoded: 0=Female, 1=Male
        'capital_gain': np.random.randint(0, 10000, n_samples),
        'capital_loss': np.random.randint(0, 1000, n_samples),
        'hours_per_week': np.random.randint(20, 80, n_samples),
        'native_country': np.random.choice(['United-States', 'Canada', 'Mexico'], n_samples),
        'income': np.random.choice([0, 1], n_samples)  # Target variable
    }

    # Make income slightly correlated with age and education
    for i in range(n_samples):
        if data['age'][i] > 40 and data['education_num'][i] > 12:
            data['income'][i] = 1 if np.random.random() > 0.3 else 0
        else:
            data['income'][i] = 1 if np.random.random() > 0.7 else 0

    return pd.DataFrame(data)


@pytest.fixture
def sample_train_val_test_data(sample_adult_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits from sample data.

    Args:
        sample_adult_data: Sample Adult dataset.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test
    train_val_df, test_df = train_test_split(
        sample_adult_data, test_size=0.3, random_state=42,
        stratify=sample_adult_data['income']
    )

    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.3, random_state=42,
        stratify=train_val_df['income']
    )

    return train_df, val_df, test_df


@pytest.fixture
def sample_preprocessor() -> FairnessAwarePreprocessor:
    """Create a sample preprocessor for testing.

    Returns:
        FairnessAwarePreprocessor instance.
    """
    return FairnessAwarePreprocessor(
        protected_attribute='sex',
        target_column='income',
        handle_missing=True,
        encode_categorical=True,
        scale_numerical=True
    )


@pytest.fixture
def fitted_preprocessor(sample_preprocessor: FairnessAwarePreprocessor,
                       sample_train_val_test_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> FairnessAwarePreprocessor:
    """Create a fitted preprocessor for testing.

    Args:
        sample_preprocessor: Unfitted preprocessor.
        sample_train_val_test_data: Train/validation/test data.

    Returns:
        Fitted FairnessAwarePreprocessor instance.
    """
    train_df, _, _ = sample_train_val_test_data
    sample_preprocessor.fit(train_df)
    return sample_preprocessor


@pytest.fixture
def sample_processed_data(fitted_preprocessor: FairnessAwarePreprocessor,
                         sample_train_val_test_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create processed data for testing.

    Args:
        fitted_preprocessor: Fitted preprocessor.
        sample_train_val_test_data: Raw train/validation/test data.

    Returns:
        Tuple of (X_processed, y_processed, protected_processed).
    """
    train_df, _, _ = sample_train_val_test_data
    X_processed, y_processed, protected_processed = fitted_preprocessor.transform(train_df)
    return X_processed, y_processed, protected_processed


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data_loader(sample_adult_data: pd.DataFrame) -> AdultIncomeLoader:
    """Create a mock data loader with sample data.

    Args:
        sample_adult_data: Sample dataset.

    Returns:
        AdultIncomeLoader with mocked load_data method.
    """
    loader = AdultIncomeLoader()

    # Override the load_data method to return sample data
    original_load_data = loader.load_data
    loader.load_data = lambda **kwargs: sample_adult_data

    return loader


@pytest.fixture
def sample_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sample predictions for testing metrics.

    Returns:
        Tuple of (y_true, y_pred, y_prob, protected_attributes).
    """
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.choice([0, 1], n_samples)

    # Create somewhat realistic predictions
    y_prob = np.random.beta(2, 2, n_samples)
    y_pred = (y_prob > 0.5).astype(int)

    protected_attributes = np.random.choice([0, 1], n_samples)

    return y_true, y_pred, y_prob, protected_attributes


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce verbosity during tests


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during testing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield