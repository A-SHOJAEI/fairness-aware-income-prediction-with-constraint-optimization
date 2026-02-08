"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.fairness_aware_income_prediction_with_constraint_optimization.data.loader import AdultIncomeLoader
from src.fairness_aware_income_prediction_with_constraint_optimization.data.preprocessing import FairnessAwarePreprocessor


class TestAdultIncomeLoader:
    """Test cases for AdultIncomeLoader class."""

    def test_init(self, temp_directory):
        """Test loader initialization."""
        loader = AdultIncomeLoader(data_dir=temp_directory)

        assert loader.data_dir == temp_directory
        assert loader.column_names == AdultIncomeLoader.COLUMN_NAMES
        assert loader.url_train == AdultIncomeLoader.URL_TRAIN
        assert loader.url_test == AdultIncomeLoader.URL_TEST

    def test_init_default_data_dir(self):
        """Test loader initialization with default data directory."""
        loader = AdultIncomeLoader()

        assert loader.data_dir == Path("data")
        assert loader.data_dir.exists()  # Should create the directory

    def test_clean_data(self, sample_adult_data):
        """Test data cleaning functionality."""
        loader = AdultIncomeLoader()

        # Add some missing values
        test_data = sample_adult_data.copy()
        test_data.loc[:5, 'workclass'] = np.nan
        test_data.loc[:3, 'age'] = np.nan

        # Add fnlwgt column that should be removed
        test_data['fnlwgt'] = np.random.randint(10000, 100000, len(test_data))

        cleaned_data = loader._clean_data(test_data)

        # Check that missing values are removed
        assert cleaned_data.isnull().sum().sum() == 0

        # Check that fnlwgt is removed
        assert 'fnlwgt' not in cleaned_data.columns

        # Check that target is properly encoded
        assert cleaned_data['income'].dtype == int
        assert set(cleaned_data['income'].unique()).issubset({0, 1})

        # Check that sex is properly encoded
        assert cleaned_data['sex'].dtype in [int, np.int64]
        assert set(cleaned_data['sex'].unique()).issubset({0, 1})

    def test_create_train_val_test_split(self, sample_adult_data):
        """Test train/validation/test split creation."""
        loader = AdultIncomeLoader()

        train_df, val_df, test_df = loader.create_train_val_test_split(
            sample_adult_data, test_size=0.2, val_size=0.2, random_state=42
        )

        total_samples = len(sample_adult_data)

        # Check sizes
        assert len(test_df) == pytest.approx(total_samples * 0.2, abs=2)
        assert len(val_df) == pytest.approx(total_samples * 0.2 * 0.8, abs=2)
        assert len(train_df) == total_samples - len(test_df) - len(val_df)

        # Check that all dataframes have same columns
        assert list(train_df.columns) == list(val_df.columns) == list(test_df.columns)

        # Check that indices don't overlap
        all_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
        assert len(all_indices) == len(train_df) + len(val_df) + len(test_df)

    def test_get_feature_info(self, sample_adult_data):
        """Test feature information extraction."""
        loader = AdultIncomeLoader()

        info = loader.get_feature_info(sample_adult_data)

        assert 'total_samples' in info
        assert 'total_features' in info
        assert 'numerical_features' in info
        assert 'categorical_features' in info
        assert 'target_column' in info
        assert 'protected_attribute' in info
        assert 'positive_class_ratio' in info

        # Check values
        assert info['total_samples'] == len(sample_adult_data)
        assert info['target_column'] == 'income'
        assert info['protected_attribute'] == 'sex'
        assert 0 <= info['positive_class_ratio'] <= 1

    @patch('urllib.request.urlretrieve')
    def test_download_file(self, mock_urlretrieve, temp_directory):
        """Test file downloading functionality."""
        loader = AdultIncomeLoader(data_dir=temp_directory)

        test_file = temp_directory / "test_file.csv"
        test_url = "http://example.com/test.csv"

        loader._download_file(test_url, test_file)

        mock_urlretrieve.assert_called_once_with(test_url, test_file)

    @patch('urllib.request.urlretrieve')
    def test_download_file_existing(self, mock_urlretrieve, temp_directory):
        """Test that existing files are not re-downloaded."""
        loader = AdultIncomeLoader(data_dir=temp_directory)

        test_file = temp_directory / "existing_file.csv"
        test_file.touch()  # Create the file

        test_url = "http://example.com/test.csv"

        loader._download_file(test_url, test_file)

        # Should not call urlretrieve for existing file
        mock_urlretrieve.assert_not_called()


class TestFairnessAwarePreprocessor:
    """Test cases for FairnessAwarePreprocessor class."""

    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = FairnessAwarePreprocessor(
            protected_attribute='sex',
            target_column='income',
            handle_missing=True,
            encode_categorical=True,
            scale_numerical=True
        )

        assert preprocessor.protected_attribute == 'sex'
        assert preprocessor.target_column == 'income'
        assert preprocessor.handle_missing is True
        assert preprocessor.encode_categorical is True
        assert preprocessor.scale_numerical is True

        assert preprocessor.numerical_features is None
        assert preprocessor.categorical_features is None
        assert preprocessor.preprocessor is None
        assert preprocessor.feature_names is None

    def test_identify_feature_types(self, sample_preprocessor, sample_adult_data):
        """Test feature type identification."""
        sample_preprocessor._identify_feature_types(sample_adult_data)

        assert sample_preprocessor.numerical_features is not None
        assert sample_preprocessor.categorical_features is not None

        # Check that target and protected attribute are handled correctly
        assert sample_preprocessor.target_column not in sample_preprocessor.numerical_features
        assert sample_preprocessor.target_column not in sample_preprocessor.categorical_features

        # Protected attribute should be in numerical features (already encoded)
        assert sample_preprocessor.protected_attribute in sample_preprocessor.numerical_features

    def test_fit(self, sample_preprocessor, sample_adult_data):
        """Test preprocessor fitting."""
        sample_preprocessor.fit(sample_adult_data)

        assert sample_preprocessor.numerical_features is not None
        assert sample_preprocessor.categorical_features is not None
        assert sample_preprocessor.preprocessor is not None
        assert sample_preprocessor.feature_names is not None

    def test_fit_missing_target_column(self, sample_preprocessor, sample_adult_data):
        """Test fitting with missing target column."""
        data_without_target = sample_adult_data.drop('income', axis=1)

        with pytest.raises(ValueError, match="Target column 'income' not found"):
            sample_preprocessor.fit(data_without_target)

    def test_fit_missing_protected_attribute(self, sample_preprocessor, sample_adult_data):
        """Test fitting with missing protected attribute."""
        data_without_protected = sample_adult_data.drop('sex', axis=1)

        with pytest.raises(ValueError, match="Protected attribute 'sex' not found"):
            sample_preprocessor.fit(data_without_protected)

    def test_transform(self, fitted_preprocessor, sample_train_val_test_data):
        """Test data transformation."""
        train_df, val_df, test_df = sample_train_val_test_data

        X_transformed, y_transformed, protected_transformed = fitted_preprocessor.transform(val_df)

        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(y_transformed, np.ndarray)
        assert isinstance(protected_transformed, np.ndarray)

        assert X_transformed.shape[0] == len(val_df)
        assert y_transformed.shape[0] == len(val_df)
        assert protected_transformed.shape[0] == len(val_df)

        # Check that features are properly transformed
        assert X_transformed.shape[1] > 0
        assert len(fitted_preprocessor.feature_names) == X_transformed.shape[1]

    def test_transform_unfitted(self, sample_preprocessor, sample_adult_data):
        """Test transformation without fitting first."""
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            sample_preprocessor.transform(sample_adult_data)

    def test_fit_transform(self, sample_preprocessor, sample_adult_data):
        """Test combined fit and transform."""
        X_transformed, y_transformed, protected_transformed = sample_preprocessor.fit_transform(sample_adult_data)

        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(y_transformed, np.ndarray)
        assert isinstance(protected_transformed, np.ndarray)

        assert X_transformed.shape[0] == len(sample_adult_data)
        assert sample_preprocessor.preprocessor is not None

    def test_get_feature_names(self, fitted_preprocessor):
        """Test feature names retrieval."""
        feature_names = fitted_preprocessor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_get_feature_names_unfitted(self, sample_preprocessor):
        """Test feature names retrieval without fitting."""
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            sample_preprocessor.get_feature_names()

    def test_get_feature_importance_analysis(self, fitted_preprocessor, sample_processed_data):
        """Test feature importance analysis."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Create mock importance scores
        n_features = X_processed.shape[1]
        importance_scores = np.random.random(n_features)

        analysis = fitted_preprocessor.get_feature_importance_analysis(importance_scores, top_k=5)

        assert 'top_features' in analysis
        assert 'top_importance' in analysis
        assert 'protected_attribute_importance' in analysis
        assert 'total_features' in analysis
        assert 'importance_concentration' in analysis

        assert len(analysis['top_features']) <= 5
        assert len(analysis['top_features']) == len(analysis['top_importance'])

    def test_get_feature_importance_analysis_wrong_length(self, fitted_preprocessor, sample_processed_data):
        """Test feature importance analysis with wrong importance array length."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Create importance scores with wrong length
        wrong_length_scores = np.random.random(5)  # Definitely not the right length

        with pytest.raises(ValueError, match="Importance scores length"):
            fitted_preprocessor.get_feature_importance_analysis(wrong_length_scores)

    def test_inverse_transform_sample(self, fitted_preprocessor, sample_processed_data):
        """Test inverse transformation of a sample."""
        X_processed, y_processed, protected_processed = sample_processed_data

        interpretation = fitted_preprocessor.inverse_transform_sample(X_processed, sample_idx=0)

        assert isinstance(interpretation, dict)
        assert len(interpretation) > 0

        # Check that we get reasonable values
        for feature_name, value in interpretation.items():
            assert isinstance(feature_name, str)
            assert value is not None

    def test_inverse_transform_sample_out_of_bounds(self, fitted_preprocessor, sample_processed_data):
        """Test inverse transformation with out-of-bounds sample index."""
        X_processed, y_processed, protected_processed = sample_processed_data

        with pytest.raises(ValueError, match="Sample index .* out of bounds"):
            fitted_preprocessor.inverse_transform_sample(X_processed, sample_idx=len(X_processed))