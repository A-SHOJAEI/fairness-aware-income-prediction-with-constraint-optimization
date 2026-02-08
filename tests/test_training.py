"""Tests for training utilities and trainer class."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.fairness_aware_income_prediction_with_constraint_optimization.training.trainer import FairnessAwareTrainer
from src.fairness_aware_income_prediction_with_constraint_optimization.utils.config import Config


class TestFairnessAwareTrainer:
    """Test cases for FairnessAwareTrainer class."""

    def test_init(self, sample_config):
        """Test trainer initialization."""
        trainer = FairnessAwareTrainer(sample_config)

        assert trainer.config == sample_config
        assert trainer.data_loader is not None
        assert trainer.preprocessor is not None
        assert trainer.model is None
        assert trainer.fairness_evaluator is not None
        assert trainer.checkpoint_dir.exists()

    def test_init_default_config(self):
        """Test trainer initialization with default config."""
        with patch('src.fairness_aware_income_prediction_with_constraint_optimization.training.trainer.Config') as mock_config:
            mock_config.return_value = MagicMock()
            trainer = FairnessAwareTrainer()
            assert trainer.config is not None

    @patch('mlflow.set_experiment')
    def test_setup_mlflow_success(self, mock_set_experiment, sample_config):
        """Test successful MLflow setup."""
        sample_config.set('training.use_mlflow', True)
        trainer = FairnessAwareTrainer(sample_config)

        mock_set_experiment.assert_called_once_with("fairness-aware-income-prediction")
        assert trainer.mlflow_enabled is True

    @patch('mlflow.set_experiment')
    def test_setup_mlflow_failure(self, mock_set_experiment, sample_config):
        """Test MLflow setup failure handling."""
        mock_set_experiment.side_effect = Exception("MLflow error")
        sample_config.set('training.use_mlflow', True)

        trainer = FairnessAwareTrainer(sample_config)

        assert trainer.mlflow_enabled is False

    def test_load_and_prepare_data(self, sample_config, mock_data_loader, sample_adult_data):
        """Test data loading and preparation."""
        with patch.object(FairnessAwareTrainer, '__init__', lambda x, config: None):
            trainer = FairnessAwareTrainer.__new__(FairnessAwareTrainer)
            trainer.config = sample_config
            trainer.data_loader = mock_data_loader
            trainer.preprocessor = MagicMock()

            # Mock the preprocessor methods
            trainer.preprocessor.fit_transform.return_value = (
                np.random.random((100, 10)),  # X_train
                np.random.randint(0, 2, 100),  # y_train
                np.random.randint(0, 2, 100)   # protected_train
            )
            trainer.preprocessor.transform.return_value = (
                np.random.random((30, 10)),   # X_val/test
                np.random.randint(0, 2, 30),  # y_val/test
                np.random.randint(0, 2, 30)   # protected_val/test
            )

            result = trainer.load_and_prepare_data()

            assert len(result) == 9  # 3 sets × 3 arrays each
            X_train, y_train, protected_train, X_val, y_val, protected_val, X_test, y_test, protected_test = result

            # Check shapes
            assert X_train.shape[0] == 100
            assert len(y_train) == 100
            assert len(protected_train) == 100

    def test_create_model(self, sample_config):
        """Test model creation."""
        trainer = FairnessAwareTrainer(sample_config)
        model = trainer.create_model()

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert model.protected_attribute_name == 'sex'

    @patch('mlflow.start_run')
    @patch('mlflow.end_run')
    def test_train_model(self, mock_end_run, mock_start_run, sample_config, sample_processed_data):
        """Test model training."""
        # Disable MLflow for this test
        sample_config.set('training.use_mlflow', False)
        sample_config.set('optimization.n_trials', 2)  # Small number for testing

        trainer = FairnessAwareTrainer(sample_config)

        X_processed, y_processed, protected_processed = sample_processed_data

        # Split data for training
        split_idx = len(X_processed) // 2
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y_processed[:split_idx], y_processed[split_idx:]
        protected_train, protected_val = protected_processed[:split_idx], protected_processed[split_idx:]

        # Train model
        trained_model = trainer.train_model(
            X_train, y_train, protected_train,
            X_val, y_val, protected_val
        )

        assert trained_model is not None
        assert trainer.model is not None
        assert hasattr(trained_model, 'predict')

    def test_evaluate_model_untrained(self, sample_config, sample_processed_data):
        """Test model evaluation with untrained model."""
        trainer = FairnessAwareTrainer(sample_config)
        X_processed, y_processed, protected_processed = sample_processed_data

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.evaluate_model(X_processed, y_processed, protected_processed)

    def test_evaluate_model_trained(self, sample_config, sample_processed_data):
        """Test model evaluation with trained model."""
        trainer = FairnessAwareTrainer(sample_config)

        # Mock a trained model
        trainer.model = MagicMock()
        trainer.model.evaluate_fairness.return_value = {
            'overall': {'accuracy': 0.85, 'auc_roc': 0.88},
            'demographic_parity_ratio': 0.87,
            'equalized_odds_difference': 0.12
        }

        X_processed, y_processed, protected_processed = sample_processed_data

        # Disable MLflow for this test
        sample_config.set('training.use_mlflow', False)
        trainer.mlflow_enabled = False

        metrics = trainer.evaluate_model(X_processed, y_processed, protected_processed, log_metrics=False)

        assert 'overall' in metrics
        assert 'demographic_parity_ratio' in metrics
        trainer.model.evaluate_fairness.assert_called_once()

    @patch('mlflow.log_metric')
    def test_log_evaluation_metrics(self, mock_log_metric, sample_config):
        """Test evaluation metrics logging to MLflow."""
        sample_config.set('training.use_mlflow', True)
        trainer = FairnessAwareTrainer(sample_config)
        trainer.mlflow_enabled = True

        metrics = {
            'overall': {'accuracy': 0.85, 'auc_roc': 0.88},
            'demographic_parity_ratio': 0.87,
            'equalized_odds_difference': 0.12,
            'composite_score': 0.86
        }

        trainer._log_evaluation_metrics(metrics)

        # Check that metrics were logged
        assert mock_log_metric.call_count > 0

    def test_cross_validate_model(self, sample_config, sample_processed_data):
        """Test cross-validation."""
        sample_config.set('training.use_mlflow', False)
        sample_config.set('optimization.n_trials', 2)

        trainer = FairnessAwareTrainer(sample_config)
        X_processed, y_processed, protected_processed = sample_processed_data

        cv_results = trainer.cross_validate_model(
            X_processed, y_processed, protected_processed, cv_folds=3
        )

        assert 'fold_results' in cv_results
        assert 'summary' in cv_results

        fold_results = cv_results['fold_results']
        assert 'accuracy' in fold_results
        assert 'auc_roc' in fold_results
        assert len(fold_results['accuracy']) == 3  # 3 folds

        summary = cv_results['summary']
        assert 'accuracy_mean' in summary
        assert 'accuracy_std' in summary

    def test_save_model_untrained(self, sample_config, temp_directory):
        """Test saving untrained model."""
        trainer = FairnessAwareTrainer(sample_config)

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.save_model(temp_directory / "test_model.pkl")

    def test_save_load_model(self, sample_config, temp_directory, sample_processed_data):
        """Test model saving and loading."""
        # Disable MLflow for this test
        sample_config.set('training.use_mlflow', False)

        trainer = FairnessAwareTrainer(sample_config)

        # Create a mock trained model
        trainer.model = MagicMock()
        trainer.training_history = [{'epoch': 1, 'loss': 0.5}]

        # Save model
        model_path = trainer.save_model(temp_directory / "test_model.pkl")

        assert model_path.exists()

        # Load model in new trainer
        new_trainer = FairnessAwareTrainer(sample_config)
        new_trainer.load_model(model_path)

        assert new_trainer.model is not None
        assert new_trainer.training_history == trainer.training_history

    def test_load_model_nonexistent(self, sample_config):
        """Test loading non-existent model file."""
        trainer = FairnessAwareTrainer(sample_config)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            trainer.load_model("nonexistent_model.pkl")

    @patch('src.fairness_aware_income_prediction_with_constraint_optimization.training.trainer.FairnessAwareTrainer.load_and_prepare_data')
    @patch('src.fairness_aware_income_prediction_with_constraint_optimization.training.trainer.FairnessAwareTrainer.train_model')
    @patch('src.fairness_aware_income_prediction_with_constraint_optimization.training.trainer.FairnessAwareTrainer.evaluate_model')
    def test_train_complete_pipeline(self, mock_evaluate, mock_train, mock_load_data, sample_config, sample_processed_data):
        """Test complete training pipeline."""
        sample_config.set('training.use_mlflow', False)

        # Mock the data loading
        X_processed, y_processed, protected_processed = sample_processed_data
        split_idx = len(X_processed) // 3
        mock_load_data.return_value = (
            X_processed[:split_idx], y_processed[:split_idx], protected_processed[:split_idx],  # train
            X_processed[split_idx:2*split_idx], y_processed[split_idx:2*split_idx], protected_processed[split_idx:2*split_idx],  # val
            X_processed[2*split_idx:], y_processed[2*split_idx:], protected_processed[2*split_idx:]  # test
        )

        # Mock the training
        mock_model = MagicMock()
        mock_train.return_value = mock_model

        # Mock the evaluation
        mock_evaluate.return_value = {
            'overall': {'accuracy': 0.85, 'auc_roc': 0.88},
            'demographic_parity_ratio': 0.87
        }

        trainer = FairnessAwareTrainer(sample_config)

        # Run complete pipeline
        results = trainer.train_complete_pipeline(save_model=False, run_cross_validation=False)

        assert 'model' in results
        assert 'test_metrics' in results
        assert 'data_info' in results

        # Check that methods were called
        mock_load_data.assert_called_once()
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()

    def test_train_complete_pipeline_with_cv(self, sample_config, sample_processed_data):
        """Test complete pipeline with cross-validation."""
        sample_config.set('training.use_mlflow', False)
        sample_config.set('optimization.n_trials', 2)

        trainer = FairnessAwareTrainer(sample_config)

        with patch.object(trainer, 'load_and_prepare_data') as mock_load_data:
            X_processed, y_processed, protected_processed = sample_processed_data
            split_idx = len(X_processed) // 3
            mock_load_data.return_value = (
                X_processed[:split_idx], y_processed[:split_idx], protected_processed[:split_idx],  # train
                X_processed[split_idx:2*split_idx], y_processed[split_idx:2*split_idx], protected_processed[split_idx:2*split_idx],  # val
                X_processed[2*split_idx:], y_processed[2*split_idx:], protected_processed[2*split_idx:]  # test
            )

            # Patch cross_validate_model to avoid long runtime
            with patch.object(trainer, 'cross_validate_model') as mock_cv:
                mock_cv.return_value = {
                    'fold_results': {'accuracy': [0.8, 0.82, 0.81]},
                    'summary': {'accuracy_mean': 0.81, 'accuracy_std': 0.01}
                }

                results = trainer.train_complete_pipeline(save_model=False, run_cross_validation=True)

                assert 'cv_results' in results
                mock_cv.assert_called_once()


class TestConfig:
    """Test cases for Config class."""

    def test_init_with_path(self, sample_config):
        """Test config initialization with path."""
        # The sample_config fixture already tests this implicitly
        assert sample_config.config is not None

    def test_init_default_path(self):
        """Test config initialization with default path."""
        # This should fail since default config doesn't exist in test environment
        with pytest.raises(FileNotFoundError):
            Config()

    def test_get_value(self, sample_config):
        """Test getting configuration values."""
        # Test existing key
        test_size = sample_config.get('data.test_size')
        assert test_size == 0.2

        # Test non-existing key with default
        non_existing = sample_config.get('non.existing.key', 'default_value')
        assert non_existing == 'default_value'

        # Test non-existing key without default
        non_existing_no_default = sample_config.get('non.existing.key')
        assert non_existing_no_default is None

    def test_set_value(self, sample_config):
        """Test setting configuration values."""
        sample_config.set('new.config.key', 'new_value')
        assert sample_config.get('new.config.key') == 'new_value'

        # Test overwriting existing value
        sample_config.set('data.test_size', 0.3)
        assert sample_config.get('data.test_size') == 0.3

    def test_update_config(self, sample_config):
        """Test updating configuration with dictionary."""
        updates = {
            'data': {'new_param': 'new_value'},
            'model': {'new_model_param': 42}
        }

        sample_config.update(updates)

        assert sample_config.get('data.new_param') == 'new_value'
        assert sample_config.get('model.new_model_param') == 42
        # Check that existing values are preserved
        assert sample_config.get('data.test_size') == 0.2

    def test_get_specialized_configs(self, sample_config):
        """Test specialized config getters."""
        data_config = sample_config.get_data_config()
        model_config = sample_config.get_model_config()
        training_config = sample_config.get_training_config()
        optimization_config = sample_config.get_optimization_config()
        evaluation_config = sample_config.get_evaluation_config()

        assert isinstance(data_config, dict)
        assert isinstance(model_config, dict)
        assert isinstance(training_config, dict)
        assert isinstance(optimization_config, dict)
        assert isinstance(evaluation_config, dict)

        assert 'test_size' in data_config
        assert 'algorithm' in model_config

    def test_save_config(self, sample_config, temp_directory):
        """Test configuration saving."""
        output_path = temp_directory / "saved_config.yaml"

        sample_config.save(output_path)

        assert output_path.exists()

        # Load and verify
        import yaml
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config['data']['test_size'] == 0.2