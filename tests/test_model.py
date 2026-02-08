"""Tests for fairness-aware model implementation."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.fairness_aware_income_prediction_with_constraint_optimization.models.model import FairnessConstrainedLGBM
from src.fairness_aware_income_prediction_with_constraint_optimization.evaluation.metrics import FairnessMetrics


class TestFairnessConstrainedLGBM:
    """Test cases for FairnessConstrainedLGBM class."""

    def test_init(self):
        """Test model initialization."""
        model = FairnessConstrainedLGBM(
            protected_attribute_name='sex',
            fairness_constraint_weight=1.0,
            demographic_parity_weight=1.0,
            equalized_odds_weight=0.5,
            max_unfairness_tolerance=0.15,
            random_state=42
        )

        assert model.protected_attribute_name == 'sex'
        assert model.fairness_constraint_weight == 1.0
        assert model.demographic_parity_weight == 1.0
        assert model.equalized_odds_weight == 0.5
        assert model.max_unfairness_tolerance == 0.15
        assert model.random_state == 42

        assert model.model is None
        assert model.best_params is None
        assert model.optimization_history == []
        assert model.feature_names is None

        # Check default parameters
        assert 'objective' in model.lgb_params
        assert model.lgb_params['objective'] == 'binary'
        assert model.lgb_params['random_state'] == 42

    def test_init_with_custom_lgb_params(self):
        """Test initialization with custom LightGBM parameters."""
        custom_params = {
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8
        }

        model = FairnessConstrainedLGBM(lgb_params=custom_params)

        # Check that custom params are merged
        assert model.lgb_params['num_leaves'] == 50
        assert model.lgb_params['learning_rate'] == 0.05
        assert model.lgb_params['feature_fraction'] == 0.8

        # Check that default params are still there
        assert model.lgb_params['objective'] == 'binary'

    def test_create_lgb_model(self):
        """Test LightGBM model creation."""
        model = FairnessConstrainedLGBM()

        params = {'num_leaves': 20, 'learning_rate': 0.05}
        lgb_model = model._create_lgb_model(params)

        assert lgb_model is not None
        assert hasattr(lgb_model, 'fit')
        assert hasattr(lgb_model, 'predict')

    def test_compute_fairness_penalty(self, sample_predictions):
        """Test fairness penalty computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        model = FairnessConstrainedLGBM()
        penalty = model._compute_fairness_penalty(y_true, y_pred, protected_attributes, y_prob)

        assert isinstance(penalty, float)
        assert penalty >= 0.0  # Penalty should be non-negative

    def test_compute_fairness_penalty_error_handling(self):
        """Test fairness penalty computation with invalid inputs."""
        model = FairnessConstrainedLGBM()

        # Test with empty arrays
        empty_array = np.array([])
        penalty = model._compute_fairness_penalty(empty_array, empty_array, empty_array)

        assert penalty == 1.0  # Should return high penalty for error cases

    def test_fit_without_optimization(self, sample_processed_data):
        """Test model fitting without hyperparameter optimization."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Split data for validation
        split_idx = len(X_processed) // 2
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y_processed[:split_idx], y_processed[split_idx:]
        protected_train, protected_val = protected_processed[:split_idx], protected_processed[split_idx:]

        model = FairnessConstrainedLGBM(random_state=42)

        # Fit without optimization (should be fast)
        model.fit(
            X_train, y_train, protected_train,
            X_val, y_val, protected_val,
            optimize_hyperparameters=False
        )

        assert model.model is not None
        assert hasattr(model.model, 'predict')

    def test_fit_with_optimization(self, sample_processed_data):
        """Test model fitting with hyperparameter optimization."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Split data for validation
        split_idx = len(X_processed) // 2
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y_processed[:split_idx], y_processed[split_idx:]
        protected_train, protected_val = protected_processed[:split_idx], protected_processed[split_idx:]

        model = FairnessConstrainedLGBM(random_state=42)

        # Fit with minimal optimization for testing
        model.fit(
            X_train, y_train, protected_train,
            X_val, y_val, protected_val,
            optimize_hyperparameters=True,
            n_trials=3,  # Very small number for testing
            timeout=10   # Short timeout
        )

        assert model.model is not None
        assert model.best_params is not None
        assert len(model.optimization_history) > 0

    def test_predict_unfitted(self, sample_processed_data):
        """Test prediction with unfitted model."""
        X_processed, y_processed, protected_processed = sample_processed_data

        model = FairnessConstrainedLGBM()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X_processed)

    def test_predict_proba_unfitted(self, sample_processed_data):
        """Test probability prediction with unfitted model."""
        X_processed, y_processed, protected_processed = sample_processed_data

        model = FairnessConstrainedLGBM()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict_proba(X_processed)

    def test_predictions_after_fitting(self, sample_processed_data):
        """Test predictions after model fitting."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Split data
        split_idx = len(X_processed) // 2
        X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_test = y_processed[:split_idx], y_processed[split_idx:]
        protected_train, protected_test = protected_processed[:split_idx], protected_processed[split_idx:]

        model = FairnessConstrainedLGBM(random_state=42)

        # Quick fit without optimization
        model.fit(
            X_train, y_train, protected_train,
            optimize_hyperparameters=False
        )

        # Test predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        assert len(y_pred) == len(X_test)
        assert y_prob.shape == (len(X_test), 2)
        assert np.all((y_pred == 0) | (y_pred == 1))  # Binary predictions
        assert np.all((y_prob >= 0) & (y_prob <= 1))  # Valid probabilities

    def test_get_feature_importance_unfitted(self):
        """Test feature importance with unfitted model."""
        model = FairnessConstrainedLGBM()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_feature_importance()

    def test_get_feature_importance_fitted(self, sample_processed_data):
        """Test feature importance with fitted model."""
        X_processed, y_processed, protected_processed = sample_processed_data

        model = FairnessConstrainedLGBM(random_state=42)

        # Quick fit
        model.fit(X_processed, y_processed, protected_processed, optimize_hyperparameters=False)

        importance = model.get_feature_importance()

        assert isinstance(importance, np.ndarray)
        assert len(importance) == X_processed.shape[1]
        assert np.all(importance >= 0)  # Importance should be non-negative

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        model = FairnessConstrainedLGBM()

        # Initially None
        assert model.get_feature_names() is None

        # Set feature names
        test_names = ['feature_1', 'feature_2', 'feature_3']
        model.feature_names = test_names

        retrieved_names = model.get_feature_names()
        assert retrieved_names == test_names
        assert retrieved_names is not test_names  # Should be a copy

    def test_get_optimization_summary_empty(self):
        """Test optimization summary with no optimization history."""
        model = FairnessConstrainedLGBM()

        summary_df = model.get_optimization_summary()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 0

    def test_get_optimization_summary_with_history(self):
        """Test optimization summary with optimization history."""
        model = FairnessConstrainedLGBM()

        # Add mock optimization history
        model.optimization_history = [
            {
                'trial_number': 0,
                'params': {'num_leaves': 10, 'learning_rate': 0.1},
                'score': 0.8,
                'auc': 0.85,
                'demographic_parity_ratio': 0.9
            },
            {
                'trial_number': 1,
                'params': {'num_leaves': 20, 'learning_rate': 0.05},
                'score': 0.82,
                'auc': 0.87,
                'demographic_parity_ratio': 0.88
            }
        ]

        summary_df = model.get_optimization_summary()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 2
        assert 'trial_number' in summary_df.columns
        assert 'score' in summary_df.columns

    def test_evaluate_fairness_unfitted(self, sample_processed_data):
        """Test fairness evaluation with unfitted model."""
        X_processed, y_processed, protected_processed = sample_processed_data

        model = FairnessConstrainedLGBM()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.evaluate_fairness(X_processed, y_processed, protected_processed)

    def test_evaluate_fairness_fitted(self, sample_processed_data):
        """Test fairness evaluation with fitted model."""
        X_processed, y_processed, protected_processed = sample_processed_data

        model = FairnessConstrainedLGBM(random_state=42)

        # Quick fit
        model.fit(X_processed, y_processed, protected_processed, optimize_hyperparameters=False)

        # Evaluate fairness
        fairness_metrics = model.evaluate_fairness(X_processed, y_processed, protected_processed)

        assert isinstance(fairness_metrics, dict)
        assert 'overall' in fairness_metrics
        assert 'demographic_parity_ratio' in fairness_metrics
        assert 'equalized_odds_difference' in fairness_metrics

    def test_evaluate_model_with_fairness(self, sample_processed_data):
        """Test model evaluation with fairness constraints."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Create a simple mock model
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_model.fit(X_processed, y_processed)

        fairness_model = FairnessConstrainedLGBM(random_state=42)

        # Test the evaluation method
        score = fairness_model._evaluate_model_with_fairness(
            mock_model, X_processed, y_processed, protected_processed
        )

        assert isinstance(score, float)
        # Score can be negative due to fairness penalties

    @patch('optuna.create_study')
    def test_optimize_hyperparameters_mock(self, mock_create_study, sample_processed_data):
        """Test hyperparameter optimization with mocked Optuna."""
        X_processed, y_processed, protected_processed = sample_processed_data

        # Split data
        split_idx = len(X_processed) // 2
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y_processed[:split_idx], y_processed[split_idx:]
        protected_train, protected_val = protected_processed[:split_idx], protected_processed[split_idx:]

        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {'num_leaves': 20, 'learning_rate': 0.1}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study

        model = FairnessConstrainedLGBM(random_state=42)

        best_params = model.optimize_hyperparameters(
            X_train, y_train, protected_train,
            X_val, y_val, protected_val,
            n_trials=5
        )

        assert best_params == mock_study.best_params
        assert model.best_params == mock_study.best_params
        mock_study.optimize.assert_called_once()


class TestFairnessMetrics:
    """Test cases for FairnessMetrics class."""

    def test_init(self):
        """Test FairnessMetrics initialization."""
        evaluator = FairnessMetrics(protected_attribute_name='sex')
        assert evaluator.protected_attribute_name == 'sex'

    def test_compute_performance_metrics(self, sample_predictions):
        """Test performance metrics computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        metrics = evaluator.compute_performance_metrics(y_true, y_pred, y_prob)

        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'average_precision', 'brier_score']

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'average_precision']:
                assert 0 <= metrics[metric] <= 1

    def test_compute_performance_metrics_without_probabilities(self, sample_predictions):
        """Test performance metrics computation without probabilities."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        metrics = evaluator.compute_performance_metrics(y_true, y_pred)

        # Should have basic metrics but not probability-based ones
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' not in metrics

    def test_compute_demographic_parity(self, sample_predictions):
        """Test demographic parity computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        dp_metrics = evaluator.compute_demographic_parity(y_pred, protected_attributes)

        assert 'demographic_parity_ratio' in dp_metrics
        assert 'demographic_parity_difference' in dp_metrics
        assert 'positive_rates' in dp_metrics

        assert 0 <= dp_metrics['demographic_parity_ratio'] <= 1
        assert 0 <= dp_metrics['demographic_parity_difference'] <= 1

    def test_compute_equalized_odds(self, sample_predictions):
        """Test equalized odds computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        eo_metrics = evaluator.compute_equalized_odds(y_true, y_pred, protected_attributes)

        assert 'equalized_odds_difference' in eo_metrics
        assert 'tpr_difference' in eo_metrics
        assert 'fpr_difference' in eo_metrics
        assert 'tprs' in eo_metrics
        assert 'fprs' in eo_metrics

        assert 0 <= eo_metrics['equalized_odds_difference'] <= 1
        assert 0 <= eo_metrics['tpr_difference'] <= 1
        assert 0 <= eo_metrics['fpr_difference'] <= 1

    def test_compute_equal_opportunity(self, sample_predictions):
        """Test equal opportunity computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        eop_metrics = evaluator.compute_equal_opportunity(y_true, y_pred, protected_attributes)

        assert 'equal_opportunity_difference' in eop_metrics
        assert 'tprs' in eop_metrics
        assert 0 <= eop_metrics['equal_opportunity_difference'] <= 1

    def test_compute_calibration_metrics(self, sample_predictions):
        """Test calibration metrics computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        cal_metrics = evaluator.compute_calibration_metrics(y_true, y_prob, protected_attributes)

        assert 'calibration_difference' in cal_metrics
        assert 'group_calibration' in cal_metrics

        # Check group calibration
        group_cal = cal_metrics['group_calibration']
        assert len(group_cal) > 0

        for group, metrics in group_cal.items():
            assert 'calibration_score' in metrics
            assert 'expected_calibration_error' in metrics
            assert 'brier_score' in metrics

    def test_compute_all_fairness_metrics(self, sample_predictions):
        """Test comprehensive fairness metrics computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        all_metrics = evaluator.compute_all_fairness_metrics(y_true, y_pred, protected_attributes, y_prob)

        # Check that all major categories are present
        assert 'overall' in all_metrics
        assert 'by_group' in all_metrics
        assert 'demographic_parity_ratio' in all_metrics
        assert 'equalized_odds_difference' in all_metrics

    def test_compute_fairness_composite_score(self, sample_predictions):
        """Test fairness composite score computation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        all_metrics = evaluator.compute_all_fairness_metrics(y_true, y_pred, protected_attributes, y_prob)

        composite_score = evaluator.compute_fairness_composite_score(all_metrics)

        assert isinstance(composite_score, float)
        assert 0 <= composite_score <= 1

    def test_evaluate_model(self, sample_predictions):
        """Test complete model evaluation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        target_metrics = {
            'auc_roc': 0.8,
            'demographic_parity_ratio': 0.85
        }

        evaluator = FairnessMetrics()
        evaluation = evaluator.evaluate_model(y_true, y_pred, protected_attributes, y_prob, target_metrics)

        assert 'composite_score' in evaluation
        assert 'target_assessment' in evaluation

        # Check target assessment
        target_assessment = evaluation['target_assessment']
        assert 'auc_roc_achieved' in target_assessment
        assert 'demographic_parity_ratio_achieved' in target_assessment

    def test_create_metrics_summary(self, sample_predictions):
        """Test metrics summary creation."""
        y_true, y_pred, y_prob, protected_attributes = sample_predictions

        evaluator = FairnessMetrics()
        metrics = evaluator.compute_all_fairness_metrics(y_true, y_pred, protected_attributes, y_prob)

        summary_df = evaluator.create_metrics_summary(metrics)

        assert isinstance(summary_df, pd.DataFrame)
        assert 'Category' in summary_df.columns
        assert 'Metric' in summary_df.columns
        assert 'Value' in summary_df.columns
        assert len(summary_df) > 0