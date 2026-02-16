"""Training utilities and fairness-aware training loops."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold

try:
    import mlflow
    import mlflow.lightgbm
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from ..data.loader import AdultIncomeLoader
from ..data.preprocessing import FairnessAwarePreprocessor
from ..models.model import FairnessConstrainedLGBM
from ..evaluation.metrics import FairnessMetrics
from ..utils.config import Config


logger = logging.getLogger(__name__)


class FairnessAwareTrainer:
    """Trainer for fairness-aware income prediction models.

    This class orchestrates the complete training pipeline including data loading,
    preprocessing, model training with fairness constraints, evaluation, and
    model persistence.

    Attributes:
        config: Configuration object.
        data_loader: Data loader instance.
        preprocessor: Data preprocessor instance.
        model: Fairness-constrained model instance.
        fairness_evaluator: Fairness metrics evaluator.
        mlflow_enabled: Whether MLflow tracking is enabled.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the fairness-aware trainer.

        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()

        # Initialize components
        data_config = self.config.get_data_config()
        self.data_loader = AdultIncomeLoader()
        self.preprocessor = FairnessAwarePreprocessor(
            protected_attribute=data_config.get('protected_attribute', 'sex'),
            target_column=data_config.get('target_column', 'income'),
            handle_missing=self.config.get('preprocessing.handle_missing', True),
            encode_categorical=self.config.get('preprocessing.encode_categorical', True),
            scale_numerical=self.config.get('preprocessing.scale_numerical', True)
        )

        self.model: Optional[FairnessConstrainedLGBM] = None
        self.fairness_evaluator = FairnessMetrics(
            data_config.get('protected_attribute', 'sex')
        )

        # MLflow setup
        self.mlflow_enabled = self.config.get('training.use_mlflow', True) and _MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            self._setup_mlflow()

        # Create directories
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Training state
        self.training_history: List[Dict[str, Any]] = []

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            mlflow.set_experiment("fairness-aware-income-prediction")
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to set up MLflow: {e}")
            self.mlflow_enabled = False

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare the data for training.

        Returns:
            Tuple of (X_train, y_train, protected_train,
                     X_val, y_val, protected_val,
                     X_test, y_test, protected_test).
        """
        logger.info("Loading and preparing data")

        # Load data
        data_config = self.config.get_data_config()
        df = self.data_loader.load_data(download_if_missing=True)

        # Create train/validation/test splits
        train_df, val_df, test_df = self.data_loader.create_train_val_test_split(
            df=df,
            test_size=data_config.get('test_size', 0.2),
            val_size=data_config.get('val_size', 0.2),
            random_state=data_config.get('random_state', 42)
        )

        # Fit preprocessor on training data
        X_train, y_train, protected_train = self.preprocessor.fit_transform(train_df)
        X_val, y_val, protected_val = self.preprocessor.transform(val_df)
        X_test, y_test, protected_test = self.preprocessor.transform(test_df)

        logger.info(f"Data preparation completed:")
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        logger.info(f"  Test set: {X_test.shape}")

        return (X_train, y_train, protected_train,
                X_val, y_val, protected_val,
                X_test, y_test, protected_test)

    def create_model(self) -> FairnessConstrainedLGBM:
        """Create a fairness-constrained model based on configuration.

        Returns:
            Configured model instance.
        """
        model_config = self.config.get_model_config()
        fairness_config = model_config.get('fairness_constraint', {})

        return FairnessConstrainedLGBM(
            protected_attribute_name=self.config.get('data.protected_attribute', 'sex'),
            fairness_constraint_weight=fairness_config.get('demographic_parity_weight', 1.0),
            demographic_parity_weight=fairness_config.get('demographic_parity_weight', 1.0),
            equalized_odds_weight=fairness_config.get('equalized_odds_weight', 0.5),
            max_unfairness_tolerance=fairness_config.get('max_unfairness_tolerance', 0.15),
            lgb_params=model_config.get('base_params', {}),
            random_state=self.config.get('data.random_state', 42)
        )

    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   protected_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   protected_val: np.ndarray) -> FairnessConstrainedLGBM:
        """Train the fairness-constrained model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            protected_train: Training protected attributes.
            X_val: Validation features.
            y_val: Validation targets.
            protected_val: Validation protected attributes.

        Returns:
            Trained model.
        """
        logger.info("Starting model training")

        # Create model
        self.model = self.create_model()

        # Get optimization configuration
        optimization_config = self.config.get_optimization_config()

        # Start MLflow run if enabled
        if self.mlflow_enabled:
            mlflow.start_run()

        try:
            # Train model with hyperparameter optimization
            self.model.fit(
                X_train, y_train, protected_train,
                X_val, y_val, protected_val,
                optimize_hyperparameters=True,
                n_trials=optimization_config.get('n_trials', 100),
                timeout=optimization_config.get('timeout')
            )

            # Log model artifacts if MLflow is enabled
            if self.mlflow_enabled:
                self._log_training_artifacts()

            logger.info("Model training completed")

        finally:
            if self.mlflow_enabled:
                mlflow.end_run()

        return self.model

    def _log_training_artifacts(self) -> None:
        """Log training artifacts to MLflow."""
        if not self.mlflow_enabled or self.model is None:
            return

        try:
            # Log model parameters
            if self.model.best_params:
                mlflow.log_params(self.model.best_params)

            # Log model configuration
            mlflow.log_params({
                'fairness_constraint_weight': self.model.fairness_constraint_weight,
                'demographic_parity_weight': self.model.demographic_parity_weight,
                'equalized_odds_weight': self.model.equalized_odds_weight,
                'max_unfairness_tolerance': self.model.max_unfairness_tolerance
            })

            # Log optimization history
            if self.model.optimization_history:
                optimization_df = self.model.get_optimization_summary()
                optimization_df.to_csv("optimization_history.csv", index=False)
                mlflow.log_artifact("optimization_history.csv")

            # Log the model itself
            if hasattr(self.model, 'model') and self.model.model is not None:
                mlflow.lightgbm.log_model(self.model.model, "model")

        except Exception as e:
            logger.warning(f"Failed to log artifacts to MLflow: {e}")

    def evaluate_model(self,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      protected_test: np.ndarray,
                      log_metrics: bool = True) -> Dict[str, Union[float, Dict]]:
        """Evaluate the trained model on test data.

        Args:
            X_test: Test features.
            y_test: Test targets.
            protected_test: Test protected attributes.
            log_metrics: Whether to log metrics to MLflow.

        Returns:
            Dictionary with evaluation metrics.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        logger.info("Evaluating model performance and fairness")

        # Get evaluation configuration
        eval_config = self.config.get_evaluation_config()
        target_metrics = eval_config.get('target_metrics', {})

        # Evaluate model
        metrics = self.model.evaluate_fairness(X_test, y_test, protected_test)

        # Add target achievement assessment
        if target_metrics:
            target_assessment = {}
            for metric_name, target_value in target_metrics.items():
                actual_value = metrics.get(metric_name)
                if actual_value is not None:
                    achieved = actual_value >= target_value
                    gap = target_value - actual_value
                    target_assessment[f"{metric_name}_achieved"] = achieved
                    target_assessment[f"{metric_name}_gap"] = gap

                    logger.info(f"{metric_name}: {actual_value:.4f} "
                              f"(target: {target_value:.4f}, "
                              f"{'✓' if achieved else '✗'})")

            metrics['target_assessment'] = target_assessment

        # Log metrics to MLflow
        if log_metrics and self.mlflow_enabled:
            self._log_evaluation_metrics(metrics)

        # Create summary
        summary_df = self.fairness_evaluator.create_metrics_summary(metrics)
        logger.info("Evaluation Summary:")
        logger.info(f"\n{summary_df.to_string(index=False)}")

        return metrics

    def _log_evaluation_metrics(self, metrics: Dict[str, Union[float, Dict]]) -> None:
        """Log evaluation metrics to MLflow.

        Args:
            metrics: Dictionary with evaluation metrics.
        """
        try:
            # Log overall performance metrics
            overall_metrics = metrics.get('overall', {})
            for metric_name, value in overall_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"overall_{metric_name}", value)

            # Log fairness metrics
            fairness_keys = [
                'demographic_parity_ratio', 'demographic_parity_difference',
                'equalized_odds_difference', 'equal_opportunity_difference',
                'composite_score'
            ]

            for key in fairness_keys:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    mlflow.log_metric(key, metrics[key])

            # Log target achievement
            target_assessment = metrics.get('target_assessment', {})
            for key, value in target_assessment.items():
                if isinstance(value, (int, float, bool)):
                    mlflow.log_metric(key, float(value))

        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {e}")

    def cross_validate_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           protected_attributes: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation with fairness evaluation.

        Args:
            X: Input features.
            y: Target labels.
            protected_attributes: Protected attribute values.
            cv_folds: Number of cross-validation folds.

        Returns:
            Dictionary with cross-validation results.
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        cv_results = {
            'accuracy': [],
            'auc_roc': [],
            'f1_score': [],
            'demographic_parity_ratio': [],
            'equalized_odds_difference': [],
            'composite_score': []
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=self.config.get('data.random_state', 42))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")

            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            protected_train_fold = protected_attributes[train_idx]
            protected_val_fold = protected_attributes[val_idx]

            # Create and train model for this fold
            fold_model = self.create_model()
            fold_model.fit(
                X_train_fold, y_train_fold, protected_train_fold,
                X_val_fold, y_val_fold, protected_val_fold,
                optimize_hyperparameters=False  # Use default params for CV
            )

            # Evaluate fold
            fold_metrics = fold_model.evaluate_fairness(
                X_val_fold, y_val_fold, protected_val_fold
            )

            # Store results
            for key in cv_results.keys():
                if key in fold_metrics:
                    cv_results[key].append(fold_metrics[key])
                elif key in fold_metrics.get('overall', {}):
                    cv_results[key].append(fold_metrics['overall'][key])
                else:
                    cv_results[key].append(np.nan)

        # Compute statistics
        cv_summary = {}
        for key, values in cv_results.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                cv_summary[f"{key}_mean"] = np.mean(valid_values)
                cv_summary[f"{key}_std"] = np.std(valid_values)

        logger.info("Cross-validation completed")
        for key, value in cv_summary.items():
            if 'mean' in key:
                metric_name = key.replace('_mean', '')
                std_value = cv_summary.get(f"{metric_name}_std", 0.0)
                logger.info(f"{metric_name}: {value:.4f} ± {std_value:.4f}")

        return {'fold_results': cv_results, 'summary': cv_summary}

    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save the trained model and preprocessor.

        Args:
            filepath: Output file path. If None, uses default naming.

        Returns:
            Path where the model was saved.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        if filepath is None:
            filepath = self.checkpoint_dir / "fairness_aware_model.pkl"
        else:
            filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model and preprocessor
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'training_history': self.training_history,
            'feature_names': self.preprocessor.get_feature_names()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, filepath: Union[str, Path]) -> 'FairnessAwareTrainer':
        """Load a trained model and preprocessor.

        Args:
            filepath: Path to the saved model file.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.training_history = model_data.get('training_history', [])

        # Update config if saved
        if 'config' in model_data:
            self.config = model_data['config']

        logger.info(f"Model loaded from {filepath}")
        return self

    def train_complete_pipeline(self,
                              save_model: bool = True,
                              run_cross_validation: bool = False) -> Dict[str, Any]:
        """Run the complete training pipeline.

        Args:
            save_model: Whether to save the trained model.
            run_cross_validation: Whether to run cross-validation.

        Returns:
            Dictionary with training results.
        """
        logger.info("Starting complete training pipeline")

        # Load and prepare data
        (X_train, y_train, protected_train,
         X_val, y_val, protected_val,
         X_test, y_test, protected_test) = self.load_and_prepare_data()

        # Train model
        trained_model = self.train_model(
            X_train, y_train, protected_train,
            X_val, y_val, protected_val
        )

        # Evaluate model
        test_metrics = self.evaluate_model(X_test, y_test, protected_test)

        results = {
            'model': trained_model,
            'test_metrics': test_metrics,
            'data_info': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'n_features': X_train.shape[1]
            }
        }

        # Run cross-validation if requested
        if run_cross_validation:
            # Combine train and validation for CV
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])
            protected_combined = np.hstack([protected_train, protected_val])

            cv_results = self.cross_validate_model(
                X_combined, y_combined, protected_combined,
                cv_folds=self.config.get('training.cv_folds', 5)
            )
            results['cv_results'] = cv_results

        # Save model if requested
        if save_model:
            model_path = self.save_model()
            results['model_path'] = model_path

        logger.info("Training pipeline completed successfully")
        return results