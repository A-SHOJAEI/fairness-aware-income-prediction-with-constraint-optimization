#!/usr/bin/env python3
"""Training script for fairness-aware income prediction model."""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairness_aware_income_prediction_with_constraint_optimization.training.trainer import FairnessAwareTrainer
from fairness_aware_income_prediction_with_constraint_optimization.utils.config import Config


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train fairness-aware income prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna optimization trials (overrides config)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optimization timeout in seconds (overrides config)"
    )

    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run cross-validation"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)

        # Override configuration with command-line arguments
        if args.n_trials is not None:
            config.set('optimization.n_trials', args.n_trials)

        if args.timeout is not None:
            config.set('optimization.timeout', args.timeout)

        if args.disable_mlflow:
            config.set('training.use_mlflow', False)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trainer
        logger.info("Initializing fairness-aware trainer")
        trainer = FairnessAwareTrainer(config)

        # Run training pipeline
        logger.info("Starting training pipeline")
        results = trainer.train_complete_pipeline(
            save_model=not args.no_save,
            run_cross_validation=args.cross_validation
        )

        # Print results summary
        logger.info("Training completed successfully!")

        # Print test metrics
        test_metrics = results['test_metrics']
        logger.info("=== Test Set Performance ===")

        overall_metrics = test_metrics.get('overall', {})
        logger.info(f"Accuracy: {overall_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"AUC-ROC: {overall_metrics.get('auc_roc', 0.0):.4f}")
        logger.info(f"F1-Score: {overall_metrics.get('f1_score', 0.0):.4f}")

        logger.info("=== Fairness Metrics ===")
        logger.info(f"Demographic Parity Ratio: {test_metrics.get('demographic_parity_ratio', 0.0):.4f}")
        logger.info(f"Equalized Odds Difference: {test_metrics.get('equalized_odds_difference', 1.0):.4f}")
        logger.info(f"Composite Score: {test_metrics.get('composite_score', 0.0):.4f}")

        # Print target achievement if available
        target_assessment = test_metrics.get('target_assessment', {})
        if target_assessment:
            logger.info("=== Target Achievement ===")
            for metric, achieved in target_assessment.items():
                if metric.endswith('_achieved'):
                    metric_name = metric.replace('_achieved', '')
                    status = "✓ Achieved" if achieved else "✗ Not achieved"
                    gap = target_assessment.get(f"{metric_name}_gap", 0.0)
                    logger.info(f"{metric_name}: {status} (gap: {gap:.4f})")

        # Print cross-validation results if available
        cv_results = results.get('cv_results', {})
        if cv_results:
            logger.info("=== Cross-Validation Results ===")
            cv_summary = cv_results.get('summary', {})

            for metric in ['accuracy', 'auc_roc', 'f1_score', 'demographic_parity_ratio', 'equalized_odds_difference']:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                if mean_key in cv_summary and std_key in cv_summary:
                    mean_val = cv_summary[mean_key]
                    std_val = cv_summary[std_key]
                    logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

        # Print model save path
        if not args.no_save and 'model_path' in results:
            logger.info(f"Model saved to: {results['model_path']}")

        # Print data information
        data_info = results.get('data_info', {})
        logger.info("=== Data Information ===")
        logger.info(f"Training samples: {data_info.get('train_size', 0)}")
        logger.info(f"Validation samples: {data_info.get('val_size', 0)}")
        logger.info(f"Test samples: {data_info.get('test_size', 0)}")
        logger.info(f"Number of features: {data_info.get('n_features', 0)}")

        # Print optimization summary if available
        if hasattr(trainer.model, 'optimization_history') and trainer.model.optimization_history:
            opt_df = trainer.model.get_optimization_summary()
            logger.info("=== Optimization Summary ===")
            logger.info(f"Total trials: {len(opt_df)}")
            logger.info(f"Best trial score: {opt_df['score'].max():.4f}")
            logger.info(f"Best trial AUC: {opt_df.loc[opt_df['score'].idxmax(), 'auc']:.4f}")
            logger.info(f"Best trial fairness ratio: {opt_df.loc[opt_df['score'].idxmax(), 'demographic_parity_ratio']:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()