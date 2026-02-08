#!/usr/bin/env python3
"""Evaluation script for fairness-aware income prediction model."""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairness_aware_income_prediction_with_constraint_optimization.training.trainer import FairnessAwareTrainer
from fairness_aware_income_prediction_with_constraint_optimization.utils.config import Config
from fairness_aware_income_prediction_with_constraint_optimization.evaluation.metrics import FairnessMetrics


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate fairness-aware income prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file"
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
        default="evaluation_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data CSV file (optional, uses default if not provided)"
    )

    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def create_fairness_plots(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Create fairness visualization plots.

    Args:
        metrics: Evaluation metrics dictionary.
        output_dir: Directory to save plots.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating fairness visualization plots")

    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            pass  # Fall back to default matplotlib style
    sns.set_palette("husl")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance metrics by group
    by_group = metrics.get('by_group', {})
    if by_group:
        group_names = list(by_group.keys())
        metrics_to_plot = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                values = [by_group[group].get(metric, 0) for group in group_names]
                bars = axes[i].bar(group_names, values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')

        # Remove unused subplot
        if len(axes) > len(metrics_to_plot):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(plots_dir / "performance_by_group.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Fairness metrics summary
    fairness_metrics = {
        'Demographic Parity Ratio': metrics.get('demographic_parity_ratio', 0),
        'Demographic Parity Diff': metrics.get('demographic_parity_difference', 1),
        'Equalized Odds Diff': metrics.get('equalized_odds_difference', 1),
        'Equal Opportunity Diff': metrics.get('equal_opportunity_difference', 1)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ratios (closer to 1 is better)
    ratio_metrics = ['Demographic Parity Ratio']
    ratio_values = [fairness_metrics[m] for m in ratio_metrics if m in fairness_metrics]
    ax1.bar(ratio_metrics, ratio_values, color='skyblue')
    ax1.set_title('Fairness Ratios (Higher is Better)')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold (0.8)')
    ax1.legend()

    # Add value labels
    for i, v in enumerate(ratio_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Differences (closer to 0 is better)
    diff_metrics = [m for m in fairness_metrics.keys() if 'Diff' in m]
    diff_values = [fairness_metrics[m] for m in diff_metrics]
    ax2.bar(diff_metrics, diff_values, color='lightcoral')
    ax2.set_title('Fairness Differences (Lower is Better)')
    ax2.set_ylabel('Difference')
    ax2.set_ylim(0, max(diff_values) * 1.1 if diff_values else 1)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    ax2.legend()

    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, v in enumerate(diff_values):
        ax2.text(i, v + max(diff_values) * 0.02 if diff_values else 0.01,
                f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plots_dir / "fairness_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Composite score visualization
    composite_score = metrics.get('composite_score', 0)
    overall_auc = metrics.get('overall', {}).get('auc_roc', 0)
    dp_ratio = metrics.get('demographic_parity_ratio', 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    categories = ['Overall AUC', 'Demographic Parity', 'Composite Score']
    values = [overall_auc, dp_ratio, composite_score]
    colors = ['steelblue', 'orange', 'green']

    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_title('Model Performance Summary')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / "model_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {plots_dir}")


def save_detailed_results(metrics: Dict[str, Any],
                         predictions_data: Dict[str, np.ndarray],
                         output_dir: Path) -> None:
    """Save detailed evaluation results.

    Args:
        metrics: Evaluation metrics dictionary.
        predictions_data: Dictionary containing predictions and related data.
        output_dir: Directory to save results.
    """
    logger = logging.getLogger(__name__)

    # Save metrics as JSON
    metrics_file = output_dir / "evaluation_metrics.json"

    # Convert numpy arrays and other non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    serializable_metrics = convert_for_json(metrics)

    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions if provided
    if predictions_data:
        predictions_df = pd.DataFrame({
            'y_true': predictions_data['y_true'],
            'y_pred': predictions_data['y_pred'],
            'y_prob': predictions_data['y_prob'],
            'protected_attribute': predictions_data['protected_attributes']
        })

        predictions_file = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")

    # Create summary report
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("FAIRNESS-AWARE INCOME PREDICTION MODEL EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Overall performance
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        overall = metrics.get('overall', {})
        for metric, value in overall.items():
            f.write(f"{metric.replace('_', ' ').title():<20}: {value:.4f}\n")

        # Fairness metrics
        f.write("\nFAIRNESS METRICS:\n")
        f.write("-" * 17 + "\n")
        fairness_keys = [
            'demographic_parity_ratio', 'demographic_parity_difference',
            'equalized_odds_difference', 'equal_opportunity_difference'
        ]
        for key in fairness_keys:
            if key in metrics:
                formatted_name = key.replace('_', ' ').title()
                f.write(f"{formatted_name:<30}: {metrics[key]:.4f}\n")

        # Composite score
        if 'composite_score' in metrics:
            f.write(f"\nComposite Score: {metrics['composite_score']:.4f}\n")

        # Target achievement
        target_assessment = metrics.get('target_assessment', {})
        if target_assessment:
            f.write("\nTARGET ACHIEVEMENT:\n")
            f.write("-" * 19 + "\n")
            for key, value in target_assessment.items():
                if key.endswith('_achieved'):
                    metric_name = key.replace('_achieved', '').replace('_', ' ').title()
                    status = "✓ Achieved" if value else "✗ Not achieved"
                    gap = target_assessment.get(key.replace('_achieved', '_gap'), 0.0)
                    f.write(f"{metric_name:<25}: {status:<12} (gap: {gap:+.4f})\n")

        # Group-wise performance
        by_group = metrics.get('by_group', {})
        if by_group:
            f.write("\nGROUP-WISE PERFORMANCE:\n")
            f.write("-" * 23 + "\n")
            for group_name, group_metrics in by_group.items():
                f.write(f"\n{group_name.upper()}:\n")
                for metric, value in group_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric.replace('_', ' ').title():<18}: {value:.4f}\n")

    logger.info(f"Summary report saved to {summary_file}")


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)

        # Load trained model
        logger.info(f"Loading trained model from {args.model_path}")
        trainer = FairnessAwareTrainer(config)
        trainer.load_model(args.model_path)

        # Prepare evaluation data
        if args.test_data:
            # Load custom test data
            logger.info(f"Loading test data from {args.test_data}")
            test_df = pd.read_csv(args.test_data)
            X_test, y_test, protected_test = trainer.preprocessor.transform(test_df)
        else:
            # Use default test split
            logger.info("Using default test data split")
            (_, _, _,
             _, _, _,
             X_test, y_test, protected_test) = trainer.load_and_prepare_data()

        # Evaluate model
        logger.info("Evaluating model performance and fairness")
        metrics = trainer.evaluate_model(X_test, y_test, protected_test, log_metrics=False)

        # Get predictions for additional analysis
        y_pred = trainer.model.predict(X_test)
        y_prob = trainer.model.predict_proba(X_test)[:, 1]

        predictions_data = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'protected_attributes': protected_test
        } if args.save_predictions else {}

        # Save detailed results
        logger.info("Saving evaluation results")
        save_detailed_results(metrics, predictions_data, output_dir)

        # Generate plots if requested
        if args.generate_plots:
            create_fairness_plots(metrics, output_dir)

        # Print summary to console
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)

        # Overall performance
        overall = metrics.get('overall', {})
        logger.info("Overall Performance:")
        for metric, value in overall.items():
            logger.info(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

        # Fairness metrics
        logger.info("\nFairness Metrics:")
        fairness_keys = [
            'demographic_parity_ratio', 'demographic_parity_difference',
            'equalized_odds_difference', 'equal_opportunity_difference'
        ]
        for key in fairness_keys:
            if key in metrics:
                logger.info(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}")

        # Composite score
        if 'composite_score' in metrics:
            logger.info(f"\nComposite Score: {metrics['composite_score']:.4f}")

        # Target achievement
        target_assessment = metrics.get('target_assessment', {})
        if target_assessment:
            logger.info("\nTarget Achievement:")
            for key, value in target_assessment.items():
                if key.endswith('_achieved'):
                    metric_name = key.replace('_achieved', '').replace('_', ' ').title()
                    status = "✓ Achieved" if value else "✗ Not achieved"
                    gap = target_assessment.get(key.replace('_achieved', '_gap'), 0.0)
                    logger.info(f"  {metric_name}: {status} (gap: {gap:+.4f})")

        logger.info(f"\nDetailed results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()