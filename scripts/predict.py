#!/usr/bin/env python3
"""Prediction script for fairness-aware income prediction model."""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairness_aware_income_prediction_with_constraint_optimization.data.loader import load_adult_dataset
from fairness_aware_income_prediction_with_constraint_optimization.data.preprocessing import AdultDataPreprocessor
from fairness_aware_income_prediction_with_constraint_optimization.utils.config import Config


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Make predictions using trained fairness-aware model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='checkpoints/fairness_aware_model.pkl',
        help='Path to trained model pickle file'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file with features (if not provided, uses test data)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (optional)'
    )

    parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Show prediction confidence scores'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of predictions to display'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def load_model(model_path: str) -> Any:
    """Load trained model from pickle file.

    Args:
        model_path: Path to model pickle file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logging.info("Model loaded successfully")
    return model


def load_input_data(input_path: Optional[str], config: Config) -> tuple:
    """Load input data for prediction.

    Args:
        input_path: Path to input CSV file (optional).
        config: Configuration object.

    Returns:
        Tuple of (X, raw_data) where X is preprocessed features and raw_data is original data.
    """
    if input_path:
        logging.info(f"Loading input data from {input_path}")
        data = pd.read_csv(input_path)

        # Initialize preprocessor
        preprocessor = AdultDataPreprocessor(
            protected_attribute=config.get('data.protected_attribute', 'sex'),
            target_column=config.get('data.target_column', 'income')
        )

        # Fit preprocessor (this should ideally be loaded from saved state)
        # For now, we'll use the data as-is
        X = preprocessor.transform(data)

        return X, data
    else:
        logging.info("No input provided, loading test data from UCI Adult dataset")

        # Load the dataset
        X_train, X_val, X_test, y_train, y_val, y_test, protected_train, protected_val, protected_test = load_adult_dataset(
            test_size=config.get('data.test_size', 0.2),
            val_size=config.get('data.val_size', 0.2),
            random_state=config.get('data.random_state', 42),
            protected_attribute=config.get('data.protected_attribute', 'sex'),
            target_column=config.get('data.target_column', 'income')
        )

        # Use test set
        return X_test, pd.DataFrame(X_test)


def make_predictions(model: Any, X: np.ndarray, show_confidence: bool = False) -> Dict[str, np.ndarray]:
    """Make predictions using the model.

    Args:
        model: Trained model.
        X: Input features.
        show_confidence: Whether to include confidence scores.

    Returns:
        Dictionary with predictions and optional confidence scores.
    """
    logging.info(f"Making predictions for {len(X)} samples")

    # Get binary predictions
    predictions = model.predict(X)

    results = {
        'predictions': predictions
    }

    # Get confidence scores if requested
    if show_confidence:
        probabilities = model.predict_proba(X)
        # Confidence is the probability of the predicted class
        confidence = np.max(probabilities, axis=1)
        results['confidence'] = confidence
        results['probabilities'] = probabilities

    logging.info("Predictions completed")
    return results


def display_predictions(results: Dict[str, np.ndarray], top_k: int = 10, show_confidence: bool = False) -> None:
    """Display prediction results.

    Args:
        results: Dictionary with predictions and optional confidence scores.
        top_k: Number of predictions to display.
        show_confidence: Whether to show confidence scores.
    """
    predictions = results['predictions']
    n_samples = len(predictions)

    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS (showing top {min(top_k, n_samples)} of {n_samples})")
    print(f"{'='*60}\n")

    # Prepare display data
    display_data = []
    for i in range(min(top_k, n_samples)):
        row = {
            'Sample': i,
            'Prediction': '>50K' if predictions[i] == 1 else '<=50K'
        }

        if show_confidence:
            confidence = results['confidence'][i]
            prob_high = results['probabilities'][i, 1]
            row['Confidence'] = f"{confidence:.2%}"
            row['Prob(>50K)'] = f"{prob_high:.2%}"

        display_data.append(row)

    # Create DataFrame for nice formatting
    df = pd.DataFrame(display_data)
    print(df.to_string(index=False))

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    n_high_income = np.sum(predictions == 1)
    n_low_income = np.sum(predictions == 0)

    print(f"Total samples: {n_samples}")
    print(f"Predicted >50K: {n_high_income} ({n_high_income/n_samples:.1%})")
    print(f"Predicted <=50K: {n_low_income} ({n_low_income/n_samples:.1%})")

    if show_confidence:
        avg_confidence = np.mean(results['confidence'])
        print(f"Average confidence: {avg_confidence:.2%}")

    print(f"{'='*60}\n")


def save_predictions(results: Dict[str, np.ndarray], output_path: str) -> None:
    """Save predictions to CSV file.

    Args:
        results: Dictionary with predictions and optional confidence scores.
        output_path: Path to save predictions.
    """
    logging.info(f"Saving predictions to {output_path}")

    # Prepare output DataFrame
    output_data = {
        'prediction': ['>=50K' if p == 1 else '<50K' for p in results['predictions']]
    }

    if 'confidence' in results:
        output_data['confidence'] = results['confidence']

    if 'probabilities' in results:
        output_data['prob_low_income'] = results['probabilities'][:, 0]
        output_data['prob_high_income'] = results['probabilities'][:, 1]

    df = pd.DataFrame(output_data)
    df.to_csv(output_path, index=False)

    logging.info(f"Predictions saved to {output_path}")


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.log_level)

    try:
        # Load configuration
        config = Config(args.config)

        # Load model
        model = load_model(args.model_path)

        # Load input data
        X, raw_data = load_input_data(args.input, config)

        # Make predictions
        results = make_predictions(model, X, show_confidence=args.show_confidence)

        # Display results
        display_predictions(results, top_k=args.top_k, show_confidence=args.show_confidence)

        # Save predictions if output path provided
        if args.output:
            save_predictions(results, args.output)

        logging.info("Prediction completed successfully")

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
