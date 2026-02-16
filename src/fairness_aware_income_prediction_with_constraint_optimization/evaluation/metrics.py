"""Evaluation metrics including fairness measures."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
from scipy import stats


logger = logging.getLogger(__name__)


class FairnessMetrics:
    """Comprehensive fairness and performance metrics evaluator.

    This class provides methods to compute various fairness metrics alongside
    traditional performance metrics for binary classification tasks.

    The implemented fairness metrics include:
    - Demographic Parity (Statistical Parity)
    - Equalized Odds
    - Equal Opportunity
    - Calibration metrics
    """

    def __init__(self, protected_attribute_name: str = 'sex') -> None:
        """Initialize the fairness metrics evaluator.

        Args:
            protected_attribute_name: Name of the protected attribute.
        """
        self.protected_attribute_name = protected_attribute_name

    def compute_performance_metrics(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute standard performance metrics.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities (optional).

        Returns:
            Dictionary containing performance metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not compute probability-based metrics: {e}")
                metrics['auc_roc'] = 0.0
                metrics['average_precision'] = 0.0
                metrics['brier_score'] = 1.0

        return metrics

    def compute_group_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            protected_attributes: np.ndarray,
                            y_prob: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each protected group.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.
            y_prob: Predicted probabilities (optional).

        Returns:
            Dictionary with metrics for each group.
        """
        unique_groups = np.unique(protected_attributes)
        group_metrics = {}

        for group in unique_groups:
            group_mask = protected_attributes == group
            group_name = f"group_{group}"

            group_metrics[group_name] = self.compute_performance_metrics(
                y_true[group_mask],
                y_pred[group_mask],
                y_prob[group_mask] if y_prob is not None else None
            )

            # Add group size information
            group_metrics[group_name]['group_size'] = np.sum(group_mask)
            group_metrics[group_name]['positive_rate'] = np.mean(y_true[group_mask])

        return group_metrics

    def compute_demographic_parity(self,
                                 y_pred: np.ndarray,
                                 protected_attributes: np.ndarray) -> Dict[str, float]:
        """Compute demographic parity metrics.

        Demographic parity is satisfied when P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1),
        where Ŷ is the prediction and A is the protected attribute.

        Args:
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.

        Returns:
            Dictionary containing demographic parity metrics.
        """
        unique_groups = np.unique(protected_attributes)
        if len(unique_groups) != 2:
            logger.warning(f"Expected 2 groups, got {len(unique_groups)}. "
                         "Demographic parity calculation may not be meaningful.")

        positive_rates = {}
        for group in unique_groups:
            group_mask = protected_attributes == group
            positive_rates[group] = np.mean(y_pred[group_mask])

        # Calculate ratio and difference
        rates = list(positive_rates.values())
        if len(rates) >= 2:
            dp_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
            dp_difference = abs(rates[0] - rates[1])
        else:
            dp_ratio = 1.0
            dp_difference = 0.0

        return {
            'demographic_parity_ratio': dp_ratio,
            'demographic_parity_difference': dp_difference,
            'positive_rates': positive_rates
        }

    def compute_equalized_odds(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             protected_attributes: np.ndarray) -> Dict[str, float]:
        """Compute equalized odds metrics.

        Equalized odds is satisfied when:
        P(Ŷ = 1 | Y = y, A = 0) = P(Ŷ = 1 | Y = y, A = 1) for y ∈ {0, 1}

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.

        Returns:
            Dictionary containing equalized odds metrics.
        """
        unique_groups = np.unique(protected_attributes)

        # Calculate TPR and FPR for each group
        tprs = {}
        fprs = {}

        for group in unique_groups:
            group_mask = protected_attributes == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]

            # True Positive Rate (Sensitivity)
            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
            fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # False Positive Rate
            fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
            tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            tprs[group] = tpr
            fprs[group] = fpr

        # Calculate differences
        tpr_values = list(tprs.values())
        fpr_values = list(fprs.values())

        tpr_difference = abs(tpr_values[0] - tpr_values[1]) if len(tpr_values) >= 2 else 0.0
        fpr_difference = abs(fpr_values[0] - fpr_values[1]) if len(fpr_values) >= 2 else 0.0

        # Equalized odds difference (average of TPR and FPR differences)
        eo_difference = (tpr_difference + fpr_difference) / 2

        return {
            'equalized_odds_difference': eo_difference,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'tprs': tprs,
            'fprs': fprs
        }

    def compute_equal_opportunity(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                protected_attributes: np.ndarray) -> Dict[str, float]:
        """Compute equal opportunity metrics.

        Equal opportunity is satisfied when:
        P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.

        Returns:
            Dictionary containing equal opportunity metrics.
        """
        equalized_odds_metrics = self.compute_equalized_odds(y_true, y_pred, protected_attributes)

        return {
            'equal_opportunity_difference': equalized_odds_metrics['tpr_difference'],
            'tprs': equalized_odds_metrics['tprs']
        }

    def compute_calibration_metrics(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  protected_attributes: np.ndarray,
                                  n_bins: int = 10) -> Dict[str, Union[float, Dict]]:
        """Compute calibration metrics for each protected group.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            protected_attributes: Protected attribute values.
            n_bins: Number of bins for calibration curve.

        Returns:
            Dictionary containing calibration metrics.
        """
        unique_groups = np.unique(protected_attributes)
        calibration_metrics = {}

        for group in unique_groups:
            group_mask = protected_attributes == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]

            if len(group_y_true) == 0:
                continue

            try:
                # Compute calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    group_y_true, group_y_prob, n_bins=n_bins
                )

                # Compute calibration score (average absolute difference)
                calibration_score = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

                # Compute expected calibration error
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]

                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()

                    if prop_in_bin > 0:
                        accuracy_in_bin = group_y_true[in_bin].mean()
                        avg_confidence_in_bin = group_y_prob[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                calibration_metrics[f'group_{group}'] = {
                    'calibration_score': calibration_score,
                    'expected_calibration_error': ece,
                    'brier_score': brier_score_loss(group_y_true, group_y_prob)
                }

            except Exception as e:
                logger.warning(f"Could not compute calibration metrics for group {group}: {e}")
                calibration_metrics[f'group_{group}'] = {
                    'calibration_score': np.inf,
                    'expected_calibration_error': np.inf,
                    'brier_score': np.inf
                }

        # Overall calibration difference
        if len(calibration_metrics) >= 2:
            cal_scores = [metrics['calibration_score'] for metrics in calibration_metrics.values()]
            calibration_difference = abs(cal_scores[0] - cal_scores[1])
        else:
            calibration_difference = 0.0

        return {
            'calibration_difference': calibration_difference,
            'group_calibration': calibration_metrics
        }

    def compute_all_fairness_metrics(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   protected_attributes: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict]]:
        """Compute all fairness metrics.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.
            y_prob: Predicted probabilities (optional).

        Returns:
            Dictionary containing all fairness metrics.
        """
        fairness_metrics = {}

        # Performance metrics overall and by group
        fairness_metrics['overall'] = self.compute_performance_metrics(y_true, y_pred, y_prob)
        fairness_metrics['by_group'] = self.compute_group_metrics(
            y_true, y_pred, protected_attributes, y_prob
        )

        # Fairness-specific metrics
        fairness_metrics.update(self.compute_demographic_parity(y_pred, protected_attributes))
        fairness_metrics.update(self.compute_equalized_odds(y_true, y_pred, protected_attributes))
        fairness_metrics.update(self.compute_equal_opportunity(y_true, y_pred, protected_attributes))

        if y_prob is not None:
            calibration_metrics = self.compute_calibration_metrics(
                y_true, y_prob, protected_attributes
            )
            fairness_metrics.update(calibration_metrics)

        return fairness_metrics

    def compute_fairness_composite_score(self,
                                       metrics: Dict[str, float],
                                       accuracy_weight: float = 0.5,
                                       fairness_weight: float = 0.5) -> float:
        """Compute a composite score combining accuracy and fairness.

        Args:
            metrics: Dictionary containing computed metrics.
            accuracy_weight: Weight for accuracy component.
            fairness_weight: Weight for fairness component.

        Returns:
            Composite score (higher is better).
        """
        # Normalize weights
        total_weight = accuracy_weight + fairness_weight
        accuracy_weight /= total_weight
        fairness_weight /= total_weight

        # Get accuracy component (AUC if available, otherwise accuracy)
        # Check both top-level and nested 'overall' dict
        overall = metrics.get('overall', {})
        accuracy_component = metrics.get('auc_roc',
                             overall.get('auc_roc',
                             metrics.get('accuracy',
                             overall.get('accuracy', 0.0))))

        # Get fairness component (average of demographic parity and equalized odds)
        dp_ratio = metrics.get('demographic_parity_ratio', 0.0)
        eo_diff = metrics.get('equalized_odds_difference', 1.0)

        # Convert equalized odds difference to a ratio (closer to 0 is better)
        eo_ratio = max(0.0, 1.0 - eo_diff)

        # Fairness component is the average of the two ratios
        fairness_component = (dp_ratio + eo_ratio) / 2

        # Composite score
        composite_score = (accuracy_weight * accuracy_component +
                         fairness_weight * fairness_component)

        return composite_score

    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      protected_attributes: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, Dict, bool]]:
        """Comprehensive model evaluation with fairness assessment.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            protected_attributes: Protected attribute values.
            y_prob: Predicted probabilities (optional).
            target_metrics: Target metric values for assessment.

        Returns:
            Dictionary containing comprehensive evaluation results.
        """
        # Compute all metrics
        metrics = self.compute_all_fairness_metrics(y_true, y_pred, protected_attributes, y_prob)

        # Compute composite score
        composite_score = self.compute_fairness_composite_score(metrics)
        metrics['composite_score'] = composite_score

        # Assess target achievement if provided
        if target_metrics:
            target_assessment = {}
            for metric_name, target_value in target_metrics.items():
                actual_value = metrics.get(metric_name)
                if actual_value is not None:
                    target_assessment[f"{metric_name}_achieved"] = actual_value >= target_value
                    target_assessment[f"{metric_name}_gap"] = target_value - actual_value

            metrics['target_assessment'] = target_assessment

        return metrics

    def create_metrics_summary(self, metrics: Dict[str, Union[float, Dict]]) -> pd.DataFrame:
        """Create a summary DataFrame of key metrics.

        Args:
            metrics: Dictionary containing computed metrics.

        Returns:
            DataFrame with metric summary.
        """
        summary_data = []

        # Overall performance metrics
        overall_metrics = metrics.get('overall', {})
        for metric, value in overall_metrics.items():
            summary_data.append({
                'Category': 'Performance',
                'Metric': metric,
                'Value': value
            })

        # Fairness metrics
        fairness_keys = [
            'demographic_parity_ratio', 'demographic_parity_difference',
            'equalized_odds_difference', 'equal_opportunity_difference'
        ]

        for key in fairness_keys:
            if key in metrics:
                summary_data.append({
                    'Category': 'Fairness',
                    'Metric': key,
                    'Value': metrics[key]
                })

        # Composite score
        if 'composite_score' in metrics:
            summary_data.append({
                'Category': 'Composite',
                'Metric': 'composite_score',
                'Value': metrics['composite_score']
            })

        return pd.DataFrame(summary_data)