"""
Fairness-Aware Income Prediction with Constraint Optimization

A machine learning project that develops an income classifier optimizing for both
predictive accuracy and demographic parity using constrained gradient boosting.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .data.loader import AdultIncomeLoader
from .data.preprocessing import FairnessAwarePreprocessor
from .models.model import FairnessConstrainedLGBM
from .training.trainer import FairnessAwareTrainer
from .evaluation.metrics import FairnessMetrics
from .utils.config import Config

__all__ = [
    "AdultIncomeLoader",
    "FairnessAwarePreprocessor",
    "FairnessConstrainedLGBM",
    "FairnessAwareTrainer",
    "FairnessMetrics",
    "Config"
]