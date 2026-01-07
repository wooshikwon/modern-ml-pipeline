# Import base class
from .base import BaseEvaluator
from .modules.causal_evaluator import CausalEvaluator

# Import all evaluator modules to trigger self-registration
from .modules.classification_evaluator import ClassificationEvaluator
from .modules.clustering_evaluator import ClusteringEvaluator
from .modules.regression_evaluator import RegressionEvaluator
from .modules.timeseries_evaluator import TimeSeriesEvaluator

# Import the registry for external use
from .registry import EvaluatorRegistry

__all__ = [
    "BaseEvaluator",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "ClusteringEvaluator",
    "CausalEvaluator",
    "TimeSeriesEvaluator",
    "EvaluatorRegistry",
]
