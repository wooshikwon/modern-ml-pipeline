# Import all evaluator modules to trigger self-registration
from .modules.classification_evaluator import ClassificationEvaluator
from .modules.regression_evaluator import RegressionEvaluator
from .modules.clustering_evaluator import ClusteringEvaluator
from .modules.causal_evaluator import CausalEvaluator

# Import the registry for external use
from .registry import EvaluatorRegistry

__all__ = [
    "ClassificationEvaluator",
    "RegressionEvaluator", 
    "ClusteringEvaluator",
    "CausalEvaluator",
    "EvaluatorRegistry",
]