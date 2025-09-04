# Import all evaluator modules to trigger self-registration
from ._modules.classification_evaluator import ClassificationEvaluator
from ._modules.regression_evaluator import RegressionEvaluator
from ._modules.clustering_evaluator import ClusteringEvaluator
from ._modules.causal_evaluator import CausalEvaluator

# Import the registry for external use
from ._registry import EvaluatorRegistry

__all__ = [
    "ClassificationEvaluator",
    "RegressionEvaluator", 
    "ClusteringEvaluator",
    "CausalEvaluator",
    "EvaluatorRegistry",
]