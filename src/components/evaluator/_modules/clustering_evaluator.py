# src/components/_evaluator/_clustering.py
from sklearn.metrics import silhouette_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClusteringEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: DataInterface):
        self.settings = data_interface_settings

    def evaluate(self, model, X, y=None, source_df=None):
        labels = model.labels_
        metrics = {
            "silhouette_score": silhouette_score(X, labels),
        }
        return metrics

# Self-registration
from .._registry import EvaluatorRegistry
EvaluatorRegistry.register("clustering", ClusteringEvaluator)
