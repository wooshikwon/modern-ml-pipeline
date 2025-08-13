# src/components/_evaluator/_steps/clustering.py
from sklearn.metrics import silhouette_score
from src.interface import BaseEvaluator
from src.settings._recipe_schema import MLTaskSettings
from src.engine import EvaluatorRegistry

class ClusteringEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: MLTaskSettings):
        self.settings = data_interface_settings

    def evaluate(self, model, X, y=None, source_df=None):
        labels = model.labels_
        metrics = {
            "silhouette_score": silhouette_score(X, labels),
        }
        return metrics

# Register plugin
EvaluatorRegistry.register("clustering", ClusteringEvaluator)