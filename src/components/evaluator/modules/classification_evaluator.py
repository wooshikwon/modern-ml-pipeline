# src/components/_evaluator/_classification.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: DataInterface):
        self.settings = data_interface_settings

    def evaluate(self, model, X, y, source_df=None):
        predictions = model.predict(X)
        
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average=self.settings.average),
            "recall": recall_score(y, predictions, average=self.settings.average),
            "f1_score": f1_score(y, predictions, average=self.settings.average),
        }
        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("classification", ClassificationEvaluator)
