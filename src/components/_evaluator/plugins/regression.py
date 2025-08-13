# src/components/_evaluator/_steps/regression.py
from sklearn.metrics import r2_score, mean_squared_error
from src.interface import BaseEvaluator
from src.settings._recipe_schema import MLTaskSettings
from src.engine import EvaluatorRegistry

class RegressionEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: MLTaskSettings):
        self.settings = data_interface_settings

    def evaluate(self, model, X, y, source_df=None):
        predictions = model.predict(X)
        metrics = {
            "r2_score": r2_score(y, predictions),
            "mean_squared_error": mean_squared_error(y, predictions),
        }
        return metrics

# Register plugin
EvaluatorRegistry.register("regression", RegressionEvaluator)