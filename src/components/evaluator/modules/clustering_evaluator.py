# src/components/_evaluator/_clustering.py
from sklearn.metrics import silhouette_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClusteringEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)

    def evaluate(self, model, X, y=None, additional_data=None):
        # 테스트 데이터에 대한 클러스터 예측
        labels = model.predict(X)
        metrics = {
            "silhouette_score": silhouette_score(X, labels),
            "inertia": model.inertia_ if hasattr(model, 'inertia_') else 0.0,
            "n_clusters": len(set(labels))
        }
        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("clustering", ClusteringEvaluator)
