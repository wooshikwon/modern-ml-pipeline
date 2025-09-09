# src/components/_evaluator/_clustering.py
from sklearn.metrics import silhouette_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClusteringEvaluator(BaseEvaluator):
    METRIC_KEYS = ["silhouette_score", "inertia", "n_clusters", "bic", "aic"]
    
    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)

    def evaluate(self, model, X, y=None, additional_data=None):
        # 모든 clustering 모델이 predict를 지원하도록 변경됨
        labels = model.predict(X)
        
        # 노이즈 포인트 처리 (필요시)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        metrics = {
            "n_clusters": n_clusters
        }
        
        # Silhouette score (2개 이상 클러스터 필요)
        if n_clusters > 1:
            metrics["silhouette_score"] = silhouette_score(X, labels)
        else:
            metrics["silhouette_score"] = 0.0
        
        # Model-specific metrics
        if hasattr(model, 'inertia_'):  # KMeans
            metrics["inertia"] = model.inertia_
        if hasattr(model, 'bic'):  # GaussianMixture
            metrics["bic"] = model.bic(X)
        if hasattr(model, 'aic'):  # GaussianMixture
            metrics["aic"] = model.aic(X)
        
        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("clustering", ClusteringEvaluator)
