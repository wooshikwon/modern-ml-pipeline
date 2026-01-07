# src/components/_evaluator/_clustering.py
from sklearn.metrics import silhouette_score

from src.components.evaluator.base import BaseEvaluator
from src.settings import DataInterface
from src.utils.core.logger import log_eval, log_eval_debug, logger


class ClusteringEvaluator(BaseEvaluator):
    METRIC_KEYS = ["silhouette_score", "inertia", "n_clusters", "bic", "aic"]
    DEFAULT_OPTIMIZATION_METRIC = "silhouette_score"

    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)
        log_eval_debug("ClusteringEvaluator 초기화 완료")

    def evaluate(self, model, X, y=None, additional_data=None):
        log_eval(f"클러스터링 모델 평가 시작 - {len(X)}샘플")

        labels = model.predict(X)

        # 노이즈 포인트 처리
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        log_eval_debug(f"클러스터 분석: {n_clusters}개 클러스터, 노이즈: {'-1' in unique_labels}")

        metrics = {"n_clusters": n_clusters}

        # Silhouette score (2개 이상 클러스터 필요)
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            metrics["silhouette_score"] = silhouette
            log_eval_debug(f"Silhouette Score: {silhouette:.4f}")
        else:
            logger.warning("[EVAL] 클러스터 수 부족 - Silhouette Score 계산 불가")
            metrics["silhouette_score"] = 0.0

        # Model-specific metrics
        model_specific_metrics = []
        if hasattr(model, "inertia_"):  # KMeans
            metrics["inertia"] = model.inertia_
            model_specific_metrics.append(f"Inertia={model.inertia_:.4f}")
        if hasattr(model, "bic"):  # GaussianMixture
            bic_score = model.bic(X)
            metrics["bic"] = bic_score
            model_specific_metrics.append(f"BIC={bic_score:.4f}")
        if hasattr(model, "aic"):  # GaussianMixture
            aic_score = model.aic(X)
            metrics["aic"] = aic_score
            model_specific_metrics.append(f"AIC={aic_score:.4f}")

        if model_specific_metrics:
            log_eval_debug(f"모델 메트릭: {', '.join(model_specific_metrics)}")

        log_eval(
            f"평가 완료 - {n_clusters}개 클러스터, Silhouette: {metrics.get('silhouette_score', 0):.4f}"
        )
        return metrics


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("clustering", ClusteringEvaluator)
