# src/components/_evaluator/_clustering.py
from sklearn.metrics import silhouette_score
from src.interface import BaseEvaluator
from src.settings import DataInterface
from src.utils.core.console_manager import get_console

class ClusteringEvaluator(BaseEvaluator):
    METRIC_KEYS = ["silhouette_score", "inertia", "n_clusters", "bic", "aic"]
    
    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)
        self.console = get_console()
        self.console.info("[ClusteringEvaluator] 초기화 완료되었습니다",
                         rich_message="✅ [ClusteringEvaluator] initialized")

    def evaluate(self, model, X, y=None, additional_data=None):
        self.console.info(f"클러스터링 모델 평가를 시작합니다 - 데이터: {len(X)}개",
                         rich_message="📐 Starting clustering model evaluation")

        # 모든 clustering 모델이 predict를 지원하도록 변경됨
        self.console.log_processing_step(
            "Clustering 레이블 예측",
            "Trained clustering 모델로 데이터 포인트 레이블 할당"
        )
        labels = model.predict(X)
        
        # 노이즈 포인트 처리 (필요시)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        self.console.log_data_operation(
            "Clustering 결과 분석",
            (len(X), n_clusters),
            f"{n_clusters}개 클러스터, 노이즈: {'-1' if -1 in unique_labels else '없음'}"
        )
        
        metrics = {
            "n_clusters": n_clusters
        }
        
        # Silhouette score (2개 이상 클러스터 필요)
        if n_clusters > 1:
            self.console.log_processing_step(
                "Silhouette Score 계산",
                f"{n_clusters}개 클러스터에 대한 내적 코헤시브니스 평가"
            )
            silhouette = silhouette_score(X, labels)
            metrics["silhouette_score"] = silhouette
            self.console.log_model_operation(
                "Silhouette Score 계산 완료",
                f"Score: {silhouette:.4f} (1에 가까울수록 좋음)"
            )
        else:
            self.console.warning(
                "[ClusteringEvaluator] 클러스터 수 부족: Silhouette Score 계산 불가",
                rich_message="⚠️ Insufficient clusters for Silhouette Score"
            )
            metrics["silhouette_score"] = 0.0
        
        # Model-specific metrics
        model_specific_metrics = []
        if hasattr(model, 'inertia_'):  # KMeans
            metrics["inertia"] = model.inertia_
            model_specific_metrics.append(f"Inertia: {model.inertia_:.4f}")
        if hasattr(model, 'bic'):  # GaussianMixture
            bic_score = model.bic(X)
            metrics["bic"] = bic_score
            model_specific_metrics.append(f"BIC: {bic_score:.4f}")
        if hasattr(model, 'aic'):  # GaussianMixture
            aic_score = model.aic(X)
            metrics["aic"] = aic_score
            model_specific_metrics.append(f"AIC: {aic_score:.4f}")

        if model_specific_metrics:
            self.console.log_processing_step(
                "Model-specific 메트릭 계산",
                ", ".join(model_specific_metrics)
            )
        
        self.console.log_model_operation(
            "Clustering 평가 완료",
            f"{n_clusters}개 클러스터, {len(metrics)}개 메트릭 계산 완료"
        )
        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("clustering", ClusteringEvaluator)
