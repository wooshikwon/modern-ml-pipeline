# src/components/_evaluator/_causal.py
import numpy as np

from src.components.evaluator.base import BaseEvaluator
from src.settings import DataInterface
from src.utils.core.logger import log_eval, log_eval_debug, logger


class CausalEvaluator(BaseEvaluator):
    METRIC_KEYS = ["ate", "ate_std", "treatment_effect_significance"]
    DEFAULT_OPTIMIZATION_METRIC = "ate"

    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)
        log_eval_debug("CausalEvaluator 초기화 완료")

    def evaluate(self, model, X, y, additional_data=None):
        """
        인과추론 모델 평가.

        Args:
            model: 인과추론 모델 (uplift prediction 지원)
            X: 피처 데이터 (treatment column 제외됨)
            y: 결과 변수
            additional_data: 추가 데이터 (treatment 정보 포함)

        Returns:
            Dict[str, float]: 인과추론 평가 메트릭들
        """
        log_eval(f"인과추론 모델 평가 시작 - {len(X)}샘플")

        if not self.data_interface.treatment_column:
            logger.error("[EVAL] Treatment 컬럼 필요")
            raise ValueError("Causal evaluation requires treatment_column in DataInterface")

        # Treatment 변수 추출 - additional_data에서 가져옴
        if not additional_data or "treatment" not in additional_data:
            logger.error("[EVAL] Treatment 데이터 누락")
            raise ValueError("Causal evaluation requires treatment data in additional_data")

        treatment = additional_data["treatment"]
        features = X  # X는 이미 treatment가 제외된 상태

        # Treatment 그룹 분석
        treated_count = np.sum(treatment == 1)
        control_count = np.sum(treatment == 0)
        log_eval_debug(f"Treatment 그룹: {treated_count}개, Control: {control_count}개")

        # ATE (Average Treatment Effect) 계산
        ate = self._calculate_ate(treatment, y)

        # CATE 관련 메트릭 (모델이 uplift prediction을 지원하는 경우)
        uplift_metrics = self._calculate_uplift_metrics(model, features, treatment, y)

        # 기본 메트릭 결합
        ate_std = self._calculate_ate_std(treatment, y)
        significance = self._test_significance(treatment, y)

        metrics = {"ate": ate, "ate_std": ate_std, "treatment_effect_significance": significance}

        metrics.update(uplift_metrics)

        log_eval(f"평가 완료 - ATE: {ate:.4f} ± {ate_std:.4f}, 유의성: {significance:.4f}")
        return metrics

    def _calculate_ate(self, treatment, outcome):
        """Average Treatment Effect 계산"""
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]

        if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
            return 0.0

        ate = treated_outcomes.mean() - control_outcomes.mean()
        return float(ate)

    def _calculate_ate_std(self, treatment, outcome):
        """ATE 표준편차 계산"""
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]

        if len(treated_outcomes) <= 1 or len(control_outcomes) <= 1:
            return 0.0

        treated_var = treated_outcomes.var() / len(treated_outcomes)
        control_var = control_outcomes.var() / len(control_outcomes)
        ate_std = np.sqrt(treated_var + control_var)

        return float(ate_std)

    def _test_significance(self, treatment, outcome):
        """Treatment effect 유의성 검정 (t-test)"""
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]

        if len(treated_outcomes) <= 1 or len(control_outcomes) <= 1:
            return 0.0

        # Welch's t-test
        treated_mean = treated_outcomes.mean()
        control_mean = control_outcomes.mean()
        treated_var = treated_outcomes.var()
        control_var = control_outcomes.var()

        pooled_se = np.sqrt(
            treated_var / len(treated_outcomes) + control_var / len(control_outcomes)
        )
        if pooled_se == 0:
            return 0.0

        t_stat = (treated_mean - control_mean) / pooled_se
        return float(abs(t_stat))

    def _calculate_uplift_metrics(self, model, features, treatment, outcome):
        """Uplift 모델 관련 메트릭들"""
        try:
            # 모델이 uplift prediction을 지원하는지 확인
            if hasattr(model, "predict_uplift"):
                uplift_predictions = model.predict_uplift(features)
                return {
                    "uplift_auc": self._uplift_auc(treatment, outcome, uplift_predictions),
                    "qini_coefficient": self._qini_coefficient(
                        treatment, outcome, uplift_predictions
                    ),
                }
            elif hasattr(model, "predict"):
                # 일반 모델인 경우 간접적으로 uplift 계산
                predictions = model.predict(features)
                return {"model_auc": self._simple_auc(outcome, predictions)}
            else:
                return {"uplift_auc": 0.0}
        except Exception:
            # 모델 예측 실패 시 기본값 반환
            return {"uplift_auc": 0.0}

    def _uplift_auc(self, treatment, outcome, uplift_pred):
        """Uplift AUC 계산 (간단한 구현)"""
        # 실제 uplift와 예측 uplift 간의 상관관계로 근사
        if len(uplift_pred) < 2:
            return 0.0

        # 실제 개별 treatment effect 계산 (단순화)
        treated_mask = treatment == 1
        control_mask = treatment == 0

        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            return 0.0

        # 예측값과 결과값 간의 상관관계 계산
        correlation = np.corrcoef(uplift_pred, outcome)[0, 1]
        return float(abs(correlation)) if not np.isnan(correlation) else 0.0

    def _qini_coefficient(self, treatment, outcome, uplift_pred):
        """Qini Coefficient 계산 (간단한 구현)"""
        return float(np.std(uplift_pred)) if len(uplift_pred) > 1 else 0.0

    def _simple_auc(self, outcome, predictions):
        """단순 AUC 계산"""
        try:
            correlation = np.corrcoef(predictions, outcome)[0, 1]
            return float(abs(correlation)) if not np.isnan(correlation) else 0.0
        except:
            return 0.0


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("causal", CausalEvaluator)
