# src/components/_evaluator/_classification.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.components.evaluator.base import BaseEvaluator
from src.settings import Settings
from src.utils.core.logger import log_eval, log_eval_debug, logger


class ClassificationEvaluator(BaseEvaluator):
    METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    DEFAULT_OPTIMIZATION_METRIC = "accuracy"

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def evaluate(self, model, X, y, source_df=None):
        log_eval(f"분류 모델 평가 시작 - {len(X)}샘플, {len(np.unique(y))}클래스")

        predictions = model.predict(X)
        metrics = {"accuracy": accuracy_score(y, predictions)}

        log_eval_debug(f"정확도: {metrics['accuracy']:.4f}")

        # 클래스별 메트릭 (average=None → 클래스별 배열 반환)
        precision_per_class = precision_score(y, predictions, average=None)
        recall_per_class = recall_score(y, predictions, average=None)
        f1_per_class = f1_score(y, predictions, average=None)

        # ROC AUC 계산 - 클래스 수에 따라 다르게 처리
        unique_classes, support_per_class = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        if n_classes == 2:
            # Binary classification - predict_proba 필요
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)[:, 1]  # 클래스 1의 확률
                    metrics["roc_auc"] = roc_auc_score(y, y_proba)
                    log_eval_debug(f"ROC AUC (이진): {metrics['roc_auc']:.4f}")
                else:
                    metrics["roc_auc"] = None
                    logger.warning("[EVAL] predict_proba 미지원 - ROC AUC 스킵")
            except Exception as e:
                metrics["roc_auc"] = None
                logger.warning(f"[EVAL] ROC AUC 계산 오류: {str(e)}")
        else:
            # Multi-class classification - probabilities 필요
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)
                    metrics["roc_auc"] = roc_auc_score(
                        y, y_proba, multi_class="ovr", average="weighted"
                    )
                    log_eval_debug(f"ROC AUC (다중): {metrics['roc_auc']:.4f}")
                else:
                    metrics["roc_auc"] = None
                    logger.warning("[EVAL] predict_proba 미지원 - ROC AUC 스킵")
            except Exception as e:
                metrics["roc_auc"] = None
                logger.warning(f"[EVAL] ROC AUC 계산 오류: {str(e)}")

        # 클래스별 메트릭을 딕셔너리에 추가
        class_metrics_summary = []
        for i, class_label in enumerate(unique_classes):
            metrics[f"class_{class_label}_precision"] = precision_per_class[i]
            metrics[f"class_{class_label}_recall"] = recall_per_class[i]
            metrics[f"class_{class_label}_f1"] = f1_per_class[i]
            metrics[f"class_{class_label}_support"] = int(support_per_class[i])

            class_metrics_summary.append(
                f"Class {class_label}: P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}"
            )

        # 클래스별 상세 로그는 DEBUG 레벨
        for summary in class_metrics_summary:
            log_eval_debug(summary)

        # 평가 완료 요약
        log_eval(
            f"평가 완료 - Acc: {metrics['accuracy']:.4f}, {len(unique_classes)}클래스, {len(metrics)}지표"
        )

        return metrics


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("classification", ClassificationEvaluator)
