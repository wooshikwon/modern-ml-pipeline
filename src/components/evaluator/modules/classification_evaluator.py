# src/components/_evaluator/_classification.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClassificationEvaluator(BaseEvaluator):
    METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)

    def evaluate(self, model, X, y, source_df=None):
        predictions = model.predict(X)
        metrics = {"accuracy": accuracy_score(y, predictions)}
        
        # 클래스별 메트릭 (average=None → 클래스별 배열 반환)
        precision_per_class = precision_score(y, predictions, average=None)
        recall_per_class = recall_score(y, predictions, average=None)
        f1_per_class = f1_score(y, predictions, average=None)
        
        # ROC AUC 계산 - 클래스 수에 따라 다르게 처리
        unique_classes, support_per_class = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification - predictions 사용 가능
            metrics["roc_auc"] = roc_auc_score(y, predictions)
        else:
            # Multi-class classification - probabilities 필요
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                else:
                    # predict_proba가 없으면 roc_auc 계산 불가
                    metrics["roc_auc"] = None
            except Exception:
                # 오류 발생 시 None 설정
                metrics["roc_auc"] = None
        
        # 클래스별 메트릭을 딕셔너리에 추가
        for i, class_label in enumerate(unique_classes):
            metrics[f"class_{class_label}_precision"] = precision_per_class[i]
            metrics[f"class_{class_label}_recall"] = recall_per_class[i]
            metrics[f"class_{class_label}_f1"] = f1_per_class[i]
            metrics[f"class_{class_label}_support"] = int(support_per_class[i])
        
        return metrics

# Self-registration
from ..registry import EvaluatorRegistry
EvaluatorRegistry.register("classification", ClassificationEvaluator)
