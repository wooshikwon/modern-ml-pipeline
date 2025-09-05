# src/components/_evaluator/_classification.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.interface import BaseEvaluator
from src.settings import DataInterface

class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, data_interface_settings: DataInterface):
        super().__init__(data_interface_settings)

    def evaluate(self, model, X, y, source_df=None):
        predictions = model.predict(X)
        metrics = {"accuracy": accuracy_score(y, predictions)}
        
        # 클래스별 메트릭 (average=None → 클래스별 배열 반환)
        precision_per_class = precision_score(y, predictions, average=None)
        recall_per_class = recall_score(y, predictions, average=None)
        f1_per_class = f1_score(y, predictions, average=None)
        
        # 클래스별 샘플 수
        unique_classes, support_per_class = np.unique(y, return_counts=True)
        
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
