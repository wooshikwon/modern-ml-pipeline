import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Clustering metrics
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

from src.interface.base_evaluator import BaseEvaluator
from src.utils.system.logger import logger


class ClassificationEvaluator(BaseEvaluator):
    """
    분류 모델 전용 평가기
    
    표준 분류 메트릭들을 계산하여 모델 성능을 종합적으로 평가합니다.
    이진 분류와 다중 클래스 분류를 모두 지원합니다.
    """
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> Dict[str, float]:
        """분류 모델 평가 메트릭 계산"""
        logger.info("분류 모델 성능 평가를 시작합니다...")
        
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 기본 메트릭 계산
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average=self.data_interface.average),
            "precision": precision_score(y_test, y_pred, average=self.data_interface.average),
            "recall": recall_score(y_test, y_pred, average=self.data_interface.average),
        }
        
        # 이진 분류인 경우 AUC 추가
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["auc"] = roc_auc_score(y_test, y_proba)
                logger.info("이진 분류 감지: AUC 메트릭 추가")
            except Exception as e:
                logger.warning(f"AUC 계산 실패: {e}")
        
        # class_weight가 설정된 경우 추가 정보 로깅
        if self.data_interface.class_weight:
            logger.info(f"Class weight 설정: {self.data_interface.class_weight}")
        
        logger.info(f"분류 모델 평가 완료: {metrics}")
        return metrics


class RegressionEvaluator(BaseEvaluator):
    """
    회귀 모델 전용 평가기
    
    표준 회귀 메트릭들을 계산하여 모델 성능을 종합적으로 평가합니다.
    """
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> Dict[str, float]:
        """회귀 모델 평가 메트릭 계산"""
        logger.info("회귀 모델 성능 평가를 시작합니다...")
        
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 회귀 메트릭 계산
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        
        # sample_weight가 설정된 경우 추가 정보 로깅
        if self.data_interface.sample_weight_col:
            logger.info(f"Sample weight 컬럼 설정: {self.data_interface.sample_weight_col}")
        
        logger.info(f"회귀 모델 평가 완료: {metrics}")
        return metrics


class ClusteringEvaluator(BaseEvaluator):
    """
    클러스터링 모델 전용 평가기
    
    클러스터링 품질을 평가하는 내재적 메트릭과 외재적 메트릭을 모두 지원합니다.
    """
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> Dict[str, float]:
        """클러스터링 모델 평가 메트릭 계산"""
        logger.info("클러스터링 모델 성능 평가를 시작합니다...")
        
        # 클러스터 예측 (라벨 할당)
        cluster_labels = model.predict(X_test)
        
        # 내재적 메트릭 계산 (실제 라벨 없이도 계산 가능)
        metrics = {
            "silhouette_score": silhouette_score(X_test, cluster_labels),
            "calinski_harabasz": calinski_harabasz_score(X_test, cluster_labels),
        }
        
        # 모델에 inertia 속성이 있으면 추가 (KMeans 등)
        if hasattr(model, 'inertia_'):
            metrics["inertia"] = model.inertia_
        
        # 실제 라벨이 있는 경우 외재적 메트릭 추가
        if self.data_interface.true_labels_col and self.data_interface.true_labels_col in test_df.columns:
            true_labels = test_df[self.data_interface.true_labels_col]
            metrics.update({
                "ari": adjusted_rand_score(true_labels, cluster_labels),
                "ami": adjusted_mutual_info_score(true_labels, cluster_labels),
            })
            logger.info("실제 라벨 발견: 외재적 메트릭 (ARI, AMI) 추가")
        
        # n_clusters 설정 정보 로깅
        if self.data_interface.n_clusters:
            logger.info(f"설정된 클러스터 수: {self.data_interface.n_clusters}")
        
        logger.info(f"클러스터링 모델 평가 완료: {metrics}")
        return metrics


class CausalEvaluator(BaseEvaluator):
    """
    인과추론/업리프트 모델 전용 평가기
    
    기존 trainer.py의 _evaluate 로직을 이동하여 
    ATE(Average Treatment Effect) 계산 등 업리프트 모델링 전용 메트릭을 제공합니다.
    """
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> Dict[str, float]:
        """인과추론/업리프트 모델 평가 메트릭 계산"""
        logger.info("인과추론/업리프트 모델 성능 평가를 시작합니다...")
        
        # 필요한 컬럼들
        treatment_col = self.data_interface.treatment_col
        target_col = self.data_interface.target_col
        treatment_value = self.data_interface.treatment_value
        
        # 업리프트 예측
        uplift_pred = model.predict(X_test)
        
        # ATE 계산
        treatment_mask = test_df[treatment_col] == treatment_value
        control_mask = ~treatment_mask
        
        # 그룹별 샘플이 하나 이상 있는지 확인
        if treatment_mask.sum() > 0 and control_mask.sum() > 0:
            actual_ate = (
                test_df.loc[treatment_mask, target_col].mean()
                - test_df.loc[control_mask, target_col].mean()
            )
        else:
            actual_ate = float("nan")
            logger.warning("처치 또는 통제 그룹 중 하나가 없어 ATE를 계산할 수 없습니다.")
        
        metrics = {
            "actual_ate": actual_ate,
            "predicted_ate": uplift_pred.mean(),
        }
        
        # 그룹별 샘플 수 로깅
        logger.info(f"처치 그룹 샘플 수: {treatment_mask.sum()}")
        logger.info(f"통제 그룹 샘플 수: {control_mask.sum()}")
        logger.info(f"인과추론 모델 평가 완료: {metrics}")
        
        return metrics 