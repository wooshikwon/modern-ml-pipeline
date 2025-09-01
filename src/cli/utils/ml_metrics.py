"""
평가 메트릭 및 튜닝 설정 관리
Registry 패턴으로 실제 Evaluator 구현에서 메트릭을 동적으로 추출
"""

import inspect
import re
from typing import Dict, List, Any, Type
import numpy as np

from src.interface import BaseEvaluator
from src.components._evaluator import (
    ClassificationEvaluator, 
    RegressionEvaluator,
    ClusteringEvaluator, 
    CausalEvaluator
)
from src.settings import DataInterface


# Evaluator Registry
EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    'classification': ClassificationEvaluator,
    'regression': RegressionEvaluator, 
    'clustering': ClusteringEvaluator,
    'causal': CausalEvaluator
}

# Task별 튜닝 설정
TUNING_CONFIGS: Dict[str, Dict[str, Any]] = {
    'classification': {
        'objective': 'f1_score',
        'direction': 'maximize',
        'n_trials': 100,
        'cv_folds': 5,
        'stratify': True
    },
    'regression': {
        'objective': 'r2_score', 
        'direction': 'maximize',
        'n_trials': 100,
        'cv_folds': 5,
        'stratify': False
    },
    'clustering': {
        'objective': 'silhouette_score',
        'direction': 'maximize', 
        'n_trials': 50,
        'cv_folds': 3,
        'stratify': False
    },
    'causal': {
        'objective': 'uplift_auc',
        'direction': 'maximize',
        'n_trials': 100,
        'cv_folds': 5,
        'stratify': False
    }
}


def _extract_metrics_from_evaluator(evaluator_class: Type[BaseEvaluator]) -> List[str]:
    """
    Evaluator 클래스의 evaluate 메서드에서 반환하는 메트릭 키들을 추출.
    
    Args:
        evaluator_class: Evaluator 클래스
        
    Returns:
        List[str]: 메트릭 키 리스트
    """
    try:
        # evaluate 메서드의 소스 코드에서 metrics dict 키들을 파싱
        source = inspect.getsource(evaluator_class.evaluate)
        
        # 정규표현식으로 metrics 딕셔너리 키 추출
        pattern = r'"([^"]+)":\s*[^,}]+'  # "key": value 패턴
        matches = re.findall(pattern, source)
        
        if matches:
            return matches
        
        # 패턴 매칭 실패 시 fallback: 실제 더미 호출 시도
        return _get_metrics_by_dummy_call(evaluator_class)
        
    except Exception:
        # 모든 방법 실패 시 빈 리스트 반환
        return []


def _get_metrics_by_dummy_call(evaluator_class: Type[BaseEvaluator]) -> List[str]:
    """
    더미 데이터로 evaluate 호출하여 실제 메트릭 키 추출.
    
    Args:
        evaluator_class: Evaluator 클래스
        
    Returns:
        List[str]: 메트릭 키 리스트
    """
    try:
        # Mock 설정으로 임시 evaluator 인스턴스 생성
        mock_settings = DataInterface(
            task_type='classification',
            target_column='target'
        )
        evaluator = evaluator_class(mock_settings)
        
        # 더미 모델과 데이터 생성
        class DummyModel:
            def predict(self, X):
                # 분류: 0/1, 회귀: 랜덤 값
                if hasattr(evaluator, 'settings') and evaluator.settings.task_type == 'regression':
                    return np.random.random(len(X))
                return np.random.randint(0, 2, len(X))
            
            @property
            def labels_(self):  # 클러스터링용
                return np.array([0, 1, 0, 1])
        
        dummy_model = DummyModel()
        X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_dummy = np.array([0, 1, 0, 1])
        
        # evaluate 호출하여 메트릭 키 추출
        metrics = evaluator.evaluate(dummy_model, X_dummy, y_dummy)
        return list(metrics.keys()) if isinstance(metrics, dict) else []
        
    except Exception:
        return []


def get_task_metrics(task_type: str) -> List[str]:
    """
    Registry에서 실제 Evaluator 구현으로부터 메트릭을 동적으로 추출.
    
    Args:
        task_type: ML task 타입 ('classification', 'regression', 등)
        
    Returns:
        List[str]: 해당 task의 메트릭 리스트
        
    Raises:
        ValueError: 지원하지 않는 task_type인 경우
    """
    if task_type.lower() not in EVALUATOR_REGISTRY:
        supported_tasks = list(EVALUATOR_REGISTRY.keys())
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {supported_tasks}")
    
    evaluator_class = EVALUATOR_REGISTRY[task_type.lower()]
    metrics = _extract_metrics_from_evaluator(evaluator_class)
    
    # 메트릭 추출 실패 시 최소 fallback (동적 추출 실패는 드물어야 함)
    if not metrics:
        from src.utils.system.logger import logger
        logger.warning(f"Dynamic metric extraction failed for {task_type}, using minimal fallback")
        # 최소한의 공통 메트릭만 제공
        common_fallbacks = {
            'classification': ['accuracy', 'f1_score'], 
            'regression': ['r2_score'],
            'clustering': ['silhouette_score'],
            'causal': ['uplift_auc']
        }
        metrics = common_fallbacks.get(task_type.lower(), [])
    
    return metrics


def get_tuning_config(task_type: str) -> Dict[str, Any]:
    """
    Task 타입에 맞는 하이퍼파라미터 튜닝 설정 반환.
    
    Args:
        task_type: ML task 타입
        
    Returns:
        Dict[str, Any]: 튜닝 설정 딕셔너리
        
    Raises:
        ValueError: 지원하지 않는 task_type인 경우
    """
    if task_type.lower() not in TUNING_CONFIGS:
        supported_tasks = list(TUNING_CONFIGS.keys())
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {supported_tasks}")
    
    return TUNING_CONFIGS[task_type.lower()].copy()


def get_primary_metric(task_type: str) -> str:
    """
    Task의 주요 최적화 메트릭 반환.
    
    Args:
        task_type: ML task 타입
        
    Returns:
        str: 주요 최적화 메트릭명
    """
    tuning_config = get_tuning_config(task_type)
    return tuning_config['objective']


def validate_custom_metrics(task_type: str, custom_metrics: List[str]) -> List[str]:
    """
    사용자 정의 메트릭이 task에 적합한지 검증.
    
    Args:
        task_type: ML task 타입
        custom_metrics: 사용자 정의 메트릭 리스트
        
    Returns:
        List[str]: 검증된 메트릭 리스트 (유효하지 않은 메트릭 제외)
    """
    available_metrics = get_task_metrics(task_type)
    
    valid_metrics = []
    for metric in custom_metrics:
        if metric in available_metrics:
            valid_metrics.append(metric)
    
    return valid_metrics


def get_evaluator_class(task_type: str) -> Type[BaseEvaluator]:
    """
    Task 타입에 해당하는 Evaluator 클래스 반환.
    
    Args:
        task_type: ML task 타입
        
    Returns:
        Type[BaseEvaluator]: Evaluator 클래스
        
    Raises:
        ValueError: 지원하지 않는 task_type인 경우
    """
    if task_type.lower() not in EVALUATOR_REGISTRY:
        supported_tasks = list(EVALUATOR_REGISTRY.keys())
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {supported_tasks}")
    
    return EVALUATOR_REGISTRY[task_type.lower()]