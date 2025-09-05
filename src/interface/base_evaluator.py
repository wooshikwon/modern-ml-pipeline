from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd

from src.settings import DataInterface


class BaseEvaluator(ABC):
    """
    모든 Evaluator가 따라야 할 표준 계약.
    각 Task Type별로 최적화된 평가 메트릭을 계산하는 Strategy Pattern의 기반.
    """
    
    def __init__(self, data_interface_settings: DataInterface):
        """
        Evaluator 초기화
        
        Args:
            data_interface_settings: DataInterface 설정 객체
        """
        self.settings = data_interface_settings
        self.task_type = data_interface_settings.task_type
    
    @abstractmethod
    def evaluate(self, model, X, y, source_df=None) -> Dict[str, float]:
        """
        모델 평가 메트릭 계산
        
        Args:
            model: 학습된 모델 객체 (외부 라이브러리 모델)
            X: 테스트 피처 데이터
            y: 테스트 타겟 데이터 (clustering의 경우 None 가능)
            source_df: 원본 테스트 데이터프레임 (추가 컬럼 접근용, 선택사항)
            
        Returns:
            Dict[str, float]: 계산된 평가 메트릭들
        """
        pass 