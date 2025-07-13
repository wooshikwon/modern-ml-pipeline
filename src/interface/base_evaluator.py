from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

from src.settings.settings import DataInterfaceSettings


class BaseEvaluator(ABC):
    """
    모든 Evaluator가 따라야 할 표준 계약.
    각 Task Type별로 최적화된 평가 메트릭을 계산하는 Strategy Pattern의 기반.
    """
    
    def __init__(self, data_interface: DataInterfaceSettings):
        """
        Evaluator 초기화
        
        Args:
            data_interface: task_type별 설정이 포함된 DataInterfaceSettings
        """
        self.data_interface = data_interface
        self.task_type = data_interface.task_type
    
    @abstractmethod
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        모델 평가 메트릭 계산
        
        Args:
            model: 학습된 모델 객체 (외부 라이브러리 모델)
            X_test: 테스트 피처 데이터
            y_test: 테스트 타겟 데이터 (clustering의 경우 None 가능)
            test_df: 원본 테스트 데이터프레임 (추가 컬럼 접근용)
            
        Returns:
            Dict[str, float]: 계산된 평가 메트릭들
        """
        pass 