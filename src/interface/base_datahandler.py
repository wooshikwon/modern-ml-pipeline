"""
BaseDataHandler - 데이터 핸들러 기본 인터페이스

각 task type별로 특화된 데이터 처리를 위한 추상 클래스입니다.
- TabularDataHandler: classification, regression, clustering, causal
- TimeseriesDataHandler: timeseries
- DeepLearningDataHandler: 향후 딥러닝용
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
from src.settings import Settings


class BaseDataHandler(ABC):
    """데이터 핸들러 기본 인터페이스"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_interface = settings.recipe.data.data_interface
        
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        데이터 준비 - 각 task type에 특화된 데이터 처리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            Tuple[X, y, additional_data]
            - X: 특성 데이터프레임
            - y: 타겟 Series (clustering의 경우 None)
            - additional_data: 추가 정보 딕셔너리
        """
        pass
        
    @abstractmethod 
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train/Test 분할 - 각 task type에 맞는 분할 방법
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            Tuple[train_df, test_df]
        """
        pass

    def split_and_prepare(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, Any, Dict[str, Any],
        pd.DataFrame, Any, Dict[str, Any]
    ]:
        """
        편의 메서드: 데이터 분할 + 각 분할에 대해 prepare_data 수행.

        Returns:
            (X_train, y_train, add_train, X_test, y_test, add_test)
        """
        train_df, test_df = self.split_data(df)
        X_train, y_train, add_train = self.prepare_data(train_df)
        X_test, y_test, add_test = self.prepare_data(test_df)
        return X_train, y_train, add_train, X_test, y_test, add_test
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        데이터 검증 (각 handler별로 구체화)
        
        Args:
            df: 검증할 데이터프레임
            
        Returns:
            검증 통과 여부
        """
        if df.empty:
            raise ValueError("데이터프레임이 비어있습니다")
        return True
        
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 메타 정보 반환
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            데이터 메타 정보 딕셔너리
        """
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_ratio": (df.isnull().sum() / len(df)).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }