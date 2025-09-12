"""
BaseDataHandler - 데이터 핸들러 기본 인터페이스

각 task type별로 특화된 데이터 처리를 위한 추상 클래스입니다.
- TabularDataHandler: classification, regression, clustering, causal
- TimeseriesDataHandler: timeseries
- DeepLearningDataHandler: 향후 딥러닝용
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
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
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        3-way 데이터 분할 - Train/Validation/Test 분할 (기본 구현)
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            Tuple[train_df, validation_df, test_df]
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train (60%) vs temp (40%)
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        
        # Second split: validation (20%) vs test (20%) from temp
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        return train_df, validation_df, test_df

    def split_and_prepare(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, Any, Dict[str, Any],  # train
        pd.DataFrame, Any, Dict[str, Any],  # validation
        pd.DataFrame, Any, Dict[str, Any],  # test
        Optional[Tuple[pd.DataFrame, Any, Dict[str, Any]]]  # calibration (None for base)
    ]:
        """
        표준화된 4-way interface: 데이터 분할 + 각 분할에 대해 prepare_data 수행
        
        모든 DataHandler가 동일한 형식으로 반환하여 Pipeline에서 일관되게 처리 가능
        Base implementation은 3-way split + calibration=None

        Returns:
            (X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data)
            calibration_data는 None (TabularDataHandler에서만 활성화)
        """
        train_df, validation_df, test_df = self.split_data(df)
        
        # 각 분할에 대해 prepare_data 수행
        X_train, y_train, add_train = self.prepare_data(train_df)
        X_val, y_val, add_val = self.prepare_data(validation_df)
        X_test, y_test, add_test = self.prepare_data(test_df)
        
        # BaseDataHandler는 calibration 미지원 (TabularDataHandler에서 오버라이드)
        calibration_data = None
        
        return X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data
        
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