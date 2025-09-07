# src/components/_preprocessor/_steps/_scaler.py
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd
from src.interface import BasePreprocessor
from ..registry import PreprocessorStepRegistry

class StandardScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: StandardScaler를 위한 래퍼
    모든 숫자형 컬럼에 자동으로 표준화를 적용합니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """표준화를 적용하고 DataFrame으로 반환합니다."""
        result_array = self.scaler.transform(X)
        return pd.DataFrame(result_array, index=X.index, columns=X.columns)
    
    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """StandardScaler는 컬럼명을 보존합니다."""
        return input_columns
    
    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존합니다."""
        return True
    
    def get_application_type(self) -> str:
        """StandardScaler는 모든 숫자형 컬럼에 자동 적용됩니다."""
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

class MinMaxScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: MinMaxScaler를 위한 래퍼
    모든 숫자형 컬럼에 자동으로 Min-Max 스케일링을 적용합니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Min-Max 스케일링을 적용하고 DataFrame으로 반환합니다."""
        result_array = self.scaler.transform(X)
        return pd.DataFrame(result_array, index=X.index, columns=X.columns)
    
    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def preserves_column_names(self) -> bool:
        return True
    
    def get_application_type(self) -> str:
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

class RobustScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: RobustScaler를 위한 래퍼
    중앙값과 사분위수를 사용하여 스케일링하므로, 이상치(outlier)에 덜 민감합니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = RobustScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Robust 스케일링을 적용하고 DataFrame으로 반환합니다."""
        result_array = self.scaler.transform(X)
        return pd.DataFrame(result_array, index=X.index, columns=X.columns)
    
    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def preserves_column_names(self) -> bool:
        return True
    
    def get_application_type(self) -> str:
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

PreprocessorStepRegistry.register("standard_scaler", StandardScalerWrapper)
PreprocessorStepRegistry.register("min_max_scaler", MinMaxScalerWrapper)
PreprocessorStepRegistry.register("robust_scaler", RobustScalerWrapper) 