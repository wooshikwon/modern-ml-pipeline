# src/components/_preprocessor/_steps/_scaler.py
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from .._registry import PreprocessorStepRegistry

class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str] = None):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

class MinMaxScalerWrapper(BaseEstimator, TransformerMixin):
    """scikit-learn의 MinMaxScaler를 위한 래퍼입니다."""
    def __init__(self, columns: List[str] = None):
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

class RobustScalerWrapper(BaseEstimator, TransformerMixin):
    """
    scikit-learn의 RobustScaler를 위한 래퍼입니다.
    중앙값과 사분위수를 사용하여 스케일링하므로, 이상치(outlier)에
    덜 민감하여 모델의 안정성을 높여줍니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

PreprocessorStepRegistry.register("standard_scaler", StandardScalerWrapper)
PreprocessorStepRegistry.register("min_max_scaler", MinMaxScalerWrapper)
PreprocessorStepRegistry.register("robust_scaler", RobustScalerWrapper) 