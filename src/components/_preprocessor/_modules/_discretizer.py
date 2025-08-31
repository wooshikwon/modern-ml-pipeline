# src/components/_preprocessor/_steps/_discretizer.py
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from .._registry import PreprocessorStepRegistry

class KBinsDiscretizerWrapper(BaseEstimator, TransformerMixin):
    """
    scikit-learn의 KBinsDiscretizer를 위한 래퍼입니다.
    연속적인 수치형 변수를 여러 구간(bin)으로 나누어 범주형 변수처럼 만듭니다.
    이를 통해 모델이 비선형 관계를 더 쉽게 학습하도록 돕습니다.
    """
    def __init__(self, n_bins: int = 5, encode: str = 'ordinal', strategy: str = 'quantile', columns: List[str] = None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.columns = columns
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy, subsample=None)

    def fit(self, X, y=None):
        """KBinsDiscretizer를 학습시킵니다."""
        self.discretizer.fit(X)
        return self

    def transform(self, X):
        """학습된 Discretizer를 사용하여 데이터를 변환합니다."""
        return self.discretizer.transform(X)

PreprocessorStepRegistry.register("kbins_discretizer", KBinsDiscretizerWrapper) 