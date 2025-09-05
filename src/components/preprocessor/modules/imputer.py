# src/components/_preprocessor/_steps/_imputer.py
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from src.interface import BasePreprocessor
from ..registry import PreprocessorStepRegistry

class SimpleImputerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = 'mean', columns: List[str] = None):
        self.strategy = strategy
        self.columns = columns
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        # 이 Wrapper는 ColumnTransformer와 함께 사용되므로,
        # fit 에서는 columns 인자를 따로 사용할 필요가 없습니다.
        # ColumnTransformer가 올바른 컬럼만 이 transformeer에 전달해줍니다.
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return self.imputer.transform(X)

PreprocessorStepRegistry.register("simple_imputer", SimpleImputerWrapper) 