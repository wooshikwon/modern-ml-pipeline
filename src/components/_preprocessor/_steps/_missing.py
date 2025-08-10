# src/components/_preprocessor/_steps/_missing.py
from sklearn.impute import MissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import numpy as np
from .._registry import PreprocessorStepRegistry

class MissingIndicatorWrapper(BaseEstimator, TransformerMixin):
    """
    scikit-learn의 MissingIndicator를 위한 래퍼입니다.
    결측치가 있었는지 여부를 나타내는 새로운 바이너리(0/1) 컬럼을 추가하여,
    결측 정보 자체를 모델이 학습할 수 있도록 합니다.
    """
    def __init__(self, columns: List[str] = None, features='missing-only'):
        self.columns = columns # 사용되지는 않지만, recipe와의 일관성을 위해 유지
        self.features = features
        self.indicator = MissingIndicator(features=self.features)

    def fit(self, X, y=None):
        """MissingIndicator를 학습시킵니다."""
        self.indicator.fit(X)
        return self

    def transform(self, X):
        """학습된 Indicator를 사용하여 데이터를 변환합니다."""
        return self.indicator.transform(X).astype(np.int64)

    def get_feature_names_out(self, input_features=None):
        """변환 후의 피처 이름을 반환합니다."""
        return self.indicator.get_feature_names_out(input_features)

PreprocessorStepRegistry.register("missing_indicator", MissingIndicatorWrapper) 