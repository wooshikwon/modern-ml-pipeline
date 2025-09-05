# src/components/_preprocessor/_steps/_encoder.py
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing import List
from src.interface import BasePreprocessor
from ..registry import PreprocessorStepRegistry

class OneHotEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """scikit-learn의 OneHotEncoder를 위한 래퍼입니다."""
    def __init__(self, handle_unknown='ignore', sparse_output=False, columns: List[str] = None):
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=self.sparse_output)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)

class OrdinalEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """scikit-learn의 OrdinalEncoder를 위한 래퍼입니다."""
    def __init__(self, handle_unknown='use_encoded_value', unknown_value=-1, columns: List[str] = None):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.columns = columns
        self.encoder = OrdinalEncoder(handle_unknown=self.handle_unknown, unknown_value=self.unknown_value)
    
    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

class CatBoostEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    category_encoders 라이브러리의 CatBoostEncoder를 위한 scikit-learn 호환 래퍼입니다.
    Target Encoding의 변종으로, 정보 누수를 방지하면서 고유 범주가 많은 변수를
    효과적으로 처리합니다.
    """
    def __init__(self, sigma: float = 0.05, columns: List[str] = None):
        self.sigma = sigma
        self.columns = columns
        self.encoder = CatBoostEncoder(sigma=self.sigma, cols=self.columns)

    def fit(self, X, y=None):
        """
        CatBoostEncoder를 학습시킵니다.
        이 인코더는 지도 학습 방식이므로 반드시 타겟 변수 y가 필요합니다.
        """
        if y is None:
            raise ValueError("CatBoostEncoder requires a target variable 'y' for fitting.")
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        """학습된 인코더를 사용하여 데이터를 변환합니다."""
        return self.encoder.transform(X) 

PreprocessorStepRegistry.register("one_hot_encoder", OneHotEncoderWrapper)
PreprocessorStepRegistry.register("ordinal_encoder", OrdinalEncoderWrapper)
PreprocessorStepRegistry.register("catboost_encoder", CatBoostEncoderWrapper) 