# src/components/_preprocessor/_steps/_tree_generator.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
import pandas as pd
import numpy as np
from typing import List
from src.interface import BasePreprocessor
from ..registry import PreprocessorStepRegistry

class TreeBasedFeatureGenerator(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    트리 앙상블 모델을 사용하여 피처를 생성합니다.
    각 데이터 포인트가 트리의 어떤 '잎사귀' 노드에 도달하는지를 원-핫 인코딩하여
    새로운 피처로 만듭니다. 이를 통해 피처 간의 비선형 상호작용을 포착합니다.
    """
    def __init__(self, n_estimators: int = 10, max_depth: int = 3, random_state: int = 42, columns: List[str] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.columns = columns
        self.tree_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.one_hot_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        트리 모델을 학습하고, 그 결과를 바탕으로 원-핫 인코더를 학습시킵니다.
        이 변환기는 지도 학습 방식이므로 반드시 타겟 변수 y가 필요합니다.
        """
        if y is None:
            raise ValueError("TreeBasedFeatureGenerator requires a target variable 'y' for fitting.")

        self.tree_model_.fit(X, y)
        leaf_indices = self.tree_model_.apply(X)
        self.one_hot_encoder_.fit(leaf_indices)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """학습된 트리와 인코더를 사용하여 데이터를 변환합니다."""
        leaf_indices = self.tree_model_.apply(X)
        return self.one_hot_encoder_.transform(leaf_indices)

    def get_feature_names_out(self, input_features=None):
        """변환 후의 피처 이름을 반환합니다."""
        return self.one_hot_encoder_.get_feature_names_out()

class PolynomialFeaturesWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    scikit-learn의 PolynomialFeatures를 위한 래퍼입니다.
    기존 피처들을 조합하여 고차항(e.g., x^2, xy)을 새로운 피처로 추가하여,
    모델이 비선형 관계를 학습하는 데 도움을 줍니다.
    """
    def __init__(self, degree: int = 2, include_bias: bool = False, interaction_only: bool = False, columns: List[str] = None):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.columns = columns
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )

    def fit(self, X, y=None):
        self.poly.fit(X)
        return self

    def transform(self, X):
        return self.poly.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.poly.get_feature_names_out(input_features)

PreprocessorStepRegistry.register("tree_based_feature_generator", TreeBasedFeatureGenerator)
PreprocessorStepRegistry.register("polynomial_features", PolynomialFeaturesWrapper) 