# src/components/_preprocessor/_steps/_tree_generator.py
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from src.components.preprocessor.base import BasePreprocessor

from ..registry import PreprocessorStepRegistry


class TreeBasedFeatureGenerator(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    트리 앙상블 모델을 사용하여 피처를 생성합니다.
    각 데이터 포인트가 트리의 어떤 '잎사귀' 노드에 도달하는지를 원-핫 인코딩하여
    새로운 피처로 만듭니다. 이를 통해 피처 간의 비선형 상호작용을 포착합니다.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 3,
        random_state: int = 42,
        columns: List[str] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.columns = columns
        self.tree_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.one_hot_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        트리 모델을 학습하고, 그 결과를 바탕으로 원-핫 인코더를 학습시킵니다.
        이 변환기는 지도 학습 방식이므로 반드시 타겟 변수 y가 필요합니다.
        """
        if y is None:
            raise ValueError(
                "TreeBasedFeatureGenerator requires a target variable 'y' for fitting."
            )

        self.tree_model_.fit(X, y)
        leaf_indices = self.tree_model_.apply(X)
        self.one_hot_encoder_.fit(leaf_indices)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 트리와 인코더를 사용하여 데이터를 변환하고 DataFrame으로 반환합니다."""
        leaf_indices = self.tree_model_.apply(X)
        result_array = self.one_hot_encoder_.transform(leaf_indices)

        # 컬럼명 생성 정책: treefeature_ 접두사 사용
        try:
            feature_names = self.one_hot_encoder_.get_feature_names_out()
            # sklearn feature names을 더 명확하게 만들기
            new_feature_names = [f"treefeature_{name}" for name in feature_names]
        except Exception:
            # 폴백: 수동으로 생성
            new_feature_names = [f"treefeature_{i}" for i in range(result_array.shape[1])]

        return pd.DataFrame(result_array, index=X.index, columns=new_feature_names)

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """변환 후 예상되는 출력 컬럼명을 반환합니다."""
        # 대략적인 추정값 (정확한 값은 학습 후에만 알 수 있음)
        estimated_features = self.n_estimators * (2**self.max_depth)
        return [f"treefeature_{i}" for i in range(estimated_features)]

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존하지 않습니다."""
        return False

    def get_application_type(self) -> str:
        """Tree Feature Generator는 특정 숫자형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


class PolynomialFeaturesWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 PolynomialFeatures를 위한 래퍼
    기존 피처들을 조합하여 고차항(e.g., x^2, xy)을 새로운 피처로 추가하여,
    모델이 비선형 관계를 학습하는 데 도움을 줍니다.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        columns: List[str] = None,
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.columns = columns
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
        )
        self._input_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self._input_columns = list(X.columns)
        self.poly.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """다항식 피처를 생성하고 DataFrame으로 반환합니다."""
        result_array = self.poly.transform(X)

        # sklearn의 get_feature_names_out을 사용하여 실제 출력 컬럼명 확인
        try:
            feature_names = self.poly.get_feature_names_out(list(X.columns))
            # polynomial feature names에 접두사 추가
            new_feature_names = [f"poly_{name}" for name in feature_names]
        except Exception:
            # 폴백: 수동으로 생성
            new_feature_names = [f"poly_feature_{i}" for i in range(result_array.shape[1])]

        return pd.DataFrame(result_array, index=X.index, columns=new_feature_names)

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """변환 후 예상되는 출력 컬럼명을 반환합니다."""
        if self._input_columns is not None:
            try:
                feature_names = self.poly.get_feature_names_out(self._input_columns)
                return [f"poly_{name}" for name in feature_names]
            except Exception:
                pass
        # 폴백: 추정값 반환
        return [f"poly_feature_{i}" for i in range(len(input_columns) * self.degree)]

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존하지 않습니다."""
        return False

    def get_application_type(self) -> str:
        """Polynomial Features는 특정 숫자형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


PreprocessorStepRegistry.register("tree_based_feature_generator", TreeBasedFeatureGenerator)
PreprocessorStepRegistry.register("polynomial_features", PolynomialFeaturesWrapper)
