# src/components/_preprocessor/_steps/_encoder.py
from typing import List

import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.components.preprocessor.base import BasePreprocessor
from src.utils.core.logger import logger

from ..registry import PreprocessorStepRegistry


class OneHotEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 OneHotEncoder를 위한 래퍼
    범주형 변수를 원-핫 인코딩으로 변환하며, 새로운 컬럼들을 생성합니다.
    """

    def __init__(self, handle_unknown="ignore", sparse_output=False, columns: List[str] = None):
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.columns = columns
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown, sparse_output=self.sparse_output
        )
        self._input_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self._input_columns = list(X.columns)
        try:
            self.encoder.fit(X)
        except ValueError as e:
            error_msg = str(e)
            if "handle_unknown" in error_msg.lower():
                logger.error(f"[OneHotEncoder] handle_unknown 설정 오류: {self.handle_unknown}")
                raise ValueError(
                    f"OneHotEncoder handle_unknown '{self.handle_unknown}' 설정에 문제가 있습니다.\n"
                    f"사용 가능한 handle_unknown:\n"
                    f"- 'error': 새로운 범주 발견 시 에러 발생\n"
                    f"- 'ignore': 새로운 범주 무시 (모든 원-핫 컬럼이 0)\n"
                    f"- 'infrequent_if_exist': infrequent 범주로 처리\n"
                    f"원본 오류: {error_msg}"
                ) from e
            else:
                raise
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """원-핫 인코딩을 적용하고 DataFrame으로 반환합니다."""
        result_array = self.encoder.transform(X)

        # sparse matrix 처리 (방어적 코딩)
        if hasattr(result_array, "toarray"):
            result_array = result_array.toarray()

        # sklearn의 get_feature_names_out을 사용하여 실제 출력 컬럼명 확인
        try:
            actual_feature_names = self.encoder.get_feature_names_out(list(X.columns))
        except Exception:
            # 폴백: 수동으로 생성
            actual_feature_names = [
                f"onehot_{col}_{i}"
                for col in X.columns
                for i in range(result_array.shape[1] // len(X.columns))
            ]

        result_df = pd.DataFrame(result_array, index=X.index, columns=actual_feature_names)
        return result_df

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """변환 후 예상되는 출력 컬럼명을 반환합니다."""
        if self._input_columns is not None:
            try:
                return list(self.encoder.get_feature_names_out(self._input_columns))
            except Exception:
                pass
        # 폴백: 추정값 반환
        return [f"onehot_{col}" for col in input_columns]

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존하지 않습니다."""
        return False

    def get_application_type(self) -> str:
        """OneHot Encoder는 특정 범주형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """범주형 컬럼만 대상으로 합니다."""
        return [
            col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "category"
        ]


class OrdinalEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 OrdinalEncoder를 위한 래퍼
    범주형 변수를 순서형 숫자로 인코딩하며, 컬럼명을 보존합니다.
    """

    def __init__(self, handle_unknown="error", unknown_value=None, columns: List[str] = None):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.columns = columns

        # sklearn 호환성: handle_unknown='error'일 때는 unknown_value 파라미터 제외
        if self.handle_unknown == "error":
            self.encoder = OrdinalEncoder(handle_unknown=self.handle_unknown)
        else:
            self.encoder = OrdinalEncoder(
                handle_unknown=self.handle_unknown, unknown_value=self.unknown_value
            )

    def fit(self, X: pd.DataFrame, y=None):
        try:
            self.encoder.fit(X)
        except ValueError as e:
            error_msg = str(e)
            if "handle_unknown" in error_msg.lower():
                logger.error(f"[OrdinalEncoder] handle_unknown 설정 오류: {self.handle_unknown}")
                raise ValueError(
                    f"OrdinalEncoder handle_unknown '{self.handle_unknown}' 설정에 문제가 있습니다.\n"
                    f"사용 가능한 handle_unknown:\n"
                    f"- 'error': 새로운 범주 발견 시 에러 발생\n"
                    f"- 'use_encoded_value': unknown_value로 지정된 값 사용\n"
                    f"원본 오류: {error_msg}"
                ) from e
            else:
                raise
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """순서형 인코딩을 적용하고 DataFrame으로 반환합니다."""
        result_array = self.encoder.transform(X)
        result_df = pd.DataFrame(result_array, index=X.index, columns=X.columns)
        return result_df

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """OrdinalEncoder는 컬럼명을 보존합니다."""
        return input_columns

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존합니다."""
        return True

    def get_application_type(self) -> str:
        """Ordinal Encoder는 특정 범주형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """범주형 컬럼만 대상으로 합니다."""
        return [
            col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "category"
        ]


class CatBoostEncoderWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: category_encoders 라이브러리의 CatBoostEncoder를 위한 래퍼
    Target Encoding의 변종으로, 정보 누수를 방지하면서 고유 범주가 많은 변수를
    효과적으로 처리합니다. 컬럼명을 보존합니다.
    """

    def __init__(self, sigma: float = 0.05, columns: List[str] = None):
        self.sigma = sigma
        self.columns = columns
        self.encoder = CatBoostEncoder(sigma=self.sigma, cols=self.columns)

    def fit(self, X: pd.DataFrame, y=None):
        """
        CatBoostEncoder를 학습시킵니다.
        이 인코더는 지도 학습 방식이므로 반드시 타겟 변수 y가 필요합니다.
        """
        if y is None:
            logger.error("[CatBoostEncoder] 타겟 변수 y가 필요합니다")
            raise ValueError("CatBoostEncoder requires a target variable 'y' for fitting.")

        self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 인코더를 사용하여 데이터를 변환하고 DataFrame으로 반환합니다."""
        result = self.encoder.transform(X)
        # CatBoostEncoder는 이미 DataFrame을 반환하지만, 확실히 하기 위해 변환
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)
        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """CatBoostEncoder는 컬럼명을 보존합니다."""
        return input_columns

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존합니다."""
        return True

    def get_application_type(self) -> str:
        """CatBoost Encoder는 특정 범주형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """범주형 컬럼만 대상으로 합니다."""
        return [
            col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "category"
        ]


PreprocessorStepRegistry.register("one_hot_encoder", OneHotEncoderWrapper)
PreprocessorStepRegistry.register("ordinal_encoder", OrdinalEncoderWrapper)
PreprocessorStepRegistry.register("catboost_encoder", CatBoostEncoderWrapper)
