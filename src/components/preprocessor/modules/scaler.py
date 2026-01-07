# src/components/_preprocessor/_steps/_scaler.py
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.components.preprocessor.base import BasePreprocessor
from src.utils.core.logger import log_prep, log_prep_debug

from ..registry import PreprocessorStepRegistry


class StandardScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: StandardScaler를 위한 래퍼
    모든 숫자형 컬럼에 자동으로 표준화를 적용합니다.
    """

    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = StandardScaler()
        self._fitted_columns = []

    def fit(self, X: pd.DataFrame, y=None):
        log_prep_debug("StandardScaler 학습 시작", "StandardScaler")

        # 적용 가능한 컬럼 필터링
        applicable_cols = self.get_applicable_columns(X)
        log_prep_debug(f"표준화 적용 가능한 숫자형 컬럼: {len(applicable_cols)}개", "StandardScaler")

        if not applicable_cols:
            # 숫자형 컬럼이 없는 경우
            log_prep_debug("표준화할 숫자형 컬럼 없음 - 스킵", "StandardScaler")
            self._fitted_columns = []
            return self

        # 엣지 케이스 필터링: 상수 컬럼과 all-NaN 컬럼 제외
        valid_cols = []
        skipped_cols = []
        for col in applicable_cols:
            col_data = X[col].dropna()
            if len(col_data) > 1 and col_data.std() > 1e-8:
                valid_cols.append(col)
            else:
                skipped_cols.append(col)

        if skipped_cols:
            log_prep_debug(f"상수값/NaN 컬럼 제외: {len(skipped_cols)}개", "StandardScaler")

        if valid_cols:
            self.scaler.fit(X[valid_cols])
            self._fitted_columns = valid_cols
            log_prep_debug(f"학습 완료 - 적용 컬럼: {len(valid_cols)}개", "StandardScaler")
        else:
            # 모든 컬럼이 상수이거나 NaN인 경우
            self._fitted_columns = []
            log_prep_debug("유효한 컬럼 없음 - 스킵", "StandardScaler")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """표준화를 적용하고 DataFrame으로 반환합니다."""
        result = X.copy()

        if self._fitted_columns:
            log_prep_debug(f"변환 적용 - {len(self._fitted_columns)}개 컬럼", "StandardScaler")
            # 피팅된 컬럼만 스케일링 적용
            fitted_data = X[self._fitted_columns]
            scaled_data = self.scaler.transform(fitted_data)
            result[self._fitted_columns] = scaled_data

            # 변환 결과 통계
            means = scaled_data.mean(axis=0)
            stds = scaled_data.std(axis=0)
            log_prep_debug(f"완료 - 평균: ~{means.mean():.4f}, 표준편차: ~{stds.mean():.4f}", "StandardScaler")

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """StandardScaler는 컬럼명을 보존합니다."""
        return input_columns

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존합니다."""
        return True

    def get_application_type(self) -> str:
        """StandardScaler는 모든 숫자형 컬럼에 자동 적용됩니다."""
        return "global"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


class MinMaxScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: MinMaxScaler를 위한 래퍼
    모든 숫자형 컬럼에 자동으로 Min-Max 스케일링을 적용합니다.
    """

    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = MinMaxScaler()
        self._fitted_columns = []

    def fit(self, X: pd.DataFrame, y=None):
        log_prep_debug("MinMaxScaler 학습 시작", "MinMaxScaler")

        # 적용 가능한 컬럼 필터링
        applicable_cols = self.get_applicable_columns(X)
        log_prep_debug(f"정규화 적용 가능한 숫자형 컬럼: {len(applicable_cols)}개", "MinMaxScaler")

        if not applicable_cols:
            log_prep_debug("정규화할 숫자형 컬럼 없음 - 스킵", "MinMaxScaler")
            self._fitted_columns = []
            return self

        # 엣지 케이스 필터링
        valid_cols = []
        skipped_cols = []
        for col in applicable_cols:
            col_data = X[col].dropna()
            if len(col_data) > 1 and col_data.std() > 1e-8:
                valid_cols.append(col)
            else:
                skipped_cols.append(col)

        if skipped_cols:
            log_prep_debug(f"상수값/NaN 컬럼 제외: {len(skipped_cols)}개", "MinMaxScaler")

        if valid_cols:
            self.scaler.fit(X[valid_cols])
            self._fitted_columns = valid_cols
            log_prep_debug(f"학습 완료 - 적용 컬럼: {len(valid_cols)}개", "MinMaxScaler")
        else:
            self._fitted_columns = []
            log_prep_debug("유효한 컬럼 없음 - 스킵", "MinMaxScaler")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Min-Max 스케일링을 적용하고 DataFrame으로 반환합니다."""
        result = X.copy()

        if self._fitted_columns:
            log_prep_debug(f"변환 적용 - {len(self._fitted_columns)}개 컬럼", "MinMaxScaler")
            fitted_data = X[self._fitted_columns]
            scaled_data = self.scaler.transform(fitted_data)
            result[self._fitted_columns] = scaled_data

            # 변환 결과 통계
            mins = scaled_data.min(axis=0)
            maxs = scaled_data.max(axis=0)
            log_prep_debug(f"완료 - 최솟값: ~{mins.mean():.4f}, 최댓값: ~{maxs.mean():.4f}", "MinMaxScaler")

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        return input_columns

    def preserves_column_names(self) -> bool:
        return True

    def get_application_type(self) -> str:
        return "global"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


class RobustScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: RobustScaler를 위한 래퍼
    중앙값과 사분위수를 사용하여 스케일링하므로, 이상치(outlier)에 덜 민감합니다.
    """

    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = RobustScaler()
        self._fitted_columns = []

    def fit(self, X: pd.DataFrame, y=None):
        # 적용 가능한 컬럼 필터링
        applicable_cols = self.get_applicable_columns(X)

        if not applicable_cols:
            self._fitted_columns = []
            return self

        # 엣지 케이스 필터링
        valid_cols = []
        for col in applicable_cols:
            col_data = X[col].dropna()
            if len(col_data) > 1 and col_data.std() > 1e-8:
                valid_cols.append(col)

        if valid_cols:
            self.scaler.fit(X[valid_cols])
            self._fitted_columns = valid_cols
        else:
            self._fitted_columns = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Robust 스케일링을 적용하고 DataFrame으로 반환합니다."""
        result = X.copy()

        if self._fitted_columns:
            fitted_data = X[self._fitted_columns]
            scaled_data = self.scaler.transform(fitted_data)
            result[self._fitted_columns] = scaled_data

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        return input_columns

    def preserves_column_names(self) -> bool:
        return True

    def get_application_type(self) -> str:
        return "global"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


PreprocessorStepRegistry.register("standard_scaler", StandardScalerWrapper)
PreprocessorStepRegistry.register("min_max_scaler", MinMaxScalerWrapper)
PreprocessorStepRegistry.register("robust_scaler", RobustScalerWrapper)
