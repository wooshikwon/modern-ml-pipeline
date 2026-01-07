# src/components/_preprocessor/_steps/_imputer.py
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator, SimpleImputer

from src.components.preprocessor.base import BasePreprocessor
from src.utils.core.logger import logger

from ..registry import PreprocessorStepRegistry


class SimpleImputerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 SimpleImputer를 위한 래퍼
    결측값을 전략에 따라 대체하며 컬럼명을 보존합니다.

    create_missing_indicators=True로 설정하면 imputation 전 결측값 위치를
    나타내는 indicator 컬럼들을 추가로 생성합니다.
    """

    def __init__(
        self,
        strategy: str = "mean",
        columns: List[str] = None,
        create_missing_indicators: bool = False,
    ):
        self.strategy = strategy
        self.columns = columns
        self.create_missing_indicators = create_missing_indicators
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.missing_indicator = None
        self._input_columns = None
        self._all_null_columns: List[str] = []
        if self.create_missing_indicators:
            self.missing_indicator = MissingIndicator(features="missing-only")

    def fit(self, X: pd.DataFrame, y=None):
        """Imputer 학습 및 필요시 MissingIndicator도 학습"""
        self._input_columns = list(X.columns)
        logger.info(
            f"[SimpleImputer] 결측값 대체 학습을 시작합니다 - strategy: {self.strategy}, 대상 컬럼: {len(X.columns)}개"
        )

        # 전체가 NaN인 컬럼 감지 → 에러 대신 경고 후 폴백 처리 (transform에서 0으로 채움)
        all_null_columns = [col for col in X.columns if X[col].isnull().all()]
        self._all_null_columns = all_null_columns
        if all_null_columns:
            logger.warning(
                f"[SimpleImputer] 전체가 결측값인 컬럼 발견: {all_null_columns}, transform 시 0으로 채움"
            )
        cols_to_fit = [c for c in X.columns if c not in self._all_null_columns]

        try:
            if cols_to_fit:
                self.imputer.fit(X[cols_to_fit])
            # else: 모든 컬럼이 all-null → fit 생략
        except ValueError as e:
            error_msg = str(e)
            if "strategy" in error_msg.lower():
                logger.error(f"[SimpleImputer] strategy 호환성 오류: {self.strategy}")
                raise ValueError(
                    f"SimpleImputer strategy '{self.strategy}'가 데이터 타입과 호환되지 않습니다.\n"
                    f"사용 가능한 strategy:\n"
                    f"- 'mean', 'median': 숫자형 컬럼만 가능\n"
                    f"- 'most_frequent': 모든 타입 가능\n"
                    f"- 'constant': 지정된 상수값 (fill_value 파라미터 필요)\n"
                    f"원본 오류: {error_msg}"
                ) from e
            else:
                raise

        # Missing indicator도 학습
        if self.create_missing_indicators and self.missing_indicator:
            logger.info("[SimpleImputer] 결측값 지시자 학습을 시작합니다")
            self.missing_indicator.fit(X)

        logger.info(
            f"[SimpleImputer] 학습 완료 - strategy: {self.strategy}, 입력 컬럼: {len(X.columns)}개"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 Imputer를 사용하여 데이터를 변환하고 DataFrame으로 반환합니다."""
        logger.info(
            f"[SimpleImputer] 결측값 대체 변환을 시작합니다 - 입력: {X.shape}, strategy: {self.strategy}"
        )

        # 1. Missing indicators 생성 (imputation 이전)
        indicator_data = None
        if self.create_missing_indicators and self.missing_indicator:
            indicator_array = self.missing_indicator.transform(X).astype(np.int64)

            # sklearn의 get_feature_names_out을 사용하여 실제 출력 컬럼명 확인
            try:
                indicator_feature_names = self.missing_indicator.get_feature_names_out(
                    list(X.columns)
                )
            except Exception:
                # 폴백: 수동으로 생성 (결측값이 있는 컬럼에 대해서만)
                indicator_feature_names = [
                    f"missingindicator_{col}" for col in X.columns if X[col].isnull().any()
                ]

            if indicator_array.shape[1] > 0:  # indicator 컬럼이 있는 경우만
                indicator_data = pd.DataFrame(
                    indicator_array, index=X.index, columns=indicator_feature_names
                )
                logger.info(
                    f"[SimpleImputer] 결측값 지시자 생성 완료 - 지시자 컬럼: {len(indicator_feature_names)}개"
                )

        # 2. Imputation 수행 (fit된 컬럼만 대상) + all-null 컬럼 폴백 채우기
        result_cols = list(X.columns)
        imputed_data = pd.DataFrame(index=X.index, columns=result_cols)
        cols_to_transform = [c for c in X.columns if c not in self._all_null_columns]
        if cols_to_transform:
            imputed_array = self.imputer.transform(X[cols_to_transform])
            imputed_data[cols_to_transform] = pd.DataFrame(
                imputed_array, index=X.index, columns=cols_to_transform
            )
        # All-NaN 컬럼은 0으로 채움 (수치형 가정). 비수치인 경우에도 0으로 안전 채움.
        for c in self._all_null_columns:
            imputed_data[c] = 0
        logger.info(f"[SimpleImputer] 결측값 대체 완료 - 처리된 데이터: {imputed_data.shape}")

        # 3. 결과 결합
        if indicator_data is not None and len(indicator_data.columns) > 0:
            # Imputed data + Missing indicators
            result_data = pd.concat([imputed_data, indicator_data], axis=1)
            logger.info(
                f"[SimpleImputer] 최종 결과 결합 완료 - 출력: {result_data.shape} (결측값 대체 + 지시자)"
            )
        else:
            # Imputed data만
            result_data = imputed_data
            logger.info(
                f"[SimpleImputer] 최종 결과 완료 - 출력: {result_data.shape} (결측값 대체만)"
            )

        return result_data

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """출력 컬럼명 반환 (원본 + missing indicator 컬럼들)"""
        output_columns = input_columns.copy()

        # Missing indicator 컬럼들 추가
        if self.create_missing_indicators:
            # 결측값이 있는 컬럼들에 대한 indicator 컬럼명 생성
            for col in input_columns:
                # 실제로는 fit/transform 시점에서 결정되지만, 예상 컬럼명 반환
                output_columns.append(f"missingindicator_{col}")

        return output_columns

    def preserves_column_names(self) -> bool:
        """Missing indicator를 생성하는 경우 컬럼이 추가되므로 False"""
        return not self.create_missing_indicators

    def get_application_type(self) -> str:
        """SimpleImputer는 사용자가 지정한 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """결측값이 있는 숫자형 컬럼만 대상으로 합니다."""
        applicable_columns = []
        for col in X.columns:
            # 숫자형이면서 결측값이 있는 컬럼만 선택
            if X[col].dtype in ["int64", "float64"] and X[col].isnull().any():
                applicable_columns.append(col)
        return applicable_columns


PreprocessorStepRegistry.register("simple_imputer", SimpleImputerWrapper)
