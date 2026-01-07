# src/components/datahandler/modules/timeseries_handler.py
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.components.datahandler.base import BaseDataHandler
from src.utils.core.logger import log_data_debug

from ..catalog_utils import load_model_catalog
from ..registry import DataHandlerRegistry


class TimeseriesDataHandler(BaseDataHandler):
    """시계열 데이터 전용 핸들러"""

    def __init__(self, settings, data_interface=None):
        super().__init__(settings)
        self._feature_columns: List[str] = []
        self._timestamp_column: str = ""
        self._target_column: str = ""

        # Catalog에서 time_features_auto 플래그 확인
        model_class_path = getattr(settings.recipe.model, "class_path", None)
        catalog = load_model_catalog(model_class_path)
        feature_reqs = catalog.get("feature_requirements", {})
        self._time_features_auto = feature_reqs.get("time_features_auto", True)

        log_data_debug(
            f"초기화 완료 - time_features_auto={self._time_features_auto}",
            "TimeseriesDataHandler",
        )

    def get_feature_columns(self) -> List[str]:
        """학습에 사용된 피처 컬럼 목록 반환"""
        return self._feature_columns

    def fit(self, df: pd.DataFrame) -> "TimeseriesDataHandler":
        """
        시계열 피처 생성 규칙 학습.
        train 데이터로 1회만 호출.
        """
        self._timestamp_column = self.data_interface.timestamp_column
        self._target_column = self.data_interface.target_column

        # 시간 피처 생성하여 피처 컬럼 목록 결정
        df_with_features = self._generate_time_features(df)

        exclude_cols = self._get_exclude_columns(df_with_features)
        auto_exclude = [self._timestamp_column, self._target_column] + exclude_cols

        if self.data_interface.feature_columns is None:
            self._feature_columns = [
                c for c in df_with_features.columns if c not in auto_exclude
            ]
        else:
            forbidden_cols = [c for c in auto_exclude if c and c in df_with_features.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(
                    f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                    f"timestamp, target, entity 컬럼은 feature로 사용할 수 없습니다."
                )
            self._feature_columns = list(self.data_interface.feature_columns)

        self._is_fitted = True
        log_data_debug(f"fit 완료 - {len(self._feature_columns)}개 피처", "TimeseriesDataHandler")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 피처 생성 및 피처 추출.
        추론 시 사용. lag/rolling 피처는 배치 데이터에서만 계산 가능.
        """
        if not self._is_fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        # timestamp 정렬
        if self._timestamp_column in df.columns:
            df = df.sort_values(self._timestamp_column).reset_index(drop=True)

        # 시간 피처 생성
        df_with_features = self._generate_time_features(df)

        # 학습된 피처 컬럼 선택
        available_cols = [c for c in self._feature_columns if c in df_with_features.columns]
        if len(available_cols) < len(self._feature_columns):
            missing = set(self._feature_columns) - set(available_cols)
            log_data_debug(f"일부 피처 누락: {missing}", "TimeseriesDataHandler")

        return df_with_features[available_cols].copy()

    def validate_data(self, df: pd.DataFrame) -> bool:
        """시계열 데이터 검증"""
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 찾을 수 없습니다")
        if target_col not in df.columns:
            raise ValueError(f"Target 컬럼 '{target_col}'을 찾을 수 없습니다")

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                log_data_debug(
                    f"Timestamp 변환 완료: {timestamp_col} → datetime", "TimeseriesDataHandler"
                )
            except Exception:
                raise ValueError(
                    f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다"
                )

        return True

    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        시계열 시간 기준 분할 (시간 순서 유지 + Data Leakage 방지)

        시계열 데이터 특성상 shuffle 없이 시간 순서대로 분할합니다.
        calibration은 시계열에서 미지원 (None 반환).

        Returns:
            Dict with keys: 'train', 'validation', 'test', 'calibration'
        """
        timestamp_col = self.data_interface.timestamp_column
        ratios = self._get_split_config()

        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)

        train_end = int(n * ratios["train"])
        val_end = int(n * (ratios["train"] + ratios["validation"]))

        train_df = df_sorted.iloc[:train_end].copy()
        validation_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        def get_period(sub_df: pd.DataFrame) -> str:
            if sub_df.empty:
                return "Empty"
            return f"{sub_df[timestamp_col].min()} ~ {sub_df[timestamp_col].max()}"

        log_data_debug(
            f"시계열 분할 완료 - Train: {len(train_df)}, Val: {len(validation_df)}, Test: {len(test_df)}",
            "TimeseriesDataHandler",
        )
        log_data_debug(f"Train 기간: {get_period(train_df)}", "TimeseriesDataHandler")
        log_data_debug(f"Val 기간: {get_period(validation_df)}", "TimeseriesDataHandler")
        log_data_debug(f"Test 기간: {get_period(test_df)}", "TimeseriesDataHandler")

        return {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
            "calibration": None,
        }

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        시계열 데이터 준비.
        내부적으로 fit/transform 패턴 사용.
        """
        self.validate_data(df)

        # fit되지 않았으면 fit 수행
        if not self._is_fitted:
            self.fit(df)

        # transform으로 X 추출
        X = self.transform(df)
        self._check_missing_values_warning(X)

        # y 및 additional_data 추출
        y = df[self._target_column] if self._target_column in df.columns else None
        additional_data = {}
        if self._timestamp_column in df.columns:
            additional_data["timestamp"] = df[self._timestamp_column]

        return X, y, additional_data

    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시계열 시간 기반 특성 자동 생성.
        catalog의 time_features_auto=false인 경우 스킵 (ARIMA 등 univariate 모델용)
        """
        if not self._time_features_auto:
            log_data_debug(
                "time_features_auto=False - 피처 생성 스킵 (univariate 모델)",
                "TimeseriesDataHandler",
            )
            return df.copy()

        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        df_copy = df.copy()

        df_copy["year"] = df_copy[timestamp_col].dt.year
        df_copy["month"] = df_copy[timestamp_col].dt.month
        df_copy["day"] = df_copy[timestamp_col].dt.day
        df_copy["dayofweek"] = df_copy[timestamp_col].dt.dayofweek
        df_copy["quarter"] = df_copy[timestamp_col].dt.quarter
        df_copy["is_weekend"] = df_copy["dayofweek"].isin([5, 6]).astype(int)

        for lag in [1, 2, 3, 7, 14]:
            df_copy[f"{target_col}_lag_{lag}"] = df_copy[target_col].shift(lag)

        for window in [3, 7, 14]:
            df_copy[f"{target_col}_rolling_mean_{window}"] = (
                df_copy[target_col].rolling(window=window).mean()
            )
            df_copy[f"{target_col}_rolling_std_{window}"] = (
                df_copy[target_col].rolling(window=window).std()
            )

        log_data_debug(
            f"시계열 특성 생성 완료: {len(df_copy.columns) - len(df)}개 신규 피처",
            "TimeseriesDataHandler",
        )
        return df_copy


# Self-registration
DataHandlerRegistry.register("timeseries", TimeseriesDataHandler)
