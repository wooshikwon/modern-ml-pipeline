# src/components/datahandler/modules/tabular_handler.py
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.datahandler.base import BaseDataHandler
from src.utils.core.logger import log_data_debug

from ..registry import DataHandlerRegistry


class TabularDataHandler(BaseDataHandler):
    """전통적인 테이블 형태 ML을 위한 데이터 핸들러 (classification, regression, clustering, causal)"""

    def __init__(self, settings, data_interface=None):
        super().__init__(settings)
        self._feature_columns: Optional[list[str]] = None
        self._target_column: Optional[str] = None
        self._treatment_column: Optional[str] = None
        log_data_debug("초기화 완료", "TabularDataHandler")

    def get_feature_columns(self) -> list[str]:
        """실제 선택된 피처 목록 반환"""
        if self._feature_columns is None:
            return self.data_interface.feature_columns or []
        return self._feature_columns

    def fit(self, df: pd.DataFrame) -> "TabularDataHandler":
        """
        피처 컬럼 목록 학습.
        train 데이터로 1회만 호출.
        """
        task_choice = self.settings.recipe.task_choice
        self._target_column = self.data_interface.target_column
        self._treatment_column = self.data_interface.treatment_column

        # 제외할 컬럼 결정
        additional_exclude = []
        if self._target_column:
            additional_exclude.append(self._target_column)
        if self._treatment_column:
            additional_exclude.append(self._treatment_column)

        exclude_cols = self._get_exclude_columns(df)
        if additional_exclude:
            exclude_cols = list(set(exclude_cols + additional_exclude))

        # 피처 컬럼 결정 및 저장
        if self.data_interface.feature_columns is None:
            self._feature_columns = [c for c in df.columns if c not in exclude_cols]
        else:
            forbidden = [c for c in exclude_cols if c in self.data_interface.feature_columns]
            if forbidden:
                raise ValueError(
                    f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {forbidden}. "
                    f"entity, timestamp, target, treatment 컬럼은 feature로 사용할 수 없습니다."
                )
            self._feature_columns = list(self.data_interface.feature_columns)

        self._is_fitted = True
        log_data_debug(f"fit 완료 - {len(self._feature_columns)}개 피처", "TabularDataHandler")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 피처 컬럼만 선택하여 반환.
        추론 시 사용.
        """
        if not self._is_fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        available_cols = [c for c in self._feature_columns if c in df.columns]
        if len(available_cols) < len(self._feature_columns):
            missing = set(self._feature_columns) - set(available_cols)
            log_data_debug(f"일부 피처 누락: {missing}", "TabularDataHandler")

        return df[available_cols].copy()

    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        4-way 데이터 분할: train/validation/test/calibration (Data Leakage 방지)

        Args:
            df: 전체 데이터프레임

        Returns:
            Dict with keys: 'train', 'validation', 'test', 'calibration'
        """
        log_data_debug(f"4-way 분할 시작 - 전체: {len(df)}행", "TabularDataHandler")

        ratios = self._get_split_config()
        random_state = self._get_random_state()

        train_ratio = ratios["train"]
        validation_ratio = ratios["validation"]
        test_ratio = ratios["test"]
        calibration_ratio = ratios["calibration"]

        stratify_series = self._get_stratify_series(df)

        # 순차적 분할 (Data Leakage 방지를 위한 올바른 순서)
        # 1. 전체 데이터에서 test 분리
        if test_ratio > 0:
            remaining_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                random_state=random_state,
                stratify=stratify_series if stratify_series is not None else None,
            )
        else:
            remaining_df = df
            test_df = pd.DataFrame()

        # 2. 남은 데이터에서 calibration 분리
        if calibration_ratio > 0:
            calib_size_from_remaining = calibration_ratio / (1 - test_ratio)
            stratify_remaining = (
                self._get_stratify_series(remaining_df) if stratify_series is not None else None
            )
            remaining_df2, calibration_df = train_test_split(
                remaining_df,
                test_size=calib_size_from_remaining,
                random_state=random_state,
                stratify=stratify_remaining,
            )
        else:
            remaining_df2 = remaining_df
            calibration_df = pd.DataFrame()

        # 3. 남은 데이터를 train/validation으로 분할
        if validation_ratio > 0:
            val_size_from_remaining = validation_ratio / (train_ratio + validation_ratio)
            stratify_remaining2 = (
                self._get_stratify_series(remaining_df2) if stratify_series is not None else None
            )
            train_df, validation_df = train_test_split(
                remaining_df2,
                test_size=val_size_from_remaining,
                random_state=random_state,
                stratify=stratify_remaining2,
            )
        else:
            train_df = remaining_df2
            validation_df = pd.DataFrame()

        log_data_debug(
            f"분할 완료 - Train: {len(train_df)}, Val: {len(validation_df)}, "
            f"Test: {len(test_df)}, Calib: {len(calibration_df)}",
            "TabularDataHandler",
        )

        return {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
            "calibration": calibration_df if calibration_ratio > 0 else None,
        }

    def _get_stratify_series(self, df: pd.DataFrame) -> pd.Series:
        """Stratification을 위한 Series 반환"""
        task_choice = self.settings.recipe.task_choice

        if task_choice == "classification":
            target_col = self.data_interface.target_column
            if target_col in df.columns:
                counts = df[target_col].value_counts()
                # 각 클래스 최소 2개, 분할 후에도 최소 1개씩 보장되는지 확인
                if len(counts) >= 2 and counts.min() >= 4:  # Increased minimum for 4-way split
                    return df[target_col]

        elif task_choice == "causal":
            treatment_col = self.data_interface.treatment_column
            if treatment_col in df.columns:
                counts = df[treatment_col].value_counts()
                if len(counts) >= 2 and counts.min() >= 4:  # Increased minimum for 4-way split
                    return df[treatment_col]

        return None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        테이블 형태 데이터 준비.
        내부적으로 fit/transform 패턴 사용.
        """
        # fit되지 않았으면 fit 수행
        if not self._is_fitted:
            self.fit(df)

        # transform으로 X 추출
        X = self.transform(df)
        self._check_missing_values_warning(X)

        # y 및 additional_data 추출
        task_choice = self.settings.recipe.task_choice

        if task_choice in ["classification", "regression", "timeseries"]:
            y = df[self._target_column] if self._target_column in df.columns else None
            return X, y, {}
        elif task_choice == "clustering":
            return X, None, {}
        elif task_choice == "causal":
            y = df[self._target_column] if self._target_column in df.columns else None
            additional_data = {}
            if self._treatment_column and self._treatment_column in df.columns:
                additional_data["treatment"] = df[self._treatment_column]
                additional_data["treatment_value"] = getattr(
                    self.data_interface, "treatment_value", 1
                )
            return X, y, additional_data
        else:
            raise ValueError(f"지원하지 않는 task_choice: {task_choice}")


# Self-registration
DataHandlerRegistry.register("tabular", TabularDataHandler)
