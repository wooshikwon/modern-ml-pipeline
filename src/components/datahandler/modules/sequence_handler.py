# src/components/datahandler/modules/sequence_handler.py
"""
Sequence DataHandler - 시퀀스 변환 전용 데이터 처리

시계열 데이터를 LSTM 등 시퀀스 모델용으로 변환합니다.
- Sliding window 기반 시퀀스 생성 (2D → 3D: samples, seq_len, features)
- 시간 기준 분할로 데이터 누수 방지
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.components.datahandler.base import BaseDataHandler
from src.utils.core.logger import log_data_debug, logger  # logger는 warning용

from ..registry import DataHandlerRegistry


class SequenceDataHandler(BaseDataHandler):
    """시퀀스 변환 전용 DataHandler - Sliding window 기반 2D→3D 변환"""

    DEFAULT_SEQUENCE_LENGTH = 30

    def __init__(self, settings):
        super().__init__(settings)
        self.task_type = settings.recipe.task_choice
        # Recipe에서 sequence_length 읽기 (None이면 기본값 사용)
        configured_seq_len = getattr(self.data_interface, "sequence_length", None)
        self.sequence_length = configured_seq_len if configured_seq_len is not None else self.DEFAULT_SEQUENCE_LENGTH
        self._feature_columns: list[str] = []
        self._original_feature_columns: list[str] = []
        self._flattened_feature_columns: list[str] = []
        self._timestamp_column: str = ""
        self._target_column: str = ""
        self._entity_columns: list[str] = self.data_interface.entity_columns or []
        self._n_features: int = 0
        self._effective_sequence_length: int = 0

        entity_info = f", entities={self._entity_columns}" if self._entity_columns else ""
        log_data_debug(
            f"초기화 완료 - seq_len={self.sequence_length}{entity_info}",
            "SequenceDataHandler",
        )

    def get_feature_columns(self) -> list[str]:
        """학습에 사용된 피처 컬럼 목록 반환 (flatten된 이름)"""
        return self._flattened_feature_columns if self._flattened_feature_columns else self._feature_columns

    def fit(self, df: pd.DataFrame) -> "SequenceDataHandler":
        """
        시퀀스 변환에 필요한 상태 학습.
        train 데이터로 1회만 호출.
        """
        self._timestamp_column = self.data_interface.timestamp_column
        self._target_column = self.data_interface.target_column

        # 피처 컬럼 결정
        exclude_cols = [self._target_column, self._timestamp_column] + (
            self.data_interface.entity_columns or []
        )

        if self.data_interface.feature_columns:
            # 명시적 feature_columns 사용
            feature_cols = self.data_interface.feature_columns
            overlap = set(feature_cols) & set(exclude_cols)
            if overlap:
                raise ValueError(
                    f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                    f"timestamp, target, entity 컬럼은 feature로 사용할 수 없습니다."
                )

            # 문자열 컬럼 검증: 시퀀스 모델은 숫자형 데이터만 지원
            string_cols = [
                col for col in feature_cols
                if col in df.columns and df[col].dtype in ["object", "string"]
            ]
            if string_cols:
                raise TypeError(
                    f"시퀀스 모델(LSTM 등)은 숫자형 데이터만 지원합니다.\n"
                    f"문자열 컬럼이 feature_columns에 포함되어 있습니다: {string_cols}\n"
                    f"[해결 방법] Recipe의 preprocessor에 인코더를 추가하세요:\n"
                    f"preprocessor:\n"
                    f"  steps:\n"
                    f"    - type: ordinal_encoder  # 또는 onehot_encoder\n"
                    f"      columns: {string_cols}"
                )
        else:
            # 자동 선택: 숫자형 컬럼만
            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]

            # 자동 제외된 문자열 컬럼이 있으면 정보 제공
            all_potential_cols = [c for c in df.columns if c not in exclude_cols]
            excluded_string_cols = [
                c for c in all_potential_cols
                if c not in feature_cols and df[c].dtype in ["object", "string"]
            ]
            if excluded_string_cols:
                logger.info(
                    f"[DATA:SequenceDataHandler] 문자열 컬럼 {len(excluded_string_cols)}개 자동 제외: "
                    f"{excluded_string_cols[:5]}{'...' if len(excluded_string_cols) > 5 else ''}"
                )

        self._original_feature_columns = list(feature_cols)
        self._feature_columns = list(feature_cols)
        self._n_features = len(feature_cols)

        # 시퀀스 길이 결정
        self._effective_sequence_length = self._compute_effective_sequence_length(df)

        # flatten된 피처 이름 생성
        self._flattened_feature_columns = [
            f"seq{t}_feat{f}"
            for t in range(self._effective_sequence_length)
            for f in range(self._n_features)
        ]

        self._is_fitted = True
        log_data_debug(
            f"fit 완료 - seq_len={self._effective_sequence_length}, "
            f"n_features={self._n_features}, "
            f"flattened={len(self._flattened_feature_columns)}개 컬럼",
            "SequenceDataHandler",
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시퀀스 변환 수행: sliding window → flatten.
        entity_columns가 설정된 경우 entity별로 시퀀스 생성 (경계 혼합 방지).
        """
        if not self._is_fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        seq_len = self._effective_sequence_length

        # Entity별 그룹화 처리
        if self._entity_columns:
            return self._transform_with_entities(df, seq_len)
        else:
            return self._transform_single(df, seq_len)

    def _transform_single(self, df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
        """단일 시계열 시퀀스 변환 (entity 구분 없음)"""
        # timestamp 정렬
        if self._timestamp_column in df.columns:
            df = df.sort_values(self._timestamp_column).reset_index(drop=True)

        X_sequences = []
        for i in range(seq_len, len(df)):
            X_seq = df.iloc[i - seq_len : i][self._original_feature_columns].values
            if not np.isnan(X_seq).any():
                X_sequences.append(X_seq)

        if not X_sequences:
            log_data_debug(
                f"시퀀스 생성 실패 - 데이터 부족 (필요: {seq_len}개 row)",
                "SequenceDataHandler",
            )
            return pd.DataFrame(columns=self._flattened_feature_columns)

        X_sequences = np.array(X_sequences)
        X_flattened = X_sequences.reshape(len(X_sequences), -1)
        X_df = pd.DataFrame(X_flattened, columns=self._flattened_feature_columns)

        log_data_debug(
            f"transform 완료 - {X_df.shape[0]}개 시퀀스",
            "SequenceDataHandler",
        )
        return X_df

    def _transform_with_entities(self, df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
        """Entity별 시퀀스 변환 (경계 혼합 방지)"""
        all_sequences = []

        for entity_key, group_df in df.groupby(self._entity_columns, sort=False):
            # Entity 내에서 timestamp 정렬
            if self._timestamp_column in group_df.columns:
                group_df = group_df.sort_values(self._timestamp_column).reset_index(drop=True)

            # Entity 내에서 sliding window 적용
            for i in range(seq_len, len(group_df)):
                X_seq = group_df.iloc[i - seq_len : i][self._original_feature_columns].values
                if not np.isnan(X_seq).any():
                    all_sequences.append(X_seq)

        if not all_sequences:
            log_data_debug(
                f"시퀀스 생성 실패 - 데이터 부족 (entity별 최소 {seq_len}개 row 필요)",
                "SequenceDataHandler",
            )
            return pd.DataFrame(columns=self._flattened_feature_columns)

        X_sequences = np.array(all_sequences)
        X_flattened = X_sequences.reshape(len(X_sequences), -1)
        X_df = pd.DataFrame(X_flattened, columns=self._flattened_feature_columns)

        n_entities = df.groupby(self._entity_columns).ngroups
        log_data_debug(
            f"transform 완료 - {n_entities}개 entity, {X_df.shape[0]}개 시퀀스",
            "SequenceDataHandler",
        )
        return X_df

    def _compute_effective_sequence_length(self, df: pd.DataFrame) -> int:
        """데이터 크기에 따라 시퀀스 길이 동적 조정"""
        n = len(df)
        return max(5, min(self.sequence_length, max(5, n // 3)))

    def validate_data(self, df: pd.DataFrame) -> bool:
        """시퀀스 데이터 검증"""
        super().validate_data(df)

        target_col = self.data_interface.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target 컬럼 '{target_col}'을 찾을 수 없습니다")

        if self.task_type == "timeseries":
            timestamp_col = self.data_interface.timestamp_column
            if not timestamp_col or timestamp_col not in df.columns:
                raise ValueError(
                    f"TimeSeries task에 필요한 timestamp_column '{timestamp_col}'을 찾을 수 없습니다"
                )

            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    log_data_debug(f"Timestamp 변환 완료: {timestamp_col}", "SequenceDataHandler")
                except:
                    raise ValueError(
                        f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다"
                    )

            seq_len = self._effective_sequence_length if self._is_fitted else self._compute_effective_sequence_length(df)
            if len(df) < seq_len + 1:
                raise ValueError(
                    f"시퀀스 생성을 위해 최소 {seq_len + 1}개 행이 필요합니다. "
                    f"현재 데이터: {len(df)}개 행"
                )

        return True

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Dict[str, Any]]:
        """
        시퀀스 데이터 준비.
        내부적으로 fit/transform 패턴 사용.
        """
        self.validate_data(df)

        if self.task_type != "timeseries":
            raise ValueError(
                f"SequenceHandler는 timeseries task 전용입니다. "
                f"현재 task: {self.task_type}. tabular 핸들러를 사용하세요."
            )

        # fit되지 않았으면 fit 수행
        if not self._is_fitted:
            self.fit(df)

        # transform으로 X 추출
        X_df = self.transform(df)

        # y 추출 (시퀀스에 맞게 조정, entity별 처리)
        seq_len = self._effective_sequence_length
        y_values = self._extract_y_values(df, seq_len)
        y_series = pd.Series(y_values, name="target") if y_values else pd.Series(dtype="float64")

        # additional_data 생성
        additional_data = {
            "sequence_length": seq_len,
            "feature_columns": self._original_feature_columns,
            "n_features": self._n_features,
            "is_timeseries": True,
            "is_timeseries_sequence": True,
            "original_sequence_shape": (len(X_df), seq_len, self._n_features),
            "data_shape": X_df.shape,
        }

        return X_df, y_series, additional_data

    def _extract_y_values(self, df: pd.DataFrame, seq_len: int) -> list:
        """
        시퀀스에 대응하는 y 값 추출.
        entity_columns가 있으면 entity별로 처리.
        """
        if self._entity_columns:
            return self._extract_y_with_entities(df, seq_len)
        else:
            return self._extract_y_single(df, seq_len)

    def _extract_y_single(self, df: pd.DataFrame, seq_len: int) -> list:
        """단일 시계열 y 추출"""
        if self._timestamp_column in df.columns:
            df_sorted = df.sort_values(self._timestamp_column).reset_index(drop=True)
        else:
            df_sorted = df

        y_values = []
        for i in range(seq_len, len(df_sorted)):
            y_val = df_sorted.iloc[i][self._target_column]
            X_seq = df_sorted.iloc[i - seq_len : i][self._original_feature_columns].values
            if not np.isnan(X_seq).any() and not np.isnan(y_val):
                y_values.append(y_val)
        return y_values

    def _extract_y_with_entities(self, df: pd.DataFrame, seq_len: int) -> list:
        """Entity별 y 추출"""
        y_values = []

        for entity_key, group_df in df.groupby(self._entity_columns, sort=False):
            if self._timestamp_column in group_df.columns:
                group_df = group_df.sort_values(self._timestamp_column).reset_index(drop=True)

            for i in range(seq_len, len(group_df)):
                y_val = group_df.iloc[i][self._target_column]
                X_seq = group_df.iloc[i - seq_len : i][self._original_feature_columns].values
                if not np.isnan(X_seq).any() and not np.isnan(y_val):
                    y_values.append(y_val)

        return y_values

    def split_and_prepare(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        Any,
        Dict[str, Any],
        pd.DataFrame,
        Any,
        Dict[str, Any],
        pd.DataFrame,
        Any,
        Dict[str, Any],
        Optional[Tuple[pd.DataFrame, Any, Dict[str, Any]]],
    ]:
        """
        Sequence용 4-way interface 구현.
        fit은 train 데이터로 1회만 수행.
        """
        train_df, validation_df, test_df = self.split_data(df)

        log_data_debug(
            f"시퀀스 분할 - train={len(train_df)}, val={len(validation_df)}, test={len(test_df)}",
            "SequenceDataHandler",
        )

        # Train: fit + transform (prepare_data 내부에서 처리)
        X_train, y_train, add_train = self.prepare_data(train_df)

        # Val/Test: transform만 (이미 fit됨)
        X_val, y_val, add_val = (
            self.prepare_data(validation_df)
            if not validation_df.empty
            else (pd.DataFrame(), pd.Series(dtype="float64"), {})
        )
        X_test, y_test, add_test = (
            self.prepare_data(test_df)
            if not test_df.empty
            else (pd.DataFrame(), pd.Series(dtype="float64"), {})
        )

        log_data_debug(
            f"시퀀스 변환 완료 - seq_len={self._effective_sequence_length}, "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}",
            "SequenceDataHandler",
        )

        calibration_data = None
        return (
            X_train, y_train, add_train,
            X_val, y_val, add_val,
            X_test, y_test, add_test,
            calibration_data,
        )

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """시계열 시간 기준 3-way 분할"""
        return self._time_based_split_3way(df)

    def _time_based_split_3way(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """시계열 시간 기준 3-way 분할 (Data Leakage 방지)"""
        timestamp_col = self.data_interface.timestamp_column
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)

        # Recipe 설정에서 분할 비율 읽기
        ratios = self._get_split_config()
        train_ratio = ratios["train"]
        validation_ratio = ratios["validation"]

        min_train_size = self.sequence_length + 10
        total_sequences = n - self.sequence_length

        if total_sequences < 30:
            raise ValueError(
                f"3-way 시퀀스 생성 후 데이터가 부족합니다. "
                f"전체 {n}행 → {total_sequences}개 시퀀스. "
                f"최소 {self.sequence_length + 30}행이 필요합니다."
            )

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        if train_end < min_train_size:
            train_end = min_train_size
            val_end = min(val_end, n - 10)
            logger.warning(f"[DATA:SequenceDataHandler] 3-way 분할 조정: train_end={train_end}")

        train_df = df_sorted.iloc[:train_end].copy()
        validation_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        train_sequences = max(0, len(train_df) - self.sequence_length)
        val_sequences = max(0, len(validation_df) - self.sequence_length)
        test_sequences = max(0, len(test_df) - self.sequence_length)

        def get_period(sub_df: pd.DataFrame) -> str:
            if sub_df.empty:
                return "Empty"
            return f"{sub_df[timestamp_col].min()} ~ {sub_df[timestamp_col].max()}"

        log_data_debug(
            f"시계열 3-way 분할 - Train: {train_sequences}, Val: {val_sequences}, Test: {test_sequences}",
            "SequenceDataHandler",
        )
        log_data_debug(f"Train 기간: {get_period(train_df)}", "SequenceDataHandler")
        log_data_debug(f"Val 기간: {get_period(validation_df)}", "SequenceDataHandler")
        log_data_debug(f"Test 기간: {get_period(test_df)}", "SequenceDataHandler")

        return train_df, validation_df, test_df

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시퀀스 데이터 메타 정보 반환"""
        base_info = super().get_data_info(df)

        sequence_info = {
            "handler_type": "sequence",
            "task_type": self.task_type,
            "sequence_length": self.sequence_length,
            "estimated_sequences": max(0, len(df) - self.sequence_length),
            "transform_description": "2D tabular → 3D sequence (sliding window)",
        }

        target_col = self.data_interface.target_column
        if target_col in df.columns:
            sequence_info["target_info"] = {
                "name": target_col,
                "dtype": str(df[target_col].dtype),
                "unique_values": df[target_col].nunique(),
                "missing_ratio": df[target_col].isnull().sum() / len(df),
            }

        timestamp_col = self.data_interface.timestamp_column
        if timestamp_col and timestamp_col in df.columns:
            sequence_info["timestamp_info"] = {
                "name": timestamp_col,
                "dtype": str(df[timestamp_col].dtype),
                "date_range": f"{df[timestamp_col].min()} ~ {df[timestamp_col].max()}",
                "frequency_hint": self._estimate_frequency(df[timestamp_col]),
            }

        return {**base_info, "sequence_specific": sequence_info}

    def _estimate_frequency(self, timestamp_series: pd.Series) -> str:
        """시계열 주기 추정"""
        try:
            sorted_ts = timestamp_series.dropna().sort_values()
            if len(sorted_ts) < 2:
                return "unknown"

            deltas = sorted_ts.diff().dropna()
            if len(deltas) == 0:
                return "unknown"

            mode_delta = deltas.mode()
            if len(mode_delta) == 0:
                return "unknown"

            mode_seconds = mode_delta.iloc[0].total_seconds()

            if mode_seconds == 86400:
                return "daily"
            elif mode_seconds == 3600:
                return "hourly"
            elif mode_seconds == 60:
                return "minutely"
            elif mode_seconds == 604800:
                return "weekly"
            elif 2505600 <= mode_seconds <= 2678400:
                return "monthly"
            else:
                return f"~{mode_seconds}s"

        except Exception:
            return "unknown"


# Registry 등록
DataHandlerRegistry.register("sequence", SequenceDataHandler)
