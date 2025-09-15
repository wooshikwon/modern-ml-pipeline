# src/components/datahandler/modules/timeseries_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from src.interface import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.core.console import get_console


class TimeseriesDataHandler(BaseDataHandler):
    """시계열 데이터 전용 핸들러"""

    def __init__(self, settings, data_interface=None):
        super().__init__(settings)
        self.console = get_console(settings)
        self.console.info("[TimeseriesDataHandler] 초기화 시작합니다")
        self.console.info("[TimeseriesDataHandler] 초기화 완료되었습니다",
                         rich_message="✅ [TimeseriesDataHandler] initialized")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """시계열 데이터 검증"""
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        
        # 필수 컬럼 존재 검증
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 찾을 수 없습니다")
        if target_col not in df.columns:
            raise ValueError(f"Target 컬럼 '{target_col}'을 찾을 수 없습니다")
            
        # 타임스탬프 데이터 타입 검증
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                self.console.log_processing_step(
                    "Timestamp 컬럼 변환",
                    f"{timestamp_col} → datetime 변환 완료"
                )
            except:
                raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다")
        
        return True
        
    # [TODO] recipe에서 split 받아서 고정값이 아니라 동적으로 분리
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        시계열 시간 기준 3-way 분할 (시간 순서 유지 + Data Leakage 방지)
        
        Returns:
            Tuple[train_df, validation_df, test_df]
        """
        timestamp_col = self.data_interface.timestamp_column
        
        # 시간 순서로 정렬 (필수)
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 3-way 시간 기준 분할: train (60%) / validation (20%) / test (20%)
        train_end = int(len(df_sorted) * 0.6)
        val_end = int(len(df_sorted) * 0.8)
        
        train_df = df_sorted.iloc[:train_end].copy()
        validation_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        train_period = f"{train_df[timestamp_col].min()} ~ {train_df[timestamp_col].max()}" if not train_df.empty else "Empty"
        val_period = f"{validation_df[timestamp_col].min()} ~ {validation_df[timestamp_col].max()}" if not validation_df.empty else "Empty"  
        test_period = f"{test_df[timestamp_col].min()} ~ {test_df[timestamp_col].max()}" if not test_df.empty else "Empty"
        
        self.console.log_data_operation(
            "TimeSeries 3-way 시간 기준 분할 완료",
            (len(train_df), len(validation_df), len(test_df)),
            f"Train: {len(train_df)}, Val: {len(validation_df)}, Test: {len(test_df)}"
        )
        self.console.info(f"  • Train 기간: {train_period}")
        self.console.info(f"  • Val 기간: {val_period}")
        self.console.info(f"  • Test 기간: {test_period}")
        
        return train_df, validation_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """시계열 데이터 준비"""
        # 데이터 검증
        self.validate_data(df)
        
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        
        # 시간 순서 정렬 (필수)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 시계열 특성 생성
        df_with_features = self._generate_time_features(df)
        
        # Feature/Target 분리
        exclude_cols = self._get_exclude_columns(df_with_features)
        
        if self.data_interface.feature_columns is None:
            # 자동 선택: timestamp, target, entity 제외
            auto_exclude = [timestamp_col, target_col] + exclude_cols
            X = df_with_features.drop(columns=[c for c in auto_exclude if c in df_with_features.columns])
            self.console.log_processing_step(
                "Timeseries Feature 컬럼 자동 선택",
                f"{len(X.columns)}개 피처: {', '.join(list(X.columns)[:3])}{'...' if len(X.columns) > 3 else ''}"
            )
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = [timestamp_col, target_col] + exclude_cols
            forbidden_cols = [c for c in forbidden_cols if c and c in df_with_features.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"timestamp, target, entity 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df_with_features[self.data_interface.feature_columns]
        
        # 숫자형 컬럼만 선택
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        self._check_missing_values_warning(X)
        
        y = df[target_col]
        
        additional_data = {
            'timestamp': df[timestamp_col]
        }
        
        return X, y, additional_data

    def split_and_prepare(self, df: pd.DataFrame):
        """Compatibility helper: return 6-tuple (train/test) for tests expecting 2-way output."""
        train_df, val_df, test_df = self.split_data(df)
        X_train, y_train, add_train = self.prepare_data(train_df)
        X_test, y_test, add_test = self.prepare_data(test_df)
        return X_train, y_train, add_train, X_test, y_test, add_test
    
    def _generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 시간 기반 특성 자동 생성"""
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        df_copy = df.copy()
        
        # 기본 시간 특성
        df_copy['year'] = df_copy[timestamp_col].dt.year
        df_copy['month'] = df_copy[timestamp_col].dt.month
        df_copy['day'] = df_copy[timestamp_col].dt.day
        df_copy['dayofweek'] = df_copy[timestamp_col].dt.dayofweek
        df_copy['quarter'] = df_copy[timestamp_col].dt.quarter
        df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features (1, 2, 3, 7, 14일 전)
        for lag in [1, 2, 3, 7, 14]:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
            
        # Rolling features (3, 7, 14일 평균)
        for window in [3, 7, 14]:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std()
        
        self.console.log_processing_step(
            "TimeSeries 특성 생성 완료",
            f"{len(df_copy.columns) - len(df)}개 신규 특성 추가 (lag, rolling, time features)"
        )
        return df_copy
    
    def _get_exclude_columns(self, df: pd.DataFrame) -> list:
        """
        데이터에서 제외할 컬럼 목록 반환
        시계열에서는 entity columns만 제외 (timestamp, target은 prepare_data에서 별도 처리)
        """
        fetcher_conf = self.settings.recipe.data.fetcher
        exclude_columns = []
        
        # Entity columns는 항상 제외
        if self.data_interface.entity_columns:
            exclude_columns.extend(self.data_interface.entity_columns)
        
        # Feature Store timestamp column 제외 (offline 모드에서)
        # Tabular 핸들러와 동일한 체크 로직 사용
        if fetcher_conf and getattr(fetcher_conf, 'type', None) == 'feature_store' and getattr(fetcher_conf, 'timestamp_column', None):
            exclude_columns.append(fetcher_conf.timestamp_column)
        
        # 실제로 존재하는 컬럼만 반환
        return [col for col in exclude_columns if col in df.columns]

    def _check_missing_values_warning(self, X: pd.DataFrame, threshold: float = 0.05):
        """
        5% 이상 결측치가 있는 컬럼을 감지하고 경고를 출력합니다.
        
        Args:
            X: 특성 데이터프레임
            threshold: 결측치 비율 임계값 (기본값: 0.05 = 5%)
        """
        if X.empty:
            return
            
        missing_info = []
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_ratio = missing_count / len(X)
            
            if missing_ratio >= threshold:
                missing_info.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_ratio': missing_ratio,
                    'total_rows': len(X)
                })
        
        if missing_info:
            missing_summary = f"{len(missing_info)}개 컬럼에 결측치 {threshold:.0%} 이상"
            self.console.warning(
                f"[TimeseriesDataHandler] 결측치 경고: {missing_summary}",
                rich_message=f"⚠️ Timeseries missing data warning: {missing_summary}"
            )
            for info in missing_info:
                self.console.warning(
                    f"  • {info['column']}: {info['missing_count']:,}개 ({info['missing_ratio']:.1%}) / 전체 {info['total_rows']:,}개 행"
                )
            self.console.warning("  💡 전처리 단계에서 결측치 처리를 고려해보세요 (Imputation, 컬럼 제거 등)")
        else:
            self.console.log_processing_step(
                "TimeSeries 결측치 검사 완료",
                f"모든 특성 컬럼의 결측치 비율이 {threshold:.0%} 미만"
            )


# Self-registration
DataHandlerRegistry.register("timeseries", TimeseriesDataHandler)