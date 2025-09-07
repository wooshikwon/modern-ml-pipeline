# src/components/datahandler/modules/timeseries_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta

from src.interface import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.system.logger import logger


class TimeseriesDataHandler(BaseDataHandler):
    """시계열 데이터 전용 핸들러"""
    
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
                logger.info(f"Timestamp 컬럼을 datetime으로 변환했습니다: {timestamp_col}")
            except:
                raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다")
        
        return True
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시간 기준 분할 (시간 순서 유지)"""
        timestamp_col = self.data_interface.timestamp_column
        test_size = 0.2
        
        # 시간 순서로 정렬 (필수)
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 시간 기준 분할점 계산
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"시계열 시간 기준 분할: Train({len(train_df)}) / Test({len(test_df)})")
        logger.info(f"Train 기간: {train_df[timestamp_col].min()} ~ {train_df[timestamp_col].max()}")
        logger.info(f"Test 기간: {test_df[timestamp_col].min()} ~ {test_df[timestamp_col].max()}")
        
        return train_df, test_df
    
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
            logger.info(f"Timeseries feature columns 자동 선택: {list(X.columns)}")
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
        
        logger.info(f"시계열 특성 생성 완료: {len(df_copy.columns) - len(df)}개 특성 추가")
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
        
        # Feature Store timestamp columns 제외 (offline 모드에서)
        if fetcher_conf.feature_store.enabled and fetcher_conf.feature_store.timestamp_columns:
            exclude_columns.extend(fetcher_conf.feature_store.timestamp_columns)
        
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
            logger.warning("⚠️  결측치가 많은 컬럼이 발견되었습니다:")
            for info in missing_info:
                logger.warning(
                    f"   - {info['column']}: {info['missing_count']:,}개 ({info['missing_ratio']:.1%}) "
                    f"/ 전체 {info['total_rows']:,}개 행"
                )
            logger.warning("   💡 전처리 단계에서 결측치 처리를 고려해보세요 (Imputation, 컬럼 제거 등)")
        else:
            logger.info(f"✅ 모든 특성 컬럼의 결측치 비율이 {threshold:.0%} 미만입니다.")


# Self-registration
DataHandlerRegistry.register("timeseries", TimeseriesDataHandler)