# src/components/datahandler/modules/deeplearning_handler.py
"""
DeepLearning DataHandler - 딥러닝 전용 데이터 처리

PyTorch 기반 딥러닝 모델들을 위한 특화된 데이터 처리를 제공합니다.
- LSTM용 시퀀스 데이터 생성 (3D: samples, seq_len, features)
- 시간 기준 분할로 데이터 누수 방지
- 딥러닝 모델에 최적화된 배치 처리
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union
from datetime import datetime

from src.interface import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.system.logger import logger
from src.utils.system.console_manager import get_console


class DeepLearningDataHandler(BaseDataHandler):
    """딥러닝 전용 DataHandler - 시퀀스 처리, 배치 생성 특화"""
    
    def __init__(self, settings):
        super().__init__(settings)
        # Recipe 스키마에 따라 task_type은 recipe.task_choice에서 가져옵니다.
        self.task_type = settings.recipe.task_choice
        self.console = get_console(settings)
        
        # 딥러닝 전용 설정들 (Recipe Schema에서 확장 예정)
        self.sequence_length = getattr(self.data_interface, 'sequence_length', 30)
        self.use_gpu = getattr(self.data_interface, 'use_gpu', True)
        
        self.console.info(f"🧠 DeepLearning DataHandler 초기화")
        self.console.info(f"   Task Type: {self.task_type}")
        self.console.info(f"   Sequence Length: {self.sequence_length}")
        self.console.info(f"   Use GPU: {self.use_gpu}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """딥러닝 데이터 검증"""
        super().validate_data(df)  # 기본 검증
        
        target_col = self.data_interface.target_column
        
        # Target 컬럼 존재 검증
        if target_col not in df.columns:
            raise ValueError(f"Target 컬럼 '{target_col}'을 찾을 수 없습니다")
        
        # Task별 추가 검증
        if self.task_type == "timeseries":
            timestamp_col = self.data_interface.timestamp_column
            if not timestamp_col or timestamp_col not in df.columns:
                raise ValueError(f"TimeSeries task에 필요한 timestamp_column '{timestamp_col}'을 찾을 수 없습니다")
            
            # 타임스탬프 데이터 타입 검증
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    self.console.info(f"Timestamp 컬럼을 datetime으로 변환했습니다: {timestamp_col}")
                except:
                    raise ValueError(f"Timestamp 컬럼 '{timestamp_col}'을 datetime으로 변환할 수 없습니다")
            
            # 시퀀스 길이 검증 (실행 고정 길이 사용)
            seq_len = self._get_sequence_length_for_run(df)
            if len(df) < seq_len + 1:
                raise ValueError(f"시퀀스 생성을 위해 최소 {seq_len + 1}개 행이 필요합니다. "
                               f"현재 데이터: {len(df)}개 행")
        
        return True
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Dict[str, Any]]:
        """딥러닝을 위한 데이터 준비"""
        # 데이터 검증
        self.validate_data(df)
        
        if self.task_type == "timeseries":
            return self._prepare_timeseries_sequences(df)
        elif self.task_type in ["classification", "regression"]:
            return self._prepare_tabular_data(df)
        else:
            raise ValueError(f"DeepLearning handler에서 지원하지 않는 task: {self.task_type}")

    def split_and_prepare(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, Any, Dict[str, Any],
        pd.DataFrame, Any, Dict[str, Any]
    ]:
        """시계열의 경우 train 분할 길이에 맞춰 시퀀스 길이를 고정한 뒤 변환합니다."""
        train_df, test_df = self.split_data(df)
        if self.task_type == "timeseries":
            # Train/Test 크기를 모두 고려하여 최소 한 개 이상의 시퀀스가 생성되도록 고정 길이 결정
            effective_train = self._effective_sequence_length(train_df)
            max_for_test = max(5, len(test_df) - 1)
            self._sequence_len_for_run = max(5, min(effective_train, max_for_test, self.sequence_length))
        X_train, y_train, add_train = self.prepare_data(train_df)
        X_test, y_test, add_test = self.prepare_data(test_df)
        return X_train, y_train, add_train, X_test, y_test, add_test
    
    def _prepare_timeseries_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """시계열 → LSTM 시퀀스 데이터 변환"""
        timestamp_col = self.data_interface.timestamp_column
        target_col = self.data_interface.target_column
        
        # 시간순 정렬 (필수)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Feature columns 추출
        exclude_cols = [target_col, timestamp_col] + (self.data_interface.entity_columns or [])
        
        if self.data_interface.feature_columns:
            # 명시적 feature 선택
            feature_cols = self.data_interface.feature_columns
            forbidden_cols = [c for c in exclude_cols if c and c in df.columns]
            overlap = set(feature_cols) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"timestamp, target, entity 컬럼은 feature로 사용할 수 없습니다.")
        else:
            # 자동 feature 선택: 수치형 컬럼에서 제외 컬럼 빼기
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_cols]
        
        self.console.info(f"📈 TimeSeries feature columns ({len(feature_cols)}): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        
        # 결측치 체크 및 경고
        missing_info = []
        for col in feature_cols:
            if col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > 0.05:  # 5% 이상
                    missing_info.append((col, missing_ratio))
        
        if missing_info:
            self.console.warning("⚠️  결측치가 많은 feature 컬럼이 발견되었습니다:")
            for col, ratio in missing_info:
                self.console.warning(f"   - {col}: {ratio:.1%}")
            self.console.warning("   💡 전처리 단계에서 결측치 처리를 고려해보세요")
        
        # Sliding window로 시퀀스 생성 (고정된 실행 시퀀스 길이 사용)
        X_sequences, y_sequences = [], []
        seq_len = self._get_sequence_length_for_run(df)
        
        for i in range(seq_len, len(df)):
            # X: 과거 seq_len개 시점의 feature들 (3D: [seq_len, n_features])
            X_seq = df.iloc[i-seq_len:i][feature_cols].values
            # y: 현재 시점의 target 값
            y_seq = df.iloc[i][target_col]
            
            # 결측치가 있는 시퀀스 스킵
            if np.isnan(X_seq).any() or np.isnan(y_seq):
                continue
                
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)  # Shape: [n_samples, seq_len, n_features]
        y_sequences = np.array(y_sequences)  # Shape: [n_samples]
        
        self.console.info(f"✅ 시퀀스 생성 완료: {X_sequences.shape} sequences → {y_sequences.shape} targets")
        self.console.info(f"   Sequence length: {seq_len}, Features: {X_sequences.shape[-1]}")
        
        # ✅ BaseModel 호환을 위해 DataFrame으로 변환 (메타데이터에 original shape 저장)
        original_shape = X_sequences.shape  # (n_samples, seq_len, n_features)
        X_flattened = X_sequences.reshape(len(X_sequences), -1)  # (n_samples, seq_len * n_features)
        
        # DataFrame으로 변환 (컬럼명: seq0_feat0, seq0_feat1, ..., seq1_feat0, ...)
        n_samples, seq_len_used, n_features = original_shape
        column_names = [f"seq{t}_feat{f}" for t in range(seq_len_used) for f in range(n_features)]
        X_df = pd.DataFrame(X_flattened, columns=column_names)
        y_series = pd.Series(y_sequences, name='target')
        
        self.console.info(f"🔄 DataFrame 변환 완료: {original_shape} → {X_df.shape} (BaseModel 호환)")
        
        additional_data = {
            'sequence_length': seq_len,
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'is_timeseries': True,
            'is_timeseries_sequence': True,  # LSTM 모델이 인식할 플래그
            'original_sequence_shape': original_shape,  # (n_samples, seq_len, n_features)
            'data_shape': X_df.shape,  # flattened shape
            'original_timestamps': df[timestamp_col].iloc[seq_len:].reset_index(drop=True)
        }
        
        return X_df, y_series, additional_data
    
    def _prepare_tabular_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """일반 테이블 데이터 → 딥러닝용 배치 처리"""
        target_col = self.data_interface.target_column
        
        # Feature selection (기존 DataHandler 로직과 동일)
        exclude_cols = [target_col] + (self.data_interface.entity_columns or [])
        
        if self.data_interface.feature_columns:
            # 명시적 feature 선택
            forbidden_cols = [c for c in exclude_cols if c and c in df.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"target, entity 컬럼은 feature로 사용할 수 없습니다.")
            X = df[self.data_interface.feature_columns]
        else:
            # 자동 feature 선택
            X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
            X = X.select_dtypes(include=[np.number])
        
        y = df[target_col]
        
        self.console.info(f"📊 Tabular data prepared: {X.shape} features → {y.shape} targets")
        
        # 결측치 체크
        missing_info = []
        for col in X.columns:
            missing_ratio = X[col].isnull().sum() / len(X)
            if missing_ratio > 0.05:  # 5% 이상
                missing_info.append((col, missing_ratio))
        
        if missing_info:
            self.console.warning("⚠️  결측치가 많은 feature 컬럼이 발견되었습니다:")
            for col, ratio in missing_info:
                self.console.warning(f"   - {col}: {ratio:.1%}")
            self.console.warning("   💡 전처리 단계에서 결측치 처리를 고려해보세요")
        
        additional_data = {
            'is_timeseries': False,
            'feature_columns': list(X.columns),
            'n_features': len(X.columns),
            'data_shape': X.shape
        }
        
        return X, y, additional_data
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 분할 (시계열은 시간 기준, 일반은 random)"""
        if self.task_type == "timeseries":
            return self._time_based_split(df)
        else:
            # 일반 데이터는 random split
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            self.console.info(f"📊 Random split 완료: Train({len(train_df)}) / Test({len(test_df)})")
            return train_df, test_df
    
    def _time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시계열 시간 기준 분할 (Data Leakage 방지)"""
        timestamp_col = self.data_interface.timestamp_column
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 시퀀스 생성에 충분한 데이터가 있는지 확인
        min_train_size = self.sequence_length + 10  # 최소한의 학습 데이터
        total_sequences = len(df_sorted) - self.sequence_length
        
        if total_sequences < 20:  # 최소 20개 시퀀스
            raise ValueError(f"시퀀스 생성 후 데이터가 부족합니다. "
                           f"전체 {len(df_sorted)}행 → {total_sequences}개 시퀀스. "
                           f"최소 {self.sequence_length + 20}행이 필요합니다.")
        
        # 80-20 분할
        split_idx = int(len(df_sorted) * 0.8)
        
        # train에서도 시퀀스 생성이 가능하도록 조정
        if split_idx < min_train_size:
            split_idx = min_train_size
            self.console.warning(f"Train 데이터가 부족하여 분할 지점을 조정했습니다: {split_idx}")
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        train_sequences = len(train_df) - self.sequence_length
        test_sequences = len(test_df) - self.sequence_length
        
        train_period = f"{train_df[timestamp_col].min()} ~ {train_df[timestamp_col].max()}"
        test_period = f"{test_df[timestamp_col].min()} ~ {test_df[timestamp_col].max()}"
        
        self.console.info(f"🕐 시계열 시간 기준 분할:")
        self.console.info(f"   Train ({len(train_df)}행 → ~{max(0, train_sequences)}개 시퀀스): {train_period}")
        self.console.info(f"   Test ({len(test_df)}행 → ~{max(0, test_sequences)}개 시퀀스): {test_period}")
        
        return train_df, test_df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """딥러닝 데이터 메타 정보 반환"""
        base_info = super().get_data_info(df)
        
        # 딥러닝 특화 정보 추가
        deep_learning_info = {
            'task_type': self.task_type,
            'sequence_length': self.sequence_length,
            'estimated_sequences': max(0, len(df) - self.sequence_length) if self.task_type == "timeseries" else None,
            'use_gpu': self.use_gpu
        }
        
        # Target 정보
        target_col = self.data_interface.target_column
        if target_col in df.columns:
            deep_learning_info['target_info'] = {
                'name': target_col,
                'dtype': str(df[target_col].dtype),
                'unique_values': df[target_col].nunique(),
                'missing_ratio': df[target_col].isnull().sum() / len(df),
                'stats': df[target_col].describe().to_dict() if df[target_col].dtype in ['int64', 'float64'] else None
            }
        
        # Timestamp 정보 (시계열인 경우)
        if self.task_type == "timeseries":
            timestamp_col = self.data_interface.timestamp_column
            if timestamp_col and timestamp_col in df.columns:
                deep_learning_info['timestamp_info'] = {
                    'name': timestamp_col,
                    'dtype': str(df[timestamp_col].dtype),
                    'date_range': f"{df[timestamp_col].min()} ~ {df[timestamp_col].max()}",
                    'frequency_hint': self._estimate_frequency(df[timestamp_col])
                }
        
        # 두 딕셔너리 병합
        return {**base_info, 'deeplearning_specific': deep_learning_info}
    
    def _estimate_frequency(self, timestamp_series: pd.Series) -> str:
        """시계열 주기 추정"""
        try:
            # 정렬된 타임스탬프의 간격들 계산
            sorted_ts = timestamp_series.dropna().sort_values()
            if len(sorted_ts) < 2:
                return "unknown"
            
            deltas = sorted_ts.diff().dropna()
            if len(deltas) == 0:
                return "unknown"
            
            # 가장 빈번한 간격 찾기
            mode_delta = deltas.mode()
            if len(mode_delta) == 0:
                return "unknown"
            
            mode_seconds = mode_delta.iloc[0].total_seconds()
            
            # 일반적인 주기들과 매칭
            if mode_seconds == 86400:  # 1 day
                return "daily"
            elif mode_seconds == 3600:  # 1 hour
                return "hourly"
            elif mode_seconds == 60:  # 1 minute
                return "minutely"
            elif mode_seconds == 604800:  # 1 week
                return "weekly"
            elif 2505600 <= mode_seconds <= 2678400:  # 29-31 days
                return "monthly"
            else:
                return f"~{mode_seconds}s"
                
        except Exception:
            return "unknown"

    def _get_sequence_length_for_run(self, df: pd.DataFrame) -> int:
        # 고정 길이가 설정되어 있으면 사용, 없으면 데이터 기반 동적 결정
        if hasattr(self, "_sequence_len_for_run") and isinstance(self._sequence_len_for_run, int):
            return self._sequence_len_for_run
        return self._effective_sequence_length(df)

    def _effective_sequence_length(self, df: pd.DataFrame) -> int:
        """데이터 크기에 따라 시퀀스 길이를 동적으로 조정합니다 (최소 5)."""
        try:
            n = len(df)
            return max(5, min(self.sequence_length, max(5, n // 3)))
        except Exception:
            return max(5, self.sequence_length)

# Registry 자동 등록
DataHandlerRegistry.register("deeplearning", DeepLearningDataHandler)