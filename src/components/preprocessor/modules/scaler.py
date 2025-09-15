# src/components/_preprocessor/_steps/_scaler.py
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd
from src.interface import BasePreprocessor
from src.utils.core.console import get_console
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
        self.console = get_console()

    def fit(self, X: pd.DataFrame, y=None):
        self.console.info("StandardScaler 학습을 시작합니다",
                         rich_message="📏 Standard Scaler training started")

        # 적용 가능한 컬럼 필터링
        applicable_cols = self.get_applicable_columns(X)
        self.console.info(f"표준화 적용 가능한 숫자형 컬럼: {len(applicable_cols)}개 발견",
                         rich_message=f"   🔍 Found [green]{len(applicable_cols)}[/green] numeric columns for standardization")

        if not applicable_cols:
            # 숫자형 컬럼이 없는 경우
            self.console.info("표준화할 숫자형 컬럼이 없어 스킵됩니다",
                            rich_message="   ⚠️  No numeric columns found - skipping standardization")
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
            self.console.info(f"상수값 또는 NaN 컬럼 제외: {skipped_cols}",
                            rich_message=f"   🚫 Excluded constant/NaN columns: [yellow]{len(skipped_cols)}[/yellow]")

        if valid_cols:
            self.scaler.fit(X[valid_cols])
            self._fitted_columns = valid_cols
            self.console.info(f"StandardScaler 학습 완료 - 적용 컬럼: {len(valid_cols)}개",
                            rich_message=f"   ✅ Fitted StandardScaler on [green]{len(valid_cols)}[/green] columns")
        else:
            # 모든 컬럼이 상수이거나 NaN인 경우
            self._fitted_columns = []
            self.console.info("유효한 컬럼이 없어 StandardScaler 학습이 스킵되었습니다",
                            rich_message="   ⚠️  No valid columns - StandardScaler training skipped")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """표준화를 적용하고 DataFrame으로 반환합니다."""
        result = X.copy()

        if self._fitted_columns:
            self.console.info(f"StandardScaler 변환 적용 중 - 대상 컬럼: {len(self._fitted_columns)}개",
                            rich_message=f"   🔄 Applying standardization to [cyan]{len(self._fitted_columns)}[/cyan] columns")
            # 피팅된 컬럼만 스케일링 적용
            fitted_data = X[self._fitted_columns]
            scaled_data = self.scaler.transform(fitted_data)
            result[self._fitted_columns] = scaled_data

            # 변환 결과 통계
            means = scaled_data.mean(axis=0)
            stds = scaled_data.std(axis=0)
            self.console.info(f"표준화 완료 - 평균: ~{means.mean():.4f}, 표준편차: ~{stds.mean():.4f}",
                            rich_message=f"   📊 Standardization completed - mean: [dim]~{means.mean():.4f}[/dim], std: [dim]~{stds.mean():.4f}[/dim]")

        return result
    
    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """StandardScaler는 컬럼명을 보존합니다."""
        return input_columns
    
    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존합니다."""
        return True
    
    def get_application_type(self) -> str:
        """StandardScaler는 모든 숫자형 컬럼에 자동 적용됩니다."""
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

class MinMaxScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: MinMaxScaler를 위한 래퍼
    모든 숫자형 컬럼에 자동으로 Min-Max 스케일링을 적용합니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = MinMaxScaler()
        self._fitted_columns = []
        self.console = get_console()

    def fit(self, X: pd.DataFrame, y=None):
        self.console.info("MinMaxScaler 학습을 시작합니다",
                         rich_message="📏 Min-Max Scaler training started")

        # 적용 가능한 컬럼 필터링
        applicable_cols = self.get_applicable_columns(X)
        self.console.info(f"정규화 적용 가능한 숫자형 컬럼: {len(applicable_cols)}개 발견",
                         rich_message=f"   🔍 Found [green]{len(applicable_cols)}[/green] numeric columns for normalization")

        if not applicable_cols:
            self.console.info("정규화할 숫자형 컬럼이 없어 스킵됩니다",
                            rich_message="   ⚠️  No numeric columns found - skipping normalization")
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
            self.console.info(f"상수값 또는 NaN 컬럼 제외: {skipped_cols}",
                            rich_message=f"   🚫 Excluded constant/NaN columns: [yellow]{len(skipped_cols)}[/yellow]")

        if valid_cols:
            self.scaler.fit(X[valid_cols])
            self._fitted_columns = valid_cols
            self.console.info(f"MinMaxScaler 학습 완료 - 적용 컬럼: {len(valid_cols)}개",
                            rich_message=f"   ✅ Fitted MinMaxScaler on [green]{len(valid_cols)}[/green] columns")
        else:
            self._fitted_columns = []
            self.console.info("유효한 컬럼이 없어 MinMaxScaler 학습이 스킵되었습니다",
                            rich_message="   ⚠️  No valid columns - MinMaxScaler training skipped")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Min-Max 스케일링을 적용하고 DataFrame으로 반환합니다."""
        result = X.copy()

        if self._fitted_columns:
            self.console.info(f"MinMaxScaler 변환 적용 중 - 대상 컬럼: {len(self._fitted_columns)}개",
                            rich_message=f"   🔄 Applying min-max normalization to [cyan]{len(self._fitted_columns)}[/cyan] columns")
            fitted_data = X[self._fitted_columns]
            scaled_data = self.scaler.transform(fitted_data)
            result[self._fitted_columns] = scaled_data

            # 변환 결과 통계
            mins = scaled_data.min(axis=0)
            maxs = scaled_data.max(axis=0)
            self.console.info(f"정규화 완료 - 최솟값: ~{mins.mean():.4f}, 최댓값: ~{maxs.mean():.4f}",
                            rich_message=f"   📊 Normalization completed - min: [dim]~{mins.mean():.4f}[/dim], max: [dim]~{maxs.mean():.4f}[/dim]")

        return result
    
    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        return input_columns
    
    def preserves_column_names(self) -> bool:
        return True
    
    def get_application_type(self) -> str:
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

class RobustScalerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: RobustScaler를 위한 래퍼
    중앙값과 사분위수를 사용하여 스케일링하므로, 이상치(outlier)에 덜 민감합니다.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns  # global 타입이므로 무시됨
        self.scaler = RobustScaler()
        self._fitted_columns = []
        self.console = get_console()

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
        return 'global'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        return [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

PreprocessorStepRegistry.register("standard_scaler", StandardScalerWrapper)
PreprocessorStepRegistry.register("min_max_scaler", MinMaxScalerWrapper)
PreprocessorStepRegistry.register("robust_scaler", RobustScalerWrapper) 