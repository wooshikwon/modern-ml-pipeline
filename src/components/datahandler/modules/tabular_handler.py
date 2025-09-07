# src/components/datahandler/modules/tabular_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from src.interface import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.system.logger import logger
from src.utils.system.console_manager import UnifiedConsole


class TabularDataHandler(BaseDataHandler):
    """전통적인 테이블 형태 ML을 위한 데이터 핸들러 (classification, regression, clustering, causal)"""
    
    def __init__(self, settings, data_interface):
        super().__init__(settings, data_interface)
        self.console = UnifiedConsole(settings)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 분할 (조건부 stratify) - 기존 split_data() 로직"""
        test_size = 0.2
        stratify_series = None
        
        if self.data_interface.task_type == "classification":
            target_col = self.data_interface.target_column
            if target_col in df.columns:
                counts = df[target_col].value_counts()
                # 각 클래스 최소 2개, 테스트 셋에 최소 1개 이상 들어갈 수 있는지 확인
                if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                    stratify_series = df[target_col]
        elif self.data_interface.task_type == "causal":
            treatment_col = self.data_interface.treatment_column
            if treatment_col in df.columns:
                counts = df[treatment_col].value_counts()
                if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                    stratify_series = df[treatment_col]

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_series
        )
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """테이블 형태 데이터 준비 (기존 prepare_training_data 로직)"""
        task_type = self.data_interface.task_type
        exclude_cols = self._get_exclude_columns(df)
        
        if task_type in ["classification", "regression"]:
            return self._prepare_supervised_data(df, exclude_cols)
        elif task_type == "clustering":
            return self._prepare_clustering_data(df, exclude_cols)  
        elif task_type == "causal":
            return self._prepare_causal_data(df, exclude_cols)
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")
    
    def _prepare_supervised_data(self, df: pd.DataFrame, exclude_cols: list) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Classification/Regression 데이터 준비"""
        target_col = self.data_interface.target_column
        
        # ✅ feature_columns null 처리 로직 (기존과 동일)
        if self.data_interface.feature_columns is None:
            # 자동 선택: target, treatment, entity 제외 모든 컬럼
            auto_exclude = [target_col] + exclude_cols
            if self.data_interface.treatment_column:
                auto_exclude.append(self.data_interface.treatment_column)
            
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            self.console.info(f"Feature columns 자동 선택: {list(X.columns)}",
                            rich_message=f"   🎯 Auto-selected features: [green]{len(X.columns)}[/green] columns")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = [target_col] + exclude_cols
            if self.data_interface.treatment_column:
                forbidden_cols.append(self.data_interface.treatment_column)
            
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"target, treatment, entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[self.data_interface.feature_columns]
            
        # 숫자형 컬럼만 사용하여 모델 입력 구성
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        self._check_missing_values_warning(X)
        
        y = df[target_col]
        additional_data = {}
        
        return X, y, additional_data
        
    def _prepare_clustering_data(self, df: pd.DataFrame, exclude_cols: list) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Clustering 데이터 준비"""
        # ✅ feature_columns null 처리
        if self.data_interface.feature_columns is None:
            auto_exclude = exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            self.console.info(f"Feature columns 자동 선택 (clustering): {list(X.columns)}",
                            rich_message=f"   🎯 Auto-selected clustering features: [green]{len(X.columns)}[/green] columns")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = exclude_cols  # entity, timestamp 컬럼만
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[self.data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        self._check_missing_values_warning(X)
        
        y = None
        additional_data = {}
        
        return X, y, additional_data
        
    def _prepare_causal_data(self, df: pd.DataFrame, exclude_cols: list) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Causal 데이터 준비"""
        target_col = self.data_interface.target_column
        treatment_col = self.data_interface.treatment_column
        
        # ✅ feature_columns null 처리
        if self.data_interface.feature_columns is None:
            auto_exclude = [target_col, treatment_col] + exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            self.console.info(f"Feature columns 자동 선택 (causal): {list(X.columns)}",
                            rich_message=f"   🎯 Auto-selected causal features: [green]{len(X.columns)}[/green] columns")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = [target_col, treatment_col] + exclude_cols
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(self.data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"target, treatment, entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[self.data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        self._check_missing_values_warning(X)
        
        y = df[target_col]
        additional_data = {
            'treatment': df[treatment_col],
            'treatment_value': getattr(self.data_interface, 'treatment_value', 1)
        }
        
        return X, y, additional_data

    def _get_exclude_columns(self, df: pd.DataFrame) -> list:
        """
        데이터에서 제외할 컬럼 목록 반환 (기존 로직)
        Entity columns와 timestamp columns만 제외 (Recipe v3.0 구조에 맞게 수정)
        """
        fetcher_conf = self.settings.recipe.data.fetcher
        
        exclude_columns = []
        
        # Entity columns는 항상 제외
        if self.data_interface.entity_columns:
            exclude_columns.extend(self.data_interface.entity_columns)
        
        # Feature Store timestamp columns 제외 (offline 모드에서)
        if fetcher_conf.type == "feature_store" and fetcher_conf.timestamp_column:
            exclude_columns.append(fetcher_conf.timestamp_column)
        
        # 실제로 존재하는 컬럼만 반환
        return [col for col in exclude_columns if col in df.columns]

    def _check_missing_values_warning(self, X: pd.DataFrame, threshold: float = 0.05):
        """
        5% 이상 결측치가 있는 컬럼을 감지하고 경고를 출력합니다. (기존 로직)
        
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
            self.console.warning("결측치가 많은 컬럼이 발견되었습니다",
                               rich_message=f"⚠️  Found [red]{len(missing_info)}[/red] columns with high missing values")
            for info in missing_info:
                self.console.warning(
                    f"   - {info['column']}: {info['missing_count']:,}개 ({info['missing_ratio']:.1%}) / 전체 {info['total_rows']:,}개 행",
                    rich_message=f"     [yellow]{info['column']}[/yellow]: [red]{info['missing_ratio']:.1%}[/red] missing"
                )
            self.console.warning("전처리 단계에서 결측치 처리를 고려해보세요",
                               rich_message="💡 Consider handling missing values in preprocessing (Imputation, column removal, etc.)",
                               suggestion="Add imputation steps or remove high-missing columns in preprocessing")
        else:
            self.console.info(f"모든 특성 컬럼의 결측치 비율이 {threshold:.0%} 미만입니다.",
                            rich_message=f"✅ All feature columns have <{threshold:.0%} missing values")


# Self-registration
DataHandlerRegistry.register("tabular", TabularDataHandler)