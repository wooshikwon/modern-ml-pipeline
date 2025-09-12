# src/components/datahandler/modules/tabular_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.interface import BaseDataHandler
from ..registry import DataHandlerRegistry
from src.utils.core.console_manager import UnifiedConsole


class TabularDataHandler(BaseDataHandler):
    """전통적인 테이블 형태 ML을 위한 데이터 핸들러 (classification, regression, clustering, causal)"""
    
    def __init__(self, settings, data_interface=None):
        # data_interface는 BaseDataHandler에서 자동으로 설정됨
        super().__init__(settings)
        self.console = UnifiedConsole(settings)
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        4-way 데이터 분할: train/validation/test/calibration (Data Leakage 방지)
        
        Args:
            df: 전체 데이터프레임
            
        Returns:
            Dict[str, pd.DataFrame]: 분할된 데이터
            - 'train': 모델 학습용
            - 'validation': 하이퍼파라미터 튜닝용  
            - 'test': 최종 평가용
            - 'calibration': 확률 보정용 (classification + calibration 활성화 시에만)
        """
        # 분할 비율 가져오기
        split_config = getattr(self.settings.recipe.data, 'split', None)
        task_choice = self.settings.recipe.task_choice
        
        # Split 설정이 없으면 오류 발생 (Recipe Builder에서 설정되어야 함)
        if not split_config:
            raise ValueError(
                "데이터 분할 설정(data.split)이 없습니다. "
                "Recipe Builder를 통해 split 비율을 설정하거나, recipe.yaml에서 data.split 섹션을 정의하세요."
            )
        
        # Split 설정에서 비율 추출 (Pydantic 모델 속성 접근)
        train_ratio = split_config.train
        validation_ratio = split_config.validation
        test_ratio = split_config.test
        calibration_ratio = split_config.calibration  # DataSplit 모델에서 기본값 0.0 설정됨
        
        # 비율 합 검증
        total_ratio = train_ratio + validation_ratio + test_ratio + calibration_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total_ratio}")
        
        # Stratification 설정
        stratify_series = self._get_stratify_series(df)
        
        # 순차적 분할 (Data Leakage 방지를 위한 올바른 순서)
        # 1. 전체 데이터에서 test 분리
        if test_ratio > 0:
            remaining_df, test_df = train_test_split(
                df, test_size=test_ratio, random_state=42, 
                stratify=stratify_series if stratify_series is not None else None
            )
        else:
            remaining_df = df
            test_df = pd.DataFrame()
        
        # 2. 남은 데이터에서 calibration 분리 (classification + calibration 활성화 시)
        if calibration_ratio > 0:
            calib_size_from_remaining = calibration_ratio / (1 - test_ratio)
            # Update stratify for remaining data
            stratify_remaining = self._get_stratify_series(remaining_df) if stratify_series is not None else None
            
            remaining_df2, calibration_df = train_test_split(
                remaining_df, test_size=calib_size_from_remaining, random_state=42,
                stratify=stratify_remaining
            )
        else:
            remaining_df2 = remaining_df
            calibration_df = pd.DataFrame()
        
        # 3. 남은 데이터를 train/validation으로 분할
        if validation_ratio > 0:
            val_size_from_remaining = validation_ratio / (train_ratio + validation_ratio)
            # Update stratify for remaining data
            stratify_remaining2 = self._get_stratify_series(remaining_df2) if stratify_series is not None else None
            
            train_df, validation_df = train_test_split(
                remaining_df2, test_size=val_size_from_remaining, random_state=42,
                stratify=stratify_remaining2
            )
        else:
            train_df = remaining_df2
            validation_df = pd.DataFrame()
        
        # 결과 로깅
        self.console.info(
            f"Data split completed - Train: {len(train_df)}, Val: {len(validation_df)}, "
            f"Test: {len(test_df)}, Calib: {len(calibration_df)}",
            rich_message=f"📊 Data split: Train([green]{len(train_df)}[/green]) "
                        f"Val([blue]{len(validation_df)}[/blue]) "
                        f"Test([yellow]{len(test_df)}[/yellow]) "
                        f"Calib([purple]{len(calibration_df)}[/purple])"
        )
        
        return {
            'train': train_df,
            'validation': validation_df, 
            'test': test_df,
            'calibration': calibration_df if calibration_ratio > 0 else None
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
        """테이블 형태 데이터 준비 (기존 prepare_training_data 로직)"""
        task_choice = self.settings.recipe.task_choice
        exclude_cols = self._get_exclude_columns(df)
        
        if task_choice in ["classification", "regression", "timeseries"]:
            return self._prepare_supervised_data(df, exclude_cols)
        elif task_choice == "clustering":
            return self._prepare_clustering_data(df, exclude_cols)  
        elif task_choice == "causal":
            return self._prepare_causal_data(df, exclude_cols)
        else:
            raise ValueError(f"지원하지 않는 task_choice: {task_choice}")
    
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
    
    def split_and_prepare(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, Any, Dict[str, Any],  # train
        pd.DataFrame, Any, Dict[str, Any],  # validation
        pd.DataFrame, Any, Dict[str, Any],  # test
        Optional[Tuple[pd.DataFrame, Any, Dict[str, Any]]]  # calibration (optional)
    ]:
        """
        4-way 데이터 분할 + 각 분할에 대해 prepare_data 수행
        
        Returns:
            (X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data)
            calibration_data는 (X_calib, y_calib, add_calib) 또는 None
        """
        split_results = self.split_data(df)
        
        # Train data 준비
        X_train, y_train, add_train = self.prepare_data(split_results['train'])
        
        # Validation data 준비
        if not split_results['validation'].empty:
            X_val, y_val, add_val = self.prepare_data(split_results['validation'])
        else:
            X_val = pd.DataFrame()
            y_val = None
            add_val = {}
        
        # Test data 준비
        if not split_results['test'].empty:
            X_test, y_test, add_test = self.prepare_data(split_results['test'])
        else:
            X_test = pd.DataFrame()
            y_test = None
            add_test = {}
        
        # Calibration data 준비 (있는 경우에만)
        calibration_data = None
        if split_results['calibration'] is not None and not split_results['calibration'].empty:
            X_calib, y_calib, add_calib = self.prepare_data(split_results['calibration'])
            calibration_data = (X_calib, y_calib, add_calib)
        
        return X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data

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
                               rich_message="💡 Consider handling missing values in preprocessing (Imputation, column removal, etc.)")
        else:
            self.console.info(f"모든 특성 컬럼의 결측치 비율이 {threshold:.0%} 미만입니다.",
                            rich_message=f"✅ All feature columns have <{threshold:.0%} missing values")


# Self-registration
DataHandlerRegistry.register("tabular", TabularDataHandler)