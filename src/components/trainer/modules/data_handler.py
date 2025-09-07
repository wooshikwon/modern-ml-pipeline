# src/components/_trainer/_data_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from src.settings import Settings
from src.utils.system.logger import logger  # ✅ logger import 추가


def split_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test 분할 (조건부 stratify)"""
    data_interface = settings.recipe.data.data_interface  # ✅ 경로 수정
    task_choice = settings.recipe.task_choice
    test_size = 0.2

    stratify_series = None
    if task_choice == "classification":
        target_col = data_interface.target_column
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            # 각 클래스 최소 2개, 테스트 셋에 최소 1개 이상 들어갈 수 있는지 확인
            if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                stratify_series = df[target_col]
    elif task_choice == "causal":
        treatment_col = data_interface.treatment_column
        if treatment_col in df.columns:
            counts = df[treatment_col].value_counts()
            if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                stratify_series = df[treatment_col]

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=stratify_series
    )
    return train_df, test_df


def _get_stratify_column_data(df: pd.DataFrame, settings: Settings):
    """Stratify용 컬럼 데이터 추출"""
    data_interface = settings.recipe.data.data_interface
    task_choice = settings.recipe.task_choice
    
    if task_choice == "classification":
        target_col = data_interface.target_column
        return df[target_col] if target_col in df.columns else None
    elif task_choice == "causal":
        treatment_col = data_interface.treatment_column
        return df[treatment_col] if treatment_col in df.columns else None
    else:
        return None


def _get_exclude_columns(settings: Settings, df: pd.DataFrame) -> list:
    """
    데이터에서 제외할 컬럼 목록 반환
    Entity columns와 timestamp columns만 제외 (Recipe v3.0 구조에 맞게 수정)
    """
    data_interface = settings.recipe.data.data_interface
    fetcher_conf = settings.recipe.data.fetcher
    
    exclude_columns = []
    
    # Entity columns 추가
    try:
        exclude_columns.extend(data_interface.entity_columns or [])
    except Exception:
        pass
    
    # Timestamp column 추가
    try:
        ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
        if ts_col:
            exclude_columns.append(ts_col)
    except Exception:
        pass

    # 실제 존재하는 컬럼만 반환
    return [c for c in exclude_columns if c in df.columns]


def prepare_training_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """동적 데이터 준비 + feature_columns null 처리"""
    data_interface = settings.recipe.data.data_interface  # ✅ 경로 수정
    task_choice = settings.recipe.task_choice
    exclude_cols = _get_exclude_columns(settings, df)
    
    if task_choice in ["classification", "regression"]:
        target_col = data_interface.target_column
        
        # ✅ feature_columns null 처리 로직
        if data_interface.feature_columns is None:
            # 자동 선택: target, treatment, entity 제외 모든 컬럼
            auto_exclude = [target_col] + exclude_cols
            if data_interface.treatment_column:
                auto_exclude.append(data_interface.treatment_column)
            
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택: {list(X.columns)}")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = [target_col] + exclude_cols
            if data_interface.treatment_column:
                forbidden_cols.append(data_interface.treatment_column)
            
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"target, treatment, entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[data_interface.feature_columns]
            
        # 숫자형 컬럼만 사용하여 모델 입력 구성
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        _check_missing_values_warning(X)
        
        y = df[target_col]
        additional_data = {}
        
    elif task_choice == "clustering":
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (clustering): {list(X.columns)}")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = exclude_cols  # entity, timestamp 컬럼만
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        _check_missing_values_warning(X)
        
        y = None
        additional_data = {}
        
    elif task_choice == "causal":
        target_col = data_interface.target_column
        treatment_col = data_interface.treatment_column
        
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = [target_col, treatment_col] + exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (causal): {list(X.columns)}")
        else:
            # 명시적 선택 - 금지된 컬럼 validation
            forbidden_cols = [target_col, treatment_col] + exclude_cols
            forbidden_cols = [c for c in forbidden_cols if c and c in df.columns]
            overlap = set(data_interface.feature_columns) & set(forbidden_cols)
            if overlap:
                raise ValueError(f"feature_columns에 금지된 컬럼이 포함되어 있습니다: {list(overlap)}. "
                               f"target, treatment, entity, timestamp 컬럼은 feature로 사용할 수 없습니다.")
            
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        
        # 5% 이상 결측 컬럼 경고
        _check_missing_values_warning(X)
        
        y = df[target_col]
        additional_data = {
            'treatment': df[treatment_col],
            'treatment_value': getattr(data_interface, 'treatment_value', 1)
        }
    else:
        raise ValueError(f"지원하지 않는 task_choice: {task_choice}")
    
    return X, y, additional_data


def _check_missing_values_warning(X: pd.DataFrame, threshold: float = 0.05):
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
