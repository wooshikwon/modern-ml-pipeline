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
    test_size = 0.2

    stratify_series = None
    if data_interface.task_type == "classification":
        target_col = data_interface.target_column
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            # 각 클래스 최소 2개, 테스트 셋에 최소 1개 이상 들어갈 수 있는지 확인
            if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                stratify_series = df[target_col]
    elif data_interface.task_type == "causal":
        treatment_col = data_interface.treatment_column
        if treatment_col in df.columns:
            counts = df[treatment_col].value_counts()
            if len(counts) >= 2 and counts.min() >= 2 and int(len(df) * test_size) >= 1:
                stratify_series = df[treatment_col]

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=stratify_series
    )
    return train_df, test_df


def _get_stratify_column_data(df: pd.DataFrame, data_interface):
    """Stratify용 컬럼 데이터 추출"""
    task_type = data_interface.task_type
    
    if task_type == "classification":
        target_col = data_interface.target_column
        return df[target_col] if target_col in df.columns else None
    elif task_type == "causal":
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
    task_type = data_interface.task_type
    exclude_cols = _get_exclude_columns(settings, df)
    
    if task_type in ["classification", "regression"]:
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
            # 명시적 선택
            X = df[data_interface.feature_columns]
            
        # 숫자형 컬럼만 사용하여 모델 입력 구성
        X = X.select_dtypes(include=[np.number])
        y = df[target_col]
        additional_data = {}
        
    elif task_type == "clustering":
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (clustering): {list(X.columns)}")
        else:
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        y = None
        additional_data = {}
        
    elif task_type == "causal":
        target_col = data_interface.target_column
        treatment_col = data_interface.treatment_column
        
        # ✅ feature_columns null 처리
        if data_interface.feature_columns is None:
            auto_exclude = [target_col, treatment_col] + exclude_cols
            X = df.drop(columns=[c for c in auto_exclude if c in df.columns])
            logger.info(f"Feature columns 자동 선택 (causal): {list(X.columns)}")
        else:
            X = df[data_interface.feature_columns]
            
        X = X.select_dtypes(include=[np.number])
        y = df[target_col]
        additional_data = {
            'treatment': df[treatment_col],
            'treatment_value': getattr(data_interface, 'treatment_value', 1)
        }
    else:
        raise ValueError(f"지원하지 않는 task_type: {task_type}")
    
    return X, y, additional_data
