# src/components/_trainer/_data_handler.py
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from src.settings import Settings


def split_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test 분할 (조건부 stratify)"""
    data_interface = settings.recipe.model.data_interface
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

def prepare_training_data(df: pd.DataFrame, settings: Settings) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """동적 데이터 준비 (task_type에 따라 다름)"""
    data_interface = settings.recipe.model.data_interface
    task_type = data_interface.task_type
    
    if task_type in ["classification", "regression"]:
        target_col = data_interface.target_column
        X = df.drop(columns=[target_col])
        y = df[target_col]
        additional_data = {}
    elif task_type == "clustering":
        X = df.copy()
        y = None
        additional_data = {}
    elif task_type == "causal":
        target_col = data_interface.target_column
        treatment_col = data_interface.treatment_column
        X = df.drop(columns=[target_col, treatment_col])
        y = df[target_col]
        additional_data = {
            'treatment': df[treatment_col],
            'treatment_value': data_interface.treatment_value
        }
    else:
        raise ValueError(f"지원하지 않는 task_type: {task_type}")
    
    return X, y, additional_data
