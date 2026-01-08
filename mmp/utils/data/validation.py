from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def create_data_interface_schema_for_storage(
    *,
    data_interface,
    df: pd.DataFrame,
    task_choice: str,
    input_feature_columns: Optional[List[str]] = None,
    model_feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    학습-추론 일관성을 위한 DataInterface 스키마 생성.

    Args:
        data_interface: Recipe의 data_interface 설정
        df: 학습 데이터프레임 (전처리 전)
        task_choice: 태스크 유형
        input_feature_columns: 전처리 전 입력 피처 목록 (추론 시 입력 데이터 검증용)
        model_feature_columns: 전처리 후 모델 피처 목록 (모델 입력 검증용)
    """
    entity_cols = getattr(data_interface, "entity_columns", []) or []
    ts_col = getattr(data_interface, "timestamp_column", None)
    target_col = getattr(data_interface, "target_column", None)

    # 입력 피처 결정 (전처리 전)
    if input_feature_columns:
        input_features = input_feature_columns
    else:
        # 명시적 feature_columns 또는 자동 추론
        explicit_feature_cols = getattr(data_interface, "feature_columns", None)
        if explicit_feature_cols:
            input_features = [c for c in explicit_feature_cols if c in df.columns]
        else:
            exclude = set(entity_cols)
            if ts_col:
                exclude.add(ts_col)
            if target_col:
                exclude.add(target_col)
            input_features = [c for c in df.columns if c not in exclude]

    # 모델 피처 결정 (전처리 후, 미제공시 입력 피처와 동일)
    model_features = model_feature_columns if model_feature_columns else input_features

    # API 서빙 시 필수 컬럼: entity + 입력 피처 (전처리 전 기준)
    required_columns = []
    if entity_cols:
        required_columns.extend(entity_cols)
    required_columns.extend(input_features)

    return {
        "task_type": task_choice,
        "entity_columns": entity_cols,
        "timestamp_column": ts_col,
        "target_column": target_col,
        # 전처리 전 피처 (추론 입력용)
        "input_feature_columns": input_features,
        # 전처리 후 피처 (모델 입력용)
        "model_feature_columns": model_features,
        # 하위 호환성 유지 (기존 코드 대응)
        "feature_columns": input_features,
        "required_columns": required_columns,
    }
