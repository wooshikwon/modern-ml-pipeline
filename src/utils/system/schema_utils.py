from __future__ import annotations
from typing import Dict, Any

# Try to import from existing location if available
try:
    from src.utils.system.schema_utils_impl import generate_training_schema_metadata  # type: ignore
except Exception:
    import pandas as pd

    def generate_training_schema_metadata(training_df: pd.DataFrame, data_interface_config: Dict[str, Any]) -> Dict[str, Any]:
        entity_cols = data_interface_config.get('entity_columns') or []
        ts_col = data_interface_config.get('timestamp_column')
        target_col = data_interface_config.get('target_column')
        exclude = set(entity_cols)
        if ts_col:
            exclude.add(ts_col)
        if target_col:
            exclude.add(target_col)
        feature_columns = [c for c in training_df.columns if c not in exclude]
        # inference_columns는 실제 추론 입력 컬럼 기준 (feature_columns와 동일하게 기본 설정)
        inference_columns = list(feature_columns)
        return {
            'schema_version': '1.0',
            'entity_columns': entity_cols,
            'timestamp_column': ts_col,
            'target_column': target_col,
            'feature_columns': feature_columns,
            'inference_columns': inference_columns,
        }