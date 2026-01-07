from __future__ import annotations

from typing import Any, Dict

# Try to import from existing location if available
try:
    from src.utils.system.schema_utils_impl import generate_training_schema_metadata  # type: ignore
except Exception:
    import pandas as pd

    def generate_training_schema_metadata(
        training_df: pd.DataFrame, data_interface_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        entity_cols = data_interface_config.get("entity_columns") or []
        ts_col = data_interface_config.get("timestamp_column")
        target_col = data_interface_config.get("target_column")

        # 명시적 feature_columns 우선 사용, 없으면 자동 도출
        explicit_feature_cols = data_interface_config.get("feature_columns")
        if explicit_feature_cols:
            feature_columns = [c for c in explicit_feature_cols if c in training_df.columns]
        else:
            exclude = set(entity_cols)
            if ts_col:
                exclude.add(ts_col)
            if target_col:
                exclude.add(target_col)
            feature_columns = [c for c in training_df.columns if c not in exclude]

        # inference_columns는 실제 추론 입력 컬럼 기준 (feature_columns와 동일하게 기본 설정)
        inference_columns = list(feature_columns)
        return {
            "schema_version": "1.0",
            "entity_columns": entity_cols,
            "timestamp_column": ts_col,
            "target_column": target_col,
            "feature_columns": feature_columns,
            "inference_columns": inference_columns,
        }
