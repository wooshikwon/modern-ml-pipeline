from __future__ import annotations
from typing import Dict, Any
import pandas as pd


def create_data_interface_schema_for_storage(*, data_interface, df: pd.DataFrame, task_choice: str) -> Dict[str, Any]:
    """
    Create a minimal DataInterface schema description for storage adapters.
    Matches Factory.create_pyfunc_wrapper call signature.
    """
    entity_cols = getattr(data_interface, 'entity_columns', []) or []
    ts_col = getattr(data_interface, 'timestamp_column', None)
    target_col = getattr(data_interface, 'target_column', None)

    exclude = set(entity_cols)
    if ts_col:
        exclude.add(ts_col)
    if target_col:
        exclude.add(target_col)

    feature_columns = [c for c in df.columns if c not in exclude]

    required_columns = []
    if entity_cols:
        required_columns.extend(entity_cols)
    if ts_col:
        required_columns.append(ts_col)
    if target_col:
        required_columns.append(target_col)

    return {
        'task_type': task_choice,
        'entity_columns': entity_cols,
        'timestamp_column': ts_col,
        'target_column': target_col,
        'feature_columns': feature_columns,
        'required_columns': required_columns,
    }