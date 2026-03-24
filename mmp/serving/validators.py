"""
서빙 엔드포인트용 입력 검증 함수.

_endpoints.py의 predict()와 predict_batch()에서 중복되던 검증 로직을 통합.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def validate_scalar_values(df: pd.DataFrame) -> None:
    """비스칼라 값(list, dict) 포함 시 HTTPException(422) 발생."""
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            raise HTTPException(
                status_code=422, detail=f"컬럼 '{col}'에 비스칼라 값이 포함되어 있습니다."
            )


def validate_numeric_types(df: pd.DataFrame, model: Any) -> None:
    """
    시그니처 기반 숫자형 컬럼 검증.

    MLflow signature과 data_interface_schema에서 숫자형으로 기대되는 컬럼에
    비수치 값이 있으면 HTTPException(422) 발생.
    실패 시 로깅 후 통과 (best-effort).
    """
    try:
        # data_interface_schema에서 feature_columns 수집
        feature_cols: set[str] = set()
        try:
            wrapped = model.unwrap_python_model()
            di = getattr(wrapped, "data_interface_schema", {}) or {}
            feature_cols = set(di.get("feature_columns") or [])
        except Exception as e:
            logger.debug(f"data_interface_schema 접근 실패 (무시): {type(e).__name__}: {e}")

        # MLflow Signature에서 컬럼 정보 추출
        expected_numeric_cols: set[str] = set()
        sig = getattr(getattr(model, "metadata", None), "signature", None)
        schema_inputs = getattr(sig, "inputs", None)

        if schema_inputs is not None and hasattr(schema_inputs, "inputs"):
            cols = getattr(schema_inputs, "inputs", []) or []
            for c in cols:
                name = getattr(c, "name", None)
                if name:
                    feature_cols.add(name)
                t = getattr(c, "type", None)
                if name and t and str(t).lower() in ("double", "float", "integer", "long"):
                    expected_numeric_cols.add(name)

        # feature 컬럼 내 숫자형 기대 컬럼 검증
        for col in feature_cols or df.columns:
            if col in df.columns and (
                not expected_numeric_cols or col in expected_numeric_cols
            ):
                val_is_numeric = (
                    df[col]
                    .apply(lambda v: isinstance(v, (int, float)) and np.isfinite(v))
                    .all()
                )
                if not val_is_numeric:
                    raise HTTPException(
                        status_code=422, detail=f"컬럼 '{col}'은 숫자형이어야 합니다."
                    )
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"숫자형 검증 fallback (무시): {type(e).__name__}: {e}")


def validate_required_columns(
    df: pd.DataFrame,
    model: Any,
    prediction_request_cls: Optional[Any] = None,
) -> None:
    """
    필수 컬럼 검증.

    DataInterface + FeatureStore fetcher + MLflow signature 기반으로 필수 컬럼을 결정하고,
    누락 시 HTTPException(422) 발생.

    prediction_request_cls가 주어지면 PredictionRequest의 model_fields도 선제 검증.
    """
    # PredictionRequest 스키마 기반 선제 검증
    if prediction_request_cls is not None:
        try:
            pr_fields = getattr(prediction_request_cls, "model_fields", {})
            if pr_fields:
                required = set(pr_fields.keys())
                missing = [c for c in sorted(required) if c not in df.columns]
                if missing:
                    raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
        except HTTPException:
            raise
        except Exception as e:
            logger.debug(f"PredictionRequest 선제 검증 fallback (무시): {type(e).__name__}: {e}")

    # DataInterface + Fetcher + MLflow signature 기반 검증
    try:
        required_cols: set[str] = set()
        wrapped = model.unwrap_python_model()
        di = getattr(wrapped, "data_interface_schema", {}) or {}
        trained_fetcher = getattr(wrapped, "trained_fetcher", None)

        # Fetcher 유무에 따라 필수 컬럼 결정
        has_feature_store_fetcher = trained_fetcher is not None and hasattr(
            trained_fetcher, "_fetcher_config"
        )

        if has_feature_store_fetcher:
            # Feature Store fetcher: entity_columns만 필수
            entity_cols = di.get("entity_columns", [])
            required_cols.update(entity_cols)
        else:
            # pass_through 또는 fetcher 없음: required_columns 필수
            required_cols.update(di.get("required_columns", []) or [])

        # MLflow signature에서 보완 (DataInterface 정보가 없는 경우)
        if not required_cols:
            sig = getattr(getattr(model, "metadata", None), "signature", None)
            schema_inputs = getattr(sig, "inputs", None)
            if schema_inputs is not None:
                try:
                    if hasattr(schema_inputs, "input_names") and callable(
                        getattr(schema_inputs, "input_names", None)
                    ):
                        for n in schema_inputs.input_names():
                            required_cols.add(n)
                except Exception as e:
                    logger.debug(f"input_names() fallback (무시): {type(e).__name__}: {e}")
                if not required_cols and hasattr(schema_inputs, "inputs"):
                    for c in getattr(schema_inputs, "inputs", []) or []:
                        name = getattr(c, "name", None)
                        if name:
                            required_cols.add(name)

        missing = [c for c in sorted(required_cols) if c not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"필수 컬럼 검증 fallback (무시): {type(e).__name__}: {e}")
