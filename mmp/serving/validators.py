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
        # object dtype만 비스칼라 가능성 있음 — 수치형은 스킵
        if df[col].dtype == object:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, dict)):
                raise HTTPException(
                    status_code=422, detail=f"컬럼 '{col}'에 비스칼라 값이 포함되어 있습니다."
                )


def validate_numeric_types(df: pd.DataFrame, model: Any) -> None:
    """
    숫자형 컬럼 검증.

    app_context 캐시가 있으면 캐시 사용, 없으면 기존 reflection 폴백.
    """
    try:
        from mmp.serving._context import app_context

        # 캐시된 feature_columns 사용
        feature_cols = app_context.feature_columns if app_context.feature_columns else None
        expected_numeric = {
            name for name, dtype in app_context.signature_type_map.items()
            if dtype in ("double", "float", "float64", "float32", "integer", "long", "int64", "int32")
        } if app_context.signature_type_map else None

        if feature_cols is None and expected_numeric is None:
            # 캐시 없으면 기존 reflection
            feature_cols, expected_numeric = _extract_numeric_info_from_model(model)

        # 벡터화 검증
        check_cols = feature_cols or set(df.columns)
        for col in check_cols:
            if col not in df.columns:
                continue
            if expected_numeric and col not in expected_numeric:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                # object 컬럼: 수치 변환 시도
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.isna().any() and not df[col].isna().any():
                    raise HTTPException(
                        status_code=422, detail=f"컬럼 '{col}'은 숫자형이어야 합니다."
                    )
            else:
                # 수치 컬럼: inf 체크 (NaN은 허용 — 모델/전처리기가 처리)
                if np.isinf(df[col]).any():
                    raise HTTPException(
                        status_code=422, detail=f"컬럼 '{col}'에 무한대 값이 포함되어 있습니다."
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

    app_context 캐시가 있으면 캐시 사용, 없으면 기존 reflection 폴백.
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

    # 캐시된 required_columns 사용
    try:
        from mmp.serving._context import app_context

        required_cols = app_context.required_columns if app_context.required_columns else None

        if required_cols is None:
            # 캐시 없으면 기존 reflection
            required_cols = _extract_required_columns_from_model(model)

        missing = [c for c in sorted(required_cols) if c not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"필수 컬럼 검증 fallback (무시): {type(e).__name__}: {e}")


# ===============================
# Fallback: 캐시 미사용 시 기존 reflection 로직
# ===============================

def _extract_numeric_info_from_model(model: Any) -> tuple[set[str] | None, set[str] | None]:
    """모델에서 feature_columns와 expected_numeric_cols를 직접 추출 (캐시 미스 시)."""
    feature_cols: set[str] = set()
    expected_numeric: set[str] = set()
    try:
        wrapped = model.unwrap_python_model()
        di = getattr(wrapped, "data_interface_schema", {}) or {}
        feature_cols = set(di.get("feature_columns") or [])

        sig = getattr(getattr(model, "metadata", None), "signature", None)
        schema_inputs = getattr(sig, "inputs", None)
        if schema_inputs and hasattr(schema_inputs, "inputs"):
            for c in getattr(schema_inputs, "inputs", []) or []:
                name = getattr(c, "name", None)
                if name:
                    feature_cols.add(name)
                t = getattr(c, "type", None)
                if name and t and str(t).lower() in ("double", "float", "integer", "long"):
                    expected_numeric.add(name)
    except Exception as e:
        logger.debug(f"모델 numeric info 추출 실패 (무시): {type(e).__name__}: {e}")

    return feature_cols or None, expected_numeric or None


def _extract_required_columns_from_model(model: Any) -> set[str]:
    """모델에서 required_columns를 직접 추출 (캐시 미스 시)."""
    required_cols: set[str] = set()
    try:
        wrapped = model.unwrap_python_model()
        di = getattr(wrapped, "data_interface_schema", {}) or {}
        trained_fetcher = getattr(wrapped, "trained_fetcher", None)
        has_fs = trained_fetcher is not None and hasattr(trained_fetcher, "_fetcher_config")

        if has_fs:
            required_cols.update(di.get("entity_columns", []))
        else:
            required_cols.update(di.get("required_columns", []) or [])

        if not required_cols:
            sig = getattr(getattr(model, "metadata", None), "signature", None)
            schema_inputs = getattr(sig, "inputs", None)
            if schema_inputs and hasattr(schema_inputs, "inputs"):
                for c in getattr(schema_inputs, "inputs", []) or []:
                    name = getattr(c, "name", None)
                    if name:
                        required_cols.add(name)
    except Exception as e:
        logger.debug(f"required columns 추출 실패 (무시): {type(e).__name__}: {e}")

    return required_cols
