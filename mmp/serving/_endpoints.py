# mmp/serving/_endpoints.py

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import HTTPException

from mmp.serving._context import app_context
from mmp.serving.schemas import (
    BatchPredictionResponse,
    HealthCheckResponse,
    HyperparameterOptimizationInfo,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    ReadyCheckResponse,
    TrainingMethodologyInfo,
)
from mmp.serving.validators import (
    validate_numeric_types,
    validate_required_columns,
    validate_scalar_values,
)
from mmp.utils.core.logger import log_warn
from mmp.utils.data.data_io import format_predictions

logger = logging.getLogger(__name__)


def _get_data_interface_schema() -> dict:
    """data_interface_schema를 반환하는 공통 헬퍼. 캐시 우선."""
    if app_context.data_interface_schema:
        return app_context.data_interface_schema
    wrapped_model = app_context.model.unwrap_python_model()
    return getattr(wrapped_model, "data_interface_schema", {}) or {}


def _convert_to_signature_types(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    MLflow Signature에 정의된 타입에 맞게 DataFrame 컬럼 타입을 변환.
    startup 캐시(signature_type_map)를 우선 사용하여 매 요청의 reflection을 제거.
    """
    if model is None:
        return df

    try:
        # 캐시 우선, 없으면 기존 reflection
        type_map = app_context.signature_type_map
        if not type_map:
            type_map = _build_signature_type_map(model)
            if not type_map:
                return df

        for col_name, col_type in type_map.items():
            if col_name not in df.columns:
                continue

            current_dtype = str(df[col_name].dtype)

            if col_type in ("double", "float", "float64", "float32"):
                if "int" in current_dtype or "object" in current_dtype:
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("float64")

            elif col_type in ("integer", "long", "int64", "int32"):
                if "float" in current_dtype:
                    df[col_name] = df[col_name].astype("int64")
                elif "object" in current_dtype:
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("int64")

    except Exception as e:
        log_warn(f"Signature 기반 타입 변환 실패: {e}", "API:TYPE")

    return df


def _build_signature_type_map(model: Any) -> dict[str, str]:
    """캐시 미스 시 signature에서 type map을 직접 구성."""
    result = {}
    try:
        sig = getattr(getattr(model, "metadata", None), "signature", None)
        schema_inputs = getattr(sig, "inputs", None)
        if schema_inputs and hasattr(schema_inputs, "inputs"):
            for col_spec in getattr(schema_inputs, "inputs", []) or []:
                name = getattr(col_spec, "name", None)
                col_type = str(getattr(col_spec, "type", "")).lower()
                if "." in col_type:
                    col_type = col_type.split(".")[-1]
                if name and col_type:
                    result[name] = col_type
    except Exception:
        pass
    return result


def health() -> HealthCheckResponse:
    """
    Liveness 체크 (K8s livenessProbe용).
    프로세스 생존 여부만 확인하는 경량 엔드포인트.
    모델 로드 상태와 무관하게 항상 200 반환.
    """
    return HealthCheckResponse(status="ok")


def ready() -> ReadyCheckResponse:
    """
    Readiness 체크 (K8s readinessProbe용).
    모델이 로드되어 트래픽을 받을 준비가 되었는지 확인.
    """
    if not app_context.is_ready:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    model_info = "unknown"
    try:
        wrapped_model = app_context.model.unwrap_python_model()
        model_info = getattr(wrapped_model, "model_class_path", "unknown")
    except Exception as e:
        logger.debug(f"모델 정보 추출 실패 (무시): {type(e).__name__}: {e}")

    return ReadyCheckResponse(
        status="ready",
        model_uri=app_context.model_uri,
        model_name=model_info,
    )


def predict_batch(request: Dict[str, Any]) -> BatchPredictionResponse:
    validated_request = app_context.BatchPredictionRequest(**request)

    input_df = pd.DataFrame([sample.model_dump() for sample in validated_request.samples])
    if input_df.empty:
        raise HTTPException(status_code=400, detail="입력 샘플이 비어있습니다.")

    # 입력 검증: 스칼라 → 숫자형 → 타입 변환 → 필수 컬럼
    validate_scalar_values(input_df)
    validate_numeric_types(input_df, app_context.model)
    input_df = _convert_to_signature_types(input_df, app_context.model)
    validate_required_columns(input_df, app_context.model)


    predict_params = {"run_mode": "serving", "return_intermediate": False, "return_dataframe": True}
    raw_predictions_df = app_context.model.predict(input_df, params=predict_params)

    # Inference 파이프라인과 동일하게 data_interface 기반 포맷 적용
    data_interface_schema = _get_data_interface_schema()
    # format_predictions에 전체 schema 전달 (top-level에 모든 필드 포함)
    predictions_df = format_predictions(raw_predictions_df, input_df, data_interface_schema)

    # JSON 직렬화 호환을 위해 numpy dtype → Python 기본형 변환 (컬럼 단위)
    for col in predictions_df.columns:
        if hasattr(predictions_df[col].dtype, "numpy_dtype") or predictions_df[col].dtype.kind in ("i", "f", "u"):
            predictions_df[col] = predictions_df[col].astype(object)

    # 출력 값 유한성 검증: NaN/Inf 포함 시 422 반환
    # 주요 예측 컬럼들 점검
    if predictions_df.map(
        lambda val: isinstance(val, (float, int)) and not np.isfinite(val)
    ).any().any():
        raise HTTPException(
            status_code=422, detail="예측 결과에 비유한 값(NaN/Inf)이 포함되어 있습니다."
        )

    return BatchPredictionResponse(
        predictions=predictions_df.to_dict(orient="records"),
        model_uri=app_context.model_uri,
        sample_count=len(predictions_df),
    )


def get_model_metadata() -> ModelMetadataResponse:
    if app_context.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    hpo_info = getattr(app_context.model, "hyperparameter_optimization", {}) or {}
    hyperparameter_optimization = HyperparameterOptimizationInfo(
        enabled=hpo_info.get("enabled", False),
        engine=hpo_info.get("engine", ""),
        best_params=hpo_info.get("best_params", {}),
        best_score=hpo_info.get("best_score", 0.0),
        total_trials=hpo_info.get("total_trials", 0),
        pruned_trials=hpo_info.get("pruned_trials", 0),
        optimization_time=str(hpo_info.get("optimization_time", "")),
    )

    tm_info = getattr(app_context.model, "training_methodology", {}) or {}
    training_methodology = TrainingMethodologyInfo(
        train_test_split_method=tm_info.get("train_test_split_method", ""),
        train_ratio=tm_info.get("train_ratio", 0.8),
        validation_strategy=tm_info.get("validation_strategy", ""),
        preprocessing_fit_scope=tm_info.get("preprocessing_fit_scope", ""),
        random_state=tm_info.get("random_state", 42),
    )

    # 🆕 Phase 5.5: DataInterface 기반 API 스키마 정보 향상
    data_interface_schema = _get_data_interface_schema()

    api_schema = {
        "input_fields": list(app_context.PredictionRequest.model_fields.keys()),
    }

    if data_interface_schema:
        api_schema.update(
            {
                "schema_generation_method": "datainterface_based",
                "entity_columns": data_interface_schema.get("entity_columns", []),
                "required_columns": data_interface_schema.get("required_columns", []),
                "task_type": data_interface_schema.get("task_type", ""),
            }
        )
    else:
        api_schema["schema_generation_method"] = "legacy_sql_parsing"

    return ModelMetadataResponse(
        model_uri=app_context.model_uri,
        model_class_path=getattr(app_context.model, "model_class_path", ""),
        hyperparameter_optimization=hyperparameter_optimization,
        training_methodology=training_methodology,
        api_schema=api_schema,
    )


def get_optimization_history() -> OptimizationHistoryResponse:
    if app_context.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    hpo_info = getattr(app_context.model, "hyperparameter_optimization", {}) or {}

    if not hpo_info.get("enabled", False):
        return OptimizationHistoryResponse(
            enabled=False,
            optimization_history=[],
            search_space={},
            convergence_info={"message": "하이퍼파라미터 최적화가 비활성화되었습니다."},
        )

    return OptimizationHistoryResponse(
        enabled=True,
        optimization_history=hpo_info.get("optimization_history", []),
        search_space=hpo_info.get("search_space", {}),
        convergence_info={
            "best_score": hpo_info.get("best_score", 0.0),
            "total_trials": hpo_info.get("total_trials", 0),
            "pruned_trials": hpo_info.get("pruned_trials", 0),
            "optimization_time": str(hpo_info.get("optimization_time", "")),
        },
    )


def get_api_schema() -> Dict[str, Any]:
    if app_context.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    # 🆕 Phase 5.5: DataInterface 스키마 정보 포함
    wrapped_model = app_context.model.unwrap_python_model()
    data_interface_schema = _get_data_interface_schema()

    schema_info = {
        "prediction_request_schema": app_context.PredictionRequest.model_json_schema(),
        "batch_prediction_request_schema": app_context.BatchPredictionRequest.model_json_schema(),
        "loader_sql_snapshot": getattr(wrapped_model, "loader_sql_snapshot", ""),
    }

    # DataInterface 스키마가 있으면 추가 정보 제공
    if data_interface_schema:
        schema_info.update(
            {
                "data_interface_schema": data_interface_schema,
                "schema_generation_method": "datainterface_based",
                "required_columns": data_interface_schema.get("required_columns", []),
                "entity_columns": data_interface_schema.get("entity_columns", []),
                "task_type": data_interface_schema.get("task_type", ""),
            }
        )
    else:
        schema_info["schema_generation_method"] = "legacy_sql_parsing"

    return schema_info


def predict(request: Dict[str, Any]) -> Dict[str, Any]:
    request_df = pd.DataFrame([request])

    # 입력 검증: PredictionRequest 선제 검증 → 스칼라 → 숫자형 → 타입 변환 → 필수 컬럼(signature)
    validate_required_columns(
        request_df, app_context.model, prediction_request_cls=app_context.PredictionRequest
    )
    validate_scalar_values(request_df)
    validate_numeric_types(request_df, app_context.model)
    request_df = _convert_to_signature_types(request_df, app_context.model)
    validate_required_columns(request_df, app_context.model)

    # 서빙 경로 강제 + DataFrame 반환 보장
    raw_predictions_df = app_context.model.predict(
        request_df,
        params={"run_mode": "serving", "return_intermediate": False, "return_dataframe": True},
    )

    # Inference 파이프라인과 동일한 포맷 적용
    data_interface_schema = _get_data_interface_schema()
    # format_predictions에 전체 schema 전달 (top-level에 모든 필드 포함)
    predictions_df = format_predictions(raw_predictions_df, request_df, data_interface_schema)

    # 최소 응답 스키마로 변환 (numpy 스칼라 → 파이썬 기본형)
    if hasattr(predictions_df, "iloc") and "prediction" in predictions_df.columns:
        value = predictions_df.iloc[0]["prediction"]
    elif hasattr(predictions_df, "iloc") and predictions_df.shape[1] >= 1:
        value = predictions_df.iloc[0, 0]
    else:
        value = None

    if isinstance(value, np.generic):
        value = value.item()

    # 출력 유한성 검증: NaN/Inf이면 422
    try:
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise HTTPException(status_code=422, detail="예측 결과가 비유한 값(NaN/Inf)입니다.")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"출력 유한성 검증 fallback (무시): {type(e).__name__}: {e}")

    return {"prediction": value, "model_uri": app_context.model_uri}
