# src/serving/_endpoints.py

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.serving._context import app_context
from src.serving.schemas import (
    BatchPredictionResponse,
    HealthCheckResponse,
    HyperparameterOptimizationInfo,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    ReadyCheckResponse,
    TrainingMethodologyInfo,
)
from src.utils.data.data_io import format_predictions

logger = logging.getLogger(__name__)


def _convert_to_signature_types(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    MLflow Signature에 정의된 타입에 맞게 DataFrame 컬럼 타입을 변환.
    int64 → float64, float → int64 등 필요한 변환 수행.
    """
    if model is None:
        return df

    try:
        metadata = getattr(model, "metadata", None)
        if metadata is None:
            return df

        sig = getattr(metadata, "signature", None)
        if sig is None:
            return df

        schema_inputs = getattr(sig, "inputs", None)
        if schema_inputs is None:
            return df

        col_specs = getattr(schema_inputs, "inputs", None)
        if col_specs is None:
            return df

        for col_spec in col_specs:
            col_name = getattr(col_spec, "name", None)
            col_type_raw = getattr(col_spec, "type", None)

            if col_name is None or col_type_raw is None:
                continue

            if col_name not in df.columns:
                continue

            # 타입 문자열 정규화 (DataType.double → double)
            col_type = str(col_type_raw).lower()
            if "." in col_type:
                col_type = col_type.split(".")[-1]

            current_dtype = str(df[col_name].dtype)

            # float/double 타입 변환 (int → float)
            if col_type in ("double", "float", "float64", "float32"):
                if "int" in current_dtype or "object" in current_dtype:
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("float64")
                    logger.debug(f"[TYPE] '{col_name}': {current_dtype} → float64")

            # int/long 타입 변환 (float → int)
            elif col_type in ("integer", "long", "int64", "int32"):
                if "float" in current_dtype:
                    df[col_name] = df[col_name].astype("int64")
                    logger.debug(f"[TYPE] '{col_name}': {current_dtype} → int64")
                elif "object" in current_dtype:
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("int64")
                    logger.debug(f"[TYPE] '{col_name}': {current_dtype} → int64")

    except Exception as e:
        logger.warning(f"[TYPE] Signature 기반 타입 변환 실패: {e}")

    return df


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
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    model_info = "unknown"
    try:
        wrapped_model = app_context.model.unwrap_python_model()
        model_info = getattr(wrapped_model, "model_class_path", "unknown")
    except Exception:
        pass

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

    # 입력 값 스칼라 검증 (list/dict 등 비스칼라 거부)
    for col in input_df.columns:
        if input_df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            raise HTTPException(
                status_code=422, detail=f"컬럼 '{col}'에 비스칼라 값이 포함되어 있습니다."
            )

    # 숫자형 컬럼 타입 검증: 학습 시그니처/데이터인터페이스 기반으로 float/int 기대 컬럼은 문자열 등 비수치 거부
    try:
        wrapped = app_context.model.unwrap_python_model()
        di = getattr(wrapped, "data_interface_schema", {}) or {}
        # data_interface_schema는 top-level에 feature_columns 포함
        feature_cols = set(di.get("feature_columns") or [])

        # MLflow Signature에서 보완
        sig = getattr(getattr(app_context.model, "metadata", None), "signature", None)
        schema_inputs = getattr(sig, "inputs", None)
        if schema_inputs is not None and hasattr(schema_inputs, "inputs"):
            cols = getattr(schema_inputs, "inputs", []) or []
            for c in cols:
                name = getattr(c, "name", None)
                if name:
                    feature_cols.add(name)

        expected_numeric_cols = set()
        if schema_inputs is not None and hasattr(schema_inputs, "inputs"):
            for c in getattr(schema_inputs, "inputs", []) or []:
                name = getattr(c, "name", None)
                t = getattr(c, "type", None)
                if name and t and str(t).lower() in ("double", "float", "integer", "long"):
                    expected_numeric_cols.add(name)

        # feature 컬럼 내 숫자형 기대 컬럼 검증
        for col in feature_cols or input_df.columns:
            if col in input_df.columns and (
                not expected_numeric_cols or col in expected_numeric_cols
            ):
                val_is_numeric = (
                    input_df[col]
                    .apply(lambda v: isinstance(v, (int, float)) and np.isfinite(v))
                    .all()
                )
                if not val_is_numeric:
                    raise HTTPException(
                        status_code=422, detail=f"컬럼 '{col}'은 숫자형이어야 합니다."
                    )
    except Exception:
        # 시그니처를 읽지 못하더라도 동작은 계속 (MLflow 내부 검증이 2차 방어)
        pass

    # MLflow 스키마 호환성을 위한 데이터 타입 변환
    input_df = _convert_to_signature_types(input_df, app_context.model)

    # 필수 컬럼 누락 검증
    # Feature Store fetcher 사용 시: entity_columns만 필수 (나머지는 Online Store에서 증강)
    # pass_through fetcher 사용 시: 모든 feature_columns 필수
    try:
        required_cols = set()
        wrapped = app_context.model.unwrap_python_model()
        di = getattr(wrapped, "data_interface_schema", {}) or {}
        trained_fetcher = getattr(wrapped, "trained_fetcher", None)

        # Fetcher 유무에 따라 필수 컬럼 결정
        has_feature_store_fetcher = trained_fetcher is not None and hasattr(
            trained_fetcher, "_fetcher_config"
        )

        if has_feature_store_fetcher:
            # Feature Store fetcher: entity_columns만 필수 (feature는 Online Store에서 조회)
            entity_cols = di.get("entity_columns", [])
            required_cols.update(entity_cols)
        else:
            # pass_through 또는 fetcher 없음: entity + feature_columns 필수
            required_cols.update(di.get("required_columns", []) or [])

        missing = [c for c in sorted(required_cols) if c not in input_df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
    except HTTPException:
        raise
    except Exception:
        pass

    predict_params = {"run_mode": "serving", "return_intermediate": False, "return_dataframe": True}
    raw_predictions_df = app_context.model.predict(input_df, params=predict_params)

    # Inference 파이프라인과 동일하게 data_interface 기반 포맷 적용
    wrapped_model = app_context.model.unwrap_python_model()
    data_interface_schema = getattr(wrapped_model, "data_interface_schema", {}) or {}
    # format_predictions에 전체 schema 전달 (top-level에 모든 필드 포함)
    predictions_df = format_predictions(raw_predictions_df, input_df, data_interface_schema)

    # JSON 직렬화 호환을 위해 numpy 스칼라를 파이썬 기본형으로 변환
    def _to_py(x):
        return x.item() if isinstance(x, np.generic) else x

    predictions_df = predictions_df.map(_to_py)

    # 출력 값 유한성 검증: NaN/Inf 포함 시 422 반환
    def _is_non_finite(val: Any) -> bool:
        try:
            return isinstance(val, (float, int)) and (not np.isfinite(val))
        except Exception:
            return False

    # 주요 예측 컬럼들 점검
    if predictions_df.map(_is_non_finite).any().any():
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
    wrapped_model = app_context.model.unwrap_python_model()
    data_interface_schema = getattr(wrapped_model, "data_interface_schema", None)

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
    data_interface_schema = getattr(wrapped_model, "data_interface_schema", None)

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

    # DataInterface 기반 PredictionRequest 스키마로 필수 컬럼 선제 검증
    try:
        pr_fields = getattr(app_context.PredictionRequest, "model_fields", {})
        if pr_fields:
            required_cols = set(pr_fields.keys())
            missing = [c for c in sorted(required_cols) if c not in request_df.columns]
            if missing:
                raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
    except HTTPException:
        raise
    except Exception:
        pass

    # MLflow 스키마 호환성을 위한 데이터 타입 변환
    request_df = _convert_to_signature_types(request_df, app_context.model)

    # 입력 값 스칼라 검증 (list/dict 등 비스칼라 거부)
    for col in request_df.columns:
        val = request_df.iloc[0][col]
        if isinstance(val, (list, dict)):
            raise HTTPException(
                status_code=422, detail=f"컬럼 '{col}'에 비스칼라 값이 포함되어 있습니다."
            )
    # 숫자형 컬럼 타입 검증: 학습 시그니처 기반으로 float/int 기대 컬럼은 문자열 등 비수치 거부
    try:
        signature_input = getattr(
            getattr(app_context.model, "metadata", None), "get_input_schema", None
        )
        expected_numeric_cols = set()
        if callable(signature_input):
            schema = signature_input()
            for col in getattr(schema, "inputs", []) or []:
                t = getattr(col, "type", None)
                name = getattr(col, "name", None)
                if name and t and str(t).lower() in ("double", "float", "integer", "long"):
                    expected_numeric_cols.add(name)
        for col in request_df.columns:
            if col in expected_numeric_cols:
                val = request_df.iloc[0][col]
                if not (isinstance(val, (int, float)) and np.isfinite(val)):
                    raise HTTPException(
                        status_code=422, detail=f"컬럼 '{col}'은 숫자형이어야 합니다."
                    )
    except Exception:
        pass

    # 필수 컬럼 누락 검증: MLflow signature 기반
    try:
        required_cols = set()
        sig = getattr(getattr(app_context.model, "metadata", None), "signature", None)
        schema_inputs = getattr(sig, "inputs", None)
        if schema_inputs is not None:
            # Try input_names() if available
            try:
                if hasattr(schema_inputs, "input_names") and callable(
                    getattr(schema_inputs, "input_names", None)
                ):
                    for n in schema_inputs.input_names():
                        required_cols.add(n)
            except Exception:
                pass
            # Fallback to inputs list
            if not required_cols and hasattr(schema_inputs, "inputs"):
                for c in getattr(schema_inputs, "inputs", []) or []:
                    name = getattr(c, "name", None)
                    if name:
                        required_cols.add(name)
        missing = [c for c in sorted(required_cols) if c not in request_df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"필수 컬럼 누락: {missing}")
    except HTTPException:
        raise
    except Exception:
        pass

    # 서빙 경로 강제 + DataFrame 반환 보장
    raw_predictions_df = app_context.model.predict(
        request_df,
        params={"run_mode": "serving", "return_intermediate": False, "return_dataframe": True},
    )

    # Inference 파이프라인과 동일한 포맷 적용
    wrapped_model = app_context.model.unwrap_python_model()
    data_interface_schema = getattr(wrapped_model, "data_interface_schema", {}) or {}
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
    except Exception:
        pass

    return {"prediction": value, "model_uri": app_context.model_uri}
