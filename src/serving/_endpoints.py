# src/serving/handlers.py

import pandas as pd
from fastapi import HTTPException
from typing import Dict, Any

from src.serving._context import app_context
from src.serving.schemas import (
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
)

def health() -> HealthCheckResponse:
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    
    model_info = "unknown"
    try:
        wrapped_model = app_context.model.unwrap_python_model()
        model_info = getattr(wrapped_model, 'model_class_path', 'unknown')
    except Exception:
        pass

    return HealthCheckResponse(
        status="healthy",
        model_uri=app_context.model_uri,
        model_name=model_info,
    )

def predict_batch(request: Dict[str, Any]) -> BatchPredictionResponse:
    validated_request = app_context.BatchPredictionRequest(**request)

    input_df = pd.DataFrame([sample.model_dump() for sample in validated_request.samples])
    if input_df.empty:
        raise HTTPException(status_code=400, detail="입력 샘플이 비어있습니다.")
    
    predict_params = { "run_mode": "serving", "return_intermediate": False }
    predictions_df = app_context.model.predict(input_df, params=predict_params)
    
    return BatchPredictionResponse(
        predictions=predictions_df.to_dict(orient="records"),
        model_uri=app_context.model_uri,
        sample_count=len(predictions_df)
    )

def get_model_metadata() -> ModelMetadataResponse:
    if app_context.model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    
    hpo_info = getattr(app_context.model, 'hyperparameter_optimization', {}) or {}
    hyperparameter_optimization = HyperparameterOptimizationInfo(
        enabled=hpo_info.get("enabled", False),
        engine=hpo_info.get("engine", ""),
        best_params=hpo_info.get("best_params", {}),
        best_score=hpo_info.get("best_score", 0.0),
        total_trials=hpo_info.get("total_trials", 0),
        pruned_trials=hpo_info.get("pruned_trials", 0),
        optimization_time=str(hpo_info.get("optimization_time", "")),
    )
    
    tm_info = getattr(app_context.model, 'training_methodology', {}) or {}
    training_methodology = TrainingMethodologyInfo(
        train_test_split_method=tm_info.get("train_test_split_method", ""),
        train_ratio=tm_info.get("train_ratio", 0.8),
        validation_strategy=tm_info.get("validation_strategy", ""),
        preprocessing_fit_scope=tm_info.get("preprocessing_fit_scope", ""),
        random_state=tm_info.get("random_state", 42),
    )
    
    api_schema = {
        "input_fields": list(app_context.PredictionRequest.model_fields.keys()),
    }
    
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
    
    hpo_info = getattr(app_context.model, 'hyperparameter_optimization', {}) or {}
    
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
    
    return {
        "prediction_request_schema": app_context.PredictionRequest.model_json_schema(),
        "batch_prediction_request_schema": app_context.BatchPredictionRequest.model_json_schema(),
        "loader_sql_snapshot": getattr(app_context.model, 'loader_sql_snapshot', ''),
    }

def predict(request: Dict[str, Any]) -> Dict[str, Any]:
    request_df = pd.DataFrame([request])
    
    try:
        wrapped_model = app_context.model.unwrap_python_model()
        if hasattr(wrapped_model, 'schema_validator') and wrapped_model.schema_validator:
            wrapped_model.schema_validator.validate_inference_consistency(request_df)
            logger.info("✅ API 실시간 스키마 검증 완료")
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"스키마 검증 실패: {e}"
        )
    except Exception as e:
        logger.warning(f"스키마 검증 중 오류 (계속 진행): {e}")
    
    predictions = app_context.model.predict(request_df)
    
    if hasattr(predictions, 'iloc'):
        prediction_result = predictions.to_dict(orient="records")[0]
    else:
        prediction_result = {"prediction": predictions[0]}
        
    return {"predictions": [prediction_result], "model_uri": app_context.model_uri} 