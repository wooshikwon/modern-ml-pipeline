import uvicorn
import pandas as pd
import mlflow
import mlflow.pyfunc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Type
from pydantic import BaseModel, create_model

from src.settings import Settings, load_settings
from src.utils.integrations import mlflow_integration as mlflow_utils
from serving.schemas import (
    get_pk_from_loader_sql,
    create_dynamic_prediction_request,
    create_batch_prediction_request,
    PredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
)
from src.utils.system.logger import logger

class AppContext:
    def __init__(self):
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_uri: str = ""
        self.settings: Settings | None = None
        self.PredictionRequest: Type[BaseModel] = create_model("DefaultPredictionRequest")
        self.BatchPredictionRequest: Type[BaseModel] = create_model("DefaultBatchPredictionRequest")

app_context = AppContext()

def setup_api_context(run_id: str, settings: Settings):
    """테스트 또는 서버 시작 시 API 컨텍스트를 설정하는 함수"""
    try:
        model_uri = f"runs:/{run_id}/model"
        app_context.model = mlflow.pyfunc.load_model(model_uri)
        app_context.model_uri = model_uri
        app_context.settings = settings
        
        wrapped_model = app_context.model.unwrap_python_model()
        loader_sql = getattr(wrapped_model, 'loader_sql_snapshot', 'SELECT user_id FROM DUAL')
        
        from src.utils.system.sql_utils import parse_select_columns
        pk_fields = parse_select_columns(loader_sql)
        
        app_context.PredictionRequest = create_dynamic_prediction_request(
            model_name="DynamicPredictionRequest", pk_fields=pk_fields
        )
        app_context.BatchPredictionRequest = create_batch_prediction_request(
            app_context.PredictionRequest
        )
        logger.info(f"API 컨텍스트 설정 완료: {model_uri}")
    except Exception as e:
        logger.error(f"API 컨텍스트 설정 실패: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 실제 서버 실행 시 사용할 lifespan
    # run_id는 환경 변수 등에서 가져와야 함
    run_id = mlflow_utils.get_latest_run_id(
        experiment_name=load_settings().mlflow.experiment_name
    )
    settings = load_settings()
    setup_api_context(run_id, settings)
    logger.info("FastAPI 애플리케이션 시작...")
    yield
    logger.info("FastAPI 애플리케이션 종료.")

# 테스트와 실제 서빙 모두에서 사용될 수 있는 최상위 app 객체
app = FastAPI(
    title="Modern ML Pipeline API",
    description="Blueprint v17.0 기반 모델 서빙 API",
    version="17.0.0",
    lifespan=lifespan  # 실제 서버 실행 시에만 lifespan이 활성화됨
)

@app.on_event("startup")
async def startup_event():
    # 실제 run_id는 run_api_server 함수에서 주입받아 설정됩니다.
    # 여기서는 FastAPI의 라이프사이클 이벤트를 보여주기 위한 예시입니다.
    pass

@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {
        "message": "Modern ML Pipeline API",
        "status": "ready" if app_context.model else "error",
        "model_uri": app_context.model_uri,
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
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

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: BaseModel) -> PredictionResponse:
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    try:
        DynamicRequest = app_context.PredictionRequest
        validated_request = DynamicRequest(**request.model_dump())
        
        input_df = pd.DataFrame([validated_request.model_dump()])
        
        predict_params = { "run_mode": "serving", "return_intermediate": True }
        
        predictions = app_context.model.predict(input_df, params=predict_params)
        
        # 결과 구조가 DataFrame이라고 가정
        prediction_value = predictions["prediction"].iloc[0]
        input_features_dict = predictions["input_features"].iloc[0]

        return PredictionResponse(
            prediction=prediction_value,
            input_features=input_features_dict,
        )
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BaseModel) -> BatchPredictionResponse:
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    try:
        DynamicBatchRequest = app_context.BatchPredictionRequest
        validated_request = DynamicBatchRequest(**request.model_dump())

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
    except Exception as e:
        logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 🆕 Blueprint v17.0: 모델 메타데이터 자기 기술 엔드포인트들
    
@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Metadata"])
def get_model_metadata() -> ModelMetadataResponse:
    """
    모델의 완전한 메타데이터 반환 (하이퍼파라미터 최적화, Data Leakage 방지 정보 포함)
    """
    try:
        if app_context.model is None:
            raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
        
        # 하이퍼파라미터 최적화 정보 구성
        hpo_info = app_context.model.hyperparameter_optimization
        hyperparameter_optimization = HyperparameterOptimizationInfo(
            enabled=hpo_info.get("enabled", False),
            engine=hpo_info.get("engine", ""),
            best_params=hpo_info.get("best_params", {}),
            best_score=hpo_info.get("best_score", 0.0),
            total_trials=hpo_info.get("total_trials", 0),
            pruned_trials=hpo_info.get("pruned_trials", 0),
            optimization_time=str(hpo_info.get("optimization_time", "")),
        )
        
        # 학습 방법론 정보 구성
        tm_info = app_context.model.training_methodology
        training_methodology = TrainingMethodologyInfo(
            train_test_split_method=tm_info.get("train_test_split_method", ""),
            train_ratio=tm_info.get("train_ratio", 0.8),
            validation_strategy=tm_info.get("validation_strategy", ""),
            preprocessing_fit_scope=tm_info.get("preprocessing_fit_scope", ""),
            random_state=tm_info.get("random_state", 42),
        )
        
        # API 스키마 정보 구성
        api_schema = {
            "input_fields": [field for field in app_context.PredictionRequest.__fields__.keys()],
            "sql_source": "loader_sql_snapshot",
            "feature_columns": app_context.model.feature_columns, # 실제 모델에서 가져옴
            "join_key": app_context.model.join_key, # 실제 모델에서 가져옴
        }
        
        return ModelMetadataResponse(
            model_uri=app_context.model_uri,
            model_class_path=getattr(app_context.model, "model_class_path", ""),
            hyperparameter_optimization=hyperparameter_optimization,
            training_methodology=training_methodology,
            training_metadata=app_context.model.training_metadata,
            api_schema=api_schema,
        )
        
    except Exception as e:
        logger.error(f"모델 메타데이터 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/optimization", response_model=OptimizationHistoryResponse, tags=["Model Metadata"])
def get_optimization_history() -> OptimizationHistoryResponse:
    """
    하이퍼파라미터 최적화 과정의 상세 히스토리 반환
    """
    try:
        if app_context.model is None:
            raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
        
        hpo_info = app_context.model.hyperparameter_optimization
        
        if not hpo_info.get("enabled", False):
            return OptimizationHistoryResponse(
                enabled=False,
                optimization_history=[],
                search_space={},
                convergence_info={"message": "하이퍼파라미터 최적화가 비활성화되었습니다."},
                timeout_occurred=False,
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
            timeout_occurred=hpo_info.get("timeout_occurred", False),
        )
        
    except Exception as e:
        logger.error(f"최적화 히스토리 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/schema", tags=["Model Metadata"])
def get_api_schema() -> Dict[str, Any]:
    """
    동적으로 생성된 API 스키마 정보 반환
    """
    try:
        if app_context.model is None:
            raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
        
        return {
            "prediction_request_schema": app_context.PredictionRequest.schema(),
            "batch_prediction_request_schema": app_context.BatchPredictionRequest.schema(),
            "loader_sql_snapshot": app_context.model.loader_sql_snapshot,
            "extracted_fields": [field for field in app_context.PredictionRequest.__fields__.keys()],
            "feature_store_info": {
                "feature_columns": app_context.model.feature_columns,
                "join_key": app_context.model.join_key,
                "feature_store_config": app_context.settings.serving.get('realtime_feature_store', {}),
            },
        }
        
    except Exception as e:
        logger.error(f"API 스키마 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def run_api_server(settings: Settings, run_id: str, host: str = "0.0.0.0", port: int = 8000):
    """
    FastAPI 서버를 실행하고, Lifespan 이벤트를 통해 모델을 로드합니다.
    """
    # Blueprint 원칙 9: 환경별 기능 분리 - API 서빙 시스템적 차단
    serving_settings = getattr(settings, 'serving', None)
    if not serving_settings or not getattr(serving_settings, 'enabled', True):
        logger.error(f"'{settings.environment.app_env}' 환경에서는 API 서빙이 비활성화되어 있습니다.")
        return

    # 모델 로드 (서버 시작 시 한 번만 실행)
    model_uri = mlflow_utils.get_model_uri(run_id)
    app_context.model = mlflow_utils.load_pyfunc_model(settings, model_uri)
    
    # 동적 라우트 생성
    PredictionRequest = create_dynamic_prediction_request(app_context.model)
    
    @app.post("/predict", tags=["Predictions"])
    def predict(request: PredictionRequest):
        # Pydantic 모델을 dict으로 변환 후, DataFrame으로 변환
        request_df = pd.DataFrame([request.dict()])
        
        # 모델 예측
        predictions = app_context.model.predict(request_df)
        return {"predictions": predictions.to_dict(orient="records")}

    uvicorn.run(app, host=host, port=port)
