# src/serving/lifespan.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow

from src.settings import Settings
from src.utils.system.logger import logger
from src.serving._context import app_context
from src.serving.schemas import create_dynamic_prediction_request, create_batch_prediction_request
from src.utils.system.sql_utils import parse_select_columns


def setup_api_context(run_id: str, settings: Settings):
    """서버 시작 시 API 컨텍스트를 설정하는 함수"""
    try:
        model_uri = f"runs:/{run_id}/model"
        app_context.model = mlflow.pyfunc.load_model(model_uri)
        app_context.model_uri = model_uri
        app_context.settings = settings
        
        wrapped_model = app_context.model.unwrap_python_model()

        # 우선 data_schema에서 entity_columns 사용, 불가 시 loader_sql에서 파싱
        data_schema = getattr(wrapped_model, 'data_schema', None)
        if isinstance(data_schema, dict) and data_schema.get('entity_columns'):
            pk_fields = list(data_schema.get('entity_columns') or [])
        else:
            loader_sql = getattr(wrapped_model, 'loader_sql_snapshot', 'SELECT user_id FROM DUAL')
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
    """FastAPI 앱의 생명주기(시작/종료)를 관리합니다."""
    logger.info("🚀 Modern ML Pipeline API 서버 시작...")
    # 여기에 서버 시작 시 필요한 로직을 추가할 수 있습니다.
    # 예를 들어, run_id와 settings를 환경 변수나 설정 파일에서 읽어와
    # setup_api_context를 호출할 수 있습니다.
    yield
    logger.info("✅ Modern ML Pipeline API 서버 종료.") 