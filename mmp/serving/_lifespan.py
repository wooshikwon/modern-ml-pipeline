# mmp/serving/lifespan.py

from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from mmp.factory import bootstrap
from mmp.serving._context import app_context
from mmp.serving.schemas import (
    create_batch_prediction_request,
    create_datainterface_based_prediction_request_v2,
    create_dynamic_prediction_request,
)
from mmp.settings import Settings
from mmp.utils.core.logger import log_api, log_api_debug, log_error, log_warn
from mmp.utils.database.sql_utils import parse_select_columns


def setup_api_context(run_id: str, settings: Settings):
    """서버 시작 시 API 컨텍스트를 설정하는 함수.

    모든 준비를 로컬 변수에서 수행한 뒤, 마지막에 app_context.initialize()를
    한 번 호출하여 스레드 안전하게 전역 상태를 설정한다.
    """
    try:
        # 부트스트랩: 레지스트리/의존성 검증 보장
        bootstrap(settings)

        # 1) 모델 로드 (로컬 변수)
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        wrapped_model = model.unwrap_python_model()

        # 2) PredictionRequest 스키마 생성 (로컬 변수)
        data_interface_schema = getattr(wrapped_model, "data_interface_schema", None)
        if data_interface_schema:
            log_api_debug("DataInterface 스키마 기반 API 스키마 생성 (target_column 자동 제외)")
            prediction_request = create_datainterface_based_prediction_request_v2(
                model_name="DataInterfacePredictionRequest",
                data_interface_schema=data_interface_schema,
                exclude_target=True,
            )
        else:
            # 폴백: 기존 방식 (data_schema 또는 SQL 파싱)
            log_warn("DataInterface 스키마 없음 - 기존 방식으로 폴백", "API")
            data_schema = getattr(wrapped_model, "data_schema", None)
            if isinstance(data_schema, dict) and data_schema.get("entity_columns"):
                pk_fields = list(data_schema.get("entity_columns") or [])
            else:
                loader_sql = getattr(
                    wrapped_model, "loader_sql_snapshot", "SELECT user_id FROM DUAL"
                )
                pk_fields = parse_select_columns(loader_sql)

            prediction_request = create_dynamic_prediction_request(
                model_name="DynamicPredictionRequest", pk_fields=pk_fields
            )

        batch_prediction_request = create_batch_prediction_request(prediction_request)

        # 3) 스레드 안전한 일괄 초기화
        app_context.initialize(
            model=model,
            model_uri=model_uri,
            settings=settings,
            prediction_request=prediction_request,
            batch_prediction_request=batch_prediction_request,
        )
        log_api(f"API 컨텍스트 설정 완료: {model_uri}")
    except Exception as e:
        log_error(f"API 컨텍스트 설정 실패: {e}", "API")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱의 생명주기(시작/종료)를 관리합니다.

    app.state.run_id / app.state.settings가 설정되어 있으면
    lifespan 시작 시 API 컨텍스트를 초기화한다.
    """
    log_api("Modern ML Pipeline API 서버 시작")

    # lifespan 내에서 모델 초기화 수행
    run_id: str | None = getattr(app.state, "run_id", None)
    settings: Settings | None = getattr(app.state, "settings", None)
    if run_id and settings:
        setup_api_context(run_id=run_id, settings=settings)

    yield
    log_api("Modern ML Pipeline API 서버 종료")
