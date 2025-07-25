import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow

from src.utils.system.logger import logger
from src.utils.system import mlflow_utils
from src.core.factory import Factory
from src.settings import Settings


def run_batch_inference(settings: Settings, run_id: str, context_params: dict = None):
    """
    지정된 Run ID의 모델을 사용하여 배치 추론을 실행합니다.
    """
    context_params = context_params or {}

    # 1. MLflow 실행 컨텍스트 시작
    with mlflow_utils.start_run(settings, run_name=f"batch_inference_{run_id}") as run:
        # 2. 모델 로드
        model_uri = mlflow_utils.get_model_uri(run_id)
        model = mlflow_utils.load_pyfunc_model(settings, model_uri)
        
        # 3. 데이터 로딩
        # Wrapper에 내장된 loader_sql_snapshot을 사용
        wrapped_model = model.unwrap_python_model()
        loader_sql = wrapped_model.loader_sql_snapshot
        
        # Factory를 통해 현재 환경에 맞는 데이터 어댑터 생성
        factory = Factory(settings)
        data_adapter = factory.create_data_adapter("loader")
        
        df = data_adapter.read(loader_sql, params=context_params)
        
        # 4. 예측 실행
        predictions_df = model.predict(df)
        
        # 5. 결과 저장
        storage_adapter = factory.create_data_adapter("storage")
        target_path = f"{settings.artifact_stores.prediction_results.base_uri}/{run.info.run_name}.parquet"
        storage_adapter.write(predictions_df, target_path)

        mlflow.log_artifact(target_path.replace("file://", ""))
        mlflow.log_metric("inference_row_count", len(predictions_df))


def _save_dataset(
    factory: Factory,
    df: pd.DataFrame,
    store_name: str,
    settings: Settings,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Factory를 통해 적절한 데이터 어댑터를 생성하고, DataFrame을 저장합니다.
    (기존 artifact_utils.save_dataset 로직을 직접 구현)
    """
    if df.empty:
        logger.warning(f"DataFrame이 비어있어, '{store_name}' 아티팩트 저장을 건너뜁니다.")
        return

    try:
        store_config = settings.artifact_stores[store_name]
    except KeyError:
        logger.error(f"'{store_name}'에 해당하는 아티팩트 스토어 설정을 찾을 수 없습니다.")
        raise

    if not store_config.enabled:
        logger.info(f"'{store_name}' 아티팩트 스토어가 비활성화되어 있어 저장을 건너뜁니다.")
        return

    base_uri = store_config.base_uri
    
    # ✅ Blueprint 원칙 3: URI 기반 동작 및 동적 팩토리 완전 구현
    # Factory가 환경별 분기와 어댑터 선택을 전담
    adapter = factory.create_data_adapter("storage")

    # 저장될 최종 경로(테이블명 또는 파일명) 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # inference에서는 model.name이 없으므로 run_id 기반으로 식별자 생성
    model_identifier = "batch_inference"
    artifact_name = f"{model_identifier}_{timestamp}"
    
    # ✅ Blueprint 원칙 3: Factory가 URI 해석 처리 - 단순한 artifact 이름만 전달
    final_target = f"{base_uri.rstrip('/')}/{artifact_name}"

    logger.info(f"'{store_name}' 아티팩트 저장 시작: {final_target}")
    adapter.write(df, final_target, options)
    logger.info(f"'{store_name}' 아티팩트 저장 완료: {final_target}")