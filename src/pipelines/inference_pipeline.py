import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime
import mlflow

from src.utils.system.logger import logger
from src.utils.system import mlflow_utils
from src.core.factory import Factory
from src.settings.settings import Settings, load_settings


def run_batch_inference(
    run_id: str,
    context_params: Optional[Dict[str, Any]] = None,
):
    """
    "완전 독립형 PyfuncWrapper"와 "통합 데이터 어댑터"를 사용하여
    투명한 배치 추론을 실행합니다.
    """
    logger.info(f"배치 추론 파이프라인 시작 (Run ID: {run_id})")
    context_params = context_params or {}

    try:
        # 1. MLflow에서 "완전 독립형" PyfuncWrapper 로드
        model_uri = f"runs:/{run_id}/model"
        wrapper = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"PyfuncWrapper 로드 완료: {model_uri}")

        # 2. (검증) 로드된 Wrapper의 recipe 정보 확인
        recipe_snapshot = wrapper.recipe_snapshot
        logger.info(f"로드된 모델 정보: {recipe_snapshot.get('class_path', 'Unknown')}")

        # 3. 데이터 로드를 위한 임시 Settings 및 Factory 생성
        #    (Adapter가 GCP Project ID 같은 환경 정보에 접근해야 하므로 필요)
        #    이 Settings는 데이터 로딩에만 사용되며, Wrapper의 로직에는 영향을 주지 않음.
        # 임시로 기존 recipe를 사용 (실제로는 환경 설정만 필요함)
        temp_settings = load_settings("xgboost_x_learner")  # 환경 설정만 사용
        factory = Factory(temp_settings)

        # 4. Wrapper의 내장 loader_uri를 사용하여 데이터 로드
        loader_uri = wrapper.loader_uri
        scheme = urlparse(loader_uri).scheme
        data_adapter = factory.create_data_adapter(scheme)
        
        input_df = data_adapter.read(loader_uri, params=context_params)
        if input_df.empty:
            logger.warning("입력 데이터가 비어있어 추론을 중단합니다.")
            return

        # 5. Wrapper를 통해 예측 실행 및 중간 산출물 얻기
        predict_params = {
            "run_mode": "batch",
            "context_params": context_params,
            "return_intermediate": True,
        }
        results = wrapper.predict(input_df, params=predict_params)

        # 6. 중간 산출물 및 최종 결과 저장
        logger.info("중간 산출물 및 최종 결과 저장을 시작합니다.")
        factory = Factory(temp_settings)
        
        if "augmented_data" in results:
            _save_dataset(factory, results["augmented_data"], "augmented_dataset", temp_settings)
        if "preprocessed_data" in results:
            _save_dataset(factory, results["preprocessed_data"], "preprocessed_dataset", temp_settings)
        if "final_results" in results:
            _save_dataset(factory, results["final_results"], "prediction_results", temp_settings)

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise


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
    parsed_uri = urlparse(base_uri)
    scheme = parsed_uri.scheme

    adapter = factory.create_data_adapter(scheme)

    # 저장될 최종 경로(테이블명 또는 파일명) 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # inference에서는 model.name이 없으므로 run_id 기반으로 식별자 생성
    model_identifier = "batch_inference"
    artifact_name = f"{model_identifier}_{timestamp}"
    
    if scheme == "bq":
        # BigQuery: 데이터셋.테이블명 형태로 구성
        dataset_table = f"{parsed_uri.netloc}.{artifact_name}"
        final_target = f"bq://{dataset_table}"
    else:
        # 다른 스토리지: 기본 URI + 아티팩트명
        final_target = f"{base_uri.rstrip('/')}/{artifact_name}"

    logger.info(f"'{store_name}' 아티팩트 저장 시작: {final_target}")
    adapter.write(df, final_target, options)
    logger.info(f"'{store_name}' 아티팩트 저장 완료: {final_target}")