import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import mlflow

from src.utils.logger import logger
from src.utils import mlflow_utils, artifact_utils
from src.core.factory import Factory
from src.settings.settings import Settings, load_settings


def run_batch_inference(
    model_name: str,
    run_id: str,
    context_params: Optional[Dict[str, Any]] = None,
):
    """
    "완전 독립형 PyfuncWrapper"와 "통합 데이터 어댑터"를 사용하여
    투명한 배치 추론을 실행합니다.
    """
    logger.info(f"배치 추론 파이프라인 시작 (모델: {model_name}, Run ID: {run_id})")
    context_params = context_params or {}

    try:
        # 1. MLflow에서 "완전 독립형" PyfuncWrapper 로드
        model_uri = f"runs:/{run_id}/{model_name}"
        wrapper = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"PyfuncWrapper 로드 완료: {model_uri}")

        # 2. (검증) 로드된 Wrapper의 모델 이름이 올바른지 확인
        recipe_snapshot = wrapper.recipe_snapshot
        wrapper_model_name = recipe_snapshot.get("name")
        if wrapper_model_name != model_name:
            raise ValueError(
                f"로드된 Wrapper의 모델 이름({wrapper_model_name})이 "
                f"입력된 모델 이름({model_name})과 일치하지 않습니다."
            )
        logger.info("Wrapper 모델 이름 검증 완료.")

        # 3. 데이터 로드를 위한 임시 Settings 및 Factory 생성
        #    (Adapter가 GCP Project ID 같은 환경 정보에 접근해야 하므로 필요)
        #    이 Settings는 데이터 로딩에만 사용되며, Wrapper의 로직에는 영향을 주지 않음.
        temp_settings = load_settings(model_name)
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
        if "augmented_data" in results:
            artifact_utils.save_dataset(
                results["augmented_data"], "augmented_dataset", temp_settings
            )
        if "preprocessed_data" in results:
            artifact_utils.save_dataset(
                results["preprocessed_data"], "preprocessed_dataset", temp_settings
            )
        if "final_results" in results:
            artifact_utils.save_dataset(
                results["final_results"], "prediction_results", temp_settings
            )

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise