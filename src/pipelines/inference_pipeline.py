import pandas as pd
from typing import Dict, Any, Optional

from src.settings.settings import Settings
from src.utils.logger import logger
from src.core.loader import get_dataset_loader
from src.core.augmenter import SQLTemplateAugmenter
from src.core.preprocessor import Preprocessor
from src.utils import mlflow_utils, artifact_utils


def run_batch_inference(
    settings: Settings,
    model_name: str,
    run_id: str,
    loader_name: str,
    context_params: Optional[Dict[str, Any]] = None,
):
    """
    투명한 배치 추론 파이프라인을 실행합니다.
    지정된 run_id에서 개별 아티팩트(모델, 전처리기)를 로드하여,
    단계별로 추론을 수행하고 중간 산출물을 저장합니다.
    """
    logger.info(f"배치 추론 파이프라인 시작: (모델: {model_name}, Run ID: {run_id})")
    context_params = context_params or {}

    try:
        # 1. 아티팩트 로드
        logger.info(f"'{run_id}' 실행에서 아티팩트를 로드합니다.")
        preprocessor_path = mlflow_utils.download_artifact(run_id, "preprocessor", settings)
        preprocessor: Preprocessor = Preprocessor.load(preprocessor_path, settings)
        
        pyfunc_wrapper = mlflow_utils.load_pyfunc_model_from_run(run_id, model_name, settings)
        raw_model = pyfunc_wrapper.model

        # 2. 데이터 로딩 (Loader)
        logger.info(f"'{loader_name}' 로더를 사용하여 데이터를 로딩합니다.")
        loader = get_dataset_loader(loader_name, settings)
        input_df = loader.load(params=context_params)
        if input_df.empty:
            logger.warning("입력 데이터가 비어있어 추론을 중단합니다.")
            return

        # 3. 피처 증강 (Augmenter)
        logger.info("피처 증강을 시작합니다.")
        augmenter_name = settings.model.augmenter
        augmenter_config = settings.augmenters[augmenter_name]
        augmenter = SQLTemplateAugmenter(config=augmenter_config, settings=settings)
        augmented_df = augmenter.augment(input_df, context_params=context_params)
        artifact_utils.save_dataset(augmented_df, 'augmented_dataset', settings)

        # 4. 데이터 전처리 (Preprocessor)
        logger.info("데이터 전처리를 시작합니다.")
        preprocessed_df = preprocessor.transform(augmented_df)
        artifact_utils.save_dataset(preprocessed_df, 'preprocessed_dataset', settings)

        # 5. 예측 (Predict)
        logger.info("모델 예측을 시작합니다.")
        predictions = raw_model.predict(preprocessed_df)
        
        # 6. 최종 결과 결합 및 저장
        logger.info("최종 결과를 결합하고 저장합니다.")
        results_df = augmented_df.copy()
        results_df['uplift_score'] = predictions
        artifact_utils.save_dataset(results_df, 'prediction_results', settings)

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise


