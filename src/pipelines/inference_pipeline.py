import pandas as pd
from typing import Dict, Any, Optional

from src.settings.settings import Settings
from src.utils.logger import logger
from src.core.loader import get_dataset_loader
from src.core.augmenter import get_augmenter
from src.core.preprocessor import Preprocessor
from src.utils import mlflow_utils, artifact_utils


def _validate_schema(df: pd.DataFrame, settings: Settings):
    """입력 데이터프레임이 data_interface에 정의된 스키마와 일치하는지 검증합니다."""
    expected_schema = settings.model.data_interface.features
    errors = []
    logger.info("모델 입력 데이터 스키마를 검증합니다...")
    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            errors.append(f"- 필수 컬럼 누락: '{col}'")
            continue
        actual_type = str(df[col].dtype)
        is_valid = False
        if expected_type == "numeric" and pd.api.types.is_numeric_dtype(df[col]):
            is_valid = True
        elif expected_type == "category" and (pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            is_valid = True
        if not is_valid:
            errors.append(f"- 컬럼 '{col}' 타입 불일치: 예상='{expected_type}', 실제='{actual_type}'")
    if errors:
        error_message = "모델 입력 데이터 스키마 검증 실패:\n" + "\n".join(errors)
        error_message += "\n\n'preprocessor' 또는 'augmenter' 설정을 확인하여 스키마를 맞추세요."
        raise TypeError(error_message)
    logger.info("스키마 검증 성공.")


def run_batch_inference(
    settings: Settings,
    model_name: str,
    run_id: str,
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
        pyfunc_wrapper = mlflow_utils.load_pyfunc_model_from_run(run_id, model_name, settings)
        raw_model = pyfunc_wrapper.model
        preprocessor = pyfunc_wrapper.preprocessor

        # 2. 데이터 로딩
        loader = get_dataset_loader(settings)
        input_df = loader.load(params=context_params)
        if input_df.empty:
            logger.warning("입력 데이터가 비어있어 추론을 중단합니다.")
            return

        # 3. 피처 증강 (선택적)
        augmented_df = input_df
        augmenter = get_augmenter(settings)
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            augmented_df = augmenter.augment(input_df, context_params=context_params)
            artifact_utils.save_dataset(augmented_df, 'augmented_dataset', settings)
        else:
            logger.info("Augmenter 설정이 없어 피처 증강을 건너뜁니다.")

        # 4. 데이터 전처리 (선택적)
        final_df = augmented_df
        if preprocessor:
            logger.info("데이터 전처리를 시작합니다.")
            final_df = preprocessor.transform(augmented_df)
            artifact_utils.save_dataset(final_df, 'preprocessed_dataset', settings)
        else:
            logger.info("Preprocessor 설정이 없어 데이터 전처리를 건너뜁니다.")

        # 5. 스키마 검증
        _validate_schema(final_df, settings)

        # 6. 예측
        logger.info("모델 예측을 시작합니다.")
        predictions = raw_model.predict(final_df)
        
        # 7. 최종 결과 결합 및 저장
        logger.info("최종 결과를 결합하고 저장합니다.")
        results_df = augmented_df.copy()
        results_df['uplift_score'] = predictions
        artifact_utils.save_dataset(results_df, 'prediction_results', settings)

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise
