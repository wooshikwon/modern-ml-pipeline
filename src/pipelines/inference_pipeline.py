from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

import mlflow

from src.factory import Factory
from src.settings import Settings
from src.settings.factory import SettingsFactory
from src.settings.mlflow_restore import restore_all_from_mlflow
from src.utils.core.console import Console
from src.utils.core.logger import log_milestone, log_phase, log_pipeline_start, logger
from src.utils.core.reproducibility import set_global_seeds
from src.utils.data.data_io import format_predictions, load_inference_data, save_output
from src.utils.integrations import mlflow_integration as mlflow_utils


def run_inference_pipeline(
    run_id: str,
    recipe_path: Optional[str] = None,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    context_params: Optional[Dict[str, Any]] = None,
):
    """
    배치 추론 파이프라인을 실행합니다.

    Args:
        run_id: 학습된 모델의 MLflow Run ID
        recipe_path: Override할 Recipe 파일 경로 (None이면 artifact에서 복원)
        config_path: Override할 Config 파일 경로 (None이면 artifact에서 복원)
        data_path: 추론할 데이터 경로 (None이면 artifact의 SQL 사용)
        context_params: SQL 렌더링에 사용할 파라미터
    """
    console = Console()
    context_params = context_params or {}

    # 설정 로드: Artifact에서 복원 또는 Override
    log_phase("Settings Loading")
    settings = _load_inference_settings(run_id, recipe_path, config_path)

    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, "computed", {}).get("seed", 42)
    set_global_seeds(seed)

    # 파이프라인 시작 로깅
    pipeline_description = (
        f"Model Run ID: {run_id} | Environment: {settings.config.environment.name}"
    )
    log_pipeline_start("Batch Inference Pipeline", pipeline_description)

    # MLflow 실행 컨텍스트 시작
    run_name = f"batch_inference_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow_utils.start_run(settings, run_name=run_name) as run:
        inference_run_id = run.info.run_id

        # Factory 생성
        factory = Factory(settings)

        # 1. 모델 로드
        log_phase("Model Loading")
        model_uri = f"runs:/{run_id}/model"
        with console.progress_tracker(
            "model_loading", 100, f"Loading model from {model_uri}"
        ) as update:
            model = mlflow.pyfunc.load_model(model_uri)
            update(100)

        log_milestone(f"Model loaded successfully from {model_uri}", "success")

        # 2. 데이터 준비
        log_phase("Data Preparation")
        data_adapter = factory.create_data_adapter()
        df = load_inference_data(
            data_adapter=data_adapter,
            data_path=data_path,
            model=model,
            run_id=run_id,
            context_params=context_params,
            console=console,
        )

        log_milestone(
            f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns", "success"
        )
        mlflow.log_metric("inference_input_rows", len(df))
        mlflow.log_metric("inference_input_columns", len(df.columns))

        # 3. 예측 실행
        log_phase("Model Inference")
        with console.progress_tracker("inference", 100, "Running model prediction") as update:
            # run_mode="batch"로 Offline Store 사용
            predictions_result = model.predict(
                df, params={"run_mode": "batch", "return_dataframe": True}
            )

            # PyfuncWrapper에서 data_interface_schema 정보 가져오기
            wrapped_model = model.unwrap_python_model()
            data_interface_schema = getattr(wrapped_model, "data_interface_schema", {}) or {}

            # data_interface_schema를 사용하여 format
            predictions_df = format_predictions(predictions_result, df, data_interface_schema)
            update(100)

        log_milestone(f"Predictions generated: {len(predictions_df)} samples", "success")
        mlflow.log_metric("inference_output_rows", len(predictions_df))

        # 4. 결과 저장
        log_phase("Output Saving")
        save_output(
            df=predictions_df,
            settings=settings,
            output_type="inference",
            factory=factory,
            run_id=inference_run_id,
            console=console,
            additional_metadata={"model_run_id": run_id},
        )

        logger.info("[Batch Inference Pipeline] 완료")
        return SimpleNamespace(
            run_id=inference_run_id, model_uri=model_uri, prediction_count=len(predictions_df)
        )


def _load_inference_settings(
    run_id: str, recipe_path: Optional[str], config_path: Optional[str]
) -> Settings:
    """
    추론용 Settings 로드.

    우선순위 (완전 덮어쓰기):
    1. recipe_path/config_path가 제공되면 해당 파일 사용
    2. 제공되지 않으면 artifact에서 복원
    """
    # Artifact에서 복원
    logger.info(f"Artifact에서 설정 복원 시작: run_id={run_id}")
    artifact_recipe, artifact_config, artifact_sql = restore_all_from_mlflow(run_id)

    # Override 적용
    factory_instance = SettingsFactory()

    if recipe_path:
        logger.info(f"Recipe override: {recipe_path}")
        recipe = factory_instance._load_recipe(recipe_path)
    else:
        recipe = artifact_recipe
        logger.info("Artifact의 Recipe 사용")

    if config_path:
        logger.info(f"Config override: {config_path}")
        config = factory_instance._load_config(config_path)
    else:
        config = artifact_config
        logger.info("Artifact의 Config 사용")

    # Settings 객체 생성
    settings = Settings(recipe=recipe, config=config)
    logger.info(f"Settings 로드 완료: env={config.environment.name}, recipe={recipe.name}")

    return settings
