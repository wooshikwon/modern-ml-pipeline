import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import mlflow

from src.factory import Factory

# 콜백 타입 정의: (event, stats) 형식
ProgressCallback = Callable[[str, str], None]
from src.settings import Settings
from src.settings.mlflow_restore import save_training_artifacts_to_mlflow
from src.utils.core.environment_check import get_pip_requirements
from src.utils.core.logger import (
    get_current_log_file,
    log_data,
    log_mlflow,
    log_pipe,
    log_train,
    logger,
)
from src.utils.core.reproducibility import set_global_seeds
from src.utils.integrations import mlflow_integration as mlflow_utils
from src.utils.integrations.mlflow_integration import (
    log_enhanced_model_with_schema,
    log_training_results,
)


def run_train_pipeline(
    settings: Settings,
    context_params: Optional[Dict[str, Any]] = None,
    record_requirements: bool = False,
    on_progress: Optional[ProgressCallback] = None,
):
    """
    모델 학습 파이프라인을 실행합니다.
    Factory를 통해 데이터 어댑터와 모든 컴포넌트를 생성하고,
    PyfuncWrapper를 생성하여 MLflow에 저장합니다.

    Args:
        on_progress: 진행 상태 콜백. (event, stats) 형식으로 호출됨.
                    event: "loading_data", "loading_data_done", "training", etc.
                    stats: "134,218 rows", "4m 23s", etc.
    """

    def emit(event: str, stats: str = "") -> None:
        """진행 상태 이벤트 발생"""
        if on_progress:
            on_progress(event, stats)

    # [3/6] Loading data 시작 (초기화 로그도 들여쓰기 적용)
    emit("loading_data")

    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, "computed", {}).get("seed", 42)
    set_global_seeds(seed)

    # 파이프라인 시작 로깅
    task_type = settings.recipe.task_choice
    model_name = getattr(settings.recipe.model, "class_path", "Unknown")

    log_pipe("========== 학습 파이프라인 시작 ==========")
    log_pipe(
        f"환경: {settings.config.environment.name} | 태스크: {task_type} | 모델: {model_name.split('.')[-1]}"
    )
    context_params = context_params or {}

    # computed 필드 초기화
    model = settings.recipe.model
    if not hasattr(model, "computed"):
        model.computed = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recipe_name = getattr(settings.recipe, "name", "run")
    model.computed.setdefault("run_name", f"{recipe_name}_{timestamp}")
    model.computed.setdefault("environment", getattr(settings.config.environment, "name", "local"))

    # MLflow 실행 컨텍스트 시작
    with mlflow_utils.start_run(
        settings, run_name=settings.recipe.model.computed.get("run_name", "training_run")
    ) as run:
        run_id = run.info.run_id

        # Factory 생성 및 컴포넌트 초기화
        factory = Factory(settings)
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        preprocessor = factory.create_preprocessor()
        trainer = factory.create_trainer()
        model = factory.create_model()
        calibrator = factory.create_calibrator()
        evaluator = factory.create_evaluator()

        # 데이터 로딩
        df = data_adapter.read(settings.recipe.data.loader.source_uri, params=context_params)
        log_data(f"로드 완료 - {len(df):,}행, {len(df.columns)}열")

        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 피처 증강
        augmented_df = fetcher.fetch(df, run_mode="train") if fetcher else df

        # 데이터 준비
        (
            X_train,
            y_train,
            add_train,
            X_val,
            y_val,
            add_val,
            X_test,
            y_test,
            add_test,
            calibration_data,
        ) = datahandler.split_and_prepare(augmented_df)

        # 전처리
        if preprocessor:
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train, dataset_name="train")
            X_val = preprocessor.transform(X_val, dataset_name="val") if not X_val.empty else X_val
            X_test = preprocessor.transform(X_test, dataset_name="test") if not X_test.empty else X_test

            if calibration_data is not None:
                X_calib, y_calib, add_calib = calibration_data
                X_calib = preprocessor.transform(X_calib, dataset_name="calib")
                calibration_data = (X_calib, y_calib, add_calib)

        emit("loading_data_done", f"{len(df):,} rows")

        # 학습
        tuning_enabled = getattr(settings.recipe.model.hyperparameters, "tuning_enabled", False)
        emit("training_optuna" if tuning_enabled else "training")
        training_start = time.time()
        trained_model, trainer_info = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=model,
            additional_data={"train": add_train, "val": add_val},
        )

        # Calibration
        trained_calibrator = None
        if calibrator and calibration_data is not None:
            X_calib, y_calib, add_calib = calibration_data
            y_prob_calib = trained_model.predict_proba(X_calib)
            trained_calibrator = calibrator.fit(y_prob_calib, y_calib)
            log_train("확률 보정 학습 완료")
        elif calibrator:
            logger.warning("[TRAIN] 보정기가 생성되었으나 보정 데이터가 없습니다")

        training_duration = time.time() - training_start
        emit("training_done", _format_duration(training_duration))

        # 평가
        emit("evaluating")
        metrics = evaluator.evaluate(trained_model, X_test, y_test, add_test)

        # Calibration 평가 (Factory로 모든 복잡한 로직 위임)
        calibration_evaluator = factory.create_calibration_evaluator(
            trained_model, trained_calibrator
        )
        if calibration_evaluator:
            calibration_metrics = calibration_evaluator.evaluate(X_test, y_test)
            metrics.update(calibration_metrics)

        training_results = {
            "evaluation_metrics": metrics,
            "trainer": trainer_info,
        }
        log_training_results(settings, metrics, training_results)

        # Task별 기본 최적화 메트릭으로 CLI 출력
        from src.components.evaluator.registry import EvaluatorRegistry

        default_metric = EvaluatorRegistry.get_default_optimization_metric(task_type)
        primary_metric = metrics.get(default_metric)
        emit("evaluating_done", f"{default_metric}: {primary_metric:.2f}" if primary_metric else "")

        # 모델 저장
        emit("saving")
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_datahandler=datahandler,
            trained_preprocessor=preprocessor,
            trained_fetcher=fetcher,
            trained_calibrator=trained_calibrator,
            training_df=augmented_df,
            training_results=training_results,
        )

        pip_reqs = get_pip_requirements() if record_requirements else []

        if not (pyfunc_wrapper.signature and pyfunc_wrapper.data_schema):
            raise ValueError(
                "Failed to generate signature and data_schema. This should not happen."
            )

        log_enhanced_model_with_schema(
            python_model=pyfunc_wrapper,
            signature=pyfunc_wrapper.signature,
            data_schema=pyfunc_wrapper.data_schema,
            input_example=augmented_df.head(5),
            pip_requirements=pip_reqs,
        )

        # Training artifacts 저장 (batch inference에서 사용)
        save_training_artifacts_to_mlflow(
            recipe=settings.recipe,
            config=settings.config,
            source_uri=settings.recipe.data.loader.source_uri,
        )

        # 로그 파일 MLflow artifact로 업로드
        logging_config = getattr(settings.config, "logging", None)
        upload_log = getattr(logging_config, "upload_to_mlflow", True) if logging_config else True
        log_artifact_uri = None
        if upload_log:
            log_file = get_current_log_file()
            if log_file and log_file.exists():
                mlflow.log_artifact(str(log_file), "training_artifacts")
                log_mlflow(f"로그 파일 업로드 완료 - {log_file.name}")
                log_artifact_uri = f"{run.info.artifact_uri}/training_artifacts/{log_file.name}"

        emit("saving_done")
        log_pipe("========== 학습 파이프라인 완료 ==========")

        # MLflow URL 및 Artifact 경로 생성
        tracking_uri = mlflow.get_tracking_uri()
        experiment_id = run.info.experiment_id
        artifact_uri = run.info.artifact_uri
        mlflow_url = None
        if tracking_uri and tracking_uri.startswith("http"):
            mlflow_url = f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"

        return SimpleNamespace(
            run_id=run_id,
            model_uri=f"runs:/{run_id}/model",
            artifact_uri=artifact_uri,
            log_artifact_uri=log_artifact_uri,
            mlflow_url=mlflow_url,
        )


def _format_duration(seconds: float) -> str:
    """초를 사람이 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
