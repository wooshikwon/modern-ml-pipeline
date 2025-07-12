import json
from pathlib import Path

import mlflow

from src.settings.settings import Settings
from src.core.loader import get_dataset_loader
from src.core.trainer import Trainer
from src.utils.logger import logger
from src.utils import mlflow_utils


def run_training(settings: Settings, loader_name: str):
    """
    모델 학습 파이프라인을 실행합니다.
    `mlflow_utils`를 사용하여 MLflow 상호작용을 중앙에서 관리합니다.
    """
    logger.info(f"'{settings.model.name}' 모델 학습 파이프라인을 시작합니다.")

    # mlflow_utils.start_run은 내부적으로 예외 처리를 포함하고,
    # 실패 시 자동으로 status 태그를 'failed'로 설정합니다.
    with mlflow_utils.start_run(settings) as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_name", settings.model.name)

        # 1. 파라미터 로깅
        logger.info("MLflow에 설정 파라미터를 로깅합니다.")
        mlflow.log_params(settings.preprocessor.params.dict())
        mlflow.log_params(settings.model.hyperparameters.__root__)
        mlflow.log_param("model_name", settings.model.name)
        mlflow.log_param("loader_name", loader_name)

        # 2. 데이터 로딩
        logger.info(f"데이터셋 로딩: '{loader_name}'")
        loader = get_dataset_loader(loader_name, settings)
        df = loader.load()
        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 3. 모델 학습
        logger.info("Trainer를 초기화하고 모델 학습을 시작합니다.")
        trainer = Trainer(settings=settings)
        preprocessor, model, results = trainer.train(df)
        logger.info("모델 학습이 완료되었습니다.")

        # 4. 결과 로깅
        logger.info("학습 결과(지표, 아티팩트)를 MLflow에 로깅합니다.")

        # 4-1. 지표 로깅
        metrics = results.get("metrics", {})
        mlflow.log_metrics(metrics)
        logger.info(f"지표 로깅 완료: {metrics}")

        # 4-2. 학습된 Preprocessor 아티팩트로 저장
        preprocessor_path = preprocessor.save(version=run_id)
        mlflow.log_artifact(preprocessor_path, "preprocessor")
        logger.info(f"Preprocessor가 '{preprocessor_path}'에 저장되고 MLflow에 로깅되었습니다.")

        # 4-3. 피처 중요도 등 추가 정보 로깅
        metadata = results.get("metadata", {})
        if metadata:
            local_dir = Path("./local/artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = local_dir / f"metadata-{run_id}.json"
            with metadata_path.open('w', encoding='utf-8') as f:
                serializable_metadata = json.loads(json.dumps(metadata, default=str))
                json.dump(serializable_metadata, f, indent=4)
            mlflow.log_artifact(str(metadata_path), "metadata")
            logger.info(f"메타데이터가 '{metadata_path}'에 저장되고 MLflow에 로깅되었습니다.")

