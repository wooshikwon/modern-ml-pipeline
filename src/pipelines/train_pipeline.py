import logging
import json
from pathlib import Path

import mlflow

from config.settings import Settings
from src.core.loader import get_dataset_loader
from src.core.trainer import Trainer
from src.utils.logger import logger


def run_training(settings: Settings):
    """
    모델 학습 파이프라인을 실행합니다.
    1. MLflow 설정 및 실행
    2. 설정 파라미터 로깅
    3. 데이터 로딩
    4. 모델 학습 (Trainer 사용)
    5. 결과(지표, 아티팩트) MLflow에 로깅
    """
    logger.info(f"'{settings.model.name}' 모델 학습 파이프라인을 시작합니다.")

    # 1. MLflow 설정
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.set_tag("model_name", settings.model.name)

        try:
            # 2. 파라미터 로깅
            logger.info("MLflow에 설정 파라미터를 로깅합니다.")
            # Pydantic 모델을 dict로 변환하여 로깅
            mlflow.log_params(settings.preprocessor.params.dict())
            mlflow.log_params(settings.model.hyperparameters.__root__)
            mlflow.log_param("model_name", settings.model.name)

            # 3. 데이터 로딩
            # loader 설정의 첫 번째 키를 데이터셋 이름으로 간주
            dataset_name = list(settings.loader.keys())[0]
            logger.info(f"데이터셋 로딩: '{dataset_name}'")
            loader = get_dataset_loader(dataset_name, settings)
            df = loader.load()
            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))

            # 4. 모델 학습
            logger.info("Trainer를 초기화하고 모델 학습을 시작합니다.")
            trainer = Trainer(settings=settings)
            # train() 메서드는 학습된 객체들과 결과 딕셔너리를 반환
            augmenter, preprocessor, model, results = trainer.train(df)
            logger.info("모델 학습이 완료되었습니다.")

            # 5. 결과 로깅
            logger.info("학습 결과(지표, 아티팩트)를 MLflow에 로��합니다.")

            # 5-1. 지표 로깅
            metrics = results.get("metrics", {})
            mlflow.log_metrics(metrics)
            logger.info(f"지표 로깅 완료: {metrics}")

            # 5-2. 학습된 Preprocessor 아티팩트로 저장
            preprocessor_path = preprocessor.save(version=run_id)
            mlflow.log_artifact(preprocessor_path, "preprocessor")
            logger.info(f"Preprocessor가 '{preprocessor_path}'에 저장되고 MLflow에 로깅되었습니다.")

            # 5-3. 피처 중요도 등 추가 정보 로깅
            metadata = results.get("metadata", {})
            if metadata:
                local_dir = Path("./local/artifacts")
                local_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = local_dir / f"metadata-{run_id}.json"
                with metadata_path.open('w', encoding='utf-8') as f:
                    # 직렬화 가능한 형태로 변환
                    serializable_metadata = json.loads(json.dumps(metadata, default=str))
                    json.dump(serializable_metadata, f, indent=4)
                mlflow.log_artifact(str(metadata_path), "metadata")
                logger.info(f"메타데이터가 '{metadata_path}'에 저장되고 MLflow에 로깅되었습니다.")

            mlflow.set_tag("status", "success")
            logger.info("학습 파이프라인이 성공적으로 완료되었습니다.")

        except Exception as e:
            mlflow.set_tag("status", "failed")
            logger.error(f"학습 파이프라인 중 오류 발생: {e}", exc_info=True)
            raise
