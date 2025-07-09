import logging
import json
from pathlib import Path

import mlflow

# config.settings에서 통합 설정 객체를 import
from config.settings import Settings
from src.core.loader import get_dataset_loader
from src.core.trainer import Trainer
from src.core.model import UpliftModelPyfuncWrapper
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
    logger.info("Uplift 모델 학습 파이프라인을 시작합니다.")

    # 1. MLflow 설정
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        try:
            # 2. 파라미터 로깅
            logger.info("MLflow에 설정 파라미터를 로깅합니다.")
            mlflow.log_params(settings.pipeline.transformer.params.model_dump())
            mlflow.log_params(settings.model.hyperparameters)
            mlflow.log_param("model_name", settings.model.name)

            # 3. 데이터 로딩
            dataset_name = list(settings.pipeline.loader.keys())[0]
            logger.info(f"데이터셋 로딩: '{dataset_name}'")
            loader = get_dataset_loader(dataset_name, settings)
            df = loader.load()
            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))
            

            # 4. 모델 학습
            logger.info("Trainer를 초기화하고 모델 학습을 시작합니다.")
            trainer = Trainer(settings=settings)
            transformer, model, results = trainer.train(df)
            logger.info("모델 학습이 완료되었습니다.")

            # 5. 결과 로깅
            logger.info("학습 결과(지표, 아티팩트)를 MLflow에 로깅합니다.")

            # 5-1. 지표 로깅
            metrics = results.get("metrics", {})
            mlflow.log_metrics(metrics)
            logger.info(f"지표 로깅 완료: {metrics}")

            # 5-2. 통합 모델(Transformer + Model)을 MLflow Model Registry에 등록
            logger.info("통합 모델을 MLflow Model Registry에 등록합니다.")
            
            pyfunc_model = UpliftModelPyfuncWrapper(model=model, transformer=transformer)
            
            model_name = settings.model.name
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=pyfunc_model,
                registered_model_name=model_name,
                code_paths=["src/"],
            )
            logger.info(f"모델 '{model_name}'이 Model Registry에 성공적으로 등록되었습니다.")

            # 5-3. 피처 중요도 등 추가 정보는 별도 아티팩트로 저장
            metadata = results.get("metadata", {})
            feature_importance = metadata.get("feature_importance", {})
            if feature_importance:
                local_dir = Path("./local/artifacts")
                local_dir.mkdir(parents=True, exist_ok=True)
                fi_path = local_dir / f"feature_importance-{run_id}.json"
                with fi_path.open('w', encoding='utf-8') as f:
                    json.dump(feature_importance, f, indent=4)
                mlflow.log_artifact(str(fi_path), "feature_importance")
                logger.info(f"피처 중요도가 '{fi_path}'에 저장되고 MLflow에 로깅되었습니다.")

            mlflow.set_tag("status", "success")
            logger.info("학습 파��프라인이 성공적으로 완료되었습니다.")

        except Exception as e:
            mlflow.set_tag("status", "failed")
            logger.error(f"학습 파이프라인 중 오류 발생: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    from config.settings import settings
    run_training(settings)
