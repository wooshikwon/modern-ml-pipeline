import json
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import mlflow

from src.settings.settings import Settings
from src.core.factory import Factory
from src.core.trainer import Trainer
from src.utils.system.logger import logger
from src.utils.system import mlflow_utils


def run_training(settings: Settings, context_params: Optional[Dict[str, Any]] = None):
    """
    모델 학습 파이프라인을 실행합니다.
    Factory를 통해 데이터 어댑터와 모든 컴포넌트를 생성하고, 최종적으로
    순수 로직 PyfuncWrapper를 생성하여 MLflow에 저장합니다.
    """
    logger.info(f"'{settings.model._computed['run_name']}' 모델 학습 파이프라인을 시작합니다.")
    context_params = context_params or {}

    factory = Factory(settings)

    with mlflow_utils.start_run(settings) as run:
        run_id = run.info.run_id
        
        # 자동 생성된 Run Name 설정
        run_name = settings.model._computed["run_name"]
        mlflow.set_tag("mlflow.runName", run_name)
        
        # 체계적인 실험 조직을 위한 추가 태그 설정
        mlflow.set_tag("model_class", settings.model._computed["model_class_name"])
        mlflow.set_tag("recipe_file", settings.model._computed["recipe_file"])
        mlflow.set_tag("experiment_type", "training")
        mlflow.set_tag("class_path", settings.model.class_path)
        mlflow.set_tag("timestamp", settings.model._computed["timestamp"])

        # 모델 설정 및 하이퍼파라미터 로깅
        mlflow.log_params(settings.model.hyperparameters.root)
        mlflow.log_param("class_path", settings.model.class_path)

        # 1. 데이터 어댑터를 사용하여 데이터 로딩
        loader_uri = settings.model.loader.source_uri
        if settings.environment.app_env == "local" and settings.model.loader.local_override_uri:
            loader_uri = settings.model.loader.local_override_uri
        
        scheme = urlparse(loader_uri).scheme
        data_adapter = factory.create_data_adapter(scheme)
        
        df = data_adapter.read(loader_uri, params=context_params)
        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 2. 학습에 사용할 컴포넌트 생성
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()

        # 3. 모델 학습
        trainer = Trainer(settings=settings)
        trained_preprocessor, trained_model, metrics = trainer.train(
            df=df,
            model=model,
            augmenter=augmenter,
            preprocessor=preprocessor,
            context_params=context_params,
        )
        
        # 4. 결과 로깅
        mlflow.log_metrics(metrics)

        # 5. 순수 로직 PyfuncWrapper 생성 및 저장
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
        )
        
        mlflow.pyfunc.log_model(
            artifact_path=settings.model.name,
            python_model=pyfunc_wrapper,
            registered_model_name=settings.model.name,
            description=f"Uplift model '{settings.model.name}' based on recipe.",
        )
        logger.info(f"순수 로직 모델 '{settings.model.name}'을 MLflow에 성공적으로 저장했습니다.")

        # 6. (선택적) 메타데이터 저장
        metadata = {"run_id": run_id, "model_name": settings.model.name}
        local_dir = Path("./local/artifacts")
        local_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = local_dir / f"metadata-{run_id}.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=str)
        mlflow.log_artifact(str(metadata_path), "metadata")