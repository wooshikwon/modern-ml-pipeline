import json
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow

from src.settings.settings import Settings
from src.core.loader import get_dataset_loader
from src.core.trainer import Trainer
from src.utils.logger import logger
from src.utils import mlflow_utils

def _log_config_artifacts(settings: Settings):
    """학습에 사용된 레시피와 SQL 파일의 스냅샷을 MLflow에 저장합니다."""
    logger.info("설정 아티팩트 저장을 시작합니다.")
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    recipe_path = base_dir / "recipes" / f"{settings.model.name}.yaml"
    if recipe_path.exists():
        mlflow.log_artifact(str(recipe_path), artifact_path="recipe_snapshot")
    
    if hasattr(settings.model.loader, 'source_sql_path') and settings.model.loader.source_sql_path:
        loader_sql_path = base_dir / settings.model.loader.source_sql_path
        if loader_sql_path.exists():
            mlflow.log_artifact(str(loader_sql_path), artifact_path="recipe_snapshot/sql")

    if settings.model.augmenter and hasattr(settings.model.augmenter, 'source_template_path') and settings.model.augmenter.source_template_path:
        augmenter_sql_path = base_dir / settings.model.augmenter.source_template_path
        if augmenter_sql_path.exists():
            mlflow.log_artifact(str(augmenter_sql_path), artifact_path="recipe_snapshot/sql")
            
    logger.info("설정 아티팩트 저장을 완료했습니다.")


def run_training(settings: Settings, context_params: Optional[Dict[str, Any]] = None):
    """
    모델 학습 파이프라인을 실행합니다.
    `mlflow_utils`를 사용하여 MLflow 상호작용을 중앙에서 관리합니다.
    """
    logger.info(f"'{settings.model.name}' 모델 학습 파이프라인을 시작합니다.")
    context_params = context_params or {}

    with mlflow_utils.start_run(settings) as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_name", settings.model.name)

        # 1. 파라미터 로깅
        logger.info("MLflow에 설정 파라미터를 로깅합니다.")
        mlflow.log_params(settings.model.dict())

        # 2. 데이터 로딩
        loader = get_dataset_loader(settings)
        df = loader.load(params=context_params)
        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 3. 모델 학습
        trainer = Trainer(settings=settings)
        preprocessor, model, results = trainer.train(df, context_params=context_params)
        
        # 4. 결과 로깅
        logger.info("학습 결과(지표, 아티팩트)를 MLflow에 로깅합니다.")
        _log_config_artifacts(settings)
        
        metrics = results.get("metrics", {})
        mlflow.log_metrics(metrics)
        
        if preprocessor:
            preprocessor_path = preprocessor.save(version=run_id)
            mlflow.log_artifact(preprocessor_path, "preprocessor")
            
        metadata = results.get("metadata", {})
        if metadata:
            local_dir = Path("./local/artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = local_dir / f"metadata-{run_id}.json"
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, default=str)
            mlflow.log_artifact(str(metadata_path), "metadata")