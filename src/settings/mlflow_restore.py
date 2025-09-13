"""MLflow 기반 완전한 Recipe 복원 시스템"""

import mlflow
import yaml
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from .recipe import Recipe
from src.utils.core.logger import logger


class MLflowRecipeRestorer:
    """MLflow에서 학습시 Recipe 완전 복원"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = mlflow.tracking.MlflowClient()

    def restore_recipe(self) -> Recipe:
        """학습시 Recipe 완전 복원"""
        try:
            logger.info(f"MLflow에서 Recipe 복원 시작: run_id={self.run_id}")
            
            # 1. recipe_snapshot.yaml 다운로드
            recipe_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/recipe_snapshot.yaml"
            )

            # 2. Recipe 데이터 로드
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = yaml.safe_load(f)

            # 3. 환경변수 치환 (현재 추론 환경 기준)
            recipe_data = self._resolve_env_variables(recipe_data)

            # 4. Recipe 객체 생성
            recipe = Recipe(**recipe_data)
            logger.info(f"Recipe 복원 성공: {recipe.name}")
            return recipe

        except FileNotFoundError:
            # Legacy Run 호환성 - 하위 호환 fallback
            logger.warning(f"Recipe snapshot을 찾을 수 없습니다: run_id={self.run_id}")
            return self._fallback_legacy_recipe()
        except Exception as e:
            logger.error(f"Recipe 복원 실패: {str(e)}")
            raise ValueError(f"Recipe 복원 실패 (run_id: {self.run_id}): {str(e)}")

    def get_training_context(self) -> Dict[str, Any]:
        """학습시 실행 컨텍스트 복원"""
        try:
            context_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/execution_context.json"
            )

            with open(context_path, 'r') as f:
                context = json.load(f)
            
            logger.debug(f"학습 컨텍스트 복원: {context.keys()}")
            return context
        except FileNotFoundError:
            logger.warning("실행 컨텍스트를 찾을 수 없습니다.")
            return {}
        except Exception as e:
            logger.warning(f"실행 컨텍스트 복원 실패: {str(e)}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 복원"""
        try:
            run = self.client.get_run(self.run_id)
            return {
                "model_uri": f"runs:/{self.run_id}/model",
                "model_stage": run.data.tags.get("mlflow.runName", "None"),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
        except Exception as e:
            logger.warning(f"모델 정보 복원 실패: {str(e)}")
            return {}

    def _resolve_env_variables(self, data: Any) -> Any:
        """환경변수 치환 (현재 추론 환경 기준)"""
        import re
        
        if isinstance(data, str):
            # ${VAR:default} 패턴 매칭
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                expr = match.group(1)
                
                if ':' in expr:
                    var_name, default_value = expr.split(':', 1)
                    var_name = var_name.strip()
                    default_value = default_value.strip()
                    return str(os.environ.get(var_name, default_value))
                else:
                    var_name = expr.strip()
                    return str(os.environ.get(var_name, match.group(0)))
            
            return re.sub(pattern, replacer, data)
        
        elif isinstance(data, dict):
            return {k: self._resolve_env_variables(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            return [self._resolve_env_variables(item) for item in data]
        
        return data

    def _fallback_legacy_recipe(self) -> Recipe:
        """Legacy Run용 기본 Recipe 생성"""
        from .recipe import (
            Recipe, Model, Data, Loader, Fetcher, DataInterface, DataSplit,
            Evaluation, Metadata, HyperparametersTuning
        )
        from datetime import datetime
        
        logger.warning("Legacy run 감지 - 기본 Recipe로 대체합니다.")
        
        # MLflow에서 기본 정보 추출 시도
        try:
            run = self.client.get_run(self.run_id)
            model_class = run.data.params.get("model_class", "sklearn.ensemble.RandomForestClassifier")
            library = run.data.params.get("library", "sklearn")
            task_type = run.data.params.get("task_type", "classification")
        except:
            model_class = "sklearn.ensemble.RandomForestClassifier"
            library = "sklearn"
            task_type = "classification"

        return Recipe(
            name=f"legacy_recipe_{self.run_id[:8]}",
            task_choice=task_type,
            model=Model(
                class_path=model_class,
                library=library,
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False,
                    values={"n_estimators": 100}
                )
            ),
            data=Data(
                loader=Loader(source_uri=None),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(
                    target_column="target",
                    entity_columns=["id"]
                ),
                split=DataSplit(train=0.8, test=0.1, validation=0.1)
            ),
            evaluation=Evaluation(metrics=["accuracy"], random_state=42),
            metadata=Metadata(
                author="MLflowRecipeRestorer",
                created_at=datetime.now().isoformat(),
                description=f"Legacy recipe restored from run {self.run_id}"
            )
        )


class MLflowRecipeSaver:
    """학습시 Recipe 완전 저장"""

    @staticmethod
    def save_recipe_snapshot(recipe: Recipe, config: Optional[Dict] = None) -> None:
        """Recipe를 MLflow 아티팩트로 저장"""
        try:
            # 1. recipe_snapshot.yaml 생성
            recipe_data = recipe.model_dump()
            recipe_yaml = yaml.dump(recipe_data, default_flow_style=False, allow_unicode=True)
            
            # 임시 파일에 저장
            temp_dir = Path("/tmp/mlflow_recipe_snapshot")
            temp_dir.mkdir(exist_ok=True)
            
            recipe_file = temp_dir / "recipe_snapshot.yaml"
            with open(recipe_file, 'w', encoding='utf-8') as f:
                f.write(recipe_yaml)
            
            # MLflow에 아티팩트로 저장
            mlflow.log_artifact(str(recipe_file), "training_artifacts")
            logger.info("Recipe snapshot을 MLflow에 저장했습니다.")
            
            # 2. execution_context.json 생성 (선택사항)
            if config:
                context_data = {
                    "config_summary": {
                        "environment": config.get("environment", {}).get("name", "unknown"),
                        "data_source": config.get("data_source", {}).get("name", "unknown"),
                        "feature_store": config.get("feature_store", {}).get("provider", "none")
                    },
                    "timestamp": recipe.metadata.created_at if recipe.metadata else None
                }
                
                context_file = temp_dir / "execution_context.json"
                with open(context_file, 'w', encoding='utf-8') as f:
                    json.dump(context_data, f, indent=2, ensure_ascii=False)
                
                mlflow.log_artifact(str(context_file), "training_artifacts")
                logger.info("실행 컨텍스트를 MLflow에 저장했습니다.")
            
            # 임시 파일 정리
            recipe_file.unlink()
            if config:
                context_file.unlink()
            temp_dir.rmdir()
            
        except Exception as e:
            logger.error(f"Recipe snapshot 저장 실패: {str(e)}")
            # 저장 실패해도 학습은 계속 진행


def save_recipe_to_mlflow(recipe: Recipe, config: Optional[Dict] = None) -> None:
    """편의 함수: Recipe를 MLflow에 저장"""
    MLflowRecipeSaver.save_recipe_snapshot(recipe, config)


def restore_recipe_from_mlflow(run_id: str) -> Recipe:
    """편의 함수: MLflow에서 Recipe 복원"""
    restorer = MLflowRecipeRestorer(run_id)
    return restorer.restore_recipe()
