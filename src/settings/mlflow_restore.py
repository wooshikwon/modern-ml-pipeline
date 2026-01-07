"""MLflow 기반 학습 Artifact 저장/복원 시스템"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import yaml

from src.utils.core.logger import logger

from .config import Config
from .recipe import Recipe


class MLflowArtifactRestorer:
    """MLflow에서 학습 시 Artifact 복원 (Recipe, Config, SQL)"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = mlflow.tracking.MlflowClient()

    def restore_recipe(self) -> Recipe:
        """Recipe 복원"""
        try:
            logger.debug(f"[MLFLOW] Recipe 복원 시작: run_id={self.run_id[:8]}...")

            recipe_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="training_artifacts/recipe_snapshot.yaml"
            )

            with open(recipe_path, "r", encoding="utf-8") as f:
                recipe_data = yaml.safe_load(f)

            recipe_data = self._resolve_env_variables(recipe_data)
            recipe = Recipe(**recipe_data)
            logger.info(f"[MLFLOW] Recipe 복원 완료: {recipe.name}")
            return recipe

        except FileNotFoundError:
            logger.error(f"[MLFLOW] Recipe snapshot 없음: run_id={self.run_id}")
            raise ValueError(f"Recipe를 찾을 수 없습니다. run_id={self.run_id}")
        except Exception as e:
            logger.error(f"[MLFLOW] Recipe 복원 실패: {str(e)}")
            raise ValueError(f"Recipe 복원 실패 (run_id: {self.run_id}): {str(e)}")

    def restore_config(self) -> Config:
        """Config 복원"""
        try:
            logger.debug(f"[MLFLOW] Config 복원 시작: run_id={self.run_id[:8]}...")

            config_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="training_artifacts/config_snapshot.yaml"
            )

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config_data = self._resolve_env_variables(config_data)
            config = Config(**config_data)
            logger.info(f"[MLFLOW] Config 복원 완료: env={config.environment.name}")
            return config

        except FileNotFoundError:
            logger.error(f"[MLFLOW] Config snapshot 없음: run_id={self.run_id}")
            raise ValueError(f"Config를 찾을 수 없습니다. run_id={self.run_id}")
        except Exception as e:
            logger.error(f"[MLFLOW] Config 복원 실패: {str(e)}")
            raise ValueError(f"Config 복원 실패 (run_id: {self.run_id}): {str(e)}")

    def restore_sql(self) -> Optional[str]:
        """SQL 쿼리 내용 복원 (학습 시 SQL을 사용한 경우에만 존재)"""
        try:
            # 먼저 artifact 존재 여부 확인
            artifacts = self.client.list_artifacts(self.run_id, path="training_artifacts")
            sql_exists = any(a.path == "training_artifacts/source_query.sql" for a in artifacts)

            if not sql_exists:
                # CSV 등 SQL이 아닌 데이터로 학습한 경우 - 정상 케이스
                logger.debug(f"[MLFLOW] SQL artifact 없음 (CSV 데이터로 학습): run_id={self.run_id[:8]}...")
                return None

            logger.debug(f"[MLFLOW] SQL 복원 시작: run_id={self.run_id[:8]}...")

            sql_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="training_artifacts/source_query.sql"
            )

            with open(sql_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            logger.info(f"[MLFLOW] SQL 복원 완료: {len(sql_content)} bytes")
            return sql_content

        except Exception as e:
            logger.warning(f"[MLFLOW] SQL 복원 실패: {str(e)}")
            return None

    def restore_all(self) -> Tuple[Recipe, Config, Optional[str]]:
        """Recipe, Config, SQL 모두 복원"""
        recipe = self.restore_recipe()
        config = self.restore_config()
        sql = self.restore_sql()
        return recipe, config, sql

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 복원"""
        try:
            run = self.client.get_run(self.run_id)
            return {
                "model_uri": f"runs:/{self.run_id}/model",
                "model_stage": run.data.tags.get("mlflow.runName", "None"),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.warning(f"[MLFLOW] 모델 정보 복원 실패: {str(e)}")
            return {}

    def _resolve_env_variables(self, data: Any) -> Any:
        """환경변수 치환"""
        import os
        import re

        if isinstance(data, str):
            pattern = r"\$\{([^}]+)\}"

            def replacer(match):
                expr = match.group(1)
                if ":" in expr:
                    var_name, default_value = expr.split(":", 1)
                    return str(os.environ.get(var_name.strip(), default_value.strip()))
                else:
                    return str(os.environ.get(expr.strip(), match.group(0)))

            return re.sub(pattern, replacer, data)

        elif isinstance(data, dict):
            return {k: self._resolve_env_variables(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [self._resolve_env_variables(item) for item in data]

        return data


class MLflowArtifactSaver:
    """학습 시 Artifact 저장 (Recipe, Config, SQL)"""

    @staticmethod
    def save_training_artifacts(
        recipe: Recipe, config: Config, source_uri: Optional[str] = None
    ) -> None:
        """Recipe, Config, SQL을 MLflow artifact로 저장"""
        temp_dir = Path("/tmp/mlflow_training_artifacts")
        temp_dir.mkdir(exist_ok=True)

        try:
            # 1. Recipe 저장
            recipe_data = recipe.model_dump()
            recipe_yaml = yaml.dump(recipe_data, default_flow_style=False, allow_unicode=True)
            recipe_file = temp_dir / "recipe_snapshot.yaml"
            with open(recipe_file, "w", encoding="utf-8") as f:
                f.write(recipe_yaml)
            mlflow.log_artifact(str(recipe_file), "training_artifacts")
            logger.info("[MLFLOW] Recipe snapshot 저장 완료")

            # 2. Config 저장
            config_data = config.model_dump()
            config_yaml = yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
            config_file = temp_dir / "config_snapshot.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_yaml)
            mlflow.log_artifact(str(config_file), "training_artifacts")
            logger.info("[MLFLOW] Config snapshot 저장 완료")

            # 3. SQL 파일 내용 저장 (source_uri가 파일 경로인 경우)
            if source_uri:
                sql_content = MLflowArtifactSaver._read_sql_content(source_uri)
                if sql_content:
                    sql_file = temp_dir / "source_query.sql"
                    with open(sql_file, "w", encoding="utf-8") as f:
                        f.write(sql_content)
                    mlflow.log_artifact(str(sql_file), "training_artifacts")
                    logger.info(f"[MLFLOW] SQL snapshot 저장 완료: {len(sql_content)} bytes")

            # 임시 파일 정리
            for f in temp_dir.iterdir():
                f.unlink()
            temp_dir.rmdir()

        except Exception as e:
            logger.error(f"[MLFLOW] Training artifacts 저장 실패: {str(e)}")

    @staticmethod
    def _read_sql_content(source_uri: str) -> Optional[str]:
        """SQL 파일 내용 읽기"""
        try:
            # 파일 경로인 경우만 처리
            if source_uri.endswith((".sql", ".sql.j2")):
                path = Path(source_uri)
                if path.exists():
                    return path.read_text(encoding="utf-8")
                else:
                    logger.warning(f"[MLFLOW] SQL 파일을 찾을 수 없습니다: {source_uri}")
            return None
        except Exception as e:
            logger.warning(f"[MLFLOW] SQL 파일 읽기 실패: {str(e)}")
            return None


# 편의 함수
def save_training_artifacts_to_mlflow(
    recipe: Recipe, config: Config, source_uri: Optional[str] = None
) -> None:
    """편의 함수: Training artifacts를 MLflow에 저장"""
    MLflowArtifactSaver.save_training_artifacts(recipe, config, source_uri)


def restore_recipe_from_mlflow(run_id: str) -> Recipe:
    """편의 함수: MLflow에서 Recipe 복원"""
    restorer = MLflowArtifactRestorer(run_id)
    return restorer.restore_recipe()


def restore_config_from_mlflow(run_id: str) -> Config:
    """편의 함수: MLflow에서 Config 복원"""
    restorer = MLflowArtifactRestorer(run_id)
    return restorer.restore_config()


def restore_all_from_mlflow(run_id: str) -> Tuple[Recipe, Config, Optional[str]]:
    """편의 함수: MLflow에서 Recipe, Config, SQL 모두 복원"""
    restorer = MLflowArtifactRestorer(run_id)
    return restorer.restore_all()
