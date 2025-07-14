# src/utils/system/mlflow_utils.py

import mlflow
from contextlib import contextmanager
from pathlib import Path

# 순환 참조를 피하기 위해 타입 힌트만 임포트
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.settings import Settings
    from mlflow.entities import Run
    from mlflow.pyfunc import PyFuncModel

from src.utils.system.logger import logger

def setup_mlflow(settings: "Settings") -> None:
    """
    config 기반 MLflow 클라이언트 설정
    환경별로 다른 tracking_uri와 experiment_name을 자동으로 적용합니다.
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    
    logger.info(f"MLflow 설정 완료:")
    logger.info(f"  - Tracking URI: {settings.mlflow.tracking_uri}")
    logger.info(f"  - Experiment: {settings.mlflow.experiment_name}")
    logger.info(f"  - Environment: {settings.environment.app_env}")

@contextmanager
def start_run(settings: "Settings") -> "Run":
    """
    MLflow 실행을 시작하고 관리하는 컨텍스트 매니저.
    `with` 구문과 함께 사용되어 실행이 끝나면 자동으로 종료되도록 합니다.
    """
    setup_mlflow(settings)
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run started: {run.info.run_id}")
        try:
            yield run
            mlflow.set_tag("status", "success")
            logger.info("MLflow Run finished successfully.")
        except Exception as e:
            mlflow.set_tag("status", "failed")
            logger.error(f"MLflow Run failed: {e}", exc_info=True)
            raise # 원래 예외를 다시 발생시켜 상위 호출자가 알 수 있도록 함

def get_model_uri_by_stage(model_name: str, stage: str) -> str:
    """
    모델 이름과 스테이지에 해당하는 모델 URI 문자열을 생성합니다.
    """
    uri = f"models:/{model_name}/{stage}"
    logger.debug(f"Generated model URI by stage: {uri}")
    return uri

def get_model_uri_by_run_id(run_id: str, model_name: str) -> str:
    """
    Run ID와 모델 이름(아티팩트 경로)에 해당하는 모델 URI 문자열을 생성합니다.
    """
    uri = f"runs:/{run_id}/{model_name}"
    logger.debug(f"Generated model URI by run_id: {uri}")
    return uri

def load_pyfunc_model_by_stage(model_name: str, stage: str, settings: "Settings") -> "PyFuncModel":
    """
    지정된 스테이지의 모델을 MLflow에서 로드하여 Pyfunc 모델 객체를 반환합니다. (주로 API 서빙용)
    """
    setup_mlflow(settings)
    model_uri = get_model_uri_by_stage(model_name, stage)
    logger.info(f"Loading model from: {model_uri}")
    try:
        return mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}", exc_info=True)
        raise

def load_pyfunc_model_from_run(run_id: str, model_name: str, settings: "Settings") -> "PyFuncModel":
    """
    지정된 Run ID에서 모델을 로드하여 Pyfunc 모델 객체를 반환합니다. (주로 배치 추론용)
    """
    setup_mlflow(settings)
    model_uri = get_model_uri_by_run_id(run_id, model_name)
    logger.info(f"Loading model from: {model_uri}")
    try:
        return mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}", exc_info=True)
        raise

def download_artifact(run_id: str, artifact_path: str, settings: "Settings") -> str:
    """
    지정된 Run ID에서 특정 아티팩트를 다운로드하고, 로컬 경로를 반환합니다.
    """
    setup_mlflow(settings)
    logger.info(f"Downloading artifact '{artifact_path}' from run '{run_id}'")
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        logger.info(f"Artifact downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download artifact '{artifact_path}' from run '{run_id}': {e}", exc_info=True)
        raise 