# src/utils/mlflow_utils.py

import mlflow
from contextlib import contextmanager

# 순환 참조를 피하기 위해 타입 힌트만 임포트
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.settings.settings import Settings
    from mlflow.entities import Run
    from mlflow.pyfunc import PyFuncModel

from src.utils.logger import logger

def setup_mlflow(settings: "Settings") -> None:
    """
    MLflow 클라이언트의 추적 URI와 실험실을 설정합니다.
    이 함수는 다른 MLflow 유틸리티 함수들에 의해 내부적으로 호출됩니다.
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    logger.debug(f"MLflow tracking URI set to: {settings.mlflow.tracking_uri}")
    logger.debug(f"MLflow experiment set to: {settings.mlflow.experiment_name}")

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

def get_model_uri(model_name: str, stage: str) -> str:
    """
    모델 이름과 스테이지에 해당하는 모델 URI 문자열을 생성합니다.
    """
    uri = f"models:/{model_name}/{stage}"
    logger.debug(f"Generated model URI: {uri}")
    return uri

def load_pyfunc_model(model_name: str, stage: str, settings: "Settings") -> "PyFuncModel":
    """
    지정된 스테이지의 모델을 MLflow에서 로드하여 Pyfunc 모델 객체를 반환합니다.
    """
    setup_mlflow(settings) # URI 설정 선행
    model_uri = get_model_uri(model_name, stage)
    logger.info(f"Loading model from: {model_uri}")
    try:
        return mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}", exc_info=True)
        raise
