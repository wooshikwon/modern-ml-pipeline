# src/utils/system/mlflow_utils.py

import mlflow
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec, ParamSpec, ParamSchema

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

def create_model_signature(input_df: pd.DataFrame, output_df: pd.DataFrame) -> ModelSignature:
    """
    입력 및 출력 데이터프레임을 기반으로 MLflow ModelSignature를 동적으로 생성합니다.
    
    Args:
        input_df (pd.DataFrame): 모델 입력 데이터프레임 (학습 시 사용된 형태)
        output_df (pd.DataFrame): 모델 출력 데이터프레임 (예측 결과 형태)
    
    Returns:
        ModelSignature: run_mode, return_intermediate 파라미터를 포함한 완전한 signature
    """
    try:
        # 입력 스키마 생성
        input_schema = Schema([
            ColSpec(
                type=_infer_pandas_dtype_to_mlflow_type(input_df[col].dtype),
                name=col
            )
            for col in input_df.columns
        ])
        
        # 출력 스키마 생성
        output_schema = Schema([
            ColSpec(
                type=_infer_pandas_dtype_to_mlflow_type(output_df[col].dtype),
                name=col
            )
            for col in output_df.columns
        ])
        
        # 파라미터 스키마 생성 (run_mode, return_intermediate 지원)
        params_schema = ParamSchema([
            ParamSpec(
                name="run_mode",
                dtype="string",
                default="batch",
                shape=None
            ),
            ParamSpec(
                name="return_intermediate",
                dtype="boolean", 
                default=False,
                shape=None
            )
        ])
        
        # ModelSignature 생성
        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=params_schema
        )
        
        logger.info(f"ModelSignature 생성 완료:")
        logger.info(f"  - 입력 컬럼: {len(input_schema.inputs)}개")
        logger.info(f"  - 출력 컬럼: {len(output_schema.inputs)}개")
        logger.info(f"  - 파라미터: run_mode, return_intermediate")
        
        return signature
        
    except Exception as e:
        logger.error(f"ModelSignature 생성 실패: {e}", exc_info=True)
        raise

def _infer_pandas_dtype_to_mlflow_type(pandas_dtype) -> str:
    """
    pandas dtype을 MLflow type으로 변환하는 헬퍼 함수
    
    Args:
        pandas_dtype: pandas 컬럼의 dtype
    
    Returns:
        str: MLflow 호환 타입 문자열
    """
    dtype_str = str(pandas_dtype)
    
    # 정수형
    if pandas_dtype.name in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        return "long"
    
    # 실수형
    elif pandas_dtype.name in ['float16', 'float32', 'float64']:
        return "double"
    
    # 불린형
    elif pandas_dtype.name == 'bool':
        return "boolean"
    
    # 문자열형
    elif pandas_dtype.name == 'object' or 'string' in dtype_str:
        return "string"
    
    # 날짜/시간형
    elif pandas_dtype.name.startswith('datetime'):
        return "datetime"
    
    # 기본값 (알 수 없는 타입)
    else:
        logger.warning(f"알 수 없는 pandas dtype: {pandas_dtype}, 'string'으로 처리")
        return "string" 