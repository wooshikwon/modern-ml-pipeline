# src/utils/system/mlflow_utils.py

import mlflow
from contextlib import contextmanager
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
    주입된 settings 객체를 기반으로 MLflow 클라이언트를 설정합니다.
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    
    logger.info(f"MLflow 설정 완료:")
    logger.info(f"  - Tracking URI: {settings.mlflow.tracking_uri}")
    logger.info(f"  - Experiment: {settings.mlflow.experiment_name}")
    logger.info(f"  - Environment: {settings.environment.app_env}")

@contextmanager
def start_run(settings: "Settings", run_name: str) -> "Run":
    """
    MLflow 실행을 시작하고 관리하는 컨텍스트 매니저.
    외부 환경 변수의 영향을 받지 않도록 tracking_uri를 명시적으로 설정합니다.
    """
    # setup_mlflow(settings) # 더 이상 전역 설정에 의존하지 않음
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run started: {run.info.run_id} ({run_name}) for experiment '{settings.mlflow.experiment_name}'")
        try:
            yield run
            mlflow.set_tag("status", "success")
            logger.info("MLflow Run finished successfully.")
        except Exception as e:
            mlflow.set_tag("status", "failed")
            logger.error(f"MLflow Run failed: {e}", exc_info=True)
            raise

def get_latest_run_id(settings: "Settings", experiment_name: str) -> str:
    """
    지정된 experiment에서 가장 최근에 성공한 run의 ID를 반환합니다.
    """
    setup_mlflow(settings)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}'을 찾을 수 없습니다.")
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.status = 'success'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs_df.empty:
            raise ValueError(f"Experiment '{experiment_name}'에서 성공한 run을 찾을 수 없습니다.")
            
        latest_run_id = runs_df.iloc[0]['run_id']
        logger.info(f"가장 최근에 성공한 Run ID 조회: {latest_run_id} (Experiment: {experiment_name})")
        return latest_run_id
        
    except Exception as e:
        logger.error(f"최근 Run ID 조회 실패: {e}")
        raise

def get_model_uri(run_id: str, artifact_path: str = "model") -> str:
    """
    Run ID와 아티팩트 경로를 사용하여 모델 URI를 생성합니다.
    """
    uri = f"runs:/{run_id}/{artifact_path}"
    logger.debug(f"생성된 모델 URI: {uri}")
    return uri

def load_pyfunc_model(settings: "Settings", model_uri: str) -> "PyFuncModel":
    """
    지정된 URI에서 모델을 로드하여 Pyfunc 모델 객체를 반환합니다.
    외부 환경 변수의 영향을 받지 않도록 MlflowClient를 직접 생성하여 사용합니다.
    """
    logger.info(f"MLflow 모델 로딩 시작: {model_uri}")
    try:
        if model_uri.startswith("runs:/"):
            # MlflowClient를 명시적으로 생성하여 아티팩트 다운로드
            from mlflow.tracking import MlflowClient
            import re

            def _parse_runs_uri(uri: str) -> tuple[str, str]:
                """'runs:/<run_id>/<artifact_path>' URI를 파싱합니다."""
                match = re.match(r"runs:/([^/]+)/(.+)", uri)
                if not match:
                    raise ValueError(f"'{uri}'는 올바른 'runs:/' URI가 아닙니다.")
                return match.group(1), match.group(2)

            client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
            run_id, artifact_path = _parse_runs_uri(model_uri)
            
            local_path = client.download_artifacts(run_id=run_id, path=artifact_path)
            logger.info(f"아티팩트를 성공적으로 다운로드했습니다: {local_path}")
            return mlflow.pyfunc.load_model(model_uri=local_path)
        else:
            # 일반 경로(local file, GCS, S3 등)는 기존 방식 사용
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            return mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"모델 로딩 실패: {model_uri}, 오류: {e}", exc_info=True)
        raise

def download_artifacts(settings: "Settings", run_id: str, artifact_path: str, dst_path: str = None) -> str:
    """
    지정된 Run ID에서 특정 아티팩트를 다운로드하고, 로컬 경로를 반환합니다.
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    logger.info(f"아티팩트 다운로드 시작: '{artifact_path}' (Run ID: '{run_id}')")
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path
        )
        logger.info(f"아티팩트 다운로드 완료: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"아티팩트 다운로드 실패: {e}", exc_info=True)
        raise

def create_model_signature(input_df: pd.DataFrame, output_df: pd.DataFrame, params: dict = None) -> ModelSignature:
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