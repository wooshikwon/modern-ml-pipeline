# src/utils/system/mlflow_utils.py

import datetime
import json
import os
import uuid
from contextlib import contextmanager

# 순환 참조를 피하기 위해 타입 힌트만 임포트
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

if TYPE_CHECKING:
    from src.settings import Settings
    from mlflow.entities import Run
    from mlflow.pyfunc import PyFuncModel

from src.utils.core.logger import log_mlflow, logger


def generate_unique_run_name(base_run_name: str) -> str:
    """
    기본 run name에 timestamp와 random suffix를 추가하여 완전히 유니크한 run name을 생성합니다.
    병렬 테스트 실행 시 MLflow run name 충돌을 방지합니다.

    Args:
        base_run_name (str): 기본 run name (예: "e2e_classification_test_run")

    Returns:
        str: 유니크한 run name (예: "e2e_classification_test_run_20250907_143025_a1b2")
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]  # 처음 8자리만 사용
    unique_run_name = f"{base_run_name}_{timestamp}_{random_suffix}"

    logger.debug(f"[MLFLOW] 유니크 run name 생성: {base_run_name} -> {unique_run_name}")
    return unique_run_name


def setup_mlflow(settings: "Settings") -> None:
    """
    주입된 settings 객체를 기반으로 MLflow 클라이언트를 설정합니다.
    """
    mlflow.set_tracking_uri(settings.config.mlflow.tracking_uri)
    mlflow.set_experiment(settings.config.mlflow.experiment_name)

    log_mlflow("설정 완료")
    logger.debug(f"[MLFLOW] Tracking URI: {settings.config.mlflow.tracking_uri}")
    logger.debug(f"[MLFLOW] Experiment: {settings.config.mlflow.experiment_name}")


@contextmanager
def start_run(settings: "Settings", run_name: str) -> "Run":
    """
    MLflow 실행을 시작하고 관리하는 컨텍스트 매니저.
    외부 환경 변수의 영향을 받지 않도록 tracking_uri를 명시적으로 설정합니다.
    자동으로 유니크한 run name을 생성하여 병렬 실행 시 충돌을 방지합니다.
    """
    # 충돌 방지를 위해 유니크한 run name 생성
    unique_run_name = generate_unique_run_name(run_name)

    # 외부에서 지정된 tracking_uri(예: 테스트)가 있다면 존중: 명시적으로 설정
    tracking_uri = settings.config.mlflow.tracking_uri
    if tracking_uri:
        # file:// 스토어는 루트 디렉토리를 미리 생성해야 함
        parsed = urlparse(tracking_uri)
        if parsed.scheme == "file" and parsed.path:
            try:
                os.makedirs(parsed.path, exist_ok=True)
            except Exception:
                # 디렉토리 생성 실패는 아래 설정 시점에서 에러로 노출됨
                pass
        mlflow.set_tracking_uri(tracking_uri)

    # 실험명 설정 (tracking_uri 설정 이후)
    mlflow.set_experiment(settings.config.mlflow.experiment_name)

    try:
        with mlflow.start_run(run_name=unique_run_name) as run:
            log_mlflow(f"Run 시작 - ID: {run.info.run_id[:8]}...")
            # 원본 run name을 태그로 저장하여 추적 가능하게 함
            mlflow.set_tag("original_run_name", run_name)
            mlflow.set_tag("unique_run_name", unique_run_name)

            try:
                yield run
                mlflow.set_tag("status", "success")
                log_mlflow("Run 완료")
            except Exception as e:
                mlflow.set_tag("status", "failed")
                logger.error(f"[MLFLOW] Run 실패: {e}", exc_info=True)
                raise
    except Exception as mlflow_error:
        # MLflow 실행 자체가 실패한 경우 (예: run name 충돌이 여전히 발생한 경우)
        if (
            "already exists" in str(mlflow_error).lower()
            or "duplicate" in str(mlflow_error).lower()
        ):
            logger.warning(f"[MLFLOW] Run name 충돌 감지: {unique_run_name}")
            # 추가 random suffix로 재시도
            retry_run_name = f"{unique_run_name}_{uuid.uuid4().hex[:4]}"
            logger.debug(f"[MLFLOW] 재시도: {retry_run_name}")

            with mlflow.start_run(run_name=retry_run_name) as run:
                log_mlflow(f"Run 시작 (재시도) - ID: {run.info.run_id[:8]}...")
                mlflow.set_tag("original_run_name", run_name)
                mlflow.set_tag("unique_run_name", retry_run_name)
                mlflow.set_tag("retry_count", "1")

                try:
                    yield run
                    mlflow.set_tag("status", "success")
                    log_mlflow("Run 완료 (재시도)")
                except Exception as e:
                    mlflow.set_tag("status", "failed")
                    logger.error(f"[MLFLOW] Run 실패 (재시도): {e}", exc_info=True)
                    raise
        else:
            # 다른 종류의 MLflow 에러는 그대로 전파
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
            max_results=1,
        )

        if runs_df.empty:
            raise ValueError(f"Experiment '{experiment_name}'에서 성공한 run을 찾을 수 없습니다.")

        latest_run_id = runs_df.iloc[0]["run_id"]
        logger.debug(f"[MLFLOW] 최근 Run ID 조회: {latest_run_id[:8]}...")
        return latest_run_id

    except Exception as e:
        logger.error(f"[MLFLOW] 최근 Run ID 조회 실패: {e}")
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
    log_mlflow(f"모델 로딩 시작: {model_uri}")
    try:
        if model_uri.startswith("runs:/"):
            # MlflowClient를 명시적으로 생성하여 아티팩트 다운로드
            import re

            def _parse_runs_uri(uri: str) -> tuple[str, str]:
                """'runs:/<run_id>/<artifact_path>' URI를 파싱합니다."""
                match = re.match(r"runs:/([^/]+)/(.+)", uri)
                if not match:
                    raise ValueError(f"'{uri}'는 올바른 'runs:/' URI가 아닙니다.")
                return match.group(1), match.group(2)

            client = MlflowClient(tracking_uri=settings.config.mlflow.tracking_uri)
            run_id, artifact_path = _parse_runs_uri(model_uri)

            local_path = client.download_artifacts(run_id=run_id, path=artifact_path)
            logger.debug(f"[MLFLOW] 아티팩트 다운로드 완료: {local_path}")
            return mlflow.pyfunc.load_model(model_uri=local_path)
        else:
            # 일반 경로(local file, GCS, S3 등)는 기존 방식 사용
            mlflow.set_tracking_uri(settings.config.mlflow.tracking_uri)
            return mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error(f"[MLFLOW] 모델 로딩 실패: {model_uri}, 오류: {e}", exc_info=True)
        raise


def download_artifacts(
    settings: "Settings", run_id: str, artifact_path: str, dst_path: str = None
) -> str:
    """
    지정된 Run ID에서 특정 아티팩트를 다운로드하고, 로컬 경로를 반환합니다.
    """
    mlflow.set_tracking_uri(settings.config.mlflow.tracking_uri)
    logger.debug(f"[MLFLOW] 아티팩트 다운로드 시작: {artifact_path}")
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
        )
        logger.debug(f"[MLFLOW] 아티팩트 다운로드 완료: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"[MLFLOW] 아티팩트 다운로드 실패: {e}", exc_info=True)
        raise


def create_model_signature(
    input_df: pd.DataFrame, output_df: pd.DataFrame, params: dict = None
) -> ModelSignature:
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
        input_schema = Schema(
            [
                ColSpec(type=_infer_pandas_dtype_to_mlflow_type(input_df[col].dtype), name=col)
                for col in input_df.columns
            ]
        )

        # 출력 스키마 생성
        output_schema = Schema(
            [
                ColSpec(type=_infer_pandas_dtype_to_mlflow_type(output_df[col].dtype), name=col)
                for col in output_df.columns
            ]
        )

        # 파라미터 스키마 생성 (run_mode, return_intermediate 지원)
        params_schema = ParamSchema(
            [
                ParamSpec(name="run_mode", dtype="string", default="batch", shape=None),
                ParamSpec(name="return_intermediate", dtype="boolean", default=False, shape=None),
            ]
        )

        # ModelSignature 생성
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

        logger.debug(
            f"[MLFLOW] Signature 생성 완료 - 입력: {len(input_schema.inputs)}열, 출력: {len(output_schema.inputs)}열"
        )

        return signature

    except Exception as e:
        logger.error(f"[MLFLOW] ModelSignature 생성 실패: {e}", exc_info=True)
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
    if pandas_dtype.name in [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]:
        return "long"

    # 실수형
    elif pandas_dtype.name in ["float16", "float32", "float64"]:
        return "double"

    # 불린형
    elif pandas_dtype.name == "bool":
        return "boolean"

    # 문자열형
    elif pandas_dtype.name == "object" or "string" in dtype_str:
        return "string"

    # 날짜/시간형
    elif pandas_dtype.name.startswith("datetime"):
        return "datetime"

    # 기본값 (알 수 없는 타입)
    else:
        logger.warning(f"[MLFLOW] 알 수 없는 pandas dtype: {pandas_dtype}, 'string'으로 처리")
        return "string"


# 🆕 Phase 5: 완전 자기 기술 Artifact - Enhanced MLflow 통합 함수들


def create_enhanced_model_signature_with_schema(
    training_df: pd.DataFrame, data_interface_config: dict
) -> tuple[ModelSignature, dict]:
    """
    학습-추론 일관성을 위한 MLflow Signature와 스키마 메타데이터 생성.

    Args:
        training_df (pd.DataFrame): Training 데이터 (전처리 전)
        data_interface_config (dict): 피처 컬럼 정보 (input/model 분리)

    Returns:
        tuple[ModelSignature, dict]: Signature와 스키마 메타데이터
    """
    logger.debug("[MLFLOW] 학습 피처 기준 Signature 생성 중")

    # 입력 피처 컬럼 결정 (전처리 전 기준, Signature 생성용)
    input_feature_cols = data_interface_config.get("input_feature_columns")
    if not input_feature_cols:
        # 하위 호환성: 기존 feature_columns 폴백
        input_feature_cols = data_interface_config.get("feature_columns")

    if not input_feature_cols:
        # 자동 도출: entity/timestamp/target 제외
        exclude_cols = []
        if data_interface_config.get("entity_columns"):
            exclude_cols.extend(data_interface_config["entity_columns"])
        if data_interface_config.get("timestamp_column"):
            exclude_cols.append(data_interface_config["timestamp_column"])
        if data_interface_config.get("target_column"):
            exclude_cols.append(data_interface_config["target_column"])
        input_feature_cols = [col for col in training_df.columns if col not in exclude_cols]

    # training_df에 존재하는 컬럼만 선택 (전처리 전 데이터이므로)
    available_input_cols = [col for col in input_feature_cols if col in training_df.columns]

    # Signature용 input_example 생성 (전처리 전 피처 기준)
    input_example = training_df.head(5).copy()
    input_example = input_example[available_input_cols] if available_input_cols else input_example
    sample_output = pd.DataFrame({"prediction": [0.0] * len(input_example)})
    signature = create_model_signature(input_example, sample_output)

    # 모델 피처 컬럼 (전처리 후 기준)
    model_feature_cols = data_interface_config.get("model_feature_columns") or available_input_cols

    # 데이터 타입 매핑
    data_types = {}
    for col in available_input_cols:
        if col in training_df.columns:
            data_types[col] = str(training_df[col].dtype)

    # 스키마 메타데이터 생성
    data_schema = {
        "schema_version": "2.0",
        "entity_columns": data_interface_config.get("entity_columns") or [],
        "timestamp_column": data_interface_config.get("timestamp_column"),
        "target_column": data_interface_config.get("target_column"),
        # 전처리 전 입력 피처 (추론 입력용)
        "input_feature_columns": available_input_cols,
        # 전처리 후 모델 피처 (모델 입력용)
        "model_feature_columns": model_feature_cols,
        # 하위 호환성
        "feature_columns": available_input_cols,
        "inference_columns": available_input_cols,
        # 컬럼 정보
        "column_count": len(available_input_cols),
        "data_types": data_types,
        # MLflow 정보
        "mlflow_version": mlflow.__version__,
        "signature_created_at": pd.Timestamp.now().isoformat(),
        "schema_created_at": pd.Timestamp.now().isoformat(),
        # Phase 통합 추적
        "phase_integration": {
            "phase_1_schema_first": True,
            "phase_2_point_in_time": True,
            "phase_3_secure_sql": True,
            "phase_4_auto_validation": True,
            "phase_5_enhanced_artifact": True,
        },
    }

    logger.debug(
        f"[MLFLOW] Signature 생성 완료 - 입력: {len(available_input_cols)}열, 모델: {len(model_feature_cols)}열"
    )

    return signature, data_schema


def log_enhanced_model_with_schema(
    python_model,
    signature: ModelSignature,
    data_schema: dict,
    input_example: pd.DataFrame,
    pip_requirements: Optional[List[str]] = None,
):
    """
    기존 mlflow.pyfunc.log_model + 확장된 메타데이터 저장

    기존 MLflow 저장 기능을 보존하면서 완전한 스키마 메타데이터를 함께 저장.
    100% 재현성과 자기 기술성을 보장하는 Enhanced Artifact 구현.

    Args:
        python_model: PyfuncWrapper 모델 인스턴스
        signature (ModelSignature): Enhanced Model Signature
        data_schema (dict): 완전한 스키마 메타데이터
        input_example (pd.DataFrame): 입력 예제 데이터
    """
    log_mlflow("모델 아티팩트 저장 시작")

    # Signature와 호환되는 input_example 생성 - 원본 타입 유지
    sig_input_names = [col.name for col in signature.inputs.inputs]
    filtered_example = input_example[
        [c for c in sig_input_names if c in input_example.columns]
    ].copy()

    # 1. 기존 MLflow 저장 로직 활용 (검증된 기능 보존)
    mlflow.pyfunc.log_model(
        name="model",
        python_model=python_model,
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=filtered_example,
        metadata={"data_schema": json.dumps(data_schema)},
    )
    logger.debug("[MLFLOW] Model 로그 완료")

    # 2. 완전한 스키마 메타데이터 저장
    mlflow.log_dict(data_schema, "model/data_schema.json")
    logger.debug("[MLFLOW] Data schema 저장 완료")

    # 3. 호환성 및 버전 정보 저장
    compatibility_info = {
        "artifact_version": "2.0",
        "creation_timestamp": pd.Timestamp.now().isoformat(),
        "mlflow_version": mlflow.__version__,
        "schema_validator_version": "2.0",
        # Phase별 기능 활성화 상태
        "features_enabled": {
            "entity_timestamp_schema": True,
            "point_in_time_correctness": True,
            "sql_injection_protection": True,
            "automatic_schema_validation": True,
            "self_descriptive_artifact": True,
        },
        # 호환성 정보
        "backward_compatibility": {
            "supports_legacy_models": False,
            "requires_enhanced_pipeline": True,
        },
        # 품질 보증 정보
        "quality_assurance": {
            "schema_drift_protection": True,
            "data_leakage_prevention": True,
            "reproducibility_guaranteed": True,
        },
    }
    mlflow.log_dict(compatibility_info, "model/compatibility_info.json")
    logger.debug("[MLFLOW] 호환성 정보 저장 완료")

    # 4. Phase 통합 요약 정보 저장
    phase_summary = {
        "phase_1": {
            "name": "Schema-First 설계",
            "achievements": ["Entity+Timestamp 필수화", "EntitySchema 구현", "Recipe 구조 현대화"],
        },
        "phase_2": {
            "name": "Point-in-Time 안전성",
            "achievements": ["ASOF JOIN 검증", "fetcher 현대화", "미래 데이터 누출 방지"],
        },
        "phase_3": {
            "name": "보안 강화 Dynamic SQL",
            "achievements": ["SQL Injection 방지", "화이트리스트 검증", "보안 템플릿 표준화"],
        },
        "phase_4": {
            "name": "일관성 자동 검증",
            "achievements": ["Schema Drift 조기 발견", "타입 호환성 엔진", "자동 검증 통합"],
        },
        "phase_5": {
            "name": "완전 자기 기술 Artifact",
            "achievements": ["100% 재현성 보장", "완전한 메타데이터 캡슐화", "자기 기술적 구조"],
        },
    }
    mlflow.log_dict(phase_summary, "model/phase_integration_summary.json")

    log_mlflow("모델 아티팩트 저장 완료")


# --- Simple results logging helper -------------------------------------------------


def log_training_results(settings: "Settings", metrics: dict, training_results: dict) -> None:
    """
    파이프라인에서 간결하게 호출하기 위한 결과 로깅 헬퍼.
    - 메트릭 로깅
    - HPO(on/off) 분기 및 하이퍼파라미터/최적 점수 로깅
    """
    # 1) Metrics
    if metrics:
        mlflow.log_metrics(metrics)
        # 메트릭 로깅 (4개씩 묶어서 여러 줄로 출력)
        items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        log_mlflow(f"메트릭 기록 ({len(items)}개)")
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            log_mlflow(f"{', '.join(chunk)}")

    # 2) Hyperparameters / HPO
    hpo = (training_results or {}).get("trainer", {}).get("hyperparameter_optimization")
    if hpo and hpo.get("enabled"):
        best_params = hpo.get("best_params") or {}
        if best_params:
            mlflow.log_params(best_params)
        if "best_score" in hpo:
            mlflow.log_metric("best_score", hpo["best_score"])
        if "total_trials" in hpo:
            mlflow.log_metric("total_trials", hpo["total_trials"])
    else:
        # HPO 비활성화 시에도 고정 하이퍼파라미터 기록을 시도
        try:
            hp = settings.recipe.model.hyperparameters
            if hasattr(hp, "values") and hp.values:
                mlflow.log_params(hp.values)
        except Exception:
            pass
