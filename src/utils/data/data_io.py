"""
Data I/O utilities for unified data operations across pipelines.

이 모듈은 파이프라인 전반에서 사용되는 공통 데이터 입출력 기능을 제공합니다:
- 데이터 로드 (SQL, Storage, Feast)
- 추론 결과 저장
- 템플릿 처리
- 예측 결과 포맷팅
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

from src.factory import Factory
from src.settings import Settings
from src.utils.core.console import Console
from src.utils.core.logger import log_milestone, logger
from src.utils.template.templating_utils import is_jinja_template, render_template_from_string


def _format_multiclass_probabilities(
    predictions_array: np.ndarray, original_df: pd.DataFrame, task_type: Optional[str] = None
) -> pd.DataFrame:
    """
    다중 클래스 확률 예측을 DataFrame으로 포맷팅하는 헬퍼 함수

    Args:
        predictions_array: 2D numpy array with class probabilities
        original_df: 원본 DataFrame (인덱스 참조용)
        task_type: 태스크 타입 (causal인 경우 CATE 컬럼명 사용)

    Returns:
        확률 컬럼이 포함된 DataFrame
    """
    n_cols = predictions_array.shape[1]

    # Causal 모델: CATE (Conditional Average Treatment Effect) 컬럼명 사용
    if task_type == "causal":
        if n_cols == 1:
            return pd.DataFrame({"cate": predictions_array[:, 0]}, index=original_df.index)
        else:
            cate_columns = {f"cate_treatment_{i}": predictions_array[:, i] for i in range(n_cols)}
            return pd.DataFrame(cate_columns, index=original_df.index)

    # Classification 모델
    if n_cols == 2:
        # Binary classification - return positive class probability
        prob_df = pd.DataFrame(
            {"prob_positive": predictions_array[:, 1], "prob_negative": predictions_array[:, 0]},
            index=original_df.index,
        )
    else:
        # Multiclass - return all class probabilities
        prob_columns = {f"prob_class_{i}": predictions_array[:, i] for i in range(n_cols)}
        prob_df = pd.DataFrame(prob_columns, index=original_df.index)

    return prob_df


def _detect_probability_predictions(pred_df: pd.DataFrame) -> bool:
    """
    DataFrame이 확률 예측을 포함하는지 감지하는 헬퍼 함수

    Args:
        pred_df: 예측 결과 DataFrame

    Returns:
        확률 예측 여부
    """
    prob_indicators = ["prob_", "probability_", "prob_class_", "prob_positive", "prob_negative"]
    return any(
        col.startswith(indicator) for col in pred_df.columns for indicator in prob_indicators
    )


def save_output(
    df: pd.DataFrame,
    settings: Settings,
    output_type: str,
    factory: Factory,
    run_id: str,
    console: Optional[Console] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    범용 출력 저장 함수.

    설정된 output adapter를 사용하여 데이터를 저장합니다.

    Args:
        df: 저장할 DataFrame
        settings: 설정 객체
        output_type: 출력 타입 (예: "inference")
        factory: 어댑터 생성을 위한 Factory 객체
        run_id: MLflow run ID
        console: 콘솔 매니저 (선택)
        additional_metadata: 추가 메타데이터 (선택)
    """
    if console is None:
        console = Console()

    # Output 설정 확인
    output_cfg = getattr(settings.config, "output", None)
    if not output_cfg:
        logger.info(f"No output configuration found, skipping {output_type} save")
        return

    target_cfg = getattr(output_cfg, output_type, None)
    if not target_cfg or not getattr(target_cfg, "enabled", True):
        logger.info(f"Output {output_type} disabled, skipping save")
        return

    # 메타데이터 추가
    if additional_metadata:
        for key, value in additional_metadata.items():
            df[key] = value

    # 기본 메타데이터 추가
    df[f"{output_type}_run_id"] = run_id
    df[f"{output_type}_timestamp"] = datetime.now()

    try:
        adapter_type = target_cfg.adapter_type

        if adapter_type == "storage":
            _save_to_storage(df, target_cfg, factory, run_id, output_type, console)

        elif adapter_type == "sql":
            _save_to_sql(df, target_cfg, factory, output_type, console)

        elif adapter_type == "bigquery":
            _save_to_bigquery(df, target_cfg, factory, output_type, console)

        else:
            logger.warning(f"Unknown output adapter type: {adapter_type}, skipping save")

    except Exception as e:
        logger.error(f"{output_type} 출력 저장 중 오류 발생: {e}")
        raise


def load_data(
    data_adapter,
    data_source: str,
    context_params: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    범용 데이터 로드 함수.

    Jinja 템플릿 처리와 다양한 데이터 소스를 지원합니다.

    Args:
        data_adapter: 데이터 어댑터 인스턴스
        data_source: 데이터 소스 (파일 경로, SQL 쿼리, 템플릿 등)
        context_params: Jinja 템플릿 파라미터 (선택)
        console: 콘솔 매니저 (선택)

    Returns:
        로드된 DataFrame
    """
    if console is None:
        console = Console()

    final_data_source = data_source

    # Jinja 템플릿 처리
    if context_params and (
        data_source.endswith(".sql.j2")
        or (data_source.endswith(".sql") and is_jinja_template(data_source))
    ):

        if data_source.endswith(".sql.j2"):
            # 파일 기반 템플릿
            template_file = Path(data_source)
            if not template_file.exists():
                raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {data_source}")

            logger.info(f"[Template] 렌더링 중: {template_file.name}")
            template_content = template_file.read_text()
            final_data_source = render_template_from_string(template_content, context_params)
        else:
            # 문자열 기반 템플릿
            final_data_source = render_template_from_string(data_source, context_params)

    # 데이터 로드
    with console.progress_tracker(
        "data_loading", 100, f"Loading data from {type(data_adapter).__name__}"
    ) as update:
        df = data_adapter.read(final_data_source)
        update(100)

    log_milestone(f"Data loaded: {len(df)} rows, {len(df.columns)} columns", "success")
    return df


def _save_to_storage(
    df: pd.DataFrame, target_cfg, factory: Factory, run_id: str, output_type: str, console: Console
):
    """Storage 어댑터를 사용한 저장."""
    storage_adapter = factory.create_data_adapter("storage")
    cfg = target_cfg.config
    base_path = (
        getattr(cfg, "base_path", None)
        if hasattr(cfg, "base_path")
        else (cfg.get("base_path") if isinstance(cfg, dict) else None)
    ) or f"./artifacts/{output_type}"

    # 파일명 생성
    if output_type == "inference":
        filename = f"predictions_{run_id}.parquet"
    else:
        # 다른 output type을 위한 일반적인 파일명 패턴
        filename = f"{output_type}_{run_id}.parquet"

    target_path = f"{base_path}/{filename}"

    with console.progress_tracker("storage_save", 100, f"Saving to {target_path}") as update:
        storage_adapter.write(df, target_path)
        update(100)

    # 로컬 파일만 MLflow artifact로 로깅
    if not target_path.startswith(("s3://", "gs://")):
        mlflow.log_artifact(target_path.replace("file://", ""))

    log_milestone(f"{output_type.capitalize()} saved to {target_path}", "success")


def _save_to_sql(
    df: pd.DataFrame, target_cfg, factory: Factory, output_type: str, console: Console
):
    """SQL 데이터베이스를 사용한 저장."""
    sql_adapter = factory.create_data_adapter("sql")
    cfg = target_cfg.config
    table = (
        getattr(cfg, "table", None)
        if hasattr(cfg, "table")
        else (cfg.get("table") if isinstance(cfg, dict) else None)
    )

    if not table:
        raise ValueError(f"output.{output_type}.config.table이 필요합니다.")

    with console.progress_tracker("sql_save", 100, f"Saving to table {table}") as update:
        sql_adapter.write(df, table, if_exists="append", index=False)
        update(100)

    log_milestone(f"{output_type.capitalize()} saved to SQL table {table}", "success")


def _save_to_bigquery(
    df: pd.DataFrame, target_cfg, factory: Factory, output_type: str, console: Console
):
    """BigQuery를 사용한 저장."""
    bq_adapter = factory.create_data_adapter("bigquery")

    cfg = target_cfg.config
    project_id = (
        getattr(cfg, "project_id", None)
        if hasattr(cfg, "project_id")
        else (cfg.get("project_id") if isinstance(cfg, dict) else None)
    )
    dataset = (
        getattr(cfg, "dataset_id", None)
        if hasattr(cfg, "dataset_id")
        else (cfg.get("dataset_id") if isinstance(cfg, dict) else None)
    )
    table = (
        getattr(cfg, "table", None)
        if hasattr(cfg, "table")
        else (cfg.get("table") if isinstance(cfg, dict) else None)
    )
    location = (
        getattr(cfg, "location", None)
        if hasattr(cfg, "location")
        else (cfg.get("location") if isinstance(cfg, dict) else None)
    )

    if not all([project_id, dataset, table]):
        raise ValueError(
            f"BigQuery {output_type} 출력에는 project_id, dataset_id, table이 필요합니다."
        )

    full_table = f"{dataset}.{table}"

    with console.progress_tracker("bigquery_save", 100, f"Saving to {full_table}") as update:
        bq_adapter.write(
            df,
            full_table,
            options={"project_id": project_id, "location": location, "if_exists": "append"},
        )
        update(100)

    log_milestone(f"{output_type.capitalize()} saved to BigQuery {full_table}", "success")


def process_template_file(
    template_path: str, context_params: Dict[str, Any], console: Optional[Console] = None
) -> str:
    """
    파일 기반 Jinja 템플릿을 처리합니다.

    Args:
        template_path: 템플릿 파일 경로
        context_params: 템플릿 렌더링 파라미터
        console: 콘솔 매니저 (선택)

    Returns:
        렌더링된 문자열 (SQL 등)

    Raises:
        FileNotFoundError: 템플릿 파일이 없는 경우
        ValueError: Jinja 템플릿에 파라미터가 없는 경우
    """
    if console is None:
        console = Console()

    template_file = Path(template_path)

    if not template_file.exists():
        raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")

    if not context_params and template_path.endswith(".sql.j2"):
        raise ValueError(f"Jinja 템플릿 파일({template_path})에는 --params가 필요합니다")

    logger.info(f"[Template] 렌더링 중: {template_file.name}")

    template_content = template_file.read_text()
    rendered_sql = render_template_from_string(template_content, context_params)

    return rendered_sql


def format_predictions(
    predictions_result: Union[pd.DataFrame, List, np.ndarray],
    original_df: pd.DataFrame,
    data_interface: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    예측 결과를 DataFrame으로 포맷팅합니다.

    data_interface에 정의된 entity_columns, timestamp_column, treatment_column을
    기반으로 예측 결과와 메타데이터를 결합합니다.
    feature_columns는 제외됩니다.

    Calibrated probability predictions를 포함하여 다양한 예측 형태를 지원합니다.

    Args:
        predictions_result: 모델 예측 결과 (DataFrame, list, array 등)
        original_df: 원본 입력 DataFrame (메타데이터 참조용)
        data_interface: 데이터 인터페이스 정의 (entity, timestamp, treatment, feature 정보)
        task_type: 태스크 타입 (classification, regression 등)

    Returns:
        포맷팅된 예측 결과 DataFrame (prediction/probabilities + entity/timestamp/treatment)
    """
    # data_interface에서 task_type 추출 (없으면 인자로 받은 값 사용)
    effective_task_type = task_type
    if data_interface and "task_type" in data_interface:
        effective_task_type = data_interface.get("task_type")

    # 예측 결과를 DataFrame으로 변환
    if isinstance(predictions_result, pd.DataFrame):
        pred_df = predictions_result.copy()
    elif isinstance(predictions_result, (list, tuple)) or hasattr(predictions_result, "tolist"):
        # 1D array/list - could be class predictions or binary probabilities
        predictions_array = np.array(predictions_result)
        if predictions_array.ndim == 1:
            # Causal 모델: CATE 컬럼명 사용
            col_name = "cate" if effective_task_type == "causal" else "prediction"
            pred_df = pd.DataFrame({col_name: predictions_array}, index=original_df.index)
        else:
            # 2D array - multiclass probabilities 또는 causal treatment effects
            pred_df = _format_multiclass_probabilities(
                predictions_array, original_df, effective_task_type
            )
    elif hasattr(predictions_result, "shape"):
        # NumPy array handling
        if predictions_result.ndim == 1:
            col_name = "cate" if effective_task_type == "causal" else "prediction"
            pred_df = pd.DataFrame({col_name: predictions_result}, index=original_df.index)
        elif predictions_result.ndim == 2:
            # 2D array - multiclass probabilities 또는 causal treatment effects
            pred_df = _format_multiclass_probabilities(
                predictions_result, original_df, effective_task_type
            )
        else:
            # Higher dimensional - flatten to 1D
            col_name = "cate" if effective_task_type == "causal" else "prediction"
            pred_df = pd.DataFrame(
                {col_name: predictions_result.flatten()[: len(original_df)]},
                index=original_df.index,
            )
    else:
        # 기타 경우 (스칼라 값 등)
        col_name = "cate" if effective_task_type == "causal" else "prediction"
        pred_df = pd.DataFrame(
            {col_name: [predictions_result] * len(original_df)}, index=original_df.index
        )

    # 확률 예측 감지 및 로깅
    is_probability_prediction = _detect_probability_predictions(pred_df)
    if is_probability_prediction:
        prob_columns = [col for col in pred_df.columns if "prob_" in col]
        logger.info(
            f"Probability predictions detected with {len(prob_columns)} probability columns: {prob_columns}"
        )

    # data_interface가 제공된 경우 정의된 컬럼만 사용
    if data_interface:
        preserve_columns = []

        # 1. Entity columns (복수 가능)
        entity_columns = data_interface.get("entity_columns", [])
        if isinstance(entity_columns, str):
            entity_columns = [entity_columns]
        for col in entity_columns:
            if col in original_df.columns:
                preserve_columns.append(col)

        # 2. Timestamp column
        timestamp_column = data_interface.get("timestamp_column")
        if timestamp_column and timestamp_column in original_df.columns:
            preserve_columns.append(timestamp_column)

        # 3. Treatment column (Causal task)
        treatment_column = data_interface.get("treatment_column")
        if treatment_column and treatment_column in original_df.columns:
            preserve_columns.append(treatment_column)

        # 4. Feature columns는 명시적으로 제외
        feature_columns = data_interface.get("feature_columns", [])
        if isinstance(feature_columns, str):
            feature_columns = [feature_columns]
        preserve_columns = [col for col in preserve_columns if col not in feature_columns]

    else:
        # data_interface가 없는 경우 fallback (deprecated, 향후 제거 예정)
        logger.warning("data_interface not provided to format_predictions. This is deprecated.")
        preserve_columns = []

    # 보존할 컬럼들을 예측 결과에 추가
    for col in preserve_columns:
        if col in original_df.columns and col not in pred_df.columns:
            pred_df[col] = original_df[col].values

    return pred_df


def load_inference_data(
    data_adapter,
    data_path: Optional[str],
    model,
    run_id: str,
    context_params: Dict[str, Any],
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """
    추론용 데이터를 로드합니다.

    CLI data_path가 있으면 우선 사용하고, 없으면 모델에 저장된 SQL을 사용합니다.

    Args:
        data_adapter: 데이터 어댑터 인스턴스
        data_path: 데이터 경로 (선택)
        model: MLflow 모델 객체
        run_id: 모델 run ID
        context_params: Jinja 템플릿 파라미터
        console: 콘솔 매니저 (선택)

    Returns:
        로드된 DataFrame

    Raises:
        ValueError: CSV 모델인데 data_path가 없거나, 보안 위반 시
    """
    if console is None:
        console = Console()

    if data_path:
        # CLI에서 지정한 data_path 사용
        final_data_source = data_path

        # Jinja 템플릿 처리
        if data_path.endswith(".sql.j2") or (data_path.endswith(".sql") and context_params):
            final_data_source = process_template_file(data_path, context_params, console)

        df = data_adapter.read(final_data_source)

    else:
        # 먼저 MLflow artifact에서 SQL 읽기 시도 (더 신뢰성 있음)
        loader_sql_template = None
        try:
            import mlflow

            sql_artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="training_artifacts/source_query.sql"
            )
            with open(sql_artifact_path, "r", encoding="utf-8") as f:
                loader_sql_template = f.read()
            logger.debug(f"[DATA] MLflow artifact에서 SQL 로드 완료: {len(loader_sql_template)} bytes")
        except Exception as e:
            logger.debug(f"[DATA] MLflow artifact SQL 로드 실패, fallback 사용: {e}")
            # fallback: 모델에 저장된 loader_sql_snapshot 사용
            wrapped_model = model.unwrap_python_model()
            loader_sql_template = wrapped_model.loader_sql_snapshot

        # CSV 기반 모델 검증
        if not loader_sql_template or not loader_sql_template.strip():
            if context_params:
                raise ValueError(
                    "보안 위반: CSV로 학습된 모델은 동적 파라미터를 지원하지 않습니다. "
                    "Jinja template (.sql.j2)로 학습하세요."
                )
            else:
                raise ValueError(f"CSV로 학습된 모델(run_id: {run_id})은 --data가 필요합니다.")

        # Jinja 템플릿 처리
        if context_params:
            if is_jinja_template(loader_sql_template):
                final_data_source = render_template_from_string(loader_sql_template, context_params)
            else:
                raise ValueError(
                    "보안 위반: 정적 SQL로 학습된 모델은 동적 파라미터를 지원하지 않습니다. "
                    "Jinja template (.sql.j2)로 학습하세요."
                )
        else:
            final_data_source = loader_sql_template

        df = data_adapter.read(final_data_source)

    return df
