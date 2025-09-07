"""
Inference Pipeline - Batch and Realtime Inference
"""
import pandas as pd
import mlflow
from datetime import datetime
from typing import Dict, Any, Optional

from src.factory import Factory
from src.utils.integrations.mlflow_integration import start_run
from src.utils.system.logger import logger
from src.settings import Settings
from src.utils.system.reproducibility import set_global_seeds


def _is_jinja_template(sql_text: str) -> bool:
    """SQL 텍스트가 Jinja 템플릿인지 확인합니다."""
    import re
    jinja_patterns = [
        r'\{\{.*?\}\}',  # {{ variable }}
        r'\{%.*?%\}',    # {% for ... %}
    ]
    return any(re.search(pattern, sql_text) for pattern in jinja_patterns)


def run_inference_pipeline(settings: Settings, run_id: str, data_path: str = None, context_params: dict = None):
    """
    지정된 Run ID의 모델을 사용하여 배치 추론을 실행합니다.
    Phase 5.3: --data-path로 직접 데이터 경로를 지정하는 방식으로 단순화
    """
    context_params = context_params or {}

    # 재현성을 위한 전역 시드 설정 (레시피 시드가 없으면 42)
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)

    # 1. MLflow 실행 컨텍스트 시작
    with start_run(settings, run_name=f"batch_inference_{run_id}") as run:
        # 2. 모델 로드
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"MLflow 모델 로딩 시작: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 3. 데이터 로딩 (CLI data_path 우선, Jinja 렌더링 지원)
        factory = Factory(settings)
        data_adapter = factory.create_data_adapter()
        
        if data_path:
            # Phase 5.3: CLI에서 지정한 data_path 사용
            final_data_source = data_path
            
            # Jinja 템플릿 렌더링 처리 (.sql.j2 또는 params가 있는 .sql)
            if data_path.endswith('.sql.j2') or (data_path.endswith('.sql') and context_params):
                from src.utils.system.templating_utils import render_template_from_string
                from pathlib import Path
                
                template_path = Path(data_path)
                if template_path.exists():
                    template_content = template_path.read_text()
                    if context_params:
                        try:
                            final_data_source = render_template_from_string(template_content, context_params)
                            logger.info(f"✅ CLI data_path Jinja 렌더링 성공: {data_path}")
                        except ValueError as e:
                            logger.error(f"🚨 CLI data_path Jinja 렌더링 실패: {e}")
                            raise ValueError(f"템플릿 렌더링 실패: {e}")
                    else:
                        # 파라미터 없이 .sql.j2 파일 → 에러
                        raise ValueError(f"Jinja 템플릿 파일({data_path})에는 --params가 필요합니다")
                else:
                    raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {data_path}")
            
            df = data_adapter.read(final_data_source)
            logger.info(f"✅ CLI data_path에서 데이터 로딩 완료: {data_path}")
            
        else:
            # Fallback: 기존 방식 (저장된 loader_sql_snapshot 사용)
            wrapped_model = model.unwrap_python_model()
            loader_sql_template = wrapped_model.loader_sql_snapshot
            
            # 기존 Jinja 렌더링 로직 (보안 강화)
            if _is_jinja_template(loader_sql_template) and context_params:
                # Jinja template + context_params → 보안 강화 동적 렌더링
                from src.utils.system.templating_utils import render_template_from_string
                try:
                    rendered_sql = render_template_from_string(loader_sql_template, context_params)
                    logger.info("✅ 동적 SQL 렌더링 성공 (보안 검증 완료)")
                    final_data_source = rendered_sql
                except ValueError as e:
                    # 보안 위반 또는 잘못된 파라미터 → 명확한 에러
                    raise ValueError(f"동적 SQL 렌더링 실패: {e}")
                    
            elif context_params:
                # 정적 SQL + context_params → 보안 에러 (명확한 안내)
                raise ValueError(
                    "🚨 보안 위반: 이 모델은 정적 SQL로 학습되어 동적 시점 변경을 지원하지 않습니다.\n"
                    "동적 Batch Inference를 원한다면 Jinja template (.sql.j2)로 학습하세요.\n"
                    f"현재 SQL: {loader_sql_template[:100]}..."
                )
            else:
                # 정적 SQL + context_params 없음 → 정상 처리
                final_data_source = loader_sql_template
            
            df = data_adapter.read(final_data_source)
            logger.info(f"✅ 기존 방식으로 데이터 로딩 완료")
        
        logger.info(f"데이터 로딩 완료: {df.shape}")
        
        # 4. 예측 실행 (PyfuncWrapper가 내부적으로 스키마 검증을 수행)
        predictions_df = model.predict(df)
        
        # 5. 핵심 메타데이터 추가 (추적성 보장)
        predictions_df['model_run_id'] = run_id  # 사용된 모델의 MLflow Run ID
        predictions_df['inference_run_id'] = run.info.run_id  # 현재 배치 추론 실행 ID
        predictions_df['inference_timestamp'] = datetime.now()  # 예측 수행 시각
        
        # 6. 결과 저장 (Output 설정 기반)
        output_cfg = getattr(settings.config, 'output', None)
        if output_cfg and getattr(output_cfg.inference, 'enabled', True):
            try:
                target = output_cfg.inference
                if target.adapter_type == "storage":
                    storage_adapter = factory.create_data_adapter("storage")
                    base_path = target.config.get('base_path', './artifacts/predictions')
                    target_path = f"{base_path}/preds_{run.info.run_id}.parquet"
                    storage_adapter.write(predictions_df, target_path)
                    # 로컬 경로만 MLflow artifact로 로깅
                    if not target_path.startswith("s3://") and not target_path.startswith("gs://"):
                        mlflow.log_artifact(target_path.replace("file://", ""))
                elif target.adapter_type == "sql":
                    sql_adapter = factory.create_data_adapter("sql")
                    table = target.config.get('table')
                    if not table:
                        raise ValueError("output.inference.config.table이 필요합니다.")
                    sql_adapter.write(predictions_df, table, if_exists='append', index=False)
                elif target.adapter_type == "bigquery":
                    bq_adapter = factory.create_data_adapter("bigquery")
                    project_id = target.config.get('project_id')
                    dataset = target.config.get('dataset_id')
                    table = target.config.get('table')
                    location = target.config.get('location')
                    if not (project_id and dataset and table):
                        raise ValueError("BigQuery 출력에는 project_id, dataset_id, table이 필요합니다.")
                    bq_adapter.write(
                        predictions_df,
                        f"{dataset}.{table}",
                        options={"project_id": project_id, "location": location, "if_exists": "append"}
                    )
                else:
                    logger.warning(f"알 수 없는 output 어댑터 타입: {target.adapter_type}. 저장을 스킵합니다.")
            except Exception as e:
                logger.error(f"출력 저장 중 오류 발생: {e}", exc_info=True)
        else:
            logger.info("Output 설정이 비활성화되어 저장을 스킵합니다.")
        
        mlflow.log_metric("inference_row_count", len(predictions_df))


def _is_jinja_template(sql: str) -> bool:
    """
    🆕 Phase 3: SQL 문자열이 Jinja2 템플릿인지 감지
    
    Args:
        sql: 검사할 SQL 문자열
        
    Returns:
        Jinja2 템플릿 패턴이 포함되어 있으면 True, 아니면 False
    """
    jinja_patterns = ['{{', '}}', '{%', '%}']
    return any(pattern in sql for pattern in jinja_patterns)