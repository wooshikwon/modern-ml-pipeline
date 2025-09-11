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
from src.utils.system.console_manager import UnifiedConsole, RichConsoleManager
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
    --data-path로 직접 데이터 경로를 지정하는 방식
    """
    context_params = context_params or {}
    console = UnifiedConsole(settings)

    # 재현성을 위한 전역 시드 설정 (레시피 시드가 없으면 42)
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)
    
    # Pipeline context start  
    rich_console = RichConsoleManager()
    with rich_console.pipeline_context("Batch Inference Pipeline", f"Model: {run_id}"):
        
        # 1. MLflow 실행 컨텍스트 시작
        with start_run(settings, run_name=f"batch_inference_{run_id}") as run:
            # 2. 모델 로드
            rich_console.log_phase("Model Loading", "🤖")
            model_uri = f"runs:/{run_id}/model"
            console.info(f"MLflow 모델 로딩 시작: {model_uri}", 
                        rich_message=f"Loading model from [cyan]{model_uri}[/cyan]")
            model = mlflow.pyfunc.load_model(model_uri)
            console.info("모델 로딩 완료", rich_message="✅ Model loaded successfully")
            
            # 3. 데이터 로딩 (CLI data_path 우선, Jinja 렌더링 지원)
            rich_console.log_phase("Data Loading", "📊")
            factory = Factory(settings)
            data_adapter = factory.create_data_adapter()
            
            if data_path:
                # CLI에서 지정한 data_path 사용
                console.info(f"CLI data_path 사용: {data_path}",
                           rich_message=f"Using CLI data path: [cyan]{data_path}[/cyan]")
                final_data_source = data_path
                
                # Jinja 템플릿 렌더링 처리 (.sql.j2 또는 params가 있는 .sql)
                if data_path.endswith('.sql.j2') or (data_path.endswith('.sql') and context_params):
                    from src.utils.system.templating_utils import render_template_from_string
                    from pathlib import Path
                    
                    rich_console.log_processing_step("Template rendering", f"Processing {Path(data_path).name}")
                    template_path = Path(data_path)
                    if template_path.exists():
                        template_content = template_path.read_text()
                        if context_params:
                            try:
                                final_data_source = render_template_from_string(template_content, context_params)
                                console.info(f"CLI data_path Jinja 렌더링 성공: {data_path}",
                                           rich_message="✅ Template rendering successful")
                            except ValueError as e:
                                console.error(f"CLI data_path Jinja 렌더링 실패: {e}",
                                           rich_message=f"❌ Template rendering failed: {e}")
                                raise ValueError(f"템플릿 렌더링 실패: {e}")
                        else:
                            # 파라미터 없이 .sql.j2 파일 → 에러
                            error_msg = f"Jinja 템플릿 파일({data_path})에는 --params가 필요합니다"
                            console.error(error_msg, suggestion="Use --params flag to provide template parameters")
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"템플릿 파일을 찾을 수 없습니다: {data_path}"
                        console.error(error_msg, context={"file_path": data_path})
                        raise FileNotFoundError(error_msg)
                
                df = data_adapter.read(final_data_source)
                console.data_operation("Data loaded from CLI path", df.shape)
                
            else:
                # Fallback: 기존 방식 (저장된 loader_sql_snapshot 사용)
                console.info("CLI data_path 없음, 저장된 SQL 사용", 
                           rich_message="Using stored SQL from model")
                wrapped_model = model.unwrap_python_model()
                loader_sql_template = wrapped_model.loader_sql_snapshot
                
                # context_params가 있는 경우 보안 검사 우선
                if context_params:
                    # CSV 기반 학습 모델 + context_params → 보안 에러 (우선)
                    if not loader_sql_template.strip():
                        error_msg = ("보안 위반: 이 모델은 정적 CSV로 학습되어 동적 시점 변경을 지원하지 않습니다.\n"
                                   "동적 Batch Inference를 원한다면 Jinja template (.sql.j2)로 학습하세요.")
                        console.error(error_msg,
                                    context={"model_run_id": run_id, "context_params": list(context_params.keys())},
                                    suggestion="Train model with Jinja template (.sql.j2) for dynamic inference")
                        raise ValueError(f"🚨 {error_msg}")
                    
                    # Jinja template + context_params → 보안 강화 동적 렌더링
                    elif _is_jinja_template(loader_sql_template):
                        # 기존 Jinja 렌더링 로직 (보안 강화)
                        from src.utils.system.templating_utils import render_template_from_string
                        rich_console.log_processing_step("Dynamic SQL rendering", "Security-validated template processing")
                        try:
                            rendered_sql = render_template_from_string(loader_sql_template, context_params)
                            console.info("동적 SQL 렌더링 성공 (보안 검증 완료)",
                                       rich_message="✅ Dynamic SQL rendering successful (security validated)")
                            final_data_source = rendered_sql
                        except ValueError as e:
                            # 보안 위반 또는 잘못된 파라미터 → 명확한 에러
                            console.error(f"동적 SQL 렌더링 실패: {e}",
                                        suggestion="Check template parameters and security constraints")
                            raise ValueError(f"동적 SQL 렌더링 실패: {e}")
                    else:
                        # 정적 SQL + context_params → 보안 에러 (명확한 안내)
                        error_msg = ("보안 위반: 이 모델은 정적 SQL로 학습되어 동적 시점 변경을 지원하지 않습니다.\n"
                                   "동적 Batch Inference를 원한다면 Jinja template (.sql.j2)로 학습하세요.")
                        console.error(error_msg, 
                                    context={"sql_preview": loader_sql_template[:100] + "..."},
                                    suggestion="Train model with Jinja template (.sql.j2) for dynamic inference")
                        raise ValueError(f"🚨 {error_msg}")
                else:
                    # context_params 없음 - CSV 기반 학습 모델의 경우 loader_sql_snapshot이 빈 문자열
                    if not loader_sql_template.strip():
                        error_msg = "이 모델은 CSV로 학습되어 data_path가 필요합니다. --data-path를 제공하세요."
                        console.error(error_msg, 
                                    suggestion="Provide --data-path for CSV-trained models",
                                    context={"model_run_id": run_id, "loader_sql_snapshot": "empty"})
                        raise ValueError(f"🚨 {error_msg}")
                    else:
                        # 정적 SQL + context_params 없음 → 정상 처리
                        final_data_source = loader_sql_template
                
                
                df = data_adapter.read(final_data_source)
                console.data_operation("Data loaded from stored SQL", df.shape)
            
            # 4. 예측 실행 (PyfuncWrapper가 내부적으로 스키마 검증을 수행)
            rich_console.log_phase("Model Inference", "🔮")
            with rich_console.progress_tracker("inference", 100, "Running model prediction") as update:
                # MLflow predict 호출 후 DataFrame으로 변환
                predictions_result = model.predict(df)
                
                # 결과가 list/array인 경우 DataFrame으로 변환
                if isinstance(predictions_result, (list, tuple)) or hasattr(predictions_result, 'tolist'):
                    predictions_df = pd.DataFrame({'prediction': predictions_result}, index=df.index)
                elif isinstance(predictions_result, pd.DataFrame):
                    predictions_df = predictions_result
                else:
                    # numpy array 등의 경우
                    predictions_df = pd.DataFrame({'prediction': predictions_result.flatten()}, index=df.index)
                    
                update(100)
        
        # 5. 핵심 메타데이터 추가 (추적성 보장)
        predictions_df['model_run_id'] = run_id  # 사용된 모델의 MLflow Run ID
        predictions_df['inference_run_id'] = run.info.run_id  # 현재 배치 추론 실행 ID
        predictions_df['inference_timestamp'] = datetime.now()  # 예측 수행 시각
        
        # 6. 결과 저장 (Output 설정 기반)
        rich_console.log_phase("Output Saving", "💾")
        output_cfg = getattr(settings.config, 'output', None)
        if output_cfg and getattr(output_cfg.inference, 'enabled', True):
            try:
                target = output_cfg.inference
                if target.adapter_type == "storage":
                    console.info("Storage 어댑터 사용하여 결과 저장",
                               rich_message="📁 Saving predictions to storage")
                    storage_adapter = factory.create_data_adapter("storage")
                    base_path = target.config.get('base_path', './artifacts/predictions')
                    target_path = f"{base_path}/preds_{run.info.run_id}.parquet"
                    storage_adapter.write(predictions_df, target_path)
                    # 로컬 경로만 MLflow artifact로 로깅
                    if not target_path.startswith("s3://") and not target_path.startswith("gs://"):
                        mlflow.log_artifact(target_path.replace("file://", ""))
                    console.info(f"예측 결과 저장 완료: {target_path}",
                               rich_message=f"✅ Predictions saved to [cyan]{target_path}[/cyan]")
                elif target.adapter_type == "sql":
                    sql_adapter = factory.create_data_adapter("sql")
                    table = target.config.get('table')
                    if not table:
                        raise ValueError("output.inference.config.table이 필요합니다.")
                    console.info(f"SQL 데이터베이스에 결과 저장: {table}",
                               rich_message=f"🗄️  Saving to database table [cyan]{table}[/cyan]")
                    sql_adapter.write(predictions_df, table, if_exists='append', index=False)
                    console.info("SQL 저장 완료", rich_message="✅ SQL save completed")
                elif target.adapter_type == "bigquery":
                    bq_adapter = factory.create_data_adapter("bigquery")
                    project_id = target.config.get('project_id')
                    dataset = target.config.get('dataset_id')
                    table = target.config.get('table')
                    location = target.config.get('location')
                    if not (project_id and dataset and table):
                        raise ValueError("BigQuery 출력에는 project_id, dataset_id, table이 필요합니다.")
                    console.info(f"BigQuery에 결과 저장: {dataset}.{table}",
                               rich_message=f"☁️  Saving to BigQuery [cyan]{dataset}.{table}[/cyan]")
                    bq_adapter.write(
                        predictions_df,
                        f"{dataset}.{table}",
                        options={"project_id": project_id, "location": location, "if_exists": "append"}
                    )
                    console.info("BigQuery 저장 완료", rich_message="✅ BigQuery save completed")
                else:
                    console.warning(f"알 수 없는 output 어댑터 타입: {target.adapter_type}. 저장을 스킵합니다.",
                                  rich_message=f"⚠️  Unknown adapter type: [yellow]{target.adapter_type}[/yellow], skipping save",
                                  context={"adapter_type": target.adapter_type})
            except Exception as e:
                console.error(f"출력 저장 중 오류 발생: {e}",
                            rich_message=f"❌ Output save failed: {e}",
                            context={"error_type": type(e).__name__, "run_id": run.info.run_id},
                            suggestion="Check output configuration and adapter connectivity")
        else:
            console.info("Output 설정이 비활성화되어 저장을 스킵합니다.",
                       rich_message="ℹ️  Output disabled, skipping save")
        
        mlflow.log_metric("inference_row_count", len(predictions_df))


def _is_jinja_template(sql: str) -> bool:
    """
    SQL 문자열이 Jinja2 템플릿인지 감지
    
    Args:
        sql: 검사할 SQL 문자열
        
    Returns:
        Jinja2 템플릿 패턴이 포함되어 있으면 True, 아니면 False
    """
    jinja_patterns = ['{{', '}}', '{%', '%}']
    return any(pattern in sql for pattern in jinja_patterns)