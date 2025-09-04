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


def run_batch_inference(settings: Settings, run_id: str, context_params: dict = None):
    """
    지정된 Run ID의 모델을 사용하여 배치 추론을 실행합니다.
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
        
        # 3. 데이터 로딩 - 🆕 Phase 3: 보안 강화 Dynamic SQL 처리
        # Wrapper에 내장된 loader_sql_snapshot을 사용
        wrapped_model = model.unwrap_python_model()
        loader_sql_template = wrapped_model.loader_sql_snapshot
        
        # Factory를 통해 현재 환경에 맞는 데이터 어댑터 생성
        factory = Factory(settings)
        
        # 🆕 Phase 3: Template SQL 보안 렌더링
        if _is_jinja_template(loader_sql_template) and context_params:
            # Jinja template + context_params → 보안 강화 동적 렌더링
            from src.utils.system.templating_utils import render_template_from_string
            try:
                rendered_sql = render_template_from_string(loader_sql_template, context_params)
                logger.info("✅ 동적 SQL 렌더링 성공 (보안 검증 완료)")
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
            rendered_sql = loader_sql_template
        
        data_adapter = factory.create_data_adapter(factory.model_config.loader.adapter)
        df = data_adapter.read(rendered_sql)
        
        # 4. 예측 실행 (PyfuncWrapper가 내부적으로 스키마 검증을 수행)
        predictions_df = model.predict(df)
        
        # 5. 핵심 메타데이터 추가 (추적성 보장)
        predictions_df['model_run_id'] = run_id  # 사용된 모델의 MLflow Run ID
        predictions_df['inference_run_id'] = run.info.run_id  # 현재 배치 추론 실행 ID
        predictions_df['inference_timestamp'] = datetime.now()  # 예측 수행 시각
        
        # 6. 결과 저장
        storage_adapter = factory.create_data_adapter("storage")
        target_path = f"{settings.artifact_stores['prediction_results'].base_uri}/preds_{run.info.run_id}.parquet"
        storage_adapter.write(predictions_df, target_path)

        # 7. PostgreSQL 저장 (설정이 활성화된 경우)
        prediction_config = settings.artifact_stores['prediction_results']
        
        if hasattr(prediction_config, 'postgres_storage') and prediction_config.postgres_storage:
            postgres_config = prediction_config.postgres_storage
            
            if postgres_config.enabled:
                try:
                    # SQL 어댑터로 PostgreSQL에 저장
                    sql_adapter = factory.create_data_adapter("sql") 
                    table_name = postgres_config.table_name
                    
                    # DataFrame을 PostgreSQL 테이블에 저장 (append 모드)
                    sql_adapter.write(predictions_df, table_name, if_exists='append', index=False)
                    logger.info(f"배치 추론 결과를 PostgreSQL 테이블 '{table_name}'에 저장 완료 ({len(predictions_df)}행)")
                    
                    mlflow.log_metric("postgres_rows_saved", len(predictions_df))
                except Exception as e:
                    logger.error(f"PostgreSQL 저장 실패: {e}")
                    # PostgreSQL 저장 실패해도 파일 저장은 성공했으므로 계속 진행

        mlflow.log_artifact(target_path.replace("file://", ""))
        mlflow.log_metric("inference_row_count", len(predictions_df))


def _save_dataset(
    factory: Factory,
    df: pd.DataFrame,
    store_name: str,
    settings: Settings,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Factory를 통해 적절한 데이터 어댑터를 생성하고, DataFrame을 저장합니다.
    (기존 artifact_utils.save_dataset 로직을 직접 구현)
    """
    if df.empty:
        logger.warning(f"DataFrame이 비어있어, '{store_name}' 아티팩트 저장을 건너뜁니다.")
        return

    try:
        # 올바른 접근 방식 적용: dict['키'] -> 결과는 Pydantic 모델
        store_config = settings.artifact_stores[store_name]
    except KeyError:
        logger.error(f"'{store_name}'에 해당하는 아티팩트 스토어 설정을 찾을 수 없습니다.")
        raise

    if not store_config.enabled:
        logger.info(f"'{store_name}' 아티팩트 스토어가 비활성화되어 있어 저장을 건너뜁니다.")
        return

    base_uri = store_config.base_uri
    
    # ✅ Blueprint 원칙 3: URI 기반 동작 및 동적 팩토리 완전 구현
    # Factory가 환경별 분기와 어댑터 선택을 전담
    adapter = factory.create_data_adapter("storage")

    # 저장될 최종 경로(테이블명 또는 파일명) 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # inference에서는 model.name이 없으므로 run_id 기반으로 식별자 생성
    model_identifier = "batch_inference"
    artifact_name = f"{model_identifier}_{timestamp}"
    
    # ✅ Blueprint 원칙 3: Factory가 URI 해석 처리 - 단순한 artifact 이름만 전달
    final_target = f"{base_uri.rstrip('/')}/{artifact_name}"

    logger.info(f"'{store_name}' 아티팩트 저장 시작: {final_target}")
    adapter.write(df, final_target, options)
    logger.info(f"'{store_name}' 아티팩트 저장 완료: {final_target}")


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