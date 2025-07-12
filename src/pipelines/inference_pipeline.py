import pandas as pd
from typing import Dict, Any, Optional

from src.settings.settings import Settings
from src.utils.logger import logger
from src.utils.sql_utils import render_sql
from src.utils.bigquery_utils import execute_query, upload_df_to_bigquery
from src.utils import mlflow_utils


def run_batch_inference(
    settings: Settings,
    model_name: str,
    model_stage: str,
    input_sql_path: str,
    output_table_id: str,
    sql_params: Optional[Dict[str, Any]] = None
):
    """
    배치 추론을 위한 전체 파이프라인을 실행합니다.
    `mlflow_utils`를 통해 지정된 스테이지의 모델을 로드하여 추론을 수행합니다.

    Args:
        settings: 프로젝트 설정 객체.
        model_name: 사용할 모델의 이름.
        model_stage: 사용할 모델의 스테이지 (e.g., "Production", "Staging").
        input_sql_path: 추론할 데이터를 로드하기 위한 SQL 파일 경로.
        output_table_id: 결과를 저장할 BigQuery 테이블 ID.
        sql_params: SQL 쿼리 템플릿에 전달할 파라미터.
    """
    logger.info(f"배치 추론 파이프라인을 시작합니다: (모델: {model_name}, 스테이지: {model_stage})")

    try:
        # 1. MLflow에서 통합 모델 로드
        model = mlflow_utils.load_pyfunc_model(
            model_name=model_name,
            stage=model_stage,
            settings=settings
        )

        # 2. 데이터 로딩
        logger.info(f"'{input_sql_path}'에서 데이터를 로딩합니다.")
        rendered_sql = render_sql(file_path=input_sql_path, params=sql_params)
        input_df = execute_query(rendered_sql, settings=settings)
        logger.info(f"총 {len(input_df)}개의 데이터를 로드했습니다.")

        if input_df.empty:
            logger.warning("입력 데이터가 비어있어 추론을 중단합니다.")
            return

        # 3. 추론 실행
        logger.info("통합 모델을 사용하여 배치 추론을 시작합니다.")
        predict_params = {"run_mode": "batch"}
        predictions = model.predict(input_df, params=predict_params)
        
        # 원본 데이터에 예측 결과 컬럼 추가
        results_df = input_df.copy()
        results_df['uplift_score'] = predictions[predictions.columns[0]]
        logger.info("추론을 완료했습니다.")

        # 4. 결과 저장
        gcp_project_id = settings.environment.gcp_project_id
        main_loader_name = list(settings.loader.keys())[0]
        dataset_id = settings.loader[main_loader_name].output.dataset_id
        
        full_table_id = f"{gcp_project_id}.{dataset_id}.{output_table_id}"

        logger.info(f"추론 결과를 BigQuery 테이블 '{full_table_id}'에 업로드합니다.")
        upload_df_to_bigquery(df=results_df, table_id=full_table_id, settings=settings)
        logger.info("결과 저장을 완료했습니다.")

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise

