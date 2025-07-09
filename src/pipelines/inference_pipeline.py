import pandas as pd
import mlflow
from typing import Dict, Any

# config.settings에서 통합 설정 객체를 import
from config.settings import Settings
from src.utils.logger import logger
from src.utils.sql_utils import render_sql
from src.utils.bigquery_utils import execute_query, upload_df_to_bigquery


def run_batch_inference(settings: Settings, model_uri: str, input_sql_path: str, output_table_id: str, sql_params: Dict[str, Any] = None):
    """
    배치 추론을 위한 전체 파이프라인을 실행합니다.
    MLflow Model Registry에서 통합 모델을 로드하여 추론을 수행합니다.

    Args:
        settings (Settings): 프로젝트 설정 객체.
        model_uri (str): 사용할 모델의 MLflow Model URI (e.g., "models:/<name>/<version>").
        input_sql_path (str): 추론할 데이터를 로드하기 위한 SQL 파일 경로.
        output_table_id (str): 결과를 저장할 BigQuery 테이블 ID.
        sql_params (Dict[str, Any], optional): SQL 쿼리 템플릿에 전달할 파라미터.
    """
    logger.info("배치 추론 파이프라인을 시작합니다.")
    logger.info(f"사용할 모델: {model_uri}")

    try:
        # 1. MLflow에서 통합 모델 로드
        logger.info("MLflow Model Registry에서 모델을 로드합니다...")
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info("모델 로드를 완료했습니다.")

        # 2. 데이터 로딩
        logger.info(f"'{input_sql_path}'에서 데이터를 로딩합니다.")
        rendered_sql = render_sql(file_path=input_sql_path, params=sql_params)
        input_df = execute_query(rendered_sql, settings=settings)
        logger.info(f"총 {len(input_df)}개의 데이터를 로드했습니다.")

        # 3. 추론 실행 (Pyfunc 모델은 내부적으로 transform과 predict를 모두 수행)
        logger.info("통합 모델을 사용하여 추론을 시작합니다.")
        predictions = model.predict(input_df)
        results_df = input_df.copy()
        results_df['uplift_score'] = predictions['uplift_score']
        logger.info("추론을 완료했습니다.")

        # 4. 결과 저장
        gcp_project_id = settings.environment.gcp_project_id
        # 첫 번째 로더의 dataset_id를 사용
        dataset_id = next(iter(settings.pipeline.loader.values())).output.dataset_id
        full_table_id = f"{gcp_project_id}.{dataset_id}.{output_table_id}"

        logger.info(f"추론 결과를 BigQuery 테이블 '{full_table_id}'에 ������합니다.")
        upload_df_to_bigquery(df=results_df, table_id=full_table_id, settings=settings)
        logger.info("결과 저장을 완료했습니다.")

        logger.info("배치 추론 파이프라인이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"배치 추론 파이프라인 중 오류 발생: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    from config.settings import settings
    # 스크립트를 직접 실행할 때 사용할 예시
    # 실제 운영 환경에서는 Airflow, Cloud Run Jobs 등에서 이 함수를 호출하게 됩니다.
    
    MODEL_URI = f"models:/{settings.model.name}/Production" 
    INPUT_SQL = "src/sql/abt_logs.sql"
    OUTPUT_TABLE = "uplift_prediction_results"
    SQL_PARAMS = {"date": "2025-07-06"}

    if "Production" not in MODEL_URI and "Staging" not in MODEL_URI and not MODEL_URI.split('/')[-1].isdigit():
        logger.warning("유효한 MLflow Model URI를 설정하고 실행해주세요. (e.g., 'models:/<name>/<version_or_stage>')")
    else:
        run_batch_inference(
            settings=settings,
            model_uri=MODEL_URI,
            input_sql_path=INPUT_SQL,
            output_table_id=OUTPUT_TABLE,
            sql_params=SQL_PARAMS
        )
