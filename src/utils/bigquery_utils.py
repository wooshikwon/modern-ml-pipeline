# uplift-virtual-coupon/utils/bigquery_utils.py

import pandas as pd
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account
from datetime import date
from google.cloud.bigquery import SchemaField
from pathlib import Path
from typing import Optional

from src.settings.settings import Settings
from src.utils.logger import logger


def get_bigquery_client(settings: Settings) -> bigquery.Client:
    """BigQuery 클라이언트를 초기화하고 반환합니다."""
    try:
        project_id = settings.environment.gcp_project_id
        credential_path = settings.environment.gcp_credential_path

        if not project_id:
            raise ValueError("GCP project ID가 설정되지 않았습니다.")

        if not credential_path:
            logger.warning("GCP credential path가 설정되지 않았습니다. 기본 인증을 사용합니다.")
            return bigquery.Client(project=project_id)

        credential_file = Path(credential_path)
        if not credential_file.exists():
            raise FileNotFoundError(f"인증 파일을 찾을 수 없습니다: {credential_path}")

        credentials = service_account.Credentials.from_service_account_file(credential_path)
        client = bigquery.Client(credentials=credentials, project=project_id)

        logger.info(f"BigQuery 클라이언트 초기화 완료 (Project: {project_id})")
        return client

    except Exception as e:
        logger.error(f"BigQuery 클라이언트 초기화 실패: {e}")
        raise


def execute_query(query: str, settings: Settings, dry_run: bool = False) -> pd.DataFrame:
    """
    BigQuery에서 쿼리를 실행하고 결과를 DataFrame으로 반환합니다.
    """
    try:
        logger.info("BigQuery 쿼리 실행 시작...")
        if dry_run:
            logger.info("DRY RUN 모드 - 쿼리 유효성만 검사합니다.")

        client = get_bigquery_client(settings)
        job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=True)
        query_job = client.query(query, job_config=job_config)

        if dry_run:
            logger.info(f"쿼리 유효성 검사 완료. 예상 처리 바이트: {query_job.total_bytes_processed:,}")
            return pd.DataFrame()

        df = query_job.to_dataframe()
        logger.info(f"쿼리 실행 완료: {len(df):,}행, {len(df.columns)}열")
        return df

    except exceptions.GoogleAPICallError as e:
        logger.error(f"BigQuery API 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"예상치 못한 쿼리 실행 오류 발생: {e}")
        raise


def _infer_bigquery_schema(df: pd.DataFrame) -> list[SchemaField]:
    """DataFrame의 데이터 타입을 기반으로 BigQuery 스키마를 추론합니다."""
    schema = []
    for col_name, dtype in df.dtypes.items():
        sample_data = df[col_name].dropna().head(100)
        if len(sample_data) == 0:
            field_type = "STRING"
        elif sample_data.apply(lambda x: isinstance(x, date)).all():
            field_type = "DATE"
        elif pd.api.types.is_integer_dtype(dtype):
            field_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            field_type = "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            field_type = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            field_type = "TIMESTAMP"
        else:
            field_type = "STRING"
        schema.append(SchemaField(name=col_name, field_type=field_type, mode="NULLABLE"))
    return schema


def _create_dataset_if_not_exists(client: bigquery.Client, dataset_id: str, settings: Settings) -> None:
    """데이터셋이 존재하지 ��으면 생성합니다."""
    project_id = settings.environment.gcp_project_id
    full_dataset_id = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(full_dataset_id)
        logger.info(f"데이터셋 {full_dataset_id} 이미 존재합니다.")
    except exceptions.NotFound:
        logger.info(f"데이터셋 {full_dataset_id} 생성 중...")
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = "US"  # Or get from settings
        client.create_dataset(dataset, timeout=30)
        logger.info(f"데이터셋 {full_dataset_id} 생성 완료")


def upload_df_to_bigquery(
    df: pd.DataFrame,
    table_id: str,
    settings: Settings,
    write_disposition: str = "WRITE_TRUNCATE"
) -> None:
    """DataFrame을 BigQuery 테이블에 업로드합니다."""
    if df.empty:
        logger.warning("업로드할 데이터가 비어있습니다.")
        return

    try:
        client = get_bigquery_client(settings)
        
        # 스키마 추론
        schema = _infer_bigquery_schema(df)

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=write_disposition,
            source_format=bigquery.SourceFormat.PARQUET,
        )

        load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        load_job.result()
        
        final_table = client.get_table(table_id)
        logger.info(f"업로드 완료: {table_id} ({final_table.num_rows:,}행)")

    except Exception as e:
        logger.error(f"BigQuery 업로드 실패: {e}")
        raise

def delete_table(table_id: str, settings: Settings) -> None:
    """BigQuery 테이블을 삭제합니다."""
    try:
        client = get_bigquery_client(settings)
        client.delete_table(table_id, not_found_ok=True)
        logger.info(f"테이블 삭제 완료: {table_id}")
    except Exception as e:
        logger.error(f"테이블 삭제 실패: {table_id}, 오류: {e}")
        raise

def set_table_expiration(table_id: str, hours: int, settings: Settings) -> None:
    """BigQuery 테이블에 만료 시간을 설정합니다."""
    try:
        client = get_bigquery_client(settings)
        table = client.get_table(table_id)
        expiration_time = pd.Timestamp.now() + pd.Timedelta(hours=hours)
        table.expires = expiration_time
        client.update_table(table, ["expires"])
        logger.info(f"테이블 만료 시간 설정 완료: {table_id}, 만료 시각: {expiration_time}")
    except Exception as e:
        logger.error(f"테이블 만료 시간 설정 실패: {table_id}, 오류: {e}")
        raise
