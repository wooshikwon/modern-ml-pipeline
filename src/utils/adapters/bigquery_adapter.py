import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from src.utils.system import sql_utils
from src.settings import Settings

class BigQueryAdapter(BaseAdapter):
    """BigQuery와의 데이터 읽기/쓰기를 처리하는 어댑터."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            self.client = self._get_client()
            self._client_available = True
        except Exception as e:
            logger.warning(f"BigQuery 클라이언트 초기화 실패 (로컬 개발 모드): {e}")
            self.client = None
            self._client_available = False

    def _get_client(self) -> bigquery.Client:
        """BigQuery 클라이언트를 초기화하고 반환합니다."""
        try:
            project_id = self.settings.environment.gcp_project_id
            credential_path = self.settings.environment.gcp_credential_path
            
            credentials = None
            if credential_path:
                credential_file = Path(credential_path)
                if credential_file.exists():
                    credentials = service_account.Credentials.from_service_account_file(credential_path)
                else:
                    logger.warning(f"인증 파일을 찾을 수 없습니다: {credential_path}. 기본 인증을 사용합니다.")
            else:
                logger.warning("GCP credential path가 설정되지 않았습니다. 기본 인증을 사용합니다.")

            client = bigquery.Client(credentials=credentials, project=project_id)
            logger.info(f"BigQuery 클라이언트 초기화 완료 (Project: {project_id})")
            return client
        except Exception as e:
            logger.error(f"BigQuery 클라이언트 초기화 실패: {e}")
            raise

    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        if not self._client_available:
            logger.warning("BigQuery 클라이언트가 없어 데이터 읽기를 건너뜁니다. 빈 DataFrame을 반환합니다.")
            return pd.DataFrame()
        
        parsed_uri = urlparse(source)
        sql_file_path = parsed_uri.path.lstrip('/')
        
        logger.info(f"BigQuery에서 데이터 읽기 시작 (SQL 파일: {sql_file_path})")

        all_params = {'gcp_project_id': self.settings.environment.gcp_project_id}
        if params:
            all_params.update(params)
            
        rendered_sql = sql_utils.render_sql(sql_file_path, all_params)
        
        query_job = self.client.query(rendered_sql, **kwargs)
        df = query_job.to_dataframe()
        logger.info(f"쿼리 실행 완료: {len(df):,}행, {len(df.columns)}열")
        return df

    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        options = options or {}
        table_id = urlparse(target).netloc + urlparse(target).path
        
        logger.info(f"BigQuery에 데이터 쓰기 시작: {table_id}")

        job_config = bigquery.LoadJobConfig()
        write_mode = options.get("write_mode", "truncate")
        job_config.write_disposition = (
            bigquery.WriteDisposition.WRITE_APPEND 
            if write_mode == "append" 
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        if options.get("allow_schema_update", False):
            job_config.schema_update_options = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
        partition_field = options.get("partition_by")
        if partition_field:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )
        
        load_job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config, **kwargs)
        load_job.result()
        logger.info(f"데이터 쓰기 완료. {load_job.output_rows} 행이 {table_id}에 저장됨.") 