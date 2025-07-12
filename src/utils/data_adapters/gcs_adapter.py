import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account

from src.interface.base_data_adapter import BaseDataAdapter
from src.utils.logger import logger
from src.settings.settings import Settings

class GCSAdapter(BaseDataAdapter):
    """Google Cloud Storage와의 데이터 읽기/쓰기를 처리하는 어댑터."""

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def _get_client(self) -> storage.Client:
        """GCS 클라이언트를 초기화하고 반환합니다."""
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

            client = storage.Client(credentials=credentials, project=project_id)
            logger.info(f"GCS 클라이언트 초기화 완료 (Project: {project_id})")
            return client
        except Exception as e:
            logger.error(f"GCS 클라이언트 초기화 실패: {e}")
            raise

    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        GCS 경로에서 Parquet 파일을 읽어옵니다.
        """
        logger.info(f"GCS에서 데이터 읽기 시작: {source}")
        # gcsfs가 설치되어 있으면 pandas가 gs:// URI를 직접 처리
        return pd.read_parquet(source, **kwargs)

    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        데이터프레임을 지정된 GCS 경로에 Parquet 형식으로 씁니다.
        """
        options = options or {}
        partition_cols = options.get("partition_by")
        
        logger.info(f"GCS에 데이터 쓰기 시작: {target}")
        
        df.to_parquet(
            target,
            index=False,
            partition_cols=partition_cols,
            **kwargs
        )
        logger.info(f"데이터 쓰기 완료. {len(df)} 행이 {target}에 저장됨.")