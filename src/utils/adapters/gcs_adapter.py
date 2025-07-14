import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account

from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from src.settings import Settings

class GCSAdapter(BaseAdapter):
    """Google Cloud Storage와의 데이터 읽기/쓰기를 처리하는 어댑터."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.client = self._get_client()

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

        Args:
            df (pd.DataFrame): 저장할 데이터프레임.
            target (str): 'gs://bucket/path/to/target_dir' 형식의 GCS URI.
            options (Optional[Dict[str, Any]]): 파티셔닝 등의 옵션.
        """
        options = options or {}
        logger.info(f"GCS에 데이터 쓰기 시작: {target}")
        # gcsfs가 설치되어 있으면 pandas가 gs:// URI를 직접 처리
        partition_cols = options.get("partition_by")
        if partition_cols:
            df.to_parquet(target, partition_cols=partition_cols, **kwargs)
        else:
            df.to_parquet(target, **kwargs)
        logger.info(f"GCS에 데이터 쓰기 완료: {target}") 