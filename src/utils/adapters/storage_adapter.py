from __future__ import annotations
import pandas as pd
import fsspec
from typing import TYPE_CHECKING, Dict, Any
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings


class StorageAdapter(BaseAdapter):
    """
    fsspec을 기반으로 하는 통합 스토리지 어댑터.
    다양한 파일 시스템(local, GCS, S3 등)과의 연결을 표준화합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.storage_options = self._get_storage_options()

    def _get_storage_options(self) -> Dict[str, Any]:
        """설정(Settings) 객체로부터 fsspec에 필요한 인증 정보를 생성합니다."""
        # 이 부분은 Phase 3에서 config 구조가 개편되면 더 동적으로 변경될 수 있습니다.
        # 현재는 gcs 설정을 우선적으로 사용한다고 가정합니다.
        try:
            # TODO: config 구조 변경 후 동적으로 storage_options를 가져오도록 수정
            # 예: gcs_config = self.settings.data_sources.gcs
            # if gcs_config.credentials: ...
            logger.info("Storage options 생성. 현재는 기본값 사용.")
            return {} # gcloud auth application-default login 등으로 인증 가정
        except Exception as e:
            logger.error(f"Storage options 생성 실패: {e}", exc_info=True)
            raise

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI에서 데이터를 읽어 DataFrame으로 반환합니다. (parquet 포맷 우선)"""
        logger.info(f"Reading from storage URI: {uri}")
        try:
            # TODO: 파일 확장자에 따라 다른 pd.read_* 함수 호출 로직 추가
            return pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)
        except Exception as e:
            logger.error(f"Storage read 작업 실패: {e}", exc_info=True)
            raise

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 씁니다. (parquet 포맷 우선)"""
        logger.info(f"Writing DataFrame to storage URI: {uri}")
        try:
            # TODO: 파일 확장자에 따라 다른 df.to_* 함수 호출 로직 추가
            df.to_parquet(uri, storage_options=self.storage_options, **kwargs)
            logger.info(f"Successfully wrote {len(df)} rows to {uri}.")
        except Exception as e:
            logger.error(f"Storage write 작업 실패: {e}", exc_info=True)
            raise 