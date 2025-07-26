from __future__ import annotations
import pandas as pd
import fsspec
from typing import TYPE_CHECKING, Dict, Any, Optional
import os
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger


class StorageAdapter(BaseAdapter):
    """
    fsspec 라이브러리를 기반으로 하는 통합 스토리지 어댑터.
    로컬 파일 시스템, GCS, S3 등 다양한 스토리지를 지원합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        super().__init__(settings, **kwargs)
        # settings 객체로부터 GCS/S3 등에 필요한 인증 정보를 읽어 
        # fsspec이 요구하는 storage_options 딕셔너리를 생성합니다.
        # 예시: self.storage_options = {"project": "my-gcp-project"}
        self.storage_options = {}
        logger.info("Storage options 생성. 현재는 기본값 사용.")

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI로부터 데이터를 읽어 DataFrame으로 반환합니다."""
        logger.info(f"StorageAdapter read from: {uri}")
        return pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 저장합니다."""
        logger.info(f"StorageAdapter write to: {uri}")
        
        # 로컬 파일 시스템의 경우, 쓰기 전에 디렉토리가 존재하는지 확인하고 생성합니다.
        if "://" not in uri or uri.startswith("file://"):
            path = Path(uri.replace("file://", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            
        df.to_parquet(uri, storage_options=self.storage_options, **kwargs) 