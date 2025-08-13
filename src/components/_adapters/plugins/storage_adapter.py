from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from src.engine import AdapterRegistry


class StorageAdapter(BaseAdapter):
    """fsspec 기반 통합 스토리지 어댑터"""
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        try:
            storage_adapter_config = self.settings.data_adapters.adapters['storage']
            self.storage_options = storage_adapter_config.config.get('storage_options', {})
            logger.info(f"StorageAdapter 초기화 완료. storage_options: {self.storage_options}")
        except Exception as e:
            logger.warning(f"Storage 어댑터 설정을 찾을 수 없거나 오류: {e}. 기본값 사용.")
            self.storage_options = {}

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        logger.info(f"StorageAdapter read from: {uri}")
        lower = uri.lower()
        if lower.endswith('.csv'):
            return pd.read_csv(uri, storage_options=self.storage_options, **kwargs)
        return pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        logger.info(f"StorageAdapter write to: {uri}")
        if "://" not in uri or uri.startswith("file://"):
            path = Path(uri.replace("file://", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
        lower = uri.lower()
        if lower.endswith('.csv'):
            df.to_csv(uri, index=False)
        else:
            df.to_parquet(uri, storage_options=self.storage_options, **kwargs)


AdapterRegistry.register("storage", StorageAdapter)

