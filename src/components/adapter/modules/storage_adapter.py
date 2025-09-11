from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.console_manager import get_console

class StorageAdapter(BaseAdapter):
    """
    fsspec 라이브러리를 기반으로 하는 통합 스토리지 어댑터.
    로컬 파일 시스템, GCS, S3 등 다양한 스토리지를 지원합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        super().__init__(settings, **kwargs)
        # 새로운 settings 스키마에서 data_source 설정을 사용
        console = get_console()
        try:
            # data_source.config에서 storage_options 추출
            if hasattr(settings.config.data_source, 'config') and isinstance(settings.config.data_source.config, dict):
                self.storage_options = settings.config.data_source.config.get('storage_options', {})
            else:
                self.storage_options = {}
            console.info(f"StorageAdapter 초기화 완료. storage_options: {self.storage_options}")
        except Exception as e:
            console.warning(f"Storage 어댑터 설정을 찾을 수 없습니다: {e}. 기본값 사용.")
            self.storage_options = {}

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI로부터 데이터를 읽어 DataFrame으로 반환합니다."""
        console = get_console()
        console.info(f"StorageAdapter read from: {uri}")
        lower = uri.lower()
        if lower.endswith('.csv'):
            return pd.read_csv(uri, storage_options=self.storage_options, **kwargs)
        return pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 저장합니다."""
        console = get_console()
        console.info(f"StorageAdapter write to: {uri}")
        
        # 로컬 파일 시스템의 경우, 쓰기 전에 디렉토리가 존재하는지 확인하고 생성합니다.
        if "://" not in uri or uri.startswith("file://"):
            path = Path(uri.replace("file://", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
        
        lower = uri.lower()
        if lower.endswith('.csv'):
            df.to_csv(uri, index=False)
        else:
            df.to_parquet(uri, storage_options=self.storage_options, **kwargs)

# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("storage", StorageAdapter)