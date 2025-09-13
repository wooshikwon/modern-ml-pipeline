from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.core.console_manager import get_console

class StorageAdapter(BaseAdapter):
    """
    fsspec 라이브러리를 기반으로 하는 통합 스토리지 어댑터.
    로컬 파일 시스템, GCS, S3 등 다양한 스토리지를 지원합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        console = get_console(settings)
        console.info("[StorageAdapter] 초기화 시작합니다")

        super().__init__(settings, **kwargs)
        # 새로운 settings 스키마에서 data_source 설정을 사용
        try:
            # data_source.config에서 storage_options 추출
            if hasattr(settings.config.data_source, 'config') and isinstance(settings.config.data_source.config, dict):
                self.storage_options = settings.config.data_source.config.get('storage_options', {})
            else:
                self.storage_options = {}
            console.info(f"[StorageAdapter] 설정 로드 완료: storage_options={len(self.storage_options)} items")
        except Exception as e:
            console.warning(f"[StorageAdapter] 설정을 찾을 수 없습니다: {e}. 기본값 사용합니다")
            self.storage_options = {}

        console.info("[StorageAdapter] 초기화 완료되었습니다",
                    rich_message="✅ [StorageAdapter] initialized")

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI로부터 데이터를 읽어 DataFrame으로 반환합니다."""
        console = get_console()
        console.info(f"[StorageAdapter] 파일 읽기를 시작합니다: {Path(uri).name}",
                    rich_message=f"📁 [StorageAdapter] Reading from: [cyan]{Path(uri).name}[/cyan]")

        lower = uri.lower()
        if lower.endswith('.csv'):
            console.info("[StorageAdapter] CSV 파일 읽기를 시작합니다")
            result = pd.read_csv(uri, storage_options=self.storage_options, **kwargs)
            console.info(f"[StorageAdapter] CSV 파일 읽기 완료되었습니다: {len(result)} rows, {len(result.columns)} columns",
                        rich_message=f"✅ [StorageAdapter] CSV loaded: [green]{len(result)} rows[/green], [blue]{len(result.columns)} columns[/blue]")
            return result

        console.info("[StorageAdapter] Parquet 파일 읽기를 시작합니다")
        result = pd.read_parquet(uri, storage_options=self.storage_options, **kwargs)
        console.info(f"[StorageAdapter] Parquet 파일 읽기 완료되었습니다: {len(result)} rows, {len(result.columns)} columns",
                    rich_message=f"✅ [StorageAdapter] Parquet loaded: [green]{len(result)} rows[/green], [blue]{len(result.columns)} columns[/blue]")
        return result

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 저장합니다."""
        console = get_console()
        console.info(f"[StorageAdapter] 파일 저장을 시작합니다: {Path(uri).name}",
                    rich_message=f"💾 [StorageAdapter] Writing to: [cyan]{Path(uri).name}[/cyan]")

        # 로컬 파일 시스템의 경우, 쓰기 전에 디렉토리가 존재하는지 확인하고 생성합니다.
        if "://" not in uri or uri.startswith("file://"):
            path = Path(uri.replace("file://", ""))
            console.info(f"[StorageAdapter] 디렉토리 생성 확인: {path.parent}")
            path.parent.mkdir(parents=True, exist_ok=True)

        lower = uri.lower()
        if lower.endswith('.csv'):
            console.info(f"[StorageAdapter] CSV 형식으로 저장합니다: {len(df)} rows")
            df.to_csv(uri, index=False)
        else:
            console.info(f"[StorageAdapter] Parquet 형식으로 저장합니다: {len(df)} rows")
            df.to_parquet(uri, storage_options=self.storage_options, **kwargs)

        console.info(f"[StorageAdapter] 파일 저장이 완료되었습니다: {Path(uri).name}",
                    rich_message=f"✅ [StorageAdapter] File saved: [cyan]{Path(uri).name}[/cyan] ([green]{len(df)} rows[/green])")

# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("storage", StorageAdapter)