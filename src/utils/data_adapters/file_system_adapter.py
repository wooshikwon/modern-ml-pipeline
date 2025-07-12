import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from src.interface.base_data_adapter import BaseDataAdapter
from src.utils.logger import logger

class FileSystemAdapter(BaseDataAdapter):
    """로컬 파일 시스템에 대한 데이터 읽기/쓰기를 처리하는 어댑터."""

    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        지정된 로컬 파일 경로에서 데이터를 읽어옵니다.

        Args:
            source (str): 'file://'로 시작하는 URI.
        """
        parsed_uri = urlparse(source)
        file_path = self._get_absolute_path(parsed_uri.path)
        
        logger.info(f"로컬 파일 시스템에서 데이터 읽기 시작: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if file_path.suffix == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"��원하지 않는 파일 형식입니다: {file_path.suffix}")

    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        데이터프레임을 지정된 로컬 파일 경로에 Parquet 형식으로 씁니다.

        Args:
            df (pd.DataFrame): 저장할 데이터프레임.
            target (str): 'file://'로 시작하는 URI.
            options (Optional[Dict[str, Any]]): 파티셔닝 등의 옵션.
        """
        options = options or {}
        parsed_uri = urlparse(target)
        target_path = self._get_absolute_path(parsed_uri.path)
        
        partition_cols = options.get("partition_by")

        logger.info(f"로컬 파일 시스템에 데이터 쓰기 시작: {target_path}")
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(target_path, index=False, partition_cols=partition_cols, **kwargs)
        logger.info(f"데이터 쓰기 완료. {len(df)} 행 저장됨.")

    def _get_absolute_path(self, path_str: str) -> Path:
        """URI 경로를 절대 경로 Path 객체로 변환합니다."""
        # URI path는 보통 맨 앞에 /가 붙으므로 제거
        path = Path(path_str.lstrip('/'))
        if not path.is_absolute():
            # 프로젝트 루트를 기준으로 경로를 재구성
            return Path(__file__).resolve().parent.parent.parent / path
        return path
