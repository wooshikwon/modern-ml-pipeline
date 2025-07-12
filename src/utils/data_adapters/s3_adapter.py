import pandas as pd
from typing import Dict, Any, Optional

from src.interface.base_data_adapter import BaseDataAdapter
from src.utils.logger import logger

class S3Adapter(BaseDataAdapter):
    """AWS S3와의 데이터 읽기/쓰기를 처리하는 어댑터."""

    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        S3 경로에서 Parquet 파일을 읽어옵니다.

        Args:
            source (str): 's3://bucket/path/to/file.parquet' 형식의 S3 URI.
        """
        logger.info(f"S3에서 데이터 읽기 시작: {source}")
        # s3fs가 설치되어 있으면 pandas가 s3:// URI를 직접 처리
        return pd.read_parquet(source, **kwargs)

    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        데이터프레임을 지정된 S3 경로에 Parquet 형식으로 씁니다.

        Args:
            df (pd.DataFrame): 저장할 데이터프레임.
            target (str): 's3://bucket/path/to/target_dir' 형식의 S3 URI.
            options (Optional[Dict[str, Any]]): 파티셔닝 등의 옵션.
        """
        options = options or {}
        partition_cols = options.get("partition_by")
        
        logger.info(f"S3에 데이터 쓰기 시작: {target}")
        
        df.to_parquet(
            target,
            index=False,
            partition_cols=partition_cols,
            **kwargs
        )
        logger.info(f"데이터 쓰기 완료. {len(df)} 행이 {target}에 저장됨.")
