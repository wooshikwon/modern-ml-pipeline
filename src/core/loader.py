import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from src.settings.settings import Settings, LoaderSettings
from src.utils.sql_utils import render_sql
from src.utils.bigquery_utils import execute_query
from src.interface.base_loader import BaseLoader

logger = logging.getLogger(__name__)

class FileLoader(BaseLoader):
    """로컬 파일 시스템에서 데이터를 로드하는 클래스 (CSV 또는 Parquet)."""
    def __init__(self, config: LoaderSettings):
        self.config = config
        # 로컬 파일 경로는 프로젝트 루트 기준 상대 경로로 가정
        self.file_path = Path(__file__).resolve().parent.parent.parent / config.local_file_path

    def load(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"로컬 파일 데이터 로딩을 시작합니다: {self.file_path}")
        if not self.file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.file_path}")

        try:
            if self.file_path.suffix == '.csv':
                df = pd.read_csv(self.file_path)
            elif self.file_path.suffix == '.parquet':
                df = pd.read_parquet(self.file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {self.file_path.suffix}")
            
            logger.info(f"파일 로딩 성공: {len(df):,} 행, {len(df.columns)} 열")
            return df
        except Exception as e:
            logger.error(f"로컬 파일 로딩 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"파일 로딩 실패: {e}") from e

class BigQueryLoader(BaseLoader):
    """BigQuery에서 SQL 쿼리를 실행하여 데이터를 로드하는 클래스."""
    def __init__(self, config: LoaderSettings, settings: Settings):
        self.config = config
        self.sql_file_path = config.sql_file_path
        self.settings = settings

    def load(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        SQL 파일을 렌더링하고 BigQuery에서 실행하여 결과를 DataFrame으로 반환합니다.
        """
        logger.info(f"BigQuery 데이터 로딩을 시작합니다: {self.sql_file_path}")
        try:
            # `params` 인자를 사용하여 SQL 템플릿을 렌더링합니다.
            rendered_sql = render_sql(self.sql_file_path, params)
            logger.debug(f"렌더링된 SQL: {rendered_sql[:200]}...")

            data = execute_query(rendered_sql, settings=self.settings)
            logger.info(f"데이터 로딩 성공: {len(data):,} 행, {len(data.columns)} 열")
            return data
        except Exception as e:
            logger.error(f"데이터 로딩 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"데이터 로딩 실패: {e}") from e


def get_dataset_loader(dataset_name: str, settings: Settings) -> BaseLoader:
    """
    설정 파일과 데이터셋 이름에 기반하여 적절한 데이터 로더 인스턴스를 반환합니다.
    'local' 환경에서는 FileLoader 사용을 우선적으로 고려합니다.
    """
    logger.info(f"'{dataset_name}'에 대한 데이터 로더를 생성합니다. (환경: {settings.environment.app_env})")
    
    loader_config = settings.loader.get(dataset_name)
    if not loader_config:
        raise ValueError(f"config.yaml의 'loader' 섹션에서 '{dataset_name}'에 대한 설정을 찾을 수 없습니다.")

    # 'local' 환경이고, local_file_path가 설정되어 있으면 FileLoader를 사용
    if settings.environment.app_env == "local" and loader_config.local_file_path:
        logger.info("로컬 환경이므로 FileLoader를 사용합니���.")
        return FileLoader(config=loader_config)
    
    # 그 외의 경우, 설정된 타입에 따라 로더 결정
    loader_type = loader_config.type
    if loader_type == "bigquery":
        logger.info("BigQueryLoader를 사용합니다.")
        return BigQueryLoader(config=loader_config, settings=settings)
    elif loader_type == "file":
        logger.info("FileLoader를 사용합니다.")
        return FileLoader(config=loader_config)
    else:
        raise ValueError(f"'{loader_type}'은(는) 지원하지 않는 로더 타입입니다 (dataset: '{dataset_name}').")
