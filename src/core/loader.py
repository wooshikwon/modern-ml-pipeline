import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from src.settings.settings import Settings, LoaderSettings
from src.utils import sql_utils, bigquery_utils
from src.interface.base_loader import BaseLoader

logger = logging.getLogger(__name__)

class FileLoader(BaseLoader):
    """로컬 파일 시스템에서 데이터를 로드하는 클래스."""
    def __init__(self, path: str):
        self.file_path = Path(path)
        if not self.file_path.is_absolute():
            self.file_path = Path(__file__).resolve().parent.parent.parent / self.file_path

    def load(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"로컬 파일 로딩: {self.file_path}")
        # ... (이전과 동일한 로직) ...

class BigQueryLoader(BaseLoader):
    """BigQuery에서 SQL 쿼리를 실행하여 데이터를 로드하는 클래스."""
    def __init__(self, sql_path: str, settings: Settings):
        self.sql_file_path = sql_path
        self.settings = settings

    def load(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"BigQuery 데이터 로딩 (SQL: {self.sql_file_path})")
        rendered_sql = sql_utils.render_sql(self.sql_file_path, params)
        return bigquery_utils.execute_query(rendered_sql, self.settings)

def get_dataset_loader(settings: Settings) -> BaseLoader:
    """
    "환경 인지" 팩토리 함수.
    레시피의 loader 설정과 현재 실행 환경(app_env)에 따라
    적절한 로더 인스턴스를 생성하여 반환합니다.
    """
    loader_config = settings.model.loader
    logger.info(f"'{loader_config.name}' 로더 생성을 시작합니다. (환경: {settings.environment.app_env})")

    is_local = settings.environment.app_env == "local"
    local_path = loader_config.local_override_path

    if is_local and local_path:
        logger.info(f"로컬 재정의 경로를 사용하여 FileLoader를 생성합니다: {local_path}")
        return FileLoader(path=local_path)
    
    # 기본 소스가 SQL 파일인 경우
    if loader_config.source_sql_path:
        logger.info(f"SQL 소스를 사용하여 BigQueryLoader를 생성합니다: {loader_config.source_sql_path}")
        return BigQueryLoader(sql_path=loader_config.source_sql_path, settings=settings)
    
    raise ValueError(f"'{loader_config.name}' 로더에 대한 유효한 소스 경로(source_sql_path)를 찾을 수 없습니다.")