import logging
from typing import Dict, Any, Optional

import pandas as pd

from src.settings.settings import Settings, LoaderSettings
from src.utils.sql_utils import render_sql
from src.utils.bigquery_utils import execute_query
from src.interface.base_loader import BaseLoader

logger = logging.getLogger(__name__)


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
    """
    logger.info(f"'{dataset_name}'에 대한 데이터 로더를 생성합니다.")
    
    # settings.loader는 이제 딕셔너리이므로 .get()으로 안전하게 접근
    loader_config = settings.loader.get(dataset_name)

    if not loader_config:
        raise ValueError(f"config.yaml의 'loader' 섹션에서 '{dataset_name}'에 대한 설정을 찾을 수 없습니다.")

    # 현재는 BigQueryLoader만 지원하지만, 향후 다른 로더를 추가할 수 있는 구조
    if loader_config.output.type == "bigquery":
        return BigQueryLoader(config=loader_config, settings=settings)
    # 예: elif loader_config.output.type == "file":
    #         return FileLoader(config=loader_config, settings=settings)
    else:
        raise ValueError(f"'{loader_config.output.type}'은(는) 지원하지 않는 로더 타입입니다 (dataset: '{dataset_name}').")
