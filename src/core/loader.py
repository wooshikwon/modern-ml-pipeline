# logger 설정은 이미 pipelines/ 의 파일에서 from src.utils.logger import logger 로 설정되어 있다고 전제함. 그러므로 여기선 로거 생성만.
import logging
logger = logging.getLogger(__name__)

import typing import Dict, Any, Optional

# config 설정과 utils 모듈 import
from config.settings import Settings, LoaderSettings
from src.utils.sql_utils import render_sql
from src.utils.bigquery_utils import execute_query

# interface import
from src.interface.base_loader import BaseLoader

import pandas as pd


class BigQueryLoader(BaseLoader):
    def __init__(self, config: LoaderSettings, settings: Settings):
        # Config.yaml에 설정된 Loader 하위의 여러 데이터셋 중 '특정' 데이터셋에 대한 설정을 dataset_config에 저장
        self.config = config
        # self.config의 sql 파일 경로를 별도로 저장
        self.sql_file_path = self.config.sql_file_path
        # utils/ 도구에 전달할 환경설정 전체 객체
        self.settings = settings

    def load(self, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"데이터 로딩을 시작합니다: {self.sql_file_path}")

        # params 인자는 Config.yaml에 설정하지 않음. 배치의 경우 날짜 파라미터가 매번 바뀌므로 이것은 진입점에서 전달해야함.
        try:
            rendered_sql = render_sql(self.sql_file_path, params)
            logger.info(f"SQL 쿼리 랜더링 성공: {rendered_sql[:100]} \n (...)")

            data = execute_query(rendered_sql, settings=self.settings)
            logger.info(f"데이터 로딩 성공: {len(data):,} 행, {len(data.columns)} 열")
            return data

        except Exception as e:
            logger.error(f"데이터 로딩 중 예측하지 못한 오류가 발생했습니다: {e}", exc_info=True)
            raise RuntimeError(f"데이터 로딩 중 예측하지 못한 오류가 발생했습니다: {e}") from e

# config의 특정 데이터셋 이름을 인자로 넣어서, BigQueryLoader 인스턴스를 반환
# pipeines/ 에서 이 함수를 호출하여 데이터셋 로더를 생성하고, 데이터셋 로더의 load() 메서드를 호출하여 데이터를 로드한다.
def get_dataset_loader(dataset_name: str, settings: Settings) -> BigQueryLoader:
    config = settings.loader.get(dataset_name)

    if not config:
        raise ValueError(f"config.yaml 파일에서 '{dataset_name}'에 대한 정의를 찾을 수 없습니다.")

    if config.output.type == "bigquery":
        return BigQueryLoader(config=config, settings=settings)
    else:
        raise ValueError(f"'{config.output.type}'은(는) 지원하지 않는 로더 타입입니다 (dataset_name: '{dataset_name}').")