from __future__ import annotations
import pandas as pd
import sqlalchemy
from typing import TYPE_CHECKING
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings


class SqlAdapter(BaseAdapter):
    """
    SQLAlchemy를 기반으로 하는 통합 SQL 어댑터.
    다양한 SQL 데이터베이스(PostgreSQL, BigQuery 등)와의 연결을 표준화합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.engine = self._create_engine()

    def _create_engine(self):
        """설정(Settings) 객체로부터 DB 연결 URI를 생성하고 SQLAlchemy 엔진을 반환합니다."""
        # 이 부분은 Phase 3에서 config 구조가 개편되면 더 동적으로 변경될 수 있습니다.
        # 현재는 dev 환경의 postgresql 설정을 우선적으로 사용한다고 가정합니다.
        try:
            # TODO: config 구조 변경 후 동적으로 connection_uri를 가져오도록 수정
            # 예시: postgresql_config = self.settings.data_sources.postgresql
            connection_uri = "postgresql://user:password@localhost:5432/mlpipeline" # 임시 하드코딩
            logger.info(f"SQLAlchemy 엔진 생성. URI: {connection_uri}")
            return sqlalchemy.create_engine(connection_uri)
        except Exception as e:
            logger.error(f"SQLAlchemy 엔진 생성 실패: {e}", exc_info=True)
            raise

    def read(self, sql_query: str, **kwargs) -> pd.DataFrame:
        """SQL 쿼리를 실행하여 결과를 DataFrame으로 반환합니다."""
        logger.info(f"Executing SQL query:\n{sql_query[:200]}...")
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(sql_query, connection, **kwargs)
        except Exception as e:
            logger.error(f"SQL read 작업 실패: {e}", exc_info=True)
            raise

    def write(self, df: pd.DataFrame, table_name: str, **kwargs):
        """DataFrame을 지정된 테이블에 씁니다."""
        logger.info(f"Writing DataFrame to table: {table_name}")
        try:
            df.to_sql(table_name, self.engine, **kwargs)
            logger.info(f"Successfully wrote {len(df)} rows to {table_name}.")
        except Exception as e:
            logger.error(f"SQL write 작업 실패: {e}", exc_info=True)
            raise 