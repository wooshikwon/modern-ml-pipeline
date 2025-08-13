from __future__ import annotations
import pandas as pd
import sqlalchemy
from typing import TYPE_CHECKING
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from src.engine import AdapterRegistry
from src.utils.system.sql_utils import prevent_select_star
from src.settings._utils import BASE_DIR

if TYPE_CHECKING:
    from src.settings import Settings


class SqlAdapter(BaseAdapter):
    """SQLAlchemy 기반 통합 SQL 어댑터"""
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.engine = self._create_engine()

    def _create_engine(self):
        try:
            sql_adapter_config = self.settings.data_adapters.adapters['sql']
            connection_uri = sql_adapter_config.config['connection_uri']
            logger.info(f"SQLAlchemy 엔진 생성. URI: {connection_uri}")
            return sqlalchemy.create_engine(connection_uri)
        except KeyError as e:
            logger.error(f"SQL 어댑터 설정을 찾을 수 없습니다: {e}")
            raise ValueError(f"SQL 어댑터 설정이 누락되었습니다: {e}")
        except Exception as e:
            logger.error(f"SQLAlchemy 엔진 생성 실패: {e}", exc_info=True)
            raise

    def _enforce_sql_guards(self, sql_query: str) -> None:
        prevent_select_star(sql_query)
        DANGEROUS = [
            "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ",
            "CREATE ", "EXEC ", "EXECUTE ", "MERGE ", "GRANT ", "REVOKE ",
        ]
        upper = sql_query.upper()
        for token in DANGEROUS:
            if token in upper:
                raise ValueError(f"보안 위반: 금지된 SQL 키워드 포함({token.strip()}).")
        if " LIMIT " not in upper:
            logger.warning("SQL LIMIT 가드: LIMIT 절이 없습니다. 대용량 쿼리일 수 있습니다.")

    def read(self, sql_query: str, **kwargs) -> pd.DataFrame:
        if sql_query.endswith('.sql'):
            sql_file_path = Path(sql_query)
            if not sql_file_path.is_absolute():
                sql_file_path = BASE_DIR / sql_query
            if sql_file_path.exists():
                sql_query = sql_file_path.read_text(encoding='utf-8')
                logger.info(f"SQL 파일 로딩: {sql_file_path}")
            else:
                logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
                raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")

        self._enforce_sql_guards(sql_query)
        logger.info(f"Executing SQL query:\n{sql_query[:200]}...")
        try:
            return pd.read_sql_query(sql_query, self.engine, **kwargs)
        except Exception as e:
            snippet = sql_query[:200].replace('\n', ' ')
            logger.error(f"SQL read 작업 실패: {e} | SQL(head): {snippet}", exc_info=True)
            raise

    def write(self, df: pd.DataFrame, table_name: str, **kwargs):
        logger.info(f"Writing DataFrame to table: {table_name}")
        try:
            df.to_sql(table_name, self.engine, **kwargs)
            logger.info(f"Successfully wrote {len(df)} rows to {table_name}.")
        except Exception as e:
            logger.error(f"SQL write 작업 실패: {e}", exc_info=True)
            raise


AdapterRegistry.register("sql", SqlAdapter)

