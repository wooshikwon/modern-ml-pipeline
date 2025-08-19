from __future__ import annotations
import pandas as pd
import sqlalchemy
from typing import TYPE_CHECKING
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from pathlib import Path
from src.settings import Settings
from src.utils.system.sql_utils import prevent_select_star

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
        try:
            # DataAdapterSettings 모델의 올바른 접근 방법: 딕셔너리 키로 접근
            sql_adapter_config = self.settings.data_adapters.adapters['sql']
            connection_uri = sql_adapter_config.config['connection_uri']
            
            logger.info(f"SQLAlchemy 엔진 생성. URI: {connection_uri}")
            # statement timeout 등은 DB별로 접속 문자열 옵션을 통해 설정 가능(여기서는 주석으로 가이드)
            return sqlalchemy.create_engine(connection_uri)
        except KeyError as e:
            logger.error(f"SQL 어댑터 설정을 찾을 수 없습니다: {e}")
            raise ValueError(f"SQL 어댑터 설정이 누락되었습니다: {e}")
        except Exception as e:
            logger.error(f"SQLAlchemy 엔진 생성 실패: {e}", exc_info=True)
            raise

    def _enforce_sql_guards(self, sql_query: str) -> None:
        """보안/신뢰성 가드 적용: SELECT * 차단, DDL/DML 금칙어 차단, LIMIT 가드(옵션)."""
        # 1) SELECT * 차단
        prevent_select_star(sql_query)

        # 2) DDL/DML 금칙어 차단(간단한 포함 검사)
        DANGEROUS = [
            "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ",
            "CREATE ", "EXEC ", "EXECUTE ", "MERGE ", "GRANT ", "REVOKE ",
        ]
        upper = sql_query.upper()
        for token in DANGEROUS:
            if token in upper:
                raise ValueError(f"보안 위반: 금지된 SQL 키워드 포함({token.strip()}).")

        # 3) LIMIT 가드(선택: 너무 큰 결과 방지). settings로 제어 가능하도록 확장 여지.
        # 여기서는 강제하지 않고 경고만 남김.
        if " LIMIT " not in upper:
            logger.warning("SQL LIMIT 가드: LIMIT 절이 없습니다. 대용량 쿼리일 수 있습니다.")

    def read(self, sql_query: str, **kwargs) -> pd.DataFrame:
        """SQL 쿼리를 실행하여 결과를 DataFrame으로 반환합니다.
        
        Args:
            sql_query: SQL 쿼리 문자열 또는 .sql 파일 경로
        """
        # 파일 경로인지 확인 (.sql 확장자로 끝나는 경우)
        if sql_query.endswith('.sql'):
            sql_file_path = Path(sql_query)
            if not sql_file_path.is_absolute():
                # 상대 경로인 경우 프로젝트 루트 기준으로 해석
                # src/utils/adapters/sql_adapter.py에서 3단계 상위 = modern-ml-pipeline/
                base_dir = Path(__file__).resolve().parent.parent.parent.parent
                sql_file_path = base_dir / sql_query
            
            if sql_file_path.exists():
                sql_query = sql_file_path.read_text(encoding='utf-8')
                logger.info(f"SQL 파일 로딩: {sql_file_path}")
            else:
                logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
                raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
        
        # 보안 가드 적용
        self._enforce_sql_guards(sql_query)
        
        logger.info(f"Executing SQL query:\n{sql_query[:200]}...")
        try:
            # Pandas + SQLAlchemy 2.x 호환: 엔진 객체를 직접 전달
            return pd.read_sql_query(sql_query, self.engine, **kwargs)
        except Exception as e:
            snippet = sql_query[:200].replace('\n', ' ')
            logger.error(f"SQL read 작업 실패: {e} | SQL(head): {snippet}", exc_info=True)
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