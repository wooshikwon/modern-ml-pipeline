from __future__ import annotations
import pandas as pd
import sqlalchemy
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
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
    다양한 SQL 데이터베이스(PostgreSQL, BigQuery, MySQL, SQLite 등)와의 연결을 표준화합니다.
    
    URI 스키마에 따라 자동으로 적절한 데이터베이스 엔진을 선택합니다:
    - bigquery:// → BigQuery 엔진
    - postgresql:// or postgres:// → PostgreSQL 엔진
    - mysql:// → MySQL 엔진
    - sqlite:// → SQLite 엔진
    """
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.engine = self._create_engine()

    def _parse_connection_uri(self, uri: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        연결 URI를 파싱하여 데이터베이스 타입과 엔진별 설정을 반환합니다.
        
        Args:
            uri: 데이터베이스 연결 URI
            
        Returns:
            (db_type, processed_uri, engine_kwargs) 튜플
        """
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        
        # 기본 엔진 설정
        engine_kwargs = {}
        
        if scheme in ('bigquery', 'bigquery+sqlalchemy'):
            db_type = 'bigquery'
            # BigQuery는 특별한 처리 필요
            engine_kwargs['pool_pre_ping'] = True
            engine_kwargs['pool_size'] = 5
            try:
                from sqlalchemy_bigquery import BigQueryDialect
                logger.info("BigQuery 엔진 설정 적용")
            except ImportError:
                logger.warning("sqlalchemy-bigquery 패키지가 설치되지 않았습니다. 기본 설정 사용.")
            processed_uri = uri
            
        elif scheme in ('postgresql', 'postgres', 'postgresql+psycopg2'):
            db_type = 'postgresql'
            # PostgreSQL 최적화 설정
            engine_kwargs['pool_size'] = 10
            engine_kwargs['max_overflow'] = 20
            engine_kwargs['pool_pre_ping'] = True
            engine_kwargs['connect_args'] = {
                'connect_timeout': 10,
                'options': '-c statement_timeout=30000'  # 30초 타임아웃
            }
            processed_uri = uri
            logger.info("PostgreSQL 엔진 설정 적용")
            
        elif scheme in ('mysql', 'mysql+pymysql', 'mysql+mysqldb'):
            db_type = 'mysql'
            # MySQL 최적화 설정
            engine_kwargs['pool_size'] = 10
            engine_kwargs['pool_recycle'] = 3600  # 1시간마다 연결 재활용
            engine_kwargs['pool_pre_ping'] = True
            processed_uri = uri
            logger.info("MySQL 엔진 설정 적용")
            
        elif scheme == 'sqlite':
            db_type = 'sqlite'
            # SQLite는 연결 풀이 필요 없음
            engine_kwargs['poolclass'] = sqlalchemy.pool.StaticPool
            engine_kwargs['connect_args'] = {'check_same_thread': False}
            processed_uri = uri
            logger.info("SQLite 엔진 설정 적용")
            
        else:
            # 알 수 없는 스키마는 기본 SQLAlchemy 처리
            db_type = 'generic'
            logger.warning(f"알 수 없는 데이터베이스 스키마: {scheme}. 기본 SQLAlchemy 설정 사용.")
            processed_uri = uri
            
        return db_type, processed_uri, engine_kwargs

    def _create_engine(self):
        """
        설정(Settings) 객체로부터 DB 연결 URI를 파싱하고 
        데이터베이스 타입에 맞는 SQLAlchemy 엔진을 생성합니다.
        """
        try:
            # DataAdapterSettings 모델의 올바른 접근 방법: 딕셔너리 키로 접근
            sql_adapter_config = self.settings.data_adapters.adapters['sql']
            connection_uri = sql_adapter_config.config['connection_uri']
            
            # URI 파싱하여 DB 타입과 엔진 설정 추출
            db_type, processed_uri, engine_kwargs = self._parse_connection_uri(connection_uri)
            
            logger.info(f"데이터베이스 타입: {db_type}")
            logger.info(f"SQLAlchemy 엔진 생성. URI: {processed_uri[:50]}...")  # 보안을 위해 URI 일부만 로깅
            
            # 데이터베이스별 특수 처리
            if db_type == 'bigquery':
                # BigQuery는 추가 설정이 필요할 수 있음
                try:
                    # BigQuery 인증 설정이 있는 경우 처리
                    if 'credentials_path' in sql_adapter_config.config:
                        import os
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sql_adapter_config.config['credentials_path']
                        logger.info("BigQuery 인증 파일 설정 완료")
                except Exception as e:
                    logger.warning(f"BigQuery 인증 설정 중 경고: {e}")
            
            # 엔진 생성
            engine = sqlalchemy.create_engine(processed_uri, **engine_kwargs)
            
            # 연결 테스트
            with engine.connect() as conn:
                logger.info(f"{db_type} 데이터베이스 연결 테스트 성공")
            
            return engine
            
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

# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("sql", SqlAdapter)