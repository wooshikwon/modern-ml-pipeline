from __future__ import annotations
import pandas as pd
import sqlalchemy
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from src.interface.base_adapter import BaseAdapter
from src.utils.system.console_manager import get_console
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
    - bigquery:// → BigQuery 엔진 (pandas_gbq 지원)
    - postgresql:// or postgres:// → PostgreSQL 엔진
    - mysql:// → MySQL 엔진
    - sqlite:// → SQLite 엔진
    """
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.db_type = None  # Will be set by _create_engine
        self.engine = self._create_engine()
        
        # BigQuery 전용 플래그 및 설정
        self.use_pandas_gbq = False
        self.project_id = None
        self.dataset_id = None
        self.location = 'US'
        
        if self.db_type == 'bigquery':
            config = self.settings.config.data_source.config
            self.use_pandas_gbq = config.get('use_pandas_gbq', False)
            self.project_id = config.get('project_id')
            self.dataset_id = config.get('dataset_id')
            self.location = config.get('location', 'US')

    def _parse_connection_uri(self, uri: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        연결 URI를 파싱하여 데이터베이스 타입과 엔진별 설정을 반환합니다.
        
        Args:
            uri: 데이터베이스 연결 URI
            
        Returns:
            (db_type, processed_uri, engine_kwargs) 튜플
        """
        console = get_console()
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
                console.info("BigQuery 엔진 설정 적용")
            except ImportError:
                console.warning("sqlalchemy-bigquery 패키지가 설치되지 않았습니다. 기본 설정 사용.")
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
            console.info("PostgreSQL 엔진 설정 적용")
            
        elif scheme in ('mysql', 'mysql+pymysql', 'mysql+mysqldb'):
            db_type = 'mysql'
            # MySQL 최적화 설정
            engine_kwargs['pool_size'] = 10
            engine_kwargs['pool_recycle'] = 3600  # 1시간마다 연결 재활용
            engine_kwargs['pool_pre_ping'] = True
            processed_uri = uri
            console.info("MySQL 엔진 설정 적용")
            
        elif scheme == 'sqlite':
            db_type = 'sqlite'
            # SQLite는 연결 풀이 필요 없음
            engine_kwargs['poolclass'] = sqlalchemy.pool.StaticPool
            engine_kwargs['connect_args'] = {'check_same_thread': False}
            processed_uri = uri
            console.info("SQLite 엔진 설정 적용")
            
        else:
            # 알 수 없는 스키마는 기본 SQLAlchemy 처리
            db_type = 'generic'
            console.warning(f"알 수 없는 데이터베이스 스키마: {scheme}. 기본 SQLAlchemy 설정 사용.")
            processed_uri = uri
            
        return db_type, processed_uri, engine_kwargs

    def _create_engine(self):
        """
        설정(Settings) 객체로부터 DB 연결 URI를 파싱하고 
        데이터베이스 타입에 맞는 SQLAlchemy 엔진을 생성합니다.
        """
        console = get_console()
        try:
            # 새로운 Settings 구조: config.data_source 접근
            data_source_config = self.settings.config.data_source.config
            connection_uri = data_source_config['connection_uri']
            
            # URI 파싱하여 DB 타입과 엔진 설정 추출
            db_type, processed_uri, engine_kwargs = self._parse_connection_uri(connection_uri)
            self.db_type = db_type  # Store db_type for later use
            
            console.info(f"데이터베이스 타입: {db_type}")
            console.info(f"SQLAlchemy 엔진 생성. URI: {processed_uri[:50]}...")  # 보안을 위해 URI 일부만 로깅
            
            # 데이터베이스별 특수 처리
            if db_type == 'bigquery':
                # BigQuery는 추가 설정이 필요할 수 있음
                try:
                    # BigQuery 인증 설정이 있는 경우 처리
                    if 'credentials_path' in data_source_config:
                        import os
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = data_source_config['credentials_path']
                        console.info("BigQuery 인증 파일 설정 완료")
                except Exception as e:
                    console.warning(f"BigQuery 인증 설정 중 경고: {e}")
            
            # 엔진 생성
            engine = sqlalchemy.create_engine(processed_uri, **engine_kwargs)
            
            # 연결 테스트
            with engine.connect() as conn:
                console.info(f"{db_type} 데이터베이스 연결 테스트 성공")
            
            return engine
            
        except KeyError as e:
            console.error(f"SQL 어댑터 설정을 찾을 수 없습니다: {e}")
            raise ValueError(f"SQL 어댑터 설정이 누락되었습니다: {e}")
        except Exception as e:
            console.error(f"SQLAlchemy 엔진 생성 실패: {e}")
            raise

    def _enforce_sql_guards(self, sql_query: str) -> None:
        """보안/신뢰성 가드 적용: SELECT * 차단, DDL/DML 금칙어 차단, LIMIT 가드(옵션)."""
        console = get_console()
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
            console.warning("SQL LIMIT 가드: LIMIT 절이 없습니다. 대용량 쿼리일 수 있습니다.")

    def read(self, source: str, params: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """SQL 쿼리를 실행하여 결과를 DataFrame으로 반환합니다.
        BigQuery의 경우 pandas_gbq를 사용할 수 있습니다.
        
        Args:
            source: SQL 쿼리 문자열 또는 .sql 파일 경로
            params: 쿼리 파라미터 (Optional)
        """
        console = get_console()
        
        # BigQuery + pandas_gbq 사용 시
        if self.db_type == 'bigquery' and self.use_pandas_gbq:
            try:
                import pandas_gbq
                console.info("BigQuery read using pandas_gbq")
                return pandas_gbq.read_gbq(
                    source, 
                    project_id=self.project_id,
                    location=self.location,
                    **kwargs
                )
            except ImportError:
                console.warning("pandas_gbq not installed, falling back to SQLAlchemy")
            except Exception as e:
                console.warning(f"pandas_gbq read failed: {e}, falling back to SQLAlchemy")
        
        # 기존 SQLAlchemy 방식 (기본값 및 fallback)
        sql_query = source  # Rename for compatibility
        
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
                console.info(f"SQL 파일 로딩: {sql_file_path}")
            else:
                console.error(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
                raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")
        
        # 보안 가드 적용
        self._enforce_sql_guards(sql_query)
        
        console.info(f"Executing SQL query:\n{sql_query[:200]}...")
        try:
            # Pandas + SQLAlchemy 2.x 호환: 엔진 객체를 직접 전달
            return pd.read_sql_query(sql_query, self.engine, params=params, **kwargs)
        except Exception as e:
            snippet = sql_query[:200].replace('\n', ' ')
            console.error(f"SQL read 작업 실패: {e} | SQL(head): {snippet}")
            raise

    def write(self, df: pd.DataFrame, target: str, **kwargs):
        """DataFrame을 지정된 테이블에 씁니다.
        BigQuery의 경우 pandas_gbq를 사용할 수 있습니다.
        
        Args:
            df: 저장할 DataFrame
            target: 대상 테이블 이름
        """
        console = get_console()
        
        # BigQuery 전용 처리
        if self.db_type == 'bigquery' and (self.use_pandas_gbq or kwargs.get('if_exists') == 'replace'):
            try:
                import pandas_gbq
                # dataset_id가 있으면 테이블명에 추가
                destination_table = f"{self.dataset_id}.{target}" if self.dataset_id and '.' not in target else target
                
                console.info(f"BigQuery write using pandas_gbq to {destination_table}")
                pandas_gbq.to_gbq(
                    df,
                    destination_table=destination_table,
                    project_id=self.project_id,
                    location=self.location,
                    if_exists=kwargs.get('if_exists', 'append'),
                    **{k: v for k, v in kwargs.items() if k not in ['if_exists']}
                )
                console.info(f"BigQuery write complete: {len(df)} rows to {destination_table}")
                return
            except ImportError:
                console.warning("pandas_gbq not installed, falling back to SQLAlchemy")
            except Exception as e:
                console.warning(f"pandas_gbq write failed: {e}, falling back to SQLAlchemy")
        
        # 기존 SQLAlchemy 방식 (기본값 및 fallback)
        console.info(f"Writing DataFrame to table: {target}")
        try:
            df.to_sql(target, self.engine, **kwargs)
            console.info(f"Successfully wrote {len(df)} rows to {target}.")
        except Exception as e:
            console.error(f"SQL write 작업 실패: {e}", context={"exception": str(e)})
            raise

# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("sql", SqlAdapter)