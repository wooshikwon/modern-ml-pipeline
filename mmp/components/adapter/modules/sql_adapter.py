from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import sqlalchemy

from mmp.components.adapter.base import BaseAdapter
from mmp.settings import Settings
from mmp.utils.core.logger import log_data_debug, logger

if TYPE_CHECKING:
    from mmp.settings import Settings


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
        log_data_debug("초기화 시작", "SqlAdapter")

        self.settings = settings
        self.db_type = None  # Will be set by _create_engine
        self.engine = self._create_engine()

        log_data_debug(f"초기화 완료 - DB 타입: {self.db_type}", "SqlAdapter")

        # BigQuery 전용 플래그 및 설정
        self.use_pandas_gbq = False
        self.project_id = None
        self.dataset_id = None
        self.location = "US"

        if self.db_type == "bigquery":
            config = self.settings.config.data_source.config
            self.use_pandas_gbq = getattr(config, "use_pandas_gbq", False)
            self.project_id = getattr(config, "project_id", None)
            self.dataset_id = getattr(config, "dataset_id", None)
            self.location = getattr(config, "location", "US")

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

        if scheme in ("bigquery", "bigquery+sqlalchemy"):
            db_type = "bigquery"
            # BigQuery는 특별한 처리 필요
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_size"] = 5
            try:
                import importlib.util

                if importlib.util.find_spec("sqlalchemy_bigquery"):
                    log_data_debug("BigQuery 엔진 설정 적용", "SqlAdapter")
            except Exception:
                logger.warning("[DATA:SqlAdapter] sqlalchemy-bigquery 미설치, 기본 설정 사용")
            processed_uri = uri

        elif scheme in ("postgresql", "postgres", "postgresql+psycopg2"):
            db_type = "postgresql"
            # PostgreSQL 최적화 설정
            engine_kwargs["pool_size"] = 10
            engine_kwargs["max_overflow"] = 20
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["connect_args"] = {
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000",  # 30초 타임아웃
            }
            processed_uri = uri
            log_data_debug("PostgreSQL 엔진 설정 적용", "SqlAdapter")

        elif scheme in ("mysql", "mysql+pymysql", "mysql+mysqldb"):
            db_type = "mysql"
            # MySQL 최적화 설정
            engine_kwargs["pool_size"] = 10
            engine_kwargs["pool_recycle"] = 3600  # 1시간마다 연결 재활용
            engine_kwargs["pool_pre_ping"] = True
            processed_uri = uri
            log_data_debug("MySQL 엔진 설정 적용", "SqlAdapter")

        elif scheme == "sqlite":
            db_type = "sqlite"
            # SQLite는 연결 풀이 필요 없음
            engine_kwargs["poolclass"] = sqlalchemy.pool.StaticPool
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            processed_uri = uri
            log_data_debug("SQLite 엔진 설정 적용", "SqlAdapter")

        else:
            # 알 수 없는 스키마는 기본 SQLAlchemy 처리
            db_type = "generic"
            logger.warning(f"[DATA:SqlAdapter] 알 수 없는 DB 스키마: {scheme}, 기본 설정 사용")
            processed_uri = uri

        return db_type, processed_uri, engine_kwargs

    def _create_engine(self):
        """
        설정(Settings) 객체로부터 DB 연결 URI를 파싱하고
        데이터베이스 타입에 맞는 SQLAlchemy 엔진을 생성합니다.
        """
        try:
            # 새로운 Settings 구조: config.data_source 접근
            data_source_config = self.settings.config.data_source.config
            # Pydantic models or dicts both supported
            connection_uri = (
                data_source_config.connection_uri
                if hasattr(data_source_config, "connection_uri")
                else data_source_config["connection_uri"]
            )

            # URI 파싱하여 DB 타입과 엔진 설정 추출
            db_type, processed_uri, engine_kwargs = self._parse_connection_uri(connection_uri)
            self.db_type = db_type  # Store db_type for later use

            log_data_debug(f"DB 타입: {db_type}", "SqlAdapter")
            log_data_debug(f"엔진 생성 - URI: {processed_uri[:50]}...", "SqlAdapter")

            # 데이터베이스별 특수 처리
            if db_type == "bigquery":
                # BigQuery 인증 설정이 있는 경우 처리
                cred_path = getattr(data_source_config, "credentials_path", None)
                if cred_path:
                    import os

                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
                    log_data_debug("BigQuery 인증 파일 설정 완료", "SqlAdapter")

            # 엔진 생성
            engine = sqlalchemy.create_engine(processed_uri, **engine_kwargs)

            # 연결 테스트
            with engine.connect():
                log_data_debug(f"{db_type} 연결 테스트 성공", "SqlAdapter")

            return engine

        except KeyError as e:
            logger.error(f"[DATA:SqlAdapter] 설정 누락: {e}")
            raise ValueError(f"SQL 어댑터 설정이 누락되었습니다: {e}")
        except Exception as e:
            logger.error(f"[DATA:SqlAdapter] 엔진 생성 실패: {e}")
            raise

    def _execute_bigquery_with_progress(self, sql_query: str, **kwargs) -> pd.DataFrame:
        """BigQuery 쿼리를 실행하며 진행 상황을 표시합니다.

        google-cloud-bigquery를 직접 사용하여 job 상태를 모니터링하고,
        주기적으로 진행 상황 로그를 출력합니다.
        """
        try:
            from google.cloud import bigquery

            # BigQuery 클라이언트 생성
            client = bigquery.Client(project=self.project_id, location=self.location)

            log_data_debug("BigQuery 쿼리 실행 시작 (진행 상황 모니터링 활성화)", "SqlAdapter")

            # 쿼리 job 생성 및 실행
            query_job = client.query(sql_query)
            job_id = query_job.job_id
            log_data_debug(f"BigQuery Job 생성됨: {job_id[:20]}...", "SqlAdapter")

            # job 완료 대기 및 진행 상황 로깅
            start_time = time.time()
            last_log_time = start_time
            log_interval = 10  # 10초마다 로그

            while not query_job.done():
                time.sleep(1)
                elapsed = time.time() - start_time

                # 주기적 진행 상황 로그
                if time.time() - last_log_time >= log_interval:
                    state = query_job.state
                    log_data_debug(
                        f"BigQuery 쿼리 실행 중... ({elapsed:.0f}초 경과, 상태: {state})",
                        "SqlAdapter",
                    )
                    last_log_time = time.time()

            # 쿼리 완료 후 결과 조회
            query_elapsed = time.time() - start_time
            log_data_debug(
                f"BigQuery 쿼리 완료 ({query_elapsed:.1f}초 소요), 결과 다운로드 중...", "SqlAdapter"
            )

            # Storage API로 DataFrame 변환 (REST API 대비 10배 이상 빠름)
            download_start = time.time()
            try:
                result = query_job.to_dataframe(create_bqstorage_client=True)
            except Exception:
                result = query_job.to_dataframe()

            download_elapsed = time.time() - download_start
            log_data_debug(f"결과 다운로드 완료 ({download_elapsed:.1f}초 소요)", "SqlAdapter")

            return result

        except ImportError as e:
            # google-cloud-bigquery 미설치 시 에러
            missing_pkg = str(e)
            logger.error(f"[DATA:SqlAdapter] BigQuery 필수 패키지 미설치: {missing_pkg}")
            raise ImportError(
                f"BigQuery 실행에 필요한 패키지가 설치되지 않았습니다.\n"
                f"오류: {missing_pkg}\n\n"
                f"해결 방법:\n"
                f"  pip install 'modern-ml-pipeline[cloud-extras]'\n"
                f"  또는: pip install google-cloud-bigquery db-dtypes"
            ) from e

        except Exception as e:
            # BigQuery 직접 실행 실패 시 에러 (db-dtypes 등)
            error_msg = str(e)
            logger.error(f"[DATA:SqlAdapter] BigQuery 실행 실패: {error_msg}")

            # db-dtypes 관련 에러인 경우 명확한 안내
            if "db-dtypes" in error_msg or "db_dtypes" in error_msg:
                raise RuntimeError(
                    f"BigQuery 결과 변환에 필요한 패키지가 설치되지 않았습니다.\n"
                    f"오류: {error_msg}\n\n"
                    f"해결 방법:\n"
                    f"  pip install 'modern-ml-pipeline[cloud-extras]'\n"
                    f"  또는: pip install db-dtypes"
                ) from e
            else:
                raise RuntimeError(f"BigQuery 쿼리 실행 실패: {error_msg}") from e

    def _enforce_sql_guards(self, sql_query: str) -> None:
        """보안/신뢰성 가드 적용: DDL/DML 금칙어 차단, LIMIT 가드(옵션)."""
        upper = sql_query.upper()

        # DDL/DML 금칙어 차단(간단한 포함 검사)
        DANGEROUS = [
            "DROP ",
            "DELETE ",
            "UPDATE ",
            "INSERT ",
            "ALTER ",
            "TRUNCATE ",
            "CREATE ",
            "EXEC ",
            "EXECUTE ",
            "MERGE ",
            "GRANT ",
            "REVOKE ",
        ]
        upper = sql_query.upper()
        for token in DANGEROUS:
            if token in upper:
                raise ValueError(f"보안 위반: 금지된 SQL 키워드 포함({token.strip()}).")

        # LIMIT 가드: 대용량 결과 방지를 위한 디버그 로그
        # CTE 내부 LIMIT, 서브쿼리 등 다양한 패턴이 있으므로 DEBUG 레벨로 처리
        if " LIMIT " not in upper:
            log_data_debug("LIMIT 절 미포함 - 대용량 쿼리 가능성", "SqlAdapter")

    def read(self, source: str, params: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """SQL 쿼리를 실행하여 결과를 DataFrame으로 반환합니다.
        BigQuery의 경우 pandas_gbq를 사용할 수 있습니다.

        Args:
            source: SQL 쿼리 문자열 또는 .sql 파일 경로
            params: 쿼리 파라미터 (Optional)
        """
        log_data_debug(f"SQL 쿼리 시작 - DB: {self.db_type}", "SqlAdapter")

        # BigQuery + pandas_gbq 사용 시
        if self.db_type == "bigquery" and self.use_pandas_gbq:
            try:
                import pandas_gbq

                log_data_debug(
                    f"BigQuery pandas_gbq 사용 - 프로젝트: {self.project_id}", "SqlAdapter"
                )
                result = pandas_gbq.read_gbq(
                    source, project_id=self.project_id, location=self.location, **kwargs
                )
                log_data_debug(
                    f"BigQuery 로딩 완료: {len(result)}행 × {len(result.columns)}열", "SqlAdapter"
                )
                return result
            except ImportError:
                logger.warning("[DATA:SqlAdapter] pandas_gbq 미설치, SQLAlchemy 폴백")
            except Exception as e:
                logger.warning(f"[DATA:SqlAdapter] pandas_gbq 실패, SQLAlchemy 폴백: {e}")

        # 기존 SQLAlchemy 방식 (기본값 및 fallback)
        sql_query = source  # Rename for compatibility

        # 파일 경로인지 확인 (.sql 또는 .sql.j2 확장자로 끝나는 경우)
        is_jinja_template = sql_query.endswith(".sql.j2")
        is_sql_file = sql_query.endswith(".sql") or is_jinja_template

        if is_sql_file:
            sql_file_path = Path(sql_query)
            if not sql_file_path.is_absolute():
                # mmp/components/adapter/modules/sql_adapter.py → 5단계 상위 = modern-ml-pipeline/
                base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
                sql_file_path = base_dir / sql_query

            if sql_file_path.exists():
                sql_query = sql_file_path.read_text(encoding="utf-8")
                log_data_debug(f"SQL 파일 로딩: {sql_file_path.name}", "SqlAdapter")

                # Jinja2 템플릿 렌더링
                if is_jinja_template:
                    from jinja2 import Template

                    template = Template(sql_query)
                    sql_query = template.render(**(params or {}))
                    log_data_debug("Jinja2 템플릿 렌더링 완료", "SqlAdapter")
                    params = None  # 템플릿 렌더링 후 params는 사용하지 않음
            else:
                logger.error(f"[DATA:SqlAdapter] SQL 파일 없음: {sql_file_path}")
                raise FileNotFoundError(f"SQL 파일을 찾을 수 없습니다: {sql_file_path}")

        # 보안 가드 적용
        self._enforce_sql_guards(sql_query)

        log_data_debug(
            f"쿼리 실행 시작 - 길이: {len(sql_query)}자, 파라미터: {len(params or {})}개",
            "SqlAdapter",
        )

        if params:
            log_data_debug(f"파라미터 바인딩: {len(params)}개", "SqlAdapter")

        try:
            start_time = time.time()

            if self.db_type == "bigquery":
                result = self._execute_bigquery_with_progress(sql_query, **kwargs)
            else:
                result = pd.read_sql_query(sql_query, self.engine, params=params, **kwargs)

            elapsed = time.time() - start_time
            data_size_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
            log_data_debug(
                f"쿼리 완료 ({elapsed:.1f}초): {len(result):,}행 × {len(result.columns)}열, "
                f"{data_size_mb:.1f}MB",
                "SqlAdapter",
            )

            # 데이터 품질 간단 체크
            null_counts = result.isnull().sum()
            if null_counts.sum() > 0:
                null_cols = null_counts[null_counts > 0]
                log_data_debug(
                    f"결측값: {len(null_cols)}개 컬럼, 총 {null_counts.sum():,}개", "SqlAdapter"
                )

            return result
        except Exception as e:
            snippet = sql_query[:100].replace("\n", " ")
            logger.error(f"[DATA:SqlAdapter] 쿼리 실패: {e}, 쿼리: {snippet}...")
            raise

    def write(self, df: pd.DataFrame, target: str, **kwargs):
        """DataFrame을 지정된 테이블에 씁니다.
        BigQuery의 경우 pandas_gbq를 사용할 수 있습니다.

        Args:
            df: 저장할 DataFrame
            target: 대상 테이블 이름
        """
        data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        log_data_debug(f"테이블 저장 시작: {target}, {len(df)}행, {data_size_mb:.1f}MB", "SqlAdapter")

        # BigQuery 전용 처리
        if self.db_type == "bigquery" and (
            self.use_pandas_gbq or kwargs.get("if_exists") == "replace"
        ):
            try:
                import pandas_gbq

                # dataset_id가 있으면 테이블명에 추가
                destination_table = (
                    f"{self.dataset_id}.{target}"
                    if self.dataset_id and "." not in target
                    else target
                )

                log_data_debug(
                    f"BigQuery pandas_gbq 저장 - 목적지: {destination_table}", "SqlAdapter"
                )
                pandas_gbq.to_gbq(
                    df,
                    destination_table=destination_table,
                    project_id=self.project_id,
                    location=self.location,
                    if_exists=kwargs.get("if_exists", "append"),
                    **{k: v for k, v in kwargs.items() if k not in ["if_exists"]},
                )
                log_data_debug(f"BigQuery 저장 완료: {len(df):,}행 → {destination_table}", "SqlAdapter")
                return
            except ImportError:
                logger.warning("[DATA:SqlAdapter] pandas_gbq 미설치, SQLAlchemy 폴백")
            except Exception as e:
                logger.warning(f"[DATA:SqlAdapter] pandas_gbq 저장 실패, SQLAlchemy 폴백: {e}")

        # 기존 SQLAlchemy 방식 (기본값 및 fallback)
        log_data_debug(f"SQLAlchemy 저장 - 테이블: {target}", "SqlAdapter")
        try:
            write_mode = kwargs.get("if_exists", "fail")
            log_data_debug(f"쓰기 모드: {write_mode}, {len(df):,}행 처리", "SqlAdapter")
            df.to_sql(target, self.engine, **kwargs)
            log_data_debug(f"테이블 저장 완료: {len(df):,}행 → {target}", "SqlAdapter")
        except Exception as e:
            logger.error(f"[DATA:SqlAdapter] 테이블 저장 실패: {target}, 오류: {e}")
            raise


# Self-registration
from ..registry import AdapterRegistry

AdapterRegistry.register("sql", SqlAdapter)
