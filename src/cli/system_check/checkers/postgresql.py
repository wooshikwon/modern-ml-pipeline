"""
PostgreSQL service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional
from urllib.parse import urlparse

from ..base import BaseServiceChecker
from ..models import CheckResult


class PostgreSQLChecker(BaseServiceChecker):
    """
    PostgreSQL 데이터베이스 연결 검사 체커.

    data_adapters.adapters.sql.config.connection_uri 설정을 기반으로
    PostgreSQL 서버 연결 상태를 검증합니다.

    지원하는 connection_uri 형식:
    - postgresql://user:password@localhost:5432/dbname
    - postgresql+psycopg2://user:password@host:port/database
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 PostgreSQL connection_uri가 설정되어 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: postgresql:// 프로토콜의 connection_uri가 설정되어 있으면 True
        """
        connection_uri = self._get_connection_uri(config)
        return connection_uri is not None and connection_uri.startswith(
            ("postgresql://", "postgresql+psycopg2://")
        )

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        PostgreSQL 데이터베이스 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: PostgreSQL 연결 검사 결과
        """
        connection_uri = self._get_connection_uri(config)

        if not connection_uri:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="PostgreSQL connection_uri가 설정되지 않음",
                recommendations=[
                    "config 파일에 data_adapters.adapters.sql.config.connection_uri 설정 추가"
                ],
            )

        try:
            # psycopg2를 사용한 실제 연결 테스트
            return self._test_connection(connection_uri)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="PostgreSQL 드라이버(psycopg2)가 설치되지 않음",
                recommendations=[
                    "psycopg2 설치: pip install psycopg2-binary",
                    "또는 uv add psycopg2-binary",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"PostgreSQL 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "PostgreSQL"

    def _get_connection_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 PostgreSQL connection_uri 추출."""
        try:
            return (
                config.get("data_adapters", {})
                .get("adapters", {})
                .get("sql", {})
                .get("config", {})
                .get("connection_uri")
            )
        except (AttributeError, TypeError):
            return None

    def _test_connection(self, connection_uri: str) -> CheckResult:
        """실제 PostgreSQL 연결 테스트."""
        try:
            import psycopg2
            from psycopg2 import OperationalError

            # URI 파싱으로 연결 정보 추출
            parsed = urlparse(connection_uri)
            connection_info = {
                "host": parsed.hostname,
                "port": parsed.port or 5432,
                "database": parsed.path.lstrip("/") if parsed.path else None,
                "user": parsed.username,
                "password": parsed.password,
            }

            # 연결 테스트
            with psycopg2.connect(
                host=connection_info["host"],
                port=connection_info["port"],
                database=connection_info["database"],
                user=connection_info["user"],
                password=connection_info["password"],
                connect_timeout=10,
            ) as conn:
                with conn.cursor() as cur:
                    # 간단한 쿼리로 연결 확인
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"PostgreSQL 연결 성공: {connection_info['host']}:{connection_info['port']}",
                    details=[
                        f"데이터베이스: {connection_info['database']}",
                        f"사용자: {connection_info['user']}",
                        f"버전: {version[:50]}...",  # 버전 정보 일부만
                    ],
                )

        except OperationalError as e:
            error_msg = str(e)
            recommendations = []

            if "could not connect to server" in error_msg:
                recommendations.extend(
                    [
                        f"PostgreSQL 서버가 실행 중인지 확인하세요: {connection_info['host']}:{connection_info['port']}",
                        "방화벽 설정을 확인하세요",
                    ]
                )
            elif "authentication failed" in error_msg:
                recommendations.extend(
                    [
                        "데이터베이스 사용자명과 비밀번호를 확인하세요",
                        "PostgreSQL pg_hba.conf 설정을 확인하세요",
                    ]
                )
            elif "does not exist" in error_msg:
                recommendations.extend(
                    [
                        f"데이터베이스가 존재하는지 확인하세요: {connection_info['database']}",
                        f"CREATE DATABASE {connection_info['database']}; 명령으로 생성하세요",
                    ]
                )
            else:
                recommendations.append("PostgreSQL 서버 상태와 설정을 확인하세요")

            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"PostgreSQL 연결 실패: {error_msg}",
                recommendations=recommendations,
            )

        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"PostgreSQL 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _generate_error_recommendations(self, error_message: str) -> list[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "PostgreSQL connection_uri 형식을 확인하세요",
            "예: postgresql://user:password@localhost:5432/dbname",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "PostgreSQL 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                ]
            )
        elif "permission" in error_message.lower():
            recommendations.append("PostgreSQL 권한 설정을 확인하세요")
        elif "module" in error_message.lower():
            recommendations.append("필요한 Python 패키지가 설치되어 있는지 확인하세요")

        return recommendations
