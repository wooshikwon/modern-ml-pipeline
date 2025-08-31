"""
MySQL service checker implementation
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


class MySQLChecker(BaseServiceChecker):
    """
    MySQL 데이터베이스 연결 검사 체커.

    data_adapters.adapters.sql.config.connection_uri 설정을 기반으로
    MySQL 서버 연결 상태를 검증합니다.

    지원하는 connection_uri 형식:
    - mysql://user:password@localhost:3306/dbname
    - mysql+pymysql://user:password@host:port/database
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 MySQL connection_uri가 설정되어 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: mysql:// 프로토콜의 connection_uri가 설정되어 있으면 True
        """
        connection_uri = self._get_connection_uri(config)
        return connection_uri is not None and connection_uri.startswith(
            ("mysql://", "mysql+pymysql://")
        )

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        MySQL 데이터베이스 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: MySQL 연결 검사 결과
        """
        connection_uri = self._get_connection_uri(config)

        if not connection_uri:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MySQL connection_uri가 설정되지 않음",
                recommendations=[
                    "config 파일에 data_adapters.adapters.sql.config.connection_uri 설정 추가"
                ],
            )

        try:
            # pymysql을 사용한 실제 연결 테스트
            return self._test_connection(connection_uri)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MySQL 드라이버(pymysql)가 설치되지 않음",
                recommendations=[
                    "pymysql 설치: pip install pymysql",
                    "또는 uv add pymysql",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MySQL 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "MySQL"

    def _get_connection_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 MySQL connection_uri 추출."""
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
        """실제 MySQL 연결 테스트."""
        try:
            import pymysql
            from pymysql import OperationalError, MySQLError

            # URI 파싱으로 연결 정보 추출
            parsed = urlparse(connection_uri)
            connection_info = {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 3306,
                "database": parsed.path.lstrip("/") if parsed.path else None,
                "user": parsed.username,
                "password": parsed.password,
            }

            # 연결 테스트
            with pymysql.connect(
                host=connection_info["host"],
                port=connection_info["port"],
                database=connection_info["database"],
                user=connection_info["user"],
                password=connection_info["password"],
                connect_timeout=10,
                charset="utf8mb4",
                autocommit=True,
            ) as connection:
                with connection.cursor() as cursor:
                    # 간단한 쿼리로 연결 확인
                    cursor.execute("SELECT VERSION()")
                    version = cursor.fetchone()[0]

                    # 데이터베이스 존재 확인
                    if connection_info["database"]:
                        cursor.execute("SELECT DATABASE()")
                        current_db = cursor.fetchone()[0]
                    else:
                        current_db = "연결됨 (DB 미지정)"

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"MySQL 연결 성공: {connection_info['host']}:{connection_info['port']}",
                    details=[
                        f"데이터베이스: {current_db}",
                        f"사용자: {connection_info['user']}",
                        f"MySQL 버전: {version[:50]}...",
                    ],
                )

        except OperationalError as e:
            error_code, error_msg = e.args
            recommendations = []

            if error_code == 2003:  # Can't connect to MySQL server
                recommendations.extend(
                    [
                        f"MySQL 서버가 실행 중인지 확인하세요: {connection_info['host']}:{connection_info['port']}",
                        "방화벽 설정을 확인하세요",
                    ]
                )
            elif error_code == 1045:  # Access denied
                recommendations.extend(
                    [
                        "MySQL 사용자명과 비밀번호를 확인하세요",
                        f"사용자 {connection_info['user']}에게 데이터베이스 접근 권한이 있는지 확인하세요",
                    ]
                )
            elif error_code == 1049:  # Unknown database
                recommendations.extend(
                    [
                        f"데이터베이스가 존재하는지 확인하세요: {connection_info['database']}",
                        f"CREATE DATABASE {connection_info['database']}; 명령으로 생성하세요",
                    ]
                )
            elif error_code == 2005:  # Unknown MySQL server host
                recommendations.extend(
                    [
                        f"MySQL 서버 호스트명을 확인하세요: {connection_info['host']}",
                        "DNS 설정이 올바른지 확인하세요",
                    ]
                )
            else:
                recommendations.append(f"MySQL 오류 코드 {error_code}를 확인하세요")

            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MySQL 연결 실패 (오류 {error_code}): {error_msg}",
                recommendations=recommendations,
            )

        except MySQLError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MySQL 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MySQL 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _generate_error_recommendations(self, error_message: str) -> list[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "MySQL connection_uri 형식을 확인하세요",
            "예: mysql://user:password@localhost:3306/dbname",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "MySQL 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                ]
            )
        elif "permission" in error_message.lower():
            recommendations.append("MySQL 사용자 권한 설정을 확인하세요")
        elif "module" in error_message.lower():
            recommendations.append("필요한 Python 패키지가 설치되어 있는지 확인하세요")
        elif "character set" in error_message.lower():
            recommendations.append("MySQL 문자 인코딩 설정을 확인하세요")

        return recommendations
