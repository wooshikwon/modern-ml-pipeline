"""
MongoDB service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from ..base import BaseServiceChecker
from ..models import CheckResult


class MongoDBChecker(BaseServiceChecker):
    """
    MongoDB 데이터베이스 연결 검사 체커.

    Config에서 mongodb:// URI가 포함된 설정을 찾아 MongoDB 서버 연결 상태를 검증합니다.

    지원하는 설정 형식:
    - mongodb://user:password@host:port/database
    - mongodb+srv://user:password@cluster.mongodb.net/database
    - data_adapters.adapters.nosql.config.connection_uri
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 MongoDB 설정이 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: MongoDB 관련 설정이 있으면 True
        """
        return self._has_mongodb_uri(config) or self._has_mongodb_nosql_config(config)

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        MongoDB 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: MongoDB 연결 검사 결과
        """
        try:
            # pymongo를 사용한 실제 연결 테스트
            return self._test_mongodb_connection(config)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MongoDB 드라이버(pymongo)가 설치되지 않음",
                recommendations=[
                    "pymongo 설치: pip install pymongo",
                    "또는 uv add pymongo",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MongoDB 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "MongoDB"

    def _has_mongodb_uri(self, config: Dict[str, Any]) -> bool:
        """Config에서 mongodb:// URI를 재귀적으로 찾기."""

        def _recursive_search(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    if _recursive_search(value):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _recursive_search(item):
                        return True
            elif isinstance(obj, str) and obj.startswith(
                ("mongodb://", "mongodb+srv://")
            ):
                return True
            return False

        return _recursive_search(config)

    def _has_mongodb_nosql_config(self, config: Dict[str, Any]) -> bool:
        """Config에 NoSQL MongoDB 설정이 있는지 확인."""
        nosql_config = self._get_nosql_config(config)
        return nosql_config and nosql_config.get("type", "").lower() == "mongodb"

    def _get_nosql_config(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Config에서 NoSQL 설정 추출."""
        try:
            return (
                config.get("data_adapters", {})
                .get("adapters", {})
                .get("nosql", {})
                .get("config", {})
            )
        except (AttributeError, TypeError):
            return None

    def _find_mongodb_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 첫 번째 mongodb:// URI 찾기."""

        def _recursive_search(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    result = _recursive_search(value)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = _recursive_search(item)
                    if result:
                        return result
            elif isinstance(obj, str) and obj.startswith(
                ("mongodb://", "mongodb+srv://")
            ):
                return obj
            return None

        return _recursive_search(config)

    def _test_mongodb_connection(self, config: Dict[str, Any]) -> CheckResult:
        """실제 MongoDB 연결 테스트."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import (
                ConnectionFailure,
                ServerSelectionTimeoutError,
                OperationFailure,
                AuthenticationFailed,
                ConfigurationError,
                NetworkTimeout,
            )

            # 연결 URI 추출
            connection_uri = self._find_mongodb_uri(config)
            if not connection_uri:
                # NoSQL config에서 연결 정보 추출
                connection_uri = self._build_mongodb_uri_from_config(config)

            if not connection_uri:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message="MongoDB 연결 정보를 찾을 수 없음",
                    recommendations=[
                        "config에 MongoDB URI 또는 연결 설정을 추가하세요"
                    ],
                )

            # MongoClient 생성 (연결 시간 제한 설정)
            client = MongoClient(
                connection_uri,
                serverSelectionTimeoutMS=10000,  # 10초
                connectTimeoutMS=10000,  # 10초
                socketTimeoutMS=10000,  # 10초
            )

            try:
                # 연결 테스트 (ping 명령 실행)
                client.admin.command("ping")

                # 서버 정보 수집
                server_info = client.server_info()
                version = server_info.get("version", "Unknown")

                # 데이터베이스 목록 (권한이 있는 경우에만)
                try:
                    db_names = client.list_database_names()
                    db_count = len(db_names)
                    db_info = f"데이터베이스 수: {db_count}"
                except OperationFailure:
                    db_info = "데이터베이스 목록 조회 권한 없음"

                # 연결된 서버 주소
                server_address = client.address

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"MongoDB 연결 성공: {server_address[0]}:{server_address[1]}",
                    details=[
                        f"MongoDB 버전: {version}",
                        db_info,
                        f"서버 주소: {server_address[0]}:{server_address[1]}",
                        f"연결 URI: {self._mask_password(connection_uri)}",
                    ],
                )

            finally:
                client.close()

        except ServerSelectionTimeoutError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MongoDB 서버를 찾을 수 없음 (서버 선택 시간 초과)",
                recommendations=[
                    "MongoDB 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                    "연결 URI의 호스트와 포트가 올바른지 확인하세요",
                    "방화벽 설정을 확인하세요",
                ],
            )
        except ConnectionFailure as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MongoDB 연결 실패: {str(e)}",
                recommendations=[
                    "MongoDB 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                    "연결 URI 형식이 올바른지 확인하세요",
                ],
            )
        except AuthenticationFailed as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MongoDB 인증 실패: {str(e)}",
                recommendations=[
                    "사용자명과 비밀번호를 확인하세요",
                    "MongoDB 사용자가 존재하는지 확인하세요",
                    "인증 데이터베이스가 올바른지 확인하세요",
                ],
            )
        except ConfigurationError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MongoDB 설정 오류: {str(e)}",
                recommendations=[
                    "MongoDB URI 형식을 확인하세요",
                    "PyMongo와 MongoDB 서버 버전 호환성을 확인하세요",
                ],
            )
        except NetworkTimeout:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MongoDB 네트워크 연결 시간 초과",
                recommendations=[
                    "네트워크 연결 상태를 확인하세요",
                    "MongoDB 서버 응답 시간을 확인하세요",
                    "방화벽이나 프록시 설정을 확인하세요",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MongoDB 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _build_mongodb_uri_from_config(self, config: Dict[str, Any]) -> Optional[str]:
        """NoSQL config에서 MongoDB URI 구성."""
        nosql_config = self._get_nosql_config(config)
        if not nosql_config:
            return None

        host = nosql_config.get("host", "localhost")
        port = nosql_config.get("port", 27017)
        database = nosql_config.get("database", "")
        username = nosql_config.get("username")
        password = nosql_config.get("password")

        # URI 구성
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}"
        else:
            uri = f"mongodb://{host}:{port}"

        if database:
            uri += f"/{database}"

        return uri

    def _mask_password(self, uri: str) -> str:
        """URI에서 비밀번호를 마스킹."""
        try:
            parsed = urlparse(uri)
            if parsed.password:
                masked_uri = uri.replace(parsed.password, "***")
                return masked_uri
        except Exception:
            pass
        return uri

    def _generate_error_recommendations(self, error_message: str) -> List[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "MongoDB URI 형식을 확인하세요",
            "예: mongodb://user:password@localhost:27017/database",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "MongoDB 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                ]
            )
        elif "auth" in error_message.lower() or "credential" in error_message.lower():
            recommendations.extend(
                [
                    "MongoDB 사용자 인증 정보를 확인하세요",
                    "인증 데이터베이스를 확인하세요",
                ]
            )
        elif "resolve" in error_message.lower() or "host" in error_message.lower():
            recommendations.extend(["호스트명을 확인하세요", "DNS 설정을 확인하세요"])
        elif "ssl" in error_message.lower() or "tls" in error_message.lower():
            recommendations.extend(
                ["SSL/TLS 설정을 확인하세요", "인증서 설정을 확인하세요"]
            )

        return recommendations
