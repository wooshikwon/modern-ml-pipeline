"""
Apache Cassandra service checker implementation
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


class CassandraChecker(BaseServiceChecker):
    """
    Apache Cassandra/ScyllaDB 연결 검사 체커.

    data_adapters.adapters.nosql.config 또는 cassandra:// URI를 기반으로
    Cassandra 클러스터 연결 상태를 검증합니다.

    지원하는 설정 형식:
    - cassandra://host:port/keyspace
    - hosts 리스트 + keyspace 설정
    - ScyllaDB 호환 설정
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 Cassandra 설정이 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: Cassandra 관련 설정이 있으면 True
        """
        return self._has_cassandra_uri(config) or self._has_cassandra_nosql_config(
            config
        )

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        Cassandra 클러스터 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: Cassandra 연결 검사 결과
        """
        try:
            # cassandra-driver를 사용한 실제 연결 테스트
            return self._test_cassandra_connection(config)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Cassandra 드라이버가 설치되지 않음",
                recommendations=[
                    "cassandra-driver 설치: pip install cassandra-driver",
                    "또는 uv add cassandra-driver",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Cassandra 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "Cassandra"

    def _has_cassandra_uri(self, config: Dict[str, Any]) -> bool:
        """Config에 cassandra:// URI가 있는지 확인."""
        connection_uri = self._get_connection_uri(config)
        return connection_uri is not None and connection_uri.startswith("cassandra://")

    def _has_cassandra_nosql_config(self, config: Dict[str, Any]) -> bool:
        """Config에 NoSQL Cassandra 설정이 있는지 확인."""
        nosql_config = self._get_nosql_config(config)
        return nosql_config and nosql_config.get("type", "").lower() in [
            "cassandra",
            "scylla",
        ]

    def _get_connection_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 connection_uri 추출."""
        try:
            return (
                config.get("data_adapters", {})
                .get("adapters", {})
                .get("nosql", {})
                .get("config", {})
                .get("connection_uri")
            )
        except (AttributeError, TypeError):
            return None

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

    def _test_cassandra_connection(self, config: Dict[str, Any]) -> CheckResult:
        """실제 Cassandra 연결 테스트."""
        try:
            from cassandra.cluster import Cluster, NoHostAvailable
            from cassandra.auth import PlainTextAuthProvider
            from cassandra import OperationTimedOut, InvalidRequest

            # 연결 정보 추출
            connection_info = self._extract_connection_info(config)

            if not connection_info["hosts"]:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message="Cassandra 호스트 정보가 없음",
                    recommendations=["config에 Cassandra 호스트 정보를 설정하세요"],
                )

            # Cluster 생성 및 연결
            cluster_kwargs = {"contact_points": connection_info["hosts"]}

            if connection_info.get("port"):
                cluster_kwargs["port"] = connection_info["port"]

            if connection_info.get("username") and connection_info.get("password"):
                auth_provider = PlainTextAuthProvider(
                    username=connection_info["username"],
                    password=connection_info["password"],
                )
                cluster_kwargs["auth_provider"] = auth_provider

            cluster = Cluster(**cluster_kwargs)

            # 연결 테스트
            with cluster:
                session = cluster.connect()

                # 간단한 시스템 쿼리로 연결 확인
                result = session.execute("SELECT release_version FROM system.local")
                version = result.one()[0] if result else "Unknown"

                # 클러스터 정보 수집
                host_count = len(cluster.metadata.all_hosts())
                keyspaces = list(cluster.metadata.keyspaces.keys())

                keyspace_info = ""
                if connection_info.get("keyspace"):
                    if connection_info["keyspace"] in keyspaces:
                        keyspace_info = (
                            f"키스페이스 '{connection_info['keyspace']}' 접근 가능"
                        )
                    else:
                        keyspace_info = (
                            f"키스페이스 '{connection_info['keyspace']}' 없음"
                        )

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"Cassandra 클러스터 연결 성공: {connection_info['hosts'][0]}",
                    details=[
                        f"클러스터 노드 수: {host_count}",
                        f"Cassandra 버전: {version}",
                        f"키스페이스 수: {len(keyspaces)}",
                        (
                            keyspace_info
                            if keyspace_info
                            else f"사용 가능한 키스페이스: {', '.join(keyspaces[:3])}..."
                        ),
                    ],
                )

        except NoHostAvailable as e:
            error_details = []
            for host, error in e.errors.items():
                error_details.append(f"{host}: {error}")

            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Cassandra 클러스터에 연결할 수 없음",
                details=error_details[:3],  # 최대 3개 호스트 오류만 표시
                recommendations=[
                    "Cassandra 클러스터가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                    "호스트 주소와 포트가 올바른지 확인하세요",
                ],
            )
        except OperationTimedOut:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Cassandra 클러스터 응답 시간 초과",
                recommendations=[
                    "Cassandra 클러스터 응답 속도를 확인하세요",
                    "네트워크 지연을 확인하세요",
                    "클러스터 부하 상태를 확인하세요",
                ],
            )
        except InvalidRequest as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Cassandra 요청 오류: {str(e)}",
                recommendations=[
                    "CQL 쿼리 권한을 확인하세요",
                    "키스페이스 접근 권한을 확인하세요",
                ],
            )
        except Exception as e:
            error_message = str(e)

            if "authentication" in error_message.lower():
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"Cassandra 인증 실패: {error_message}",
                    recommendations=[
                        "Cassandra 사용자명과 비밀번호를 확인하세요",
                        "Cassandra 인증 설정을 확인하세요",
                    ],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"Cassandra 연결 테스트 중 예상치 못한 오류: {error_message}",
                    recommendations=self._generate_error_recommendations(error_message),
                )

    def _extract_connection_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Config에서 Cassandra 연결 정보 추출."""
        connection_info = {"hosts": ["127.0.0.1"], "port": 9042}

        # 1. URI 방식 (cassandra://host:port/keyspace)
        connection_uri = self._get_connection_uri(config)
        if connection_uri and connection_uri.startswith("cassandra://"):
            parsed = urlparse(connection_uri)
            connection_info.update(
                {
                    "hosts": [parsed.hostname] if parsed.hostname else ["127.0.0.1"],
                    "port": parsed.port or 9042,
                    "keyspace": parsed.path.lstrip("/") if parsed.path else None,
                    "username": parsed.username,
                    "password": parsed.password,
                }
            )

        # 2. NoSQL config 방식
        nosql_config = self._get_nosql_config(config)
        if nosql_config:
            if "hosts" in nosql_config:
                hosts = nosql_config["hosts"]
                if isinstance(hosts, str):
                    connection_info["hosts"] = [hosts]
                elif isinstance(hosts, list):
                    connection_info["hosts"] = hosts

            connection_info.update(
                {
                    "port": nosql_config.get("port", 9042),
                    "keyspace": nosql_config.get("keyspace"),
                    "username": nosql_config.get("username"),
                    "password": nosql_config.get("password"),
                }
            )

        return connection_info

    def _generate_error_recommendations(self, error_message: str) -> List[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "Cassandra 클러스터가 실행 중인지 확인하세요",
            "호스트와 포트 설정을 확인하세요",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "Cassandra 클러스터 응답 시간을 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                ]
            )
        elif "auth" in error_message.lower():
            recommendations.extend(
                ["Cassandra 인증 설정을 확인하세요", "사용자명과 비밀번호를 확인하세요"]
            )
        elif "keyspace" in error_message.lower():
            recommendations.append("키스페이스가 존재하는지 확인하세요")
        elif "ssl" in error_message.lower():
            recommendations.append("SSL/TLS 설정을 확인하세요")

        return recommendations
