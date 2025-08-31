"""
Elasticsearch service checker implementation
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


class ElasticsearchChecker(BaseServiceChecker):
    """
    Elasticsearch 검색 엔진 연결 검사 체커.

    Config에서 elasticsearch:// 또는 http(s):// URI가 포함된 설정을 찾아
    Elasticsearch 클러스터 연결 상태를 검증합니다.

    지원하는 설정 형식:
    - http://localhost:9200
    - https://user:password@elasticsearch.example.com:9200
    - cloud_id를 이용한 Elastic Cloud 연결
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 Elasticsearch 설정이 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: Elasticsearch 관련 설정이 있으면 True
        """
        return self._has_elasticsearch_uri(config) or self._has_elasticsearch_config(
            config
        )

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        Elasticsearch 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: Elasticsearch 연결 검사 결과
        """
        try:
            # elasticsearch-py를 사용한 실제 연결 테스트
            return self._test_elasticsearch_connection(config)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Elasticsearch 드라이버가 설치되지 않음",
                recommendations=[
                    "elasticsearch 설치: pip install elasticsearch",
                    "또는 uv add elasticsearch",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "Elasticsearch"

    def _has_elasticsearch_uri(self, config: Dict[str, Any]) -> bool:
        """Config에서 Elasticsearch URI를 재귀적으로 찾기."""

        def _recursive_search(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    if _recursive_search(value):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _recursive_search(item):
                        return True
            elif isinstance(obj, str):
                # Elasticsearch 관련 URI 패턴 확인
                lower_str = obj.lower()
                if "elasticsearch" in lower_str or (
                    obj.startswith(("http://", "https://"))
                    and (":9200" in obj or "elasticsearch" in lower_str)
                ):
                    return True
            return False

        return _recursive_search(config)

    def _has_elasticsearch_config(self, config: Dict[str, Any]) -> bool:
        """Config에 Elasticsearch 관련 설정이 있는지 확인."""

        # search, logging, vector_store 등에서 elasticsearch 설정 찾기
        def _search_elasticsearch_keys(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if "elasticsearch" in key.lower() or "elastic" in key.lower():
                        return True
                    if _search_elasticsearch_keys(value, current_path):
                        return True
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if _search_elasticsearch_keys(item, f"{path}[{i}]"):
                        return True
            return False

        return _search_elasticsearch_keys(config)

    def _find_elasticsearch_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 첫 번째 Elasticsearch URI 찾기."""

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
            elif isinstance(obj, str):
                lower_str = obj.lower()
                if "elasticsearch" in lower_str or (
                    obj.startswith(("http://", "https://"))
                    and (":9200" in obj or "elasticsearch" in lower_str)
                ):
                    return obj
            return None

        return _recursive_search(config)

    def _test_elasticsearch_connection(self, config: Dict[str, Any]) -> CheckResult:
        """실제 Elasticsearch 연결 테스트."""
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch.exceptions import (
                ConnectionError,
                ConnectionTimeout,
                TransportError,
                AuthenticationException,
                AuthorizationException,
                SSLError,
            )

            # 연결 정보 추출
            connection_info = self._extract_connection_info(config)

            if not connection_info["hosts"]:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message="Elasticsearch 호스트 정보가 없음",
                    recommendations=["config에 Elasticsearch 호스트 정보를 설정하세요"],
                )

            # Elasticsearch 클라이언트 생성
            client_kwargs = {
                "hosts": connection_info["hosts"],
                "request_timeout": 10,  # 10초 타임아웃
                "max_retries": 1,
                "retry_on_timeout": True,
            }

            # 인증 설정
            if connection_info.get("username") and connection_info.get("password"):
                client_kwargs["basic_auth"] = (
                    connection_info["username"],
                    connection_info["password"],
                )
            elif connection_info.get("api_key"):
                client_kwargs["api_key"] = connection_info["api_key"]

            # SSL 설정
            if any(host.startswith("https://") for host in connection_info["hosts"]):
                client_kwargs["verify_certs"] = False  # 개발환경을 위한 설정

            client = Elasticsearch(**client_kwargs)

            # 클러스터 연결 테스트
            cluster_info = client.info()
            cluster_health = client.cluster.health()

            # 버전 및 클러스터 정보 추출
            es_version = cluster_info.get("version", {}).get("number", "Unknown")
            cluster_name = cluster_info.get("cluster_name", "Unknown")

            # 클러스터 상태
            cluster_status = cluster_health.get("status", "unknown")
            active_nodes = cluster_health.get("number_of_nodes", 0)

            status_color = {"green": "정상", "yellow": "경고", "red": "위험"}.get(
                cluster_status, "알 수 없음"
            )

            return CheckResult(
                is_healthy=True,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 클러스터 연결 성공: {cluster_name}",
                details=[
                    f"클러스터 상태: {status_color} ({cluster_status})",
                    f"Elasticsearch 버전: {es_version}",
                    f"활성 노드 수: {active_nodes}",
                    f"연결 호스트: {', '.join(connection_info['hosts'])}",
                ],
            )

        except ConnectionError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 연결 실패: {str(e)}",
                recommendations=[
                    "Elasticsearch 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                    "호스트 주소와 포트가 올바른지 확인하세요",
                ],
            )
        except ConnectionTimeout:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Elasticsearch 연결 시간 초과",
                recommendations=[
                    "Elasticsearch 서버 응답 시간을 확인하세요",
                    "네트워크 지연을 확인하세요",
                    "클러스터 부하 상태를 확인하세요",
                ],
            )
        except AuthenticationException as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 인증 실패: {str(e)}",
                recommendations=[
                    "사용자명과 비밀번호를 확인하세요",
                    "API 키가 유효한지 확인하세요",
                    "Elasticsearch 보안 설정을 확인하세요",
                ],
            )
        except AuthorizationException as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 권한 없음: {str(e)}",
                recommendations=[
                    "사용자 권한 설정을 확인하세요",
                    "인덱스 접근 권한을 확인하세요",
                    "클러스터 관리 권한을 확인하세요",
                ],
            )
        except SSLError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch SSL 오류: {str(e)}",
                recommendations=[
                    "SSL 인증서 설정을 확인하세요",
                    "CA 인증서 경로를 확인하세요",
                    "TLS 버전 호환성을 확인하세요",
                ],
            )
        except TransportError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 전송 오류: {str(e)}",
                recommendations=[
                    "네트워크 연결을 확인하세요",
                    "서버 구성을 확인하세요",
                    "방화벽 설정을 확인하세요",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Elasticsearch 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _extract_connection_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Config에서 Elasticsearch 연결 정보 추출."""
        connection_info = {"hosts": ["http://localhost:9200"]}

        # URI 방식으로 찾기
        elasticsearch_uri = self._find_elasticsearch_uri(config)
        if elasticsearch_uri:
            # URI에서 인증 정보 추출
            parsed = urlparse(elasticsearch_uri)

            # 호스트 정보 구성
            if parsed.hostname:
                scheme = parsed.scheme or "http"
                port = parsed.port or (9200 if scheme == "http" else 9200)
                host = f"{scheme}://{parsed.hostname}:{port}"
                connection_info["hosts"] = [host]

            # 인증 정보
            if parsed.username:
                connection_info["username"] = parsed.username
            if parsed.password:
                connection_info["password"] = parsed.password

        # 구조화된 설정에서 추가 정보 찾기
        self._extract_structured_config(config, connection_info)

        return connection_info

    def _extract_structured_config(
        self, config: Dict[str, Any], connection_info: Dict[str, Any]
    ) -> None:
        """구조화된 config에서 Elasticsearch 설정 추출."""

        def _search_elasticsearch_config(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key

                    # elasticsearch 관련 키 찾기
                    if "elasticsearch" in key.lower() or "elastic" in key.lower():
                        if isinstance(value, dict):
                            self._update_connection_from_dict(value, connection_info)
                        elif isinstance(value, str) and value.startswith(
                            ("http://", "https://")
                        ):
                            connection_info["hosts"] = [value]

                    if isinstance(value, (dict, list)):
                        _search_elasticsearch_config(value, current_path)

        _search_elasticsearch_config(config)

    def _update_connection_from_dict(
        self, es_config: Dict[str, Any], connection_info: Dict[str, Any]
    ) -> None:
        """Elasticsearch 설정 딕셔너리에서 연결 정보 업데이트."""
        if "hosts" in es_config:
            hosts = es_config["hosts"]
            if isinstance(hosts, str):
                connection_info["hosts"] = [hosts]
            elif isinstance(hosts, list):
                connection_info["hosts"] = hosts

        if "host" in es_config:
            host = es_config["host"]
            port = es_config.get("port", 9200)
            scheme = es_config.get("scheme", "http")
            connection_info["hosts"] = [f"{scheme}://{host}:{port}"]

        # 인증 정보
        if "username" in es_config:
            connection_info["username"] = es_config["username"]
        if "password" in es_config:
            connection_info["password"] = es_config["password"]
        if "api_key" in es_config:
            connection_info["api_key"] = es_config["api_key"]
        if "cloud_id" in es_config:
            connection_info["cloud_id"] = es_config["cloud_id"]

    def _generate_error_recommendations(self, error_message: str) -> List[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "Elasticsearch 서버가 실행 중인지 확인하세요",
            "연결 URI 형식을 확인하세요",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "Elasticsearch 클러스터 응답 시간을 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                ]
            )
        elif "auth" in error_message.lower() or "credential" in error_message.lower():
            recommendations.extend(
                [
                    "Elasticsearch 인증 설정을 확인하세요",
                    "사용자명과 비밀번호 또는 API 키를 확인하세요",
                ]
            )
        elif "ssl" in error_message.lower() or "tls" in error_message.lower():
            recommendations.extend(
                ["SSL/TLS 설정을 확인하세요", "인증서 설정을 확인하세요"]
            )
        elif "connection" in error_message.lower():
            recommendations.extend(
                ["네트워크 연결을 확인하세요", "방화벽 설정을 확인하세요"]
            )
        elif "transport" in error_message.lower():
            recommendations.extend(
                ["서버 구성을 확인하세요", "클러스터 상태를 확인하세요"]
            )

        return recommendations
