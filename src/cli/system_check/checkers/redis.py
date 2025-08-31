"""
Redis service checker implementation
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


class RedisChecker(BaseServiceChecker):
    """
    Redis 캐시 서버 연결 검사 체커.

    Config에서 redis:// URI가 포함된 설정을 찾아 Redis 서버 연결 상태를 검증합니다.

    지원하는 설정 형식:
    - redis://localhost:6379/0
    - redis://user:password@localhost:6379/0
    - rediss://localhost:6380/0 (SSL)
    - data_adapters.adapters.cache.config 설정
    - feature_store.feast_config.online_store 설정
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 Redis 설정이 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: Redis 관련 설정이 있으면 True
        """
        return (
            self._has_redis_uri(config)
            or self._has_redis_cache_config(config)
            or self._has_redis_online_store_config(config)
        )

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        Redis 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: Redis 연결 검사 결과
        """
        try:
            # redis-py를 사용한 실제 연결 테스트
            return self._test_redis_connection(config)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Redis 드라이버(redis-py)가 설치되지 않음",
                recommendations=["redis 설치: pip install redis", "또는 uv add redis"],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "Redis"

    def _has_redis_uri(self, config: Dict[str, Any]) -> bool:
        """Config에서 redis:// URI를 재귀적으로 찾기."""

        def _recursive_search(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    if _recursive_search(value):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _recursive_search(item):
                        return True
            elif isinstance(obj, str) and obj.startswith(("redis://", "rediss://")):
                return True
            return False

        return _recursive_search(config)

    def _has_redis_cache_config(self, config: Dict[str, Any]) -> bool:
        """Config에 Redis 캐시 설정이 있는지 확인."""
        cache_config = self._get_cache_config(config)
        return cache_config and cache_config.get("type", "").lower() == "redis"

    def _has_redis_online_store_config(self, config: Dict[str, Any]) -> bool:
        """Config에 Redis online_store 설정이 있는지 확인."""
        online_store_config = self._get_online_store_config(config)
        return (
            online_store_config
            and online_store_config.get("type", "").lower() == "redis"
        )

    def _get_cache_config(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Config에서 캐시 설정 추출."""
        try:
            return (
                config.get("data_adapters", {})
                .get("adapters", {})
                .get("cache", {})
                .get("config", {})
            )
        except (AttributeError, TypeError):
            return None

    def _get_online_store_config(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Config에서 online_store 설정 추출."""
        try:
            return (
                config.get("feature_store", {})
                .get("feast_config", {})
                .get("online_store", {})
            )
        except (AttributeError, TypeError):
            return None

    def _find_redis_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 첫 번째 redis:// URI 찾기."""

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
            elif isinstance(obj, str) and obj.startswith(("redis://", "rediss://")):
                return obj
            return None

        return _recursive_search(config)

    def _test_redis_connection(self, config: Dict[str, Any]) -> CheckResult:
        """실제 Redis 연결 테스트."""
        try:
            import redis
            from redis.exceptions import (
                ConnectionError,
                TimeoutError,
                AuthenticationError,
                ResponseError,
                RedisError,
            )

            # 연결 정보 추출
            connection_info = self._extract_connection_info(config)

            if not connection_info:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message="Redis 연결 정보를 찾을 수 없음",
                    recommendations=["config에 Redis URI 또는 연결 설정을 추가하세요"],
                )

            # Redis 클라이언트 생성
            try:
                if connection_info.get("connection_uri"):
                    client = redis.from_url(
                        connection_info["connection_uri"],
                        decode_responses=True,
                        socket_connect_timeout=10,
                        socket_timeout=10,
                        health_check_interval=30,
                    )
                else:
                    client_kwargs = {
                        "host": connection_info.get("host", "localhost"),
                        "port": connection_info.get("port", 6379),
                        "db": connection_info.get("db", 0),
                        "decode_responses": True,
                        "socket_connect_timeout": 10,
                        "socket_timeout": 10,
                        "health_check_interval": 30,
                    }

                    if connection_info.get("password"):
                        client_kwargs["password"] = connection_info["password"]
                    if connection_info.get("username"):
                        client_kwargs["username"] = connection_info["username"]

                    client = redis.Redis(**client_kwargs)

                # 연결 테스트
                client.ping()

                # 서버 정보 수집
                server_info = client.info()
                redis_version = server_info.get("redis_version", "Unknown")
                used_memory = server_info.get("used_memory_human", "Unknown")
                connected_clients = server_info.get("connected_clients", 0)

                # 데이터베이스 키 수 확인 (권한이 있는 경우)
                try:
                    db_size = client.dbsize()
                    db_info = f"DB 키 수: {db_size}"
                except ResponseError:
                    db_info = "DB 정보 조회 권한 없음"

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"Redis 서버 연결 성공: {connection_info.get('host', 'localhost')}:{connection_info.get('port', 6379)}",
                    details=[
                        f"Redis 버전: {redis_version}",
                        f"사용 메모리: {used_memory}",
                        f"연결된 클라이언트 수: {connected_clients}",
                        db_info,
                        f"연결 URI: {self._mask_password(connection_info.get('connection_uri', ''))}",
                    ],
                )

            finally:
                # 연결 정리
                try:
                    client.close()
                except Exception:
                    pass

        except ConnectionError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 연결 실패: {str(e)}",
                recommendations=[
                    "Redis 서버가 실행 중인지 확인하세요",
                    "네트워크 연결 상태를 확인하세요",
                    "호스트 주소와 포트가 올바른지 확인하세요",
                    "방화벽 설정을 확인하세요",
                ],
            )
        except AuthenticationError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 인증 실패: {str(e)}",
                recommendations=[
                    "Redis 비밀번호를 확인하세요",
                    "사용자명이 올바른지 확인하세요",
                    "Redis AUTH 설정을 확인하세요",
                ],
            )
        except TimeoutError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Redis 연결 시간 초과",
                recommendations=[
                    "Redis 서버 응답 시간을 확인하세요",
                    "네트워크 지연을 확인하세요",
                    "서버 부하 상태를 확인하세요",
                ],
            )
        except ResponseError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 응답 오류: {str(e)}",
                recommendations=[
                    "Redis 명령어 권한을 확인하세요",
                    "데이터베이스 번호가 올바른지 확인하세요",
                    "Redis 설정을 확인하세요",
                ],
            )
        except RedisError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"Redis 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _extract_connection_info(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Config에서 Redis 연결 정보 추출."""
        connection_info = {}

        # 1. URI 방식으로 찾기
        redis_uri = self._find_redis_uri(config)
        if redis_uri:
            connection_info["connection_uri"] = redis_uri
            # URI에서 기본 정보도 추출
            parsed = urlparse(redis_uri)
            connection_info.update(
                {
                    "host": parsed.hostname or "localhost",
                    "port": parsed.port or 6379,
                    "db": (
                        int(parsed.path.lstrip("/"))
                        if parsed.path.lstrip("/").isdigit()
                        else 0
                    ),
                    "username": parsed.username,
                    "password": parsed.password,
                }
            )
            return connection_info

        # 2. 캐시 설정 방식
        cache_config = self._get_cache_config(config)
        if cache_config:
            connection_info.update(
                {
                    "host": cache_config.get("host", "localhost"),
                    "port": cache_config.get("port", 6379),
                    "db": cache_config.get("db", 0),
                    "password": cache_config.get("password"),
                    "username": cache_config.get("username"),
                }
            )
            return connection_info

        # 3. online_store 설정 방식
        online_store_config = self._get_online_store_config(config)
        if (
            online_store_config
            and online_store_config.get("type", "").lower() == "redis"
        ):
            connection_info.update(
                {
                    "host": online_store_config.get("host", "localhost"),
                    "port": online_store_config.get("port", 6379),
                    "db": online_store_config.get("db", 0),
                    "password": online_store_config.get("password"),
                    "username": online_store_config.get("username"),
                }
            )

            # connection_string 처리
            connection_string = online_store_config.get("connection_string")
            if connection_string and connection_string.startswith(
                ("redis://", "rediss://")
            ):
                connection_info["connection_uri"] = connection_string

            return connection_info

        # 4. 기타 Redis 관련 설정 찾기
        redis_settings = self._find_redis_settings(config)
        if redis_settings:
            return redis_settings

        return None

    def _find_redis_settings(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Config에서 Redis 관련 설정을 재귀적으로 찾기."""

        def _search_redis_config(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key

                    # redis 관련 키 찾기
                    if "redis" in key.lower():
                        if isinstance(value, dict):
                            # Redis 설정 딕셔너리 발견
                            return {
                                "host": value.get("host", "localhost"),
                                "port": value.get("port", 6379),
                                "db": value.get("db", 0),
                                "password": value.get("password"),
                                "username": value.get("username"),
                            }
                        elif isinstance(value, str) and value.startswith(
                            ("redis://", "rediss://")
                        ):
                            return {"connection_uri": value}

                    if isinstance(value, (dict, list)):
                        result = _search_redis_config(value, current_path)
                        if result:
                            return result
            return None

        return _search_redis_config(config)

    def _mask_password(self, uri: str) -> str:
        """URI에서 비밀번호를 마스킹."""
        if not uri:
            return ""
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
            "Redis 서버가 실행 중인지 확인하세요",
            "연결 URI 형식을 확인하세요: redis://localhost:6379/0",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                ["Redis 서버 응답 시간을 확인하세요", "네트워크 연결 상태를 확인하세요"]
            )
        elif "auth" in error_message.lower() or "credential" in error_message.lower():
            recommendations.extend(
                ["Redis 인증 설정을 확인하세요", "비밀번호가 올바른지 확인하세요"]
            )
        elif "connection" in error_message.lower():
            recommendations.extend(
                [
                    "네트워크 연결을 확인하세요",
                    "방화벽 설정을 확인하세요",
                    "Redis 서버 포트를 확인하세요",
                ]
            )
        elif "ssl" in error_message.lower() or "tls" in error_message.lower():
            recommendations.extend(
                ["SSL/TLS 설정을 확인하세요", "인증서 설정을 확인하세요"]
            )
        elif "permission" in error_message.lower() or "denied" in error_message.lower():
            recommendations.extend(
                ["Redis ACL 권한을 확인하세요", "사용자 권한 설정을 확인하세요"]
            )

        return recommendations
