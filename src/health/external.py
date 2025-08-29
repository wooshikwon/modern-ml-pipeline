"""
External Services Health Check Implementation
Blueprint v17.0 - External service connectivity validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 예외 처리 및 로깅
"""

import time
import subprocess
from typing import Optional

try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.health.models import CheckResult, CheckCategory, HealthCheckError, ConnectionTestResult, HealthCheckConfig


class ExternalServicesHealthCheck:
    """
    외부 서비스 연결성 검사를 수행하는 클래스.
    
    PostgreSQL, Redis, Feast Feature Store 등의 연결 상태를 확인합니다.
    """
    
    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """
        ExternalServicesHealthCheck 인스턴스를 초기화합니다.
        
        Args:
            config: 건강 검사 설정
        """
        self.category = CheckCategory.EXTERNAL_SERVICES
        self.config = config or HealthCheckConfig()
    
    def check_postgresql(self) -> CheckResult:
        """
        PostgreSQL 데이터베이스 연결을 확인합니다.
        
        Returns:
            CheckResult: PostgreSQL 연결 검사 결과
        """
        if not PSYCOPG_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="psycopg 패키지가 설치되지 않음",
                details=["❌ PostgreSQL 드라이버를 찾을 수 없음"],
                recommendations=["uv add psycopg[binary]"]
            )
        
        try:
            connection_result = self._test_postgresql_connection()
            
            if connection_result.is_connected:
                return CheckResult(
                    is_healthy=True,
                    message=f"PostgreSQL 연결 성공 ({connection_result.response_time_ms:.1f}ms)",
                    details=[
                        f"호스트: {self.config.postgres_host}:{self.config.postgres_port}",
                        f"데이터베이스: {self.config.postgres_database}",
                        f"응답 시간: {connection_result.response_time_ms:.1f}ms",
                        f"성능: {connection_result.performance_rating}",
                        f"서버 버전: {connection_result.service_version or 'Unknown'}",
                        "✅ 데이터베이스 연결 정상"
                    ]
                )
            else:
                recommendations = [
                    "PostgreSQL 서버가 실행 중인지 확인하세요",
                    f"연결 설정 확인: {self.config.postgres_host}:{self.config.postgres_port}",
                    "데이터베이스 및 사용자 권한 확인",
                    "네트워크 연결 및 방화벽 설정 확인",
                    "환경 변수 설정: MMP_HEALTH_POSTGRES_HOST, MMP_HEALTH_POSTGRES_DATABASE"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="PostgreSQL 연결 실패",
                    details=[
                        f"호스트: {self.config.postgres_host}:{self.config.postgres_port}",
                        f"오류: {connection_result.error_message}",
                        "❌ 데이터베이스 접근 불가"
                    ],
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"PostgreSQL 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_redis(self) -> CheckResult:
        """
        Redis 서버 연결을 확인합니다.
        
        Returns:
            CheckResult: Redis 연결 검사 결과
        """
        if not REDIS_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="redis 패키지가 설치되지 않음",
                details=["❌ Redis 클라이언트를 찾을 수 없음"],
                recommendations=["uv add redis"]
            )
        
        try:
            connection_result = self._test_redis_connection()
            
            if connection_result.is_connected:
                return CheckResult(
                    is_healthy=True,
                    message=f"Redis 연결 성공 ({connection_result.response_time_ms:.1f}ms)",
                    details=[
                        f"호스트: {self.config.redis_host}:{self.config.redis_port}",
                        f"응답 시간: {connection_result.response_time_ms:.1f}ms",
                        f"성능: {connection_result.performance_rating}",
                        f"서버 버전: {connection_result.service_version or 'Unknown'}",
                        "✅ 캐시 서버 연결 정상"
                    ]
                )
            else:
                recommendations = [
                    "Redis 서버가 실행 중인지 확인하세요",
                    f"연결 설정 확인: {self.config.redis_host}:{self.config.redis_port}",
                    "Redis 서버 상태 확인: redis-cli ping",
                    "네트워크 연결 및 방화벽 설정 확인",
                    "환경 변수 설정: MMP_HEALTH_REDIS_HOST, MMP_HEALTH_REDIS_PORT"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="Redis 연결 실패",
                    details=[
                        f"호스트: {self.config.redis_host}:{self.config.redis_port}",
                        f"오류: {connection_result.error_message}",
                        "❌ 캐시 서버 접근 불가"
                    ],
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"Redis 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_feast(self) -> CheckResult:
        """
        Feast Feature Store 상태를 확인합니다.
        
        Returns:
            CheckResult: Feast 검사 결과
        """
        try:
            # Feast CLI 명령어 실행 가능 여부 확인
            result = subprocess.run(
                ['feast', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                recommendations = [
                    "Feast를 설치하세요: uv add feast",
                    "또는 시스템 전역 설치: pip install feast",
                    "설치 후 PATH 환경변수 확인"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="Feast CLI를 찾을 수 없음",
                    details=[
                        "❌ Feast 명령어 실행 불가",
                        f"오류: {result.stderr.strip() if result.stderr else 'Unknown error'}"
                    ],
                    recommendations=recommendations
                )
            
            feast_version = "Available (help command successful)"
            
            # Feature Store Repository 확인
            repo_path = self.config.feast_repo_path
            if repo_path:
                repo_result = self._test_feast_repository(repo_path)
                
                details = [
                    f"Feast 버전: {feast_version}",
                    f"Repository 경로: {repo_path}"
                ]
                
                if repo_result.is_connected:
                    details.extend([
                        "✅ Feature Store repository 접근 가능",
                        f"응답 시간: {repo_result.response_time_ms:.1f}ms"
                    ])
                    
                    return CheckResult(
                        is_healthy=True,
                        message="Feast Feature Store 정상",
                        details=details
                    )
                else:
                    details.extend([
                        "❌ Feature Store repository 접근 불가",
                        f"오류: {repo_result.error_message}"
                    ])
                    
                    recommendations = [
                        f"Repository 경로 확인: {repo_path}",
                        "Feast 초기화: feast init <repo_name>",
                        "Feature store 설정 파일 확인",
                        "권한 및 네트워크 연결 확인"
                    ]
                    
                    return CheckResult(
                        is_healthy=False,
                        message="Feast repository 접근 실패",
                        details=details,
                        recommendations=recommendations
                    )
            else:
                return CheckResult(
                    is_healthy=True,
                    message="Feast CLI 사용 가능 (Repository 미설정)",
                    details=[
                        f"Feast 버전: {feast_version}",
                        "⚠️ Repository 경로가 설정되지 않음",
                        "기본 기능만 확인됨"
                    ]
                )
                
        except subprocess.TimeoutExpired:
            return CheckResult(
                is_healthy=False,
                message="Feast 명령어 실행 시간 초과",
                details=["❌ Feast 응답 없음 (10초 초과)"],
                recommendations=["Feast 재설치를 고려하세요"]
            )
        except FileNotFoundError:
            recommendations = [
                "Feast를 설치하세요: uv add feast",
                "또는 시스템 전역 설치: pip install feast",
                "설치 후 터미널 재시작"
            ]
            
            return CheckResult(
                is_healthy=False,
                message="Feast가 설치되지 않음",
                details=["❌ feast 명령어를 찾을 수 없음"],
                recommendations=recommendations
            )
        except Exception as e:
            raise HealthCheckError(
                message=f"Feast 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def _test_postgresql_connection(self) -> ConnectionTestResult:
        """
        PostgreSQL 연결을 실제로 테스트합니다.
        
        Returns:
            ConnectionTestResult: PostgreSQL 연결 테스트 결과
        """
        start_time = time.time()
        
        try:
            # 기본 연결 문자열 구성
            import os
            postgres_user = os.getenv('MMP_HEALTH_POSTGRES_USER', 'mluser')
            postgres_password = os.getenv('PGPASSWORD', os.getenv('MMP_HEALTH_POSTGRES_PASSWORD', 'mysecretpassword'))
            
            conn_string = (
                f"host={self.config.postgres_host} "
                f"port={self.config.postgres_port} "
                f"dbname={self.config.postgres_database} "
                f"user={postgres_user} password={postgres_password}"
            )
            
            with psycopg.connect(
                conn_string, 
                connect_timeout=self.config.connection_timeout
            ) as conn:
                # 버전 정보 조회
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    result = cur.fetchone()
                    version_info = result[0] if result else None
                
                response_time_ms = (time.time() - start_time) * 1000
                
                return ConnectionTestResult(
                    service_name="PostgreSQL",
                    is_connected=True,
                    response_time_ms=response_time_ms,
                    service_version=version_info.split(' ')[1] if version_info else None
                )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="PostgreSQL",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    def _test_redis_connection(self) -> ConnectionTestResult:
        """
        Redis 연결을 실제로 테스트합니다.
        
        Returns:
            ConnectionTestResult: Redis 연결 테스트 결과
        """
        start_time = time.time()
        
        try:
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                socket_timeout=self.config.connection_timeout,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            # PING 명령으로 연결 테스트
            ping_result = r.ping()
            
            # 서버 정보 조회
            info = r.info('server')
            redis_version = info.get('redis_version', 'Unknown')
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if ping_result:
                return ConnectionTestResult(
                    service_name="Redis",
                    is_connected=True,
                    response_time_ms=response_time_ms,
                    service_version=redis_version
                )
            else:
                return ConnectionTestResult(
                    service_name="Redis",
                    is_connected=False,
                    response_time_ms=response_time_ms,
                    error_message="PING 명령 실패"
                )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Redis",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    def _test_feast_repository(self, repo_path: str) -> ConnectionTestResult:
        """
        Feast Repository 접근성을 테스트합니다.
        
        Args:
            repo_path: Feast repository 경로
            
        Returns:
            ConnectionTestResult: Feast repository 테스트 결과
        """
        start_time = time.time()
        
        try:
            # Repository 상태 확인
            result = subprocess.run(
                ['feast', 'list', 'feature-views'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.config.connection_timeout
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                return ConnectionTestResult(
                    service_name="Feast Repository",
                    is_connected=True,
                    response_time_ms=response_time_ms,
                    additional_info={'output': result.stdout.strip()}
                )
            else:
                return ConnectionTestResult(
                    service_name="Feast Repository",
                    is_connected=False,
                    response_time_ms=response_time_ms,
                    error_message=result.stderr.strip() or "Unknown error"
                )
                
        except subprocess.TimeoutExpired:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Feast Repository",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=f"시간 초과 ({self.config.connection_timeout}초)"
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Feast Repository",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )