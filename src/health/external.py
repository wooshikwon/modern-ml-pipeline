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
    
    # M04-2-4 Enhanced External Services Validation Methods
    
    def check_services_selectively(self) -> CheckResult:
        """
        설정에 따라 선택적으로 외부 서비스를 검증합니다.
        
        Returns:
            CheckResult: 선택적 검증 결과
        """
        details = []
        recommendations = []
        all_healthy = True
        services_checked = 0
        services_skipped = 0
        
        # PostgreSQL 선택적 검증
        if self.config.skip_postgresql:
            details.append("🚫 PostgreSQL 검증이 스킵됨 (설정에 따라)")
            services_skipped += 1
        else:
            try:
                result = self.check_postgresql()
                services_checked += 1
                if not result.is_healthy:
                    all_healthy = False
                    details.append(f"❌ PostgreSQL: {result.message}")
                else:
                    details.append(f"✅ PostgreSQL: {result.message}")
            except Exception as e:
                all_healthy = False
                details.append(f"❌ PostgreSQL 검증 오류: {e}")
        
        # Redis 선택적 검증
        if self.config.skip_redis:
            details.append("🚫 Redis 검증이 스킵됨 (설정에 따라)")
            services_skipped += 1
        else:
            try:
                result = self.check_redis()
                services_checked += 1
                if not result.is_healthy:
                    all_healthy = False
                    details.append(f"❌ Redis: {result.message}")
                else:
                    details.append(f"✅ Redis: {result.message}")
            except Exception as e:
                all_healthy = False
                details.append(f"❌ Redis 검증 오류: {e}")
        
        # Feast 선택적 검증
        if self.config.skip_feast:
            details.append("🚫 Feast 검증이 스킵됨 (설정에 따라)")
            services_skipped += 1
        else:
            try:
                result = self.check_feast()
                services_checked += 1
                if not result.is_healthy:
                    all_healthy = False
                    details.append(f"❌ Feast: {result.message}")
                else:
                    details.append(f"✅ Feast: {result.message}")
            except Exception as e:
                all_healthy = False
                details.append(f"❌ Feast 검증 오류: {e}")
        
        # 결과 생성
        if services_skipped > 0:
            details.insert(0, f"📊 검증 통계: {services_checked}개 검증됨, {services_skipped}개 스킵됨")
        
        if services_checked == 0:
            return CheckResult(
                is_healthy=True,
                message="모든 외부 서비스 검증이 스킵됨",
                details=details,
                recommendations=["외부 서비스 사용 시 skip_* 설정을 해제하세요"]
            )
        
        message = f"선택적 외부 서비스 검증 {'완료' if all_healthy else '실패'}"
        return CheckResult(
            is_healthy=all_healthy,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_docker_integration(self) -> CheckResult:
        """
        mmp-local-dev Docker 컨테이너 상태를 통합하여 검증합니다.
        
        Returns:
            CheckResult: Docker 통합 검증 결과
        """
        if not self.config.enable_docker_integration:
            return CheckResult(
                is_healthy=True,
                message="Docker 통합 검증이 비활성화됨",
                details=["⚙️ enable_docker_integration=False"]
            )
        
        details = []
        recommendations = []
        all_healthy = True
        
        try:
            # mmp-local-dev 디렉토리에서 docker-compose ps 실행
            from pathlib import Path
            
            mmp_local_dev_path = Path(self.config.mmp_local_dev_path or "../mmp-local-dev")
            
            if not mmp_local_dev_path.exists():
                return CheckResult(
                    is_healthy=False,
                    message="mmp-local-dev 디렉토리를 찾을 수 없음",
                    details=[f"❌ 경로: {mmp_local_dev_path.absolute()}"],
                    recommendations=[
                        f"mmp-local-dev를 {mmp_local_dev_path.absolute()}에 클론하세요",
                        "또는 MMP_HEALTH_MMP_LOCAL_DEV_PATH 환경변수를 설정하세요"
                    ]
                )
            
            # Docker 컨테이너 상태 확인
            result = subprocess.run(
                ['docker-compose', 'ps'],
                cwd=mmp_local_dev_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                details.append(f"❌ docker-compose ps 실행 실패: {result.stderr}")
                all_healthy = False
                recommendations.extend([
                    "Docker가 실행 중인지 확인하세요",
                    "mmp-local-dev에서 docker-compose up -d 실행하세요"
                ])
            else:
                # 컨테이너 상태 파싱
                output_lines = result.stdout.strip().split('\n')
                container_statuses = {}
                
                for line in output_lines[1:]:  # 헤더 제외
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 6:
                            container_name = parts[0]
                            status = parts[5] if len(parts) > 5 else "unknown"
                            container_statuses[container_name] = status
                
                details.append(f"🐳 Docker 컨테이너 상태 ({len(container_statuses)}개)")
                
                for container, status in container_statuses.items():
                    if 'healthy' in status.lower():
                        details.append(f"✅ {container}: {status}")
                    elif 'unhealthy' in status.lower():
                        details.append(f"⚠️ {container}: {status}")
                        recommendations.append(f"{container} 컨테이너 로그 확인: docker logs {container}")
                    else:
                        details.append(f"ℹ️ {container}: {status}")
            
            message = "Docker 통합 검증 완료" if all_healthy else "Docker 컨테이너 일부 문제 발견"
            
        except subprocess.TimeoutExpired:
            all_healthy = False
            details.append("❌ docker-compose 명령 시간 초과")
            recommendations.append("Docker 데몬 상태를 확인하세요")
            message = "Docker 통합 검증 시간 초과"
        except FileNotFoundError:
            all_healthy = False
            details.append("❌ docker-compose 명령을 찾을 수 없음")
            recommendations.extend([
                "Docker Compose를 설치하세요",
                "PATH에 docker-compose가 포함되어 있는지 확인하세요"
            ])
            message = "Docker Compose 미설치"
        except Exception as e:
            raise HealthCheckError(
                message=f"Docker 통합 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
        
        return CheckResult(
            is_healthy=all_healthy,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_postgresql_detailed(self) -> CheckResult:
        """
        PostgreSQL의 세부 기능을 검증합니다 (실제 쿼리 실행).
        
        Returns:
            CheckResult: PostgreSQL 세부 검증 결과
        """
        if not PSYCOPG_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="psycopg 패키지가 설치되지 않음",
                details=["❌ PostgreSQL 세부 검증 불가능"],
                recommendations=["uv add psycopg[binary]"]
            )
        
        try:
            # 기본 연결 테스트
            connection_result = self._test_postgresql_connection()
            
            if not connection_result.is_connected:
                return CheckResult(
                    is_healthy=False,
                    message="PostgreSQL 기본 연결 실패",
                    details=[f"❌ 연결 오류: {connection_result.error_message}"],
                    recommendations=self._get_postgresql_recommendations()
                )
            
            # 세부 기능 테스트
            detailed_result = self._test_postgresql_detailed_functionality()
            
            details = [
                f"✅ 기본 연결: {connection_result.response_time_ms:.1f}ms",
                f"📊 서버 버전: {connection_result.service_version or 'Unknown'}"
            ]
            
            if detailed_result.is_connected:
                details.extend([
                    "✅ 테이블 생성/삭제 테스트 통과",
                    "✅ 쿼리 실행 테스트 통과",
                    f"📈 세부 테스트: {detailed_result.response_time_ms:.1f}ms"
                ])
                
                if detailed_result.additional_info:
                    for key, value in detailed_result.additional_info.items():
                        details.append(f"📋 {key}: {value}")
                
                return CheckResult(
                    is_healthy=True,
                    message=f"PostgreSQL 세부 기능 검증 완료 ({detailed_result.response_time_ms:.1f}ms)",
                    details=details
                )
            else:
                details.extend([
                    f"❌ 세부 기능 테스트 실패: {detailed_result.error_message}",
                    "⚠️ 기본 연결은 가능하지만 쿼리 실행에 문제 있음"
                ])
                
                recommendations = [
                    "PostgreSQL 사용자 권한을 확인하세요",
                    "데이터베이스 스키마 권한을 확인하세요",
                    "PostgreSQL 로그를 확인하세요"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="PostgreSQL 세부 기능 검증 실패",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"PostgreSQL 세부 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_redis_detailed(self) -> CheckResult:
        """
        Redis의 세부 캐싱 기능을 검증합니다 (실제 set/get 작업).
        
        Returns:
            CheckResult: Redis 세부 검증 결과
        """
        if not REDIS_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="redis 패키지가 설치되지 않음",
                details=["❌ Redis 세부 검증 불가능"],
                recommendations=["uv add redis"]
            )
        
        try:
            # 기본 연결 테스트
            connection_result = self._test_redis_connection()
            
            if not connection_result.is_connected:
                return CheckResult(
                    is_healthy=False,
                    message="Redis 기본 연결 실패",
                    details=[f"❌ 연결 오류: {connection_result.error_message}"],
                    recommendations=self._get_redis_recommendations()
                )
            
            # 세부 캐싱 기능 테스트
            caching_result = self._test_redis_caching_functionality()
            
            details = [
                f"✅ 기본 연결: {connection_result.response_time_ms:.1f}ms",
                f"📊 서버 버전: {connection_result.service_version or 'Unknown'}"
            ]
            
            if caching_result.is_connected:
                details.extend([
                    "✅ 캐시 저장 (SET) 테스트 통과",
                    "✅ 캐시 조회 (GET) 테스트 통과",
                    "✅ 캐시 삭제 (DEL) 테스트 통과",
                    f"📈 캐싱 테스트: {caching_result.response_time_ms:.1f}ms"
                ])
                
                if caching_result.additional_info:
                    for key, value in caching_result.additional_info.items():
                        details.append(f"📋 {key}: {value}")
                
                return CheckResult(
                    is_healthy=True,
                    message=f"Redis 세부 캐싱 기능 검증 완료 ({caching_result.response_time_ms:.1f}ms)",
                    details=details
                )
            else:
                details.extend([
                    f"❌ 캐싱 기능 테스트 실패: {caching_result.error_message}",
                    "⚠️ 기본 연결은 가능하지만 캐시 작업에 문제 있음"
                ])
                
                recommendations = [
                    "Redis 서버 메모리 상태를 확인하세요",
                    "Redis 설정 파일을 확인하세요",
                    "Redis 로그를 확인하세요"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="Redis 세부 캐싱 기능 검증 실패",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"Redis 세부 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_feast_detailed(self) -> CheckResult:
        """
        Feast Feature Store의 세부 기능을 검증합니다 (실제 feature 조회).
        
        Returns:
            CheckResult: Feast 세부 검증 결과
        """
        try:
            # Feast 버전 확인
            version_result = subprocess.run(
                ['feast', 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if version_result.returncode != 0:
                return CheckResult(
                    is_healthy=False,
                    message="Feast 명령어 실행 불가",
                    details=[f"❌ 버전 확인 실패: {version_result.stderr}"],
                    recommendations=["Feast 설치 확인: uv add feast"]
                )
            
            feast_version = version_result.stdout.strip()
            details = [f"📊 Feast 버전: {feast_version}"]
            
            # Repository 설정 확인
            repo_path = self.config.feast_repo_path
            if not repo_path:
                # mmp-local-dev의 feast 디렉토리 확인
                from pathlib import Path
                default_repo = Path(self.config.mmp_local_dev_path or "../mmp-local-dev") / "feast"
                if default_repo.exists():
                    repo_path = str(default_repo)
                    details.append(f"📂 Repository: {repo_path} (자동 감지)")
                else:
                    details.append("⚠️ Feature Store repository 경로가 설정되지 않음")
                    return CheckResult(
                        is_healthy=True,
                        message="Feast CLI 사용 가능 (Repository 미설정)",
                        details=details,
                        recommendations=["MMP_HEALTH_FEAST_REPO_PATH 환경변수 설정"]
                    )
            
            # Feature Store 세부 기능 테스트
            feature_result = self._test_feast_feature_functionality(repo_path)
            
            if feature_result.is_connected:
                details.extend([
                    f"✅ Repository 접근: {feature_result.response_time_ms:.1f}ms",
                    "✅ Feature Views 조회 가능"
                ])
                
                if feature_result.additional_info:
                    output = feature_result.additional_info.get('output', '')
                    if output:
                        feature_views = [line.strip() for line in output.split('\n') if line.strip()]
                        details.append(f"📋 발견된 Feature Views: {len(feature_views)}개")
                        for fv in feature_views[:3]:  # 처음 3개만 표시
                            details.append(f"  • {fv}")
                        if len(feature_views) > 3:
                            details.append(f"  • ... 외 {len(feature_views) - 3}개 더")
                
                # Materialization 상태 확인 시도
                try:
                    materialize_result = subprocess.run(
                        ['feast', 'materialize-incremental', '--help'],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if materialize_result.returncode == 0:
                        details.append("✅ Materialization 기능 사용 가능")
                except:
                    details.append("⚠️ Materialization 기능 확인 불가")
                
                return CheckResult(
                    is_healthy=True,
                    message=f"Feast 세부 기능 검증 완료 ({feature_result.response_time_ms:.1f}ms)",
                    details=details
                )
            else:
                details.extend([
                    f"❌ Feature Store 기능 테스트 실패: {feature_result.error_message}",
                    f"📂 Repository 경로: {repo_path}"
                ])
                
                recommendations = [
                    f"Feast repository 초기화: feast init -t {repo_path}",
                    "Feature definitions 파일 확인",
                    "PostgreSQL/Redis 연결 설정 확인"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="Feast 세부 기능 검증 실패",
                    details=details,
                    recommendations=recommendations
                )
                
        except subprocess.TimeoutExpired:
            return CheckResult(
                is_healthy=False,
                message="Feast 명령 시간 초과",
                details=["❌ Feast 응답 없음 (10초 초과)"],
                recommendations=["Feast 재설치 고려"]
            )
        except Exception as e:
            raise HealthCheckError(
                message=f"Feast 세부 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_mmp_local_dev_compatibility(self) -> CheckResult:
        """
        mmp-local-dev 환경과의 호환성을 검증합니다.
        
        Returns:
            CheckResult: mmp-local-dev 호환성 검증 결과
        """
        details = []
        recommendations = []
        compatibility_issues = 0
        
        # 1. 포트 설정 호환성 확인
        expected_ports = {
            "PostgreSQL": (self.config.postgres_port, 5432),
            "Redis": (self.config.redis_port, 6379),
            # MLflow는 mmp-local-dev에서 5002 포트 사용
        }
        
        details.append("🔧 포트 설정 호환성 확인:")
        for service, (actual, expected) in expected_ports.items():
            if actual == expected:
                details.append(f"  ✅ {service}: {actual} (표준)")
            else:
                details.append(f"  ⚠️ {service}: {actual} (표준: {expected})")
                compatibility_issues += 1
                recommendations.append(f"{service} 포트를 {expected}로 설정하세요")
        
        # 2. mmp-local-dev 디렉토리 구조 확인
        from pathlib import Path
        
        mmp_local_dev_path = Path(self.config.mmp_local_dev_path or "../mmp-local-dev")
        
        details.append("📁 mmp-local-dev 디렉토리 구조 확인:")
        if mmp_local_dev_path.exists():
            details.append(f"  ✅ 기본 경로: {mmp_local_dev_path.absolute()}")
            
            # 중요한 파일들 확인
            important_files = {
                "docker-compose.yml": "Docker 서비스 정의",
                "dev-contract.yml": "개발 계약서",
                "feast/": "Feature Store 설정",
                ".env": "환경 변수 설정"
            }
            
            for file_path, description in important_files.items():
                full_path = mmp_local_dev_path / file_path
                if full_path.exists():
                    details.append(f"  ✅ {file_path}: {description}")
                else:
                    details.append(f"  ❌ {file_path}: {description} (누락)")
                    compatibility_issues += 1
                    recommendations.append(f"{file_path} 파일을 확인하세요")
        else:
            details.append(f"  ❌ 경로 없음: {mmp_local_dev_path.absolute()}")
            compatibility_issues += 1
            recommendations.extend([
                f"mmp-local-dev를 {mmp_local_dev_path.parent}에 클론하세요",
                "또는 MMP_HEALTH_MMP_LOCAL_DEV_PATH 설정을 확인하세요"
            ])
        
        # 3. 환경 변수 호환성 확인
        import os
        
        details.append("🌍 환경 변수 호환성 확인:")
        env_checks = {
            "MMP_HEALTH_POSTGRES_HOST": ("localhost", "PostgreSQL 호스트"),
            "MMP_HEALTH_REDIS_HOST": ("localhost", "Redis 호스트"),
        }
        
        for env_var, (expected_value, description) in env_checks.items():
            actual_value = os.getenv(env_var, "미설정")
            if actual_value == expected_value or actual_value == "미설정":
                details.append(f"  ✅ {env_var}: {actual_value} ({description})")
            else:
                details.append(f"  ⚠️ {env_var}: {actual_value} (권장: {expected_value})")
        
        # 4. 전체 호환성 평가
        if compatibility_issues == 0:
            message = "mmp-local-dev 완전 호환"
            is_healthy = True
            details.insert(0, "🎯 완벽한 mmp-local-dev 호환성을 확인했습니다!")
        elif compatibility_issues <= 2:
            message = f"mmp-local-dev 부분 호환 ({compatibility_issues}개 문제)"
            is_healthy = True
            details.insert(0, f"⚠️ {compatibility_issues}개 호환성 문제가 발견되었지만 동작 가능합니다.")
        else:
            message = f"mmp-local-dev 호환성 문제 ({compatibility_issues}개)"
            is_healthy = False
            details.insert(0, f"❌ {compatibility_issues}개 호환성 문제로 인해 일부 기능에 제한이 있을 수 있습니다.")
        
        return CheckResult(
            is_healthy=is_healthy,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    # Helper methods for detailed functionality testing
    
    def _test_postgresql_detailed_functionality(self) -> ConnectionTestResult:
        """PostgreSQL 세부 기능 테스트를 수행합니다."""
        start_time = time.time()
        
        try:
            import os
            postgres_user = os.getenv('MMP_HEALTH_POSTGRES_USER', 'mluser')
            postgres_password = os.getenv('PGPASSWORD', os.getenv('MMP_HEALTH_POSTGRES_PASSWORD', 'mysecretpassword'))
            
            conn_string = (
                f"host={self.config.postgres_host} "
                f"port={self.config.postgres_port} "
                f"dbname={self.config.postgres_database} "
                f"user={postgres_user} password={postgres_password}"
            )
            
            with psycopg.connect(conn_string, connect_timeout=self.config.connection_timeout) as conn:
                with conn.cursor() as cur:
                    # 테스트 테이블 생성
                    test_table = "health_check_test_" + str(int(time.time()))
                    cur.execute(f"CREATE TEMPORARY TABLE {test_table} (id SERIAL, test_data TEXT)")
                    
                    # 데이터 삽입
                    cur.execute(f"INSERT INTO {test_table} (test_data) VALUES ('health_check')")
                    
                    # 데이터 조회
                    cur.execute(f"SELECT COUNT(*) FROM {test_table}")
                    count = cur.fetchone()[0]
                    
                    # 성능 정보 조회
                    cur.execute("SELECT current_setting('shared_buffers')")
                    shared_buffers = cur.fetchone()[0]
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return ConnectionTestResult(
                        service_name="PostgreSQL Detailed",
                        is_connected=True,
                        response_time_ms=response_time_ms,
                        additional_info={
                            'query_test': 'success',
                            'table_access': 'ok', 
                            'test_records': count,
                            'shared_buffers': shared_buffers
                        }
                    )
                    
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="PostgreSQL Detailed",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    def _test_redis_caching_functionality(self) -> ConnectionTestResult:
        """Redis 캐싱 기능 테스트를 수행합니다."""
        start_time = time.time()
        
        try:
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                socket_timeout=self.config.connection_timeout,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            test_key = f"health_check_test_{int(time.time())}"
            test_value = "health_check_value"
            
            # SET 테스트
            r.set(test_key, test_value, ex=30)  # 30초 후 만료
            
            # GET 테스트
            retrieved_value = r.get(test_key)
            if retrieved_value.decode('utf-8') != test_value:
                raise Exception("캐시 데이터 불일치")
            
            # 메모리 정보 조회
            info = r.info('memory')
            used_memory = info.get('used_memory_human', 'Unknown')
            
            # DEL 테스트 (정리)
            r.delete(test_key)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return ConnectionTestResult(
                service_name="Redis Caching",
                is_connected=True,
                response_time_ms=response_time_ms,
                additional_info={
                    'cache_test': 'success',
                    'set_get': 'ok',
                    'used_memory': used_memory,
                    'operations': 'set, get, del'
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Redis Caching",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    def _test_feast_feature_functionality(self, repo_path: str) -> ConnectionTestResult:
        """Feast Feature Store 기능 테스트를 수행합니다."""
        start_time = time.time()
        
        try:
            # Feature Views 목록 조회 (올바른 feast 명령어 구문)
            result = subprocess.run(
                ['feast', 'feature-views', 'list'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.config.connection_timeout
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                return ConnectionTestResult(
                    service_name="Feast Features",
                    is_connected=True,
                    response_time_ms=response_time_ms,
                    additional_info={'output': result.stdout.strip()}
                )
            else:
                return ConnectionTestResult(
                    service_name="Feast Features",
                    is_connected=False,
                    response_time_ms=response_time_ms,
                    error_message=result.stderr.strip() or "Unknown error"
                )
                
        except subprocess.TimeoutExpired:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Feast Features",
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message="시간 초과"
            )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service_name="Feast Features", 
                is_connected=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    def _get_postgresql_recommendations(self) -> list:
        """PostgreSQL 연결 실패 시 추천 사항을 반환합니다."""
        return [
            "PostgreSQL 서버가 실행 중인지 확인하세요",
            f"연결 설정 확인: {self.config.postgres_host}:{self.config.postgres_port}",
            "데이터베이스 및 사용자 권한 확인",
            "mmp-local-dev: cd ../mmp-local-dev && docker-compose up -d",
            "환경 변수 설정: MMP_HEALTH_POSTGRES_HOST, MMP_HEALTH_POSTGRES_DATABASE"
        ]
    
    def _get_redis_recommendations(self) -> list:
        """Redis 연결 실패 시 추천 사항을 반환합니다."""
        return [
            "Redis 서버가 실행 중인지 확인하세요",
            f"연결 설정 확인: {self.config.redis_host}:{self.config.redis_port}",
            "Redis 서버 상태 확인: redis-cli ping",
            "mmp-local-dev: cd ../mmp-local-dev && docker-compose up -d",
            "환경 변수 설정: MMP_HEALTH_REDIS_HOST, MMP_HEALTH_REDIS_PORT"
        ]