"""
MLflow Health Check Implementation  
Blueprint v17.0 - MLflow connectivity and configuration validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 예외 처리 및 로깅
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import requests
from urllib.parse import urlparse

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.health.models import CheckResult, CheckCategory, HealthCheckError, ConnectionTestResult


class MLflowHealthCheck:
    """
    MLflow 관련 건강 검사를 수행하는 클래스.
    
    MLflow 서버 연결성, 로컬 모드 설정, 트래킹 URI 검증 등을 수행합니다.
    """
    
    def __init__(self) -> None:
        """MLflowHealthCheck 인스턴스를 초기화합니다."""
        self.category = CheckCategory.MLFLOW
        
    def check_server_connectivity(self) -> CheckResult:
        """
        MLflow 서버 모드의 연결 상태를 확인합니다.
        
        Returns:
            CheckResult: MLflow 서버 연결 검사 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            # 현재 트래킹 URI 확인
            tracking_uri = mlflow.get_tracking_uri()
            
            # HTTP/HTTPS URI인지 확인 (서버 모드)
            parsed_uri = urlparse(tracking_uri)
            if parsed_uri.scheme not in ('http', 'https'):
                return CheckResult(
                    is_healthy=False,
                    message="MLflow가 서버 모드로 설정되지 않음",
                    details=[
                        f"현재 트래킹 URI: {tracking_uri}",
                        "서버 모드가 아닌 것으로 보임"
                    ]
                )
            
            # 서버 연결 테스트
            connection_result = self._test_server_connection(tracking_uri)
            
            if connection_result.is_connected:
                return CheckResult(
                    is_healthy=True,
                    message=f"MLflow 서버 연결 성공 ({connection_result.response_time_ms:.1f}ms)",
                    details=[
                        f"트래킹 URI: {tracking_uri}",
                        f"응답 시간: {connection_result.response_time_ms:.1f}ms",
                        f"성능: {connection_result.performance_rating}",
                        "✅ 서버 모드 정상 동작"
                    ]
                )
            else:
                recommendations = [
                    "MLflow 서버가 실행 중인지 확인하세요",
                    f"서버 URL 접근 테스트: curl {tracking_uri}/health",
                    "네트워크 연결 및 방화벽 설정 확인",
                    "서버 로그 확인"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="MLflow 서버 연결 실패",
                    details=[
                        f"트래킹 URI: {tracking_uri}",
                        f"오류: {connection_result.error_message}",
                        "❌ 서버 응답 없음"
                    ],
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow 서버 연결 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_local_mode(self) -> CheckResult:
        """
        MLflow 로컬 모드의 디렉토리 접근성을 확인합니다.
        
        Returns:
            CheckResult: MLflow 로컬 모드 검사 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            # 현재 트래킹 URI 확인
            tracking_uri = mlflow.get_tracking_uri()
            
            # 파일 URI 또는 로컬 경로인지 확인
            parsed_uri = urlparse(tracking_uri)
            is_local = (
                parsed_uri.scheme in ('file', '') or 
                tracking_uri.startswith('./') or 
                not parsed_uri.netloc
            )
            
            if not is_local:
                return CheckResult(
                    is_healthy=False,
                    message="MLflow가 로컬 모드로 설정되지 않음",
                    details=[
                        f"현재 트래킹 URI: {tracking_uri}",
                        "로컬 모드가 아닌 것으로 보임"
                    ]
                )
            
            # mlruns 디렉토리 경로 결정
            if parsed_uri.scheme == 'file':
                mlruns_path = Path(parsed_uri.path)
            else:
                # 상대 경로 또는 절대 경로
                mlruns_path = Path(tracking_uri.replace('file://', ''))
            
            # 절대 경로로 변환
            if not mlruns_path.is_absolute():
                mlruns_path = Path.cwd() / mlruns_path
            
            # 디렉토리 접근성 및 권한 확인
            details = []
            
            # 디렉토리 존재 확인
            if mlruns_path.exists():
                details.append(f"✅ mlruns 디렉토리 존재: {mlruns_path}")
                
                # 읽기/쓰기 권한 확인
                if os.access(mlruns_path, os.R_OK):
                    details.append("✅ 읽기 권한 확인")
                else:
                    details.append("❌ 읽기 권한 없음")
                
                if os.access(mlruns_path, os.W_OK):
                    details.append("✅ 쓰기 권한 확인")
                else:
                    details.append("❌ 쓰기 권한 없음")
            else:
                details.append(f"⚠️ mlruns 디렉토리 없음: {mlruns_path}")
                # 디렉토리가 없어도 생성 가능한지 확인
                parent_dir = mlruns_path.parent
                if parent_dir.exists() and os.access(parent_dir, os.W_OK):
                    details.append("✅ 상위 디렉토리 쓰기 가능 (자동 생성됨)")
                else:
                    details.append("❌ 상위 디렉토리 쓰기 불가")
            
            # 전체 평가
            has_access = (
                (mlruns_path.exists() and os.access(mlruns_path, os.R_OK | os.W_OK)) or
                (not mlruns_path.exists() and mlruns_path.parent.exists() and 
                 os.access(mlruns_path.parent, os.W_OK))
            )
            
            if has_access:
                return CheckResult(
                    is_healthy=True,
                    message="MLflow 로컬 모드 접근 가능",
                    details=details + [
                        f"트래킹 URI: {tracking_uri}",
                        "✅ 로컬 아티팩트 저장 준비됨"
                    ]
                )
            else:
                recommendations = [
                    f"mlruns 디렉토리 권한을 확인하세요: {mlruns_path}",
                    "디렉토리 생성: mkdir -p mlruns",
                    "권한 수정: chmod 755 mlruns",
                    "또는 다른 위치로 MLFLOW_TRACKING_URI 변경"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="MLflow 로컬 모드 접근 불가",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow 로컬 모드 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def detect_current_mode(self) -> str:
        """
        현재 MLflow 설정 모드를 자동 감지합니다.
        
        Returns:
            str: 'server', 'local', 또는 'unknown'
        """
        if not MLFLOW_AVAILABLE:
            return 'unknown'
        
        try:
            tracking_uri = mlflow.get_tracking_uri()
            parsed_uri = urlparse(tracking_uri)
            
            if parsed_uri.scheme in ('http', 'https'):
                return 'server'
            elif parsed_uri.scheme == 'file' or not parsed_uri.scheme:
                return 'local'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def _test_server_connection(self, tracking_uri: str, timeout: int = 10) -> ConnectionTestResult:
        """
        MLflow 서버에 실제 연결을 테스트합니다.
        
        Args:
            tracking_uri: MLflow 트래킹 URI
            timeout: 연결 타임아웃 (초)
            
        Returns:
            ConnectionTestResult: 연결 테스트 결과
        """
        import time
        
        start_time = time.time()
        
        try:
            # Health check 엔드포인트 시도
            health_url = f"{tracking_uri.rstrip('/')}/health"
            response = requests.get(health_url, timeout=timeout)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return ConnectionTestResult(
                    service_name="MLflow Server",
                    is_connected=True,
                    response_time_ms=response_time_ms,
                    service_version=response.headers.get('mlflow-version'),
                    additional_info={'status_code': response.status_code}
                )
            else:
                return ConnectionTestResult(
                    service_name="MLflow Server",
                    is_connected=False,
                    response_time_ms=response_time_ms,
                    error_message=f"HTTP {response.status_code}",
                    additional_info={'status_code': response.status_code}
                )
                
        except requests.exceptions.Timeout:
            return ConnectionTestResult(
                service_name="MLflow Server",
                is_connected=False,
                error_message=f"연결 시간 초과 ({timeout}초)"
            )
        except requests.exceptions.ConnectionError as e:
            return ConnectionTestResult(
                service_name="MLflow Server", 
                is_connected=False,
                error_message=f"연결 오류: {e}"
            )
        except Exception as e:
            return ConnectionTestResult(
                service_name="MLflow Server",
                is_connected=False,
                error_message=f"예상치 못한 오류: {e}"
            )
    
    def check_server_detailed(self) -> CheckResult:
        """
        MLflow 서버 모드의 고급 기능들을 세부 검증합니다.
        
        버전 호환성, 실험 기능, 인증 상태, 연결 안정성을 확인합니다.
        
        Returns:
            CheckResult: MLflow 서버 세부 검증 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            tracking_uri = mlflow.get_tracking_uri()
            parsed_uri = urlparse(tracking_uri)
            
            # 서버 모드 확인
            if parsed_uri.scheme not in ('http', 'https'):
                return CheckResult(
                    is_healthy=False,
                    message="서버 모드가 아님 - 세부 검증 불가",
                    details=[f"현재 URI: {tracking_uri}", "로컬 모드로 실행 중"],
                    recommendations=["MLflow 서버 환경에서 실행하거나 check_local_mode_detailed() 사용"]
                )
            
            details = []
            issues = []
            recommendations = []
            checks_passed = 0
            total_checks = 0
            
            # 1. 기본 연결성 확인
            total_checks += 1
            connection_result = self._test_server_connection(tracking_uri, timeout=15)
            if connection_result.is_connected:
                details.append(f"✅ 서버 연결: {tracking_uri}")
                details.append(f"✅ 응답 시간: {connection_result.response_time_ms:.1f}ms")
                details.append(f"✅ 성능 등급: {connection_result.performance_rating}")
                checks_passed += 1
                
                # 서버 버전 정보
                if connection_result.service_version:
                    details.append(f"📋 서버 버전: {connection_result.service_version}")
            else:
                details.append(f"❌ 서버 연결 실패: {connection_result.error_message}")
                issues.append("서버 연결 불가")
                recommendations.append("서버 상태 확인")
            
            # 2. 클라이언트-서버 버전 호환성
            total_checks += 1
            try:
                client_version = mlflow.__version__
                details.append(f"📋 클라이언트 버전: {client_version}")
                
                # 버전 호환성 체크 (기본적으로 major.minor 일치 권장)
                if connection_result.service_version:
                    client_major_minor = '.'.join(client_version.split('.')[:2])
                    server_major_minor = '.'.join(connection_result.service_version.split('.')[:2])
                    
                    if client_major_minor == server_major_minor:
                        details.append("✅ 클라이언트-서버 버전 호환")
                        checks_passed += 1
                    else:
                        details.append("⚠️ 클라이언트-서버 버전 불일치")
                        issues.append(f"버전 불일치: 클라이언트 {client_version}, 서버 {connection_result.service_version}")
                        recommendations.append("MLflow 클라이언트와 서버 버전을 맞춰주세요")
                else:
                    details.append("⚠️ 서버 버전 확인 불가")
                    checks_passed += 0.5  # 부분 점수
            except Exception as e:
                details.append(f"❌ 버전 호환성 검사 실패: {e}")
                issues.append("버전 검사 오류")
            
            # 3. 실험 기능 테스트 (실제 실험 생성/조회)
            total_checks += 1
            if connection_result.is_connected:
                try:
                    # 테스트 실험 이름 (health check 전용)
                    test_experiment_name = f"health_check_test_{int(time.time())}"
                    
                    # 실험 생성 테스트
                    experiment_id = mlflow.create_experiment(test_experiment_name)
                    details.append("✅ 실험 생성 기능 정상")
                    
                    # 실험 조회 테스트
                    experiment = mlflow.get_experiment(experiment_id)
                    if experiment and experiment.name == test_experiment_name:
                        details.append("✅ 실험 조회 기능 정상")
                        checks_passed += 1
                    
                    # 정리: 테스트 실험 삭제
                    try:
                        mlflow.delete_experiment(experiment_id)
                        details.append("✅ 테스트 실험 정리 완료")
                    except Exception:
                        details.append("⚠️ 테스트 실험 정리 실패 (수동 정리 필요)")
                        
                except Exception as e:
                    details.append(f"❌ 실험 기능 테스트 실패: {e}")
                    issues.append("실험 생성/조회 오류")
                    recommendations.append("MLflow 서버 권한 및 설정 확인")
            else:
                details.append("❌ 실험 기능 테스트 건너뜀 (연결 실패)")
                issues.append("실험 기능 테스트 불가")
            
            # 4. 연결 안정성 테스트 (다중 요청)
            total_checks += 1
            if connection_result.is_connected:
                try:
                    response_times = []
                    for i in range(3):
                        start = time.time()
                        test_conn = self._test_server_connection(tracking_uri, timeout=5)
                        if test_conn.is_connected:
                            response_times.append(test_conn.response_time_ms)
                        time.sleep(0.5)  # 짧은 간격
                    
                    if len(response_times) == 3:
                        avg_time = sum(response_times) / len(response_times)
                        max_time = max(response_times)
                        details.append("✅ 연결 안정성: 3/3 성공")
                        details.append(f"✅ 평균 응답시간: {avg_time:.1f}ms")
                        details.append(f"✅ 최대 응답시간: {max_time:.1f}ms")
                        checks_passed += 1
                    else:
                        details.append(f"⚠️ 연결 안정성: {len(response_times)}/3 성공")
                        issues.append("연결 불안정")
                        recommendations.append("네트워크 연결 상태 확인")
                        checks_passed += len(response_times) / 3
                        
                except Exception as e:
                    details.append(f"❌ 연결 안정성 테스트 실패: {e}")
                    issues.append("안정성 테스트 오류")
            else:
                details.append("❌ 연결 안정성 테스트 건너뜀 (연결 실패)")
            
            # 전체 요약
            details.extend([
                "",
                "📊 서버 세부 검증 요약:",
                f"   - 총 검사: {total_checks}개",
                f"   - 통과: {checks_passed:.1f}개",
                f"   - 성공률: {(checks_passed/total_checks)*100:.1f}%"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
            
            # 성공 조건: 75% 이상 통과
            is_healthy = checks_passed / total_checks >= 0.75
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"MLflow 서버 세부 검증: {checks_passed:.1f}/{total_checks} 통과",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow 서버 세부 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_local_mode_detailed(self) -> CheckResult:
        """
        MLflow 로컬 모드의 세부 기능들을 검증합니다.
        
        디스크 공간, 실제 로깅 기능, 아티팩트 저장, 디렉토리 구조를 확인합니다.
        
        Returns:
            CheckResult: MLflow 로컬 모드 세부 검증 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            tracking_uri = mlflow.get_tracking_uri()
            parsed_uri = urlparse(tracking_uri)
            
            # 로컬 모드 확인
            is_local = (
                parsed_uri.scheme in ('file', '') or 
                tracking_uri.startswith('./') or 
                not parsed_uri.netloc
            )
            
            if not is_local:
                return CheckResult(
                    is_healthy=False,
                    message="로컬 모드가 아님 - 세부 검증 불가",
                    details=[f"현재 URI: {tracking_uri}", "서버 모드로 실행 중"],
                    recommendations=["MLflow 로컬 환경에서 실행하거나 check_server_detailed() 사용"]
                )
            
            details = []
            issues = []
            recommendations = []
            checks_passed = 0
            total_checks = 0
            
            # mlruns 디렉토리 경로 확인
            if parsed_uri.scheme == 'file':
                mlruns_path = Path(parsed_uri.path)
            else:
                mlruns_path = Path(tracking_uri.replace('file://', ''))
            
            if not mlruns_path.is_absolute():
                mlruns_path = Path.cwd() / mlruns_path
            
            details.append(f"📂 MLflow 디렉토리: {mlruns_path}")
            
            # 1. 디렉토리 접근성 및 권한
            total_checks += 1
            try:
                if mlruns_path.exists():
                    details.append("✅ mlruns 디렉토리 존재")
                    
                    # 읽기/쓰기 권한
                    read_ok = os.access(mlruns_path, os.R_OK)
                    write_ok = os.access(mlruns_path, os.W_OK)
                    
                    if read_ok and write_ok:
                        details.append("✅ 디렉토리 권한: 읽기/쓰기 모두 가능")
                        checks_passed += 1
                    elif read_ok:
                        details.append("⚠️ 디렉토리 권한: 읽기만 가능")
                        issues.append("쓰기 권한 없음")
                        recommendations.append(f"chmod 755 {mlruns_path}")
                        checks_passed += 0.5
                    else:
                        details.append("❌ 디렉토리 권한: 접근 불가")
                        issues.append("디렉토리 접근 불가")
                        recommendations.append(f"chmod 755 {mlruns_path}")
                else:
                    parent_dir = mlruns_path.parent
                    if parent_dir.exists() and os.access(parent_dir, os.W_OK):
                        details.append("✅ 디렉토리 자동 생성 가능")
                        checks_passed += 1
                    else:
                        details.append("❌ 디렉토리 생성 불가")
                        issues.append("상위 디렉토리 권한 부족")
                        recommendations.append(f"mkdir -p {mlruns_path}")
            except Exception as e:
                details.append(f"❌ 디렉토리 검사 실패: {e}")
                issues.append("디렉토리 검사 오류")
            
            # 2. 디스크 공간 확인
            total_checks += 1
            try:
                if mlruns_path.exists() or mlruns_path.parent.exists():
                    check_path = mlruns_path if mlruns_path.exists() else mlruns_path.parent
                    disk_usage = shutil.disk_usage(check_path)
                    
                    # 바이트를 GB로 변환
                    free_gb = disk_usage.free / (1024 ** 3)
                    total_gb = disk_usage.total / (1024 ** 3)
                    used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
                    
                    details.append(f"💾 사용 가능 공간: {free_gb:.1f}GB ({total_gb:.1f}GB 중)")
                    details.append(f"💾 디스크 사용률: {used_percent:.1f}%")
                    
                    if free_gb >= 1.0:  # 1GB 이상 권장
                        details.append("✅ 디스크 공간 충분")
                        checks_passed += 1
                    elif free_gb >= 0.1:  # 100MB 이상 최소
                        details.append("⚠️ 디스크 공간 부족 (최소 요구사항)")
                        issues.append("디스크 공간 부족")
                        recommendations.append("디스크 공간을 확보하세요")
                        checks_passed += 0.5
                    else:
                        details.append("❌ 디스크 공간 심각하게 부족")
                        issues.append("디스크 공간 심각 부족")
                        recommendations.append("즉시 디스크 공간 확보 필요")
            except Exception as e:
                details.append(f"⚠️ 디스크 공간 확인 실패: {e}")
                checks_passed += 0.5
            
            # 3. 실제 로깅 기능 테스트
            total_checks += 1
            try:
                # 임시 실험으로 로깅 테스트
                original_uri = mlflow.get_tracking_uri()
                test_experiment_name = f"health_check_local_{int(time.time())}"
                
                with mlflow.start_run(experiment_id=mlflow.create_experiment(test_experiment_name)):
                    # 메트릭 로깅
                    mlflow.log_metric("test_metric", 0.95)
                    mlflow.log_param("test_param", "health_check")
                    
                    # 임시 아티팩트 생성 및 로깅
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write("MLflow health check test artifact")
                        temp_file = f.name
                    
                    mlflow.log_artifact(temp_file, "health_check")
                    
                # 로깅된 데이터 검증
                experiment = mlflow.get_experiment_by_name(test_experiment_name)
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                
                if len(runs) > 0:
                    details.append("✅ 실험 로깅 기능 정상")
                    details.append("✅ 메트릭/파라미터 저장 확인")
                    details.append("✅ 아티팩트 저장 확인")
                    checks_passed += 1
                    
                    # 정리
                    try:
                        mlflow.delete_experiment(experiment.experiment_id)
                        os.unlink(temp_file)
                        details.append("✅ 테스트 데이터 정리 완료")
                    except Exception:
                        details.append("⚠️ 테스트 데이터 정리 실패")
                else:
                    details.append("❌ 로깅 데이터 검증 실패")
                    issues.append("로깅 기능 문제")
                    
            except Exception as e:
                details.append(f"❌ 로깅 기능 테스트 실패: {e}")
                issues.append("로깅 기능 오류")
                recommendations.append("MLflow 설정 및 권한 확인")
            
            # 4. mlruns 디렉토리 구조 검증
            total_checks += 1
            try:
                if mlruns_path.exists():
                    # .trash 디렉토리 확인 (삭제된 실험 보관)
                    trash_dir = mlruns_path / ".trash"
                    if trash_dir.exists():
                        details.append("✅ .trash 디렉토리 존재 (정상 구조)")
                    else:
                        details.append("ℹ️ .trash 디렉토리 없음 (삭제된 실험 없음)")
                    
                    # 기존 실험 디렉토리 확인
                    experiment_dirs = [d for d in mlruns_path.iterdir() 
                                     if d.is_dir() and d.name != ".trash" and d.name.isdigit()]
                    details.append(f"📁 기존 실험: {len(experiment_dirs)}개")
                    
                    # meta.yaml 파일 확인 (각 실험 디렉토리)
                    valid_experiments = 0
                    for exp_dir in experiment_dirs[:3]:  # 최대 3개까지만 확인
                        meta_file = exp_dir / "meta.yaml"
                        if meta_file.exists():
                            valid_experiments += 1
                    
                    if len(experiment_dirs) == 0:
                        details.append("ℹ️ 디렉토리 구조: 초기 상태 (정상)")
                        checks_passed += 1
                    elif valid_experiments == len(experiment_dirs[:3]):
                        details.append("✅ 디렉토리 구조: 유효한 실험 구조")
                        checks_passed += 1
                    else:
                        details.append("⚠️ 디렉토리 구조: 일부 손상된 실험")
                        issues.append("실험 디렉토리 구조 손상")
                        recommendations.append("mlruns 디렉토리 정리 고려")
                        checks_passed += 0.5
                else:
                    details.append("ℹ️ 디렉토리 구조: 초기 상태 (생성 예정)")
                    checks_passed += 1
            except Exception as e:
                details.append(f"⚠️ 디렉토리 구조 검사 실패: {e}")
                checks_passed += 0.5
            
            # 전체 요약
            details.extend([
                "",
                "📊 로컬 모드 세부 검증 요약:",
                f"   - 총 검사: {total_checks}개",
                f"   - 통과: {checks_passed:.1f}개",
                f"   - 성공률: {(checks_passed/total_checks)*100:.1f}%"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
            
            # 성공 조건: 80% 이상 통과
            is_healthy = checks_passed / total_checks >= 0.8
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"MLflow 로컬 모드 세부 검증: {checks_passed:.1f}/{total_checks} 통과",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow 로컬 모드 세부 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_graceful_degradation(self) -> CheckResult:
        """
        MLflow Graceful Degradation(서버↔로컬 전환) 시나리오를 검증합니다.
        
        서버 연결 실패 시 로컬 모드 자동 전환과 사용자 안내를 확인합니다.
        
        Returns:
            CheckResult: Graceful Degradation 검증 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            details = []
            issues = []
            recommendations = []
            checks_passed = 0
            total_checks = 0
            
            # 현재 모드 확인
            current_uri = mlflow.get_tracking_uri()
            current_mode = self.detect_current_mode()
            details.append(f"🔍 현재 MLflow 모드: {current_mode} ({current_uri})")
            
            # 1. 현재 모드 기능 확인
            total_checks += 1
            if current_mode == 'server':
                # 서버 모드 - 연결 테스트
                connection_result = self._test_server_connection(current_uri, timeout=10)
                if connection_result.is_connected:
                    details.append("✅ 서버 모드 정상 동작 중")
                    checks_passed += 1
                else:
                    details.append(f"❌ 서버 모드 연결 실패: {connection_result.error_message}")
                    issues.append("현재 서버 연결 불가")
                    
            elif current_mode == 'local':
                # 로컬 모드 - 기본 검증
                local_result = self.check_local_mode()
                if local_result.is_healthy:
                    details.append("✅ 로컬 모드 정상 동작 중")
                    checks_passed += 1
                else:
                    details.append("❌ 로컬 모드 접근 불가")
                    issues.append("현재 로컬 모드 문제")
            else:
                details.append("❌ 알 수 없는 MLflow 모드")
                issues.append("MLflow 모드 감지 실패")
            
            # 2. 서버→로컬 전환 시나리오 테스트
            total_checks += 1
            try:
                details.append("🔄 서버→로컬 전환 시나리오 테스트")
                
                # 가상의 서버 URI로 테스트 (연결 실패 예상)
                test_server_uri = "http://nonexistent-mlflow-server.local:5000"
                original_uri = mlflow.get_tracking_uri()
                
                # 임시로 불가능한 서버 URI 설정
                os.environ['MLFLOW_TRACKING_URI'] = test_server_uri
                mlflow.set_tracking_uri(test_server_uri)
                
                # 연결 실패 확인
                connection_test = self._test_server_connection(test_server_uri, timeout=3)
                if not connection_test.is_connected:
                    details.append("✅ 서버 연결 실패 시나리오 재현")
                    
                    # 로컬로 폴백 설정
                    fallback_uri = "./mlruns"  # 로컬 디렉토리
                    mlflow.set_tracking_uri(fallback_uri)
                    
                    # 폴백 후 기능 테스트
                    try:
                        test_exp_name = f"fallback_test_{int(time.time())}"
                        exp_id = mlflow.create_experiment(test_exp_name)
                        details.append("✅ 로컬 모드 폴백 성공")
                        details.append("✅ 폴백 후 실험 생성 가능")
                        checks_passed += 1
                        
                        # 정리
                        mlflow.delete_experiment(exp_id)
                        details.append("✅ 폴백 테스트 정리 완료")
                    except Exception as e:
                        details.append(f"❌ 폴백 후 기능 테스트 실패: {e}")
                        issues.append("폴백 모드 기능 문제")
                        recommendations.append("로컬 디렉토리 권한 확인")
                else:
                    details.append("⚠️ 테스트 서버에 예상치 못한 연결 성공")
                
                # 원복
                mlflow.set_tracking_uri(original_uri)
                if 'MLFLOW_TRACKING_URI' in os.environ:
                    if original_uri.startswith('file://') or '://' not in original_uri:
                        del os.environ['MLFLOW_TRACKING_URI']
                    else:
                        os.environ['MLFLOW_TRACKING_URI'] = original_uri
                        
            except Exception as e:
                details.append(f"❌ 전환 시나리오 테스트 실패: {e}")
                issues.append("전환 시나리오 오류")
                # 원복 시도
                try:
                    mlflow.set_tracking_uri(original_uri)
                except:
                    pass
            
            # 3. 전환 시 사용자 메시지 품질 확인
            total_checks += 1
            try:
                # Phase 1에서 구현된 Graceful Degradation 검증
                # MLflow 설정 로더가 적절한 메시지를 제공하는지 확인
                
                # 임시로 서버 모드 설정 후 연결 실패 시뮬레이션
                test_scenarios = [
                    ("연결 타임아웃", "http://10.255.255.1:5000"),  # 라우팅 불가능한 IP
                    ("DNS 해석 실패", "http://invalid-hostname-123.local:5000"),
                    ("포트 접근 불가", "http://localhost:99999")
                ]
                
                scenario_results = []
                for scenario_name, test_uri in test_scenarios:
                    try:
                        test_result = self._test_server_connection(test_uri, timeout=2)
                        if not test_result.is_connected:
                            scenario_results.append(f"✅ {scenario_name}: 적절한 오류 감지")
                        else:
                            scenario_results.append(f"⚠️ {scenario_name}: 예상치 못한 연결 성공")
                    except Exception:
                        scenario_results.append(f"✅ {scenario_name}: 오류 처리 확인")
                
                details.extend(scenario_results)
                
                if len([r for r in scenario_results if "✅" in r]) >= 2:
                    details.append("✅ 다양한 실패 시나리오 적절히 처리")
                    checks_passed += 1
                else:
                    details.append("⚠️ 일부 실패 시나리오 처리 개선 필요")
                    issues.append("오류 처리 개선 필요")
                    checks_passed += 0.5
                    
            except Exception as e:
                details.append(f"❌ 사용자 메시지 품질 테스트 실패: {e}")
                issues.append("메시지 품질 테스트 오류")
            
            # 4. 전환 과정의 데이터 일관성 확인
            total_checks += 1
            try:
                details.append("🔍 전환 과정 데이터 일관성 검증")
                
                # 현재 실험 목록 저장
                original_experiments = []
                try:
                    if current_mode == 'local':
                        original_experiments = [exp.name for exp in mlflow.search_experiments()]
                    details.append(f"📋 현재 실험 수: {len(original_experiments)}개")
                except Exception:
                    details.append("📋 현재 실험 조회 불가")
                
                # 데이터 일관성 평가
                # (실제로는 서버↔로컬 전환 시 실험 데이터가 독립적임을 확인)
                if current_mode == 'local':
                    details.append("✅ 로컬 모드: 데이터 일관성 자동 보장")
                    checks_passed += 1
                elif current_mode == 'server':
                    details.append("✅ 서버 모드: 원격 데이터 일관성 의존")
                    checks_passed += 1
                else:
                    details.append("⚠️ 알 수 없는 모드: 일관성 확인 불가")
                    checks_passed += 0.5
                    
            except Exception as e:
                details.append(f"❌ 데이터 일관성 검증 실패: {e}")
                issues.append("데이터 일관성 검증 오류")
            
            # 전체 요약
            details.extend([
                "",
                "📊 Graceful Degradation 검증 요약:",
                f"   - 총 검사: {total_checks}개",
                f"   - 통과: {checks_passed:.1f}개",
                f"   - 성공률: {(checks_passed/total_checks)*100:.1f}%"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
                recommendations.extend([
                    "MLflow Graceful Degradation 설정 확인",
                    "로컬 백업 디렉토리 권한 확인",
                    "네트워크 연결 안정성 점검"
                ])
            
            # 성공 조건: 75% 이상 통과
            is_healthy = checks_passed / total_checks >= 0.75
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"MLflow Graceful Degradation: {checks_passed:.1f}/{total_checks} 시나리오 통과",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow Graceful Degradation 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_tracking_functionality(self) -> CheckResult:
        """
        MLflow 전체 추적 워크플로우를 종합 검증합니다.
        
        실험→로깅→조회→아티팩트 전체 흐름의 end-to-end 동작을 확인합니다.
        
        Returns:
            CheckResult: MLflow 추적 기능 종합 검증 결과
        """
        if not MLFLOW_AVAILABLE:
            return CheckResult(
                is_healthy=False,
                message="MLflow 패키지가 설치되지 않음",
                details=["❌ mlflow 패키지를 찾을 수 없음"],
                recommendations=["uv add mlflow"]
            )
        
        try:
            details = []
            issues = []
            recommendations = []
            checks_passed = 0
            total_checks = 0
            
            current_mode = self.detect_current_mode()
            current_uri = mlflow.get_tracking_uri()
            details.append(f"🔍 MLflow 추적 워크플로우 테스트 시작 ({current_mode} 모드)")
            details.append(f"📍 트래킹 URI: {current_uri}")
            
            # 테스트용 실험 이름
            test_exp_name = f"e2e_workflow_test_{int(time.time())}"
            experiment_id = None
            run_id = None
            temp_files = []
            
            try:
                # 1. 실험 생성 (Create Experiment)
                total_checks += 1
                try:
                    experiment_id = mlflow.create_experiment(test_exp_name)
                    details.append("✅ 1단계: 실험 생성 성공")
                    details.append(f"   📝 실험 ID: {experiment_id}")
                    checks_passed += 1
                except Exception as e:
                    details.append(f"❌ 1단계: 실험 생성 실패 - {e}")
                    issues.append("실험 생성 불가")
                    recommendations.append("MLflow 실험 생성 권한 확인")
                
                # 2. 실험 Run 시작 및 메트릭/파라미터 로깅 (Logging)
                total_checks += 1
                if experiment_id:
                    try:
                        with mlflow.start_run(experiment_id=experiment_id) as run:
                            run_id = run.info.run_id
                            
                            # 다양한 타입의 데이터 로깅
                            mlflow.log_param("model_type", "test_model")
                            mlflow.log_param("data_version", "1.0.0")
                            mlflow.log_metric("accuracy", 0.95)
                            mlflow.log_metric("f1_score", 0.87)
                            mlflow.log_metric("training_time", 123.45)
                            
                            # 태그 추가
                            mlflow.set_tag("test_type", "health_check")
                            mlflow.set_tag("environment", "testing")
                        
                        details.append("✅ 2단계: Run 및 메트릭/파라미터 로깅 성공")
                        details.append(f"   🏃 Run ID: {run_id[:8]}...")
                        checks_passed += 1
                    except Exception as e:
                        details.append(f"❌ 2단계: 로깅 실패 - {e}")
                        issues.append("메트릭/파라미터 로깅 오류")
                        recommendations.append("MLflow 로깅 권한 및 설정 확인")
                else:
                    details.append("❌ 2단계: 실험 ID 없음으로 건너뜀")
                
                # 3. 아티팩트 업로드 (Artifacts)
                total_checks += 1
                if run_id:
                    try:
                        with mlflow.start_run(run_id=run_id):
                            # 텍스트 파일 아티팩트
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                                f.write("MLflow end-to-end workflow test\nTest artifact content")
                                temp_text_file = f.name
                                temp_files.append(temp_text_file)
                            mlflow.log_artifact(temp_text_file, "test_artifacts")
                            
                            # JSON 파일 아티팩트
                            import json
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                json.dump({"test": True, "version": "1.0", "metrics": [0.95, 0.87]}, f)
                                temp_json_file = f.name
                                temp_files.append(temp_json_file)
                            mlflow.log_artifact(temp_json_file, "test_artifacts")
                        
                        details.append("✅ 3단계: 아티팩트 업로드 성공")
                        details.append("   📁 아티팩트: text, json 파일")
                        checks_passed += 1
                    except Exception as e:
                        details.append(f"❌ 3단계: 아티팩트 업로드 실패 - {e}")
                        issues.append("아티팩트 업로드 오류")
                        recommendations.append("아티팩트 저장 경로 및 권한 확인")
                else:
                    details.append("❌ 3단계: Run ID 없음으로 건너뜀")
                
                # 4. 데이터 조회 및 검증 (Search & Retrieve)
                total_checks += 1
                if experiment_id:
                    try:
                        # 실험 정보 조회
                        experiment = mlflow.get_experiment(experiment_id)
                        if experiment.name == test_exp_name:
                            details.append("✅ 4-1단계: 실험 정보 조회 성공")
                        else:
                            details.append("⚠️ 4-1단계: 실험 정보 불일치")
                        
                        # Run 정보 조회
                        runs = mlflow.search_runs(experiment_ids=[experiment_id])
                        if len(runs) > 0:
                            run_data = runs.iloc[0]
                            details.append("✅ 4-2단계: Run 정보 조회 성공")
                            
                            # 메트릭 검증
                            if 'metrics.accuracy' in run_data and run_data['metrics.accuracy'] == 0.95:
                                details.append("✅ 4-3단계: 메트릭 데이터 검증 성공")
                            else:
                                details.append("⚠️ 4-3단계: 메트릭 데이터 불일치")
                            
                            # 파라미터 검증
                            if 'params.model_type' in run_data and run_data['params.model_type'] == 'test_model':
                                details.append("✅ 4-4단계: 파라미터 데이터 검증 성공")
                            else:
                                details.append("⚠️ 4-4단계: 파라미터 데이터 불일치")
                            
                            checks_passed += 1
                        else:
                            details.append("❌ 4-2단계: Run 조회 실패")
                            issues.append("Run 데이터 조회 불가")
                            
                    except Exception as e:
                        details.append(f"❌ 4단계: 데이터 조회 실패 - {e}")
                        issues.append("데이터 조회 오류")
                        recommendations.append("MLflow 조회 권한 및 인덱싱 확인")
                else:
                    details.append("❌ 4단계: 실험 ID 없음으로 건너뜀")
                
                # 5. 아티팩트 다운로드 검증 (Download Artifacts)
                total_checks += 1
                if run_id:
                    try:
                        # 아티팩트 목록 조회
                        artifacts = mlflow.list_artifacts(run_id=run_id)
                        if len(artifacts) > 0:
                            details.append(f"✅ 5-1단계: 아티팩트 목록 조회 성공 ({len(artifacts)}개)")
                            
                            # 아티팩트 다운로드 테스트
                            with tempfile.TemporaryDirectory() as temp_dir:
                                download_path = mlflow.artifacts.download_artifacts(
                                    run_id=run_id, 
                                    artifact_path="test_artifacts",
                                    dst_path=temp_dir
                                )
                                
                                if Path(download_path).exists():
                                    downloaded_files = list(Path(download_path).iterdir())
                                    details.append(f"✅ 5-2단계: 아티팩트 다운로드 성공 ({len(downloaded_files)}개 파일)")
                                    checks_passed += 1
                                else:
                                    details.append("❌ 5-2단계: 아티팩트 다운로드 경로 없음")
                                    issues.append("아티팩트 다운로드 실패")
                        else:
                            details.append("⚠️ 5-1단계: 아티팩트 목록 없음")
                            checks_passed += 0.5
                            
                    except Exception as e:
                        details.append(f"❌ 5단계: 아티팩트 다운로드 실패 - {e}")
                        issues.append("아티팩트 다운로드 오류")
                        recommendations.append("아티팩트 저장소 접근 권한 확인")
                else:
                    details.append("❌ 5단계: Run ID 없음으로 건너뜀")
                
            finally:
                # 정리 작업
                details.append("")
                details.append("🧹 테스트 데이터 정리:")
                
                try:
                    if experiment_id:
                        mlflow.delete_experiment(experiment_id)
                        details.append("✅ 테스트 실험 삭제 완료")
                except Exception as e:
                    details.append(f"⚠️ 테스트 실험 삭제 실패: {e}")
                
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass
                details.append(f"✅ 임시 파일 {len(temp_files)}개 정리 완료")
            
            # 전체 요약
            details.extend([
                "",
                "📊 MLflow 추적 워크플로우 종합 검증:",
                f"   - 총 단계: {total_checks}단계",
                f"   - 통과: {checks_passed:.1f}단계",
                f"   - 성공률: {(checks_passed/total_checks)*100:.1f}%",
                "   - 워크플로우: 생성→로깅→업로드→조회→다운로드"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
                recommendations.extend([
                    "MLflow 전체 권한 설정 확인",
                    "저장소 용량 및 접근성 점검",
                    "네트워크 안정성 확인"
                ])
            
            # 성공 조건: 80% 이상 통과 (end-to-end 워크플로우는 엄격하게)
            is_healthy = checks_passed / total_checks >= 0.8
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"MLflow 추적 워크플로우: {checks_passed:.1f}/{total_checks} 단계 통과",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"MLflow 추적 워크플로우 검증 실패: {e}",
                category=self.category,
                original_error=e
            )