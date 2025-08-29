"""
MLflow Health Check Implementation  
Blueprint v17.0 - MLflow connectivity and configuration validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 예외 처리 및 로깅
"""

import os
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