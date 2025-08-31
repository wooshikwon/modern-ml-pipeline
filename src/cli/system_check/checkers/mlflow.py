"""
MLflow service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional
import requests

from ..base import BaseServiceChecker
from ..models import CheckResult


class MLflowChecker(BaseServiceChecker):
    """
    MLflow Tracking Server 연결 검사 체커.

    ml_tracking.tracking_uri 설정을 기반으로 MLflow 서버 연결 상태를 검증합니다.
    지원하는 tracking_uri 형식:
    - http://localhost:5000 (로컬 서버)
    - https://mlflow.example.com (원격 서버)
    - file:///path/to/mlruns (로컬 파일 시스템)
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 MLflow tracking_uri가 설정되어 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: ml_tracking.tracking_uri가 설정되어 있으면 True
        """
        tracking_uri = self._get_tracking_uri(config)
        return tracking_uri is not None and tracking_uri.strip() != ""

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        MLflow Tracking Server 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: MLflow 연결 검사 결과
        """
        tracking_uri = self._get_tracking_uri(config)

        if not tracking_uri:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="MLflow tracking_uri가 설정되지 않음",
                recommendations=["config 파일에 ml_tracking.tracking_uri 설정 추가"],
            )

        try:
            # file:// URI는 로컬 파일 시스템이므로 별도 처리
            if tracking_uri.startswith("file://"):
                return self._check_file_uri(tracking_uri)

            # HTTP/HTTPS URI는 API 엔드포인트 호출로 검증
            elif tracking_uri.startswith(("http://", "https://")):
                return self._check_http_uri(tracking_uri)

            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"지원하지 않는 tracking_uri 형식: {tracking_uri}",
                    recommendations=[
                        "http://, https://, file:// 형식의 URI를 사용하세요",
                        "예: http://localhost:5000, file:///path/to/mlruns",
                    ],
                )

        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MLflow 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "MLflow"

    def _get_tracking_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 MLflow tracking_uri 추출."""
        try:
            return config.get("ml_tracking", {}).get("tracking_uri")
        except (AttributeError, TypeError):
            return None

    def _check_file_uri(self, tracking_uri: str) -> CheckResult:
        """file:// URI 형식의 MLflow 검사."""
        try:
            from pathlib import Path

            # file:///path/to/mlruns에서 경로 추출
            file_path = tracking_uri.replace("file://", "")
            mlruns_path = Path(file_path)

            if mlruns_path.exists():
                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"MLflow 로컬 스토리지 접근 성공: {file_path}",
                    details=[f"MLruns 디렉토리: {mlruns_path.absolute()}"],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"MLflow 로컬 스토리지 경로가 존재하지 않음: {file_path}",
                    recommendations=[
                        f"디렉토리를 생성하세요: mkdir -p {file_path}",
                        "또는 다른 경로로 tracking_uri를 변경하세요",
                    ],
                )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MLflow 로컬 스토리지 검사 실패: {str(e)}",
                recommendations=["tracking_uri의 file:// 경로를 확인하세요"],
            )

    def _check_http_uri(self, tracking_uri: str) -> CheckResult:
        """HTTP/HTTPS URI 형식의 MLflow 서버 검사."""
        try:
            # MLflow API 엔드포인트 호출 (/api/2.0/mlflow/experiments/search)
            api_url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/experiments/search"

            response = requests.get(
                api_url, timeout=10, params={"max_results": 1}  # 최소한의 요청
            )

            if response.status_code == 200:
                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"MLflow 서버 연결 성공: {tracking_uri}",
                    details=[
                        f"응답 코드: {response.status_code}",
                        f"응답 시간: {response.elapsed.total_seconds():.3f}초",
                    ],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"MLflow 서버 응답 오류: HTTP {response.status_code}",
                    recommendations=[
                        f"MLflow 서버 상태 확인: {tracking_uri}",
                        "서버가 올바르게 실행 중인지 확인하세요",
                    ],
                )

        except requests.exceptions.ConnectionError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MLflow 서버에 연결할 수 없음: {tracking_uri}",
                recommendations=[
                    "MLflow 서버가 실행 중인지 확인하세요",
                    f"서버 주소가 올바른지 확인하세요: {tracking_uri}",
                    "네트워크 연결 상태를 확인하세요",
                ],
            )
        except requests.exceptions.Timeout:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"MLflow 서버 응답 시간 초과: {tracking_uri}",
                recommendations=[
                    "MLflow 서버 응답 속도를 확인하세요",
                    "네트워크 상태를 확인하세요",
                ],
            )

    def _generate_error_recommendations(self, error_message: str) -> list[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "MLflow tracking_uri 설정을 확인하세요",
            "MLflow 서버가 실행 중인지 확인하세요",
        ]

        if "connection" in error_message.lower():
            recommendations.append("네트워크 연결 상태를 확인하세요")
        elif "timeout" in error_message.lower():
            recommendations.append("서버 응답 시간을 확인하세요")
        elif "permission" in error_message.lower():
            recommendations.append("파일 시스템 권한을 확인하세요")

        return recommendations
