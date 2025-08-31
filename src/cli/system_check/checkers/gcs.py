"""
Google Cloud Storage service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional, List
import re

from ..base import BaseServiceChecker
from ..models import CheckResult


class GCSChecker(BaseServiceChecker):
    """
    Google Cloud Storage 연결 검사 체커.

    Config에서 gs:// URI가 포함된 설정을 찾아 GCS 접근성을 검증합니다.

    검사 대상 설정:
    - artifacts.model_registry.storage_uri
    - data_adapters에서 gs:// 경로들
    - 기타 gs:// 프로토콜을 사용하는 모든 설정
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 gs:// URI가 포함되어 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: gs:// URI가 설정되어 있으면 True
        """
        gcs_uris = self._find_gcs_uris(config)
        return len(gcs_uris) > 0

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        Google Cloud Storage 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: GCS 연결 검사 결과
        """
        gcs_uris = self._find_gcs_uris(config)

        if not gcs_uris:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="GCS URI 설정이 없음",
                recommendations=["config에 gs:// 형식의 URI를 설정하세요"],
            )

        try:
            # Google Cloud Storage 클라이언트를 사용한 연결 테스트
            return self._test_gcs_connection(gcs_uris)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="Google Cloud Storage 클라이언트 라이브러리가 설치되지 않음",
                recommendations=[
                    "Google Cloud Storage 설치: pip install google-cloud-storage",
                    "또는 uv add google-cloud-storage",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"GCS 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "Google Cloud Storage"

    def _find_gcs_uris(self, config: Dict[str, Any]) -> List[str]:
        """Config에서 모든 gs:// URI를 재귀적으로 찾기."""
        gcs_uris = []

        def _recursive_search(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    _recursive_search(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _recursive_search(item, f"{path}[{i}]")
            elif isinstance(obj, str) and obj.startswith("gs://"):
                gcs_uris.append(obj)

        _recursive_search(config)
        return gcs_uris

    def _test_gcs_connection(self, gcs_uris: List[str]) -> CheckResult:
        """실제 GCS 연결 테스트."""
        try:
            from google.cloud import storage
            from google.cloud.exceptions import NotFound, Forbidden

            # GCS 클라이언트 생성
            client = storage.Client()

            # 첫 번째 URI를 사용해서 버킷 접근 테스트
            test_uri = gcs_uris[0]
            bucket_name = self._extract_bucket_name(test_uri)

            if not bucket_name:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"잘못된 GCS URI 형식: {test_uri}",
                    recommendations=["올바른 gs://bucket-name/path 형식을 사용하세요"],
                )

            # 버킷 존재 여부 및 접근 권한 확인
            bucket = client.bucket(bucket_name)
            bucket.reload()  # 이 호출이 권한 검사

            # 성공한 경우 버킷 정보 수집
            bucket_location = bucket.location
            storage_class = bucket.storage_class

            return CheckResult(
                is_healthy=True,
                service_name=self.get_service_name(),
                message=f"GCS 버킷 접근 성공: {bucket_name}",
                details=[
                    f"버킷 위치: {bucket_location}",
                    f"스토리지 클래스: {storage_class}",
                    f"총 GCS URI 수: {len(gcs_uris)}",
                    f"테스트 URI: {test_uri}",
                ],
            )

        except NotFound:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"GCS 버킷을 찾을 수 없음: {bucket_name}",
                recommendations=[
                    f"버킷이 존재하는지 확인하세요: {bucket_name}",
                    "Google Cloud Console에서 버킷을 확인하세요",
                ],
            )
        except Forbidden:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"GCS 버킷 접근 권한 없음: {bucket_name}",
                recommendations=[
                    "Google Cloud 인증을 확인하세요: gcloud auth login",
                    "서비스 계정 권한을 확인하세요",
                    f"버킷 {bucket_name}에 대한 Storage Object Viewer 권한이 필요합니다",
                ],
            )
        except Exception as e:
            error_message = str(e)

            if (
                "authentication" in error_message.lower()
                or "credential" in error_message.lower()
            ):
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"GCS 인증 실패: {error_message}",
                    recommendations=[
                        "Google Cloud 인증을 설정하세요: gcloud auth application-default login",
                        "또는 GOOGLE_APPLICATION_CREDENTIALS 환경변수를 설정하세요",
                    ],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"GCS 연결 테스트 중 예상치 못한 오류: {error_message}",
                    recommendations=self._generate_error_recommendations(error_message),
                )

    def _extract_bucket_name(self, gcs_uri: str) -> Optional[str]:
        """gs:// URI에서 버킷 이름 추출."""
        # gs://bucket-name/path/to/file => bucket-name
        match = re.match(r"gs://([^/]+)", gcs_uri)
        return match.group(1) if match else None

    def _generate_error_recommendations(self, error_message: str) -> List[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "GCS URI 형식을 확인하세요: gs://bucket-name/path",
            "Google Cloud 프로젝트와 권한을 확인하세요",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                ["네트워크 연결 상태를 확인하세요", "GCS API 응답 시간을 확인하세요"]
            )
        elif "quota" in error_message.lower():
            recommendations.append("Google Cloud Storage API 할당량을 확인하세요")
        elif "billing" in error_message.lower():
            recommendations.extend(
                [
                    "Google Cloud 프로젝트의 결제 설정을 확인하세요",
                    "Cloud Storage API가 활성화되어 있는지 확인하세요",
                ]
            )

        return recommendations
