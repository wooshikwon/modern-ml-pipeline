"""
BigQuery service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional

from ..base import BaseServiceChecker
from ..models import CheckResult


class BigQueryChecker(BaseServiceChecker):
    """
    Google BigQuery 연결 검사 체커.

    다음 설정을 기반으로 BigQuery 연결 상태를 검증합니다:
    1. data_adapters.adapters.sql.config.connection_uri에서 bigquery:// 프로토콜
    2. feature_store.feast_config.offline_store.type == "bigquery"

    지원하는 connection_uri 형식:
    - bigquery://project-id/dataset-id
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 BigQuery 설정이 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: BigQuery 관련 설정이 있으면 True
        """
        return self._has_bigquery_connection_uri(
            config
        ) or self._has_bigquery_offline_store(config)

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        BigQuery 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: BigQuery 연결 검사 결과
        """
        try:
            # Google Cloud BigQuery 클라이언트를 사용한 연결 테스트
            return self._test_bigquery_connection(config)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="BigQuery 클라이언트 라이브러리가 설치되지 않음",
                recommendations=[
                    "Google Cloud BigQuery 설치: pip install google-cloud-bigquery",
                    "또는 uv add google-cloud-bigquery",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"BigQuery 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "BigQuery"

    def _has_bigquery_connection_uri(self, config: Dict[str, Any]) -> bool:
        """Config에 bigquery:// connection_uri가 있는지 확인."""
        connection_uri = self._get_connection_uri(config)
        return connection_uri is not None and connection_uri.startswith("bigquery://")

    def _has_bigquery_offline_store(self, config: Dict[str, Any]) -> bool:
        """Config에 BigQuery offline_store 설정이 있는지 확인."""
        offline_store_config = self._get_offline_store_config(config)
        return (
            offline_store_config
            and offline_store_config.get("type", "").lower() == "bigquery"
        )

    def _get_connection_uri(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 SQL connection_uri 추출."""
        try:
            return (
                config.get("data_adapters", {})
                .get("adapters", {})
                .get("sql", {})
                .get("config", {})
                .get("connection_uri")
            )
        except (AttributeError, TypeError):
            return None

    def _get_offline_store_config(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Config에서 offline_store 설정 추출."""
        try:
            return (
                config.get("feature_store", {})
                .get("feast_config", {})
                .get("offline_store", {})
            )
        except (AttributeError, TypeError):
            return None

    def _test_bigquery_connection(self, config: Dict[str, Any]) -> CheckResult:
        """실제 BigQuery 연결 테스트."""
        try:
            from google.cloud import bigquery
            from google.cloud.exceptions import NotFound, Forbidden

            # 프로젝트 ID 추출
            project_id = self._extract_project_id(config)
            if not project_id:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message="BigQuery 프로젝트 ID를 찾을 수 없음",
                    recommendations=[
                        "connection_uri에 프로젝트 ID를 포함하세요: bigquery://project-id",
                        "또는 GOOGLE_CLOUD_PROJECT 환경변수를 설정하세요",
                    ],
                )

            # BigQuery 클라이언트 생성 및 테스트
            client = bigquery.Client(project=project_id)

            # 데이터셋 목록 조회로 권한 및 연결 확인
            datasets = list(client.list_datasets(max_results=5))

            return CheckResult(
                is_healthy=True,
                service_name=self.get_service_name(),
                message=f"BigQuery 연결 성공 - Project: {project_id}",
                details=[
                    f"접근 가능한 데이터셋 수: {len(datasets)}",
                    f"프로젝트 ID: {project_id}",
                    (
                        f"데이터셋 예시: {', '.join([ds.dataset_id for ds in datasets[:3]])}"
                        if datasets
                        else "데이터셋 없음"
                    ),
                ],
            )

        except Forbidden as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"BigQuery 접근 권한 없음: {str(e)}",
                recommendations=[
                    "Google Cloud 인증을 확인하세요: gcloud auth login",
                    "서비스 계정 키 파일을 설정하세요: GOOGLE_APPLICATION_CREDENTIALS",
                    f"BigQuery API가 활성화되어 있는지 확인하세요: {project_id}",
                ],
            )
        except NotFound as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"BigQuery 리소스를 찾을 수 없음: {str(e)}",
                recommendations=[
                    f"프로젝트 ID가 올바른지 확인하세요: {project_id}",
                    "프로젝트에 BigQuery API가 활성화되어 있는지 확인하세요",
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
                    message=f"BigQuery 인증 실패: {error_message}",
                    recommendations=[
                        "Google Cloud 인증을 설정하세요: gcloud auth application-default login",
                        "또는 GOOGLE_APPLICATION_CREDENTIALS 환경변수를 설정하세요",
                    ],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"BigQuery 연결 테스트 중 예상치 못한 오류: {error_message}",
                    recommendations=self._generate_error_recommendations(error_message),
                )

    def _extract_project_id(self, config: Dict[str, Any]) -> Optional[str]:
        """Config에서 BigQuery 프로젝트 ID 추출."""
        import os

        # 1. connection_uri에서 추출 (bigquery://project-id/dataset)
        connection_uri = self._get_connection_uri(config)
        if connection_uri and connection_uri.startswith("bigquery://"):
            uri_parts = connection_uri.replace("bigquery://", "").split("/")
            if uri_parts and uri_parts[0]:
                return uri_parts[0]

        # 2. offline_store 설정에서 추출
        offline_store = self._get_offline_store_config(config)
        if offline_store and offline_store.get("project_id"):
            return offline_store["project_id"]

        # 3. 환경변수에서 추출
        return os.getenv("GOOGLE_CLOUD_PROJECT")

    def _generate_error_recommendations(self, error_message: str) -> list[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "BigQuery 설정을 확인하세요",
            "Google Cloud 프로젝트가 올바른지 확인하세요",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                [
                    "네트워크 연결 상태를 확인하세요",
                    "BigQuery API 응답 시간을 확인하세요",
                ]
            )
        elif "quota" in error_message.lower():
            recommendations.append("BigQuery API 할당량을 확인하세요")
        elif "billing" in error_message.lower():
            recommendations.extend(
                [
                    "Google Cloud 프로젝트의 결제 설정을 확인하세요",
                    "BigQuery API가 활성화되어 있는지 확인하세요",
                ]
            )

        return recommendations
