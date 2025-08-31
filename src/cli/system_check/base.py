"""
Base interface for service checkers
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from .models import CheckResult


class BaseServiceChecker(ABC):
    """
    Base abstract class for all service checkers.

    모든 서비스 체커는 이 인터페이스를 구현해야 합니다.
    각 체커는 config를 분석해서 해당 서비스 사용 여부를 판단하고,
    실제 연결 검사를 수행합니다.

    Supported services:
    - PostgreSQL (connection_uri 기반)
    - BigQuery (bigquery:// URI 및 feast offline_store 기반)
    - Google Cloud Storage (gs:// URI 기반)
    - AWS S3 (s3:// URI 기반)
    - Redis (online_store 설정 기반)
    - MLflow (tracking_uri 기반)
    - Feast (feast_config 기반)
    """

    @abstractmethod
    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        주어진 config에서 이 체커가 검사할 수 있는 서비스가 설정되어 있는지 판단.

        Args:
            config: 전체 설정 딕셔너리 (config/*.yaml 파일들을 병합한 것)

        Returns:
            bool: 이 체커가 검사할 서비스가 설정되어 있으면 True

        Examples:
            PostgreSQL checker는 data_adapters.adapters.sql.config.connection_uri에서
            postgresql:// 프로토콜을 찾으면 True 반환

            BigQuery checker는 bigquery:// URI나
            feature_store.feast_config.offline_store.type == "bigquery"를 찾으면 True 반환
        """
        pass

    @abstractmethod
    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        실제 서비스 연결 검사를 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: 검사 결과 (성공/실패, 메시지, 상세정보, 권장사항 포함)

        Raises:
            Exception: 연결 실패 시 CheckResult에 실패 정보를 담아서 반환해야 함
            (Exception을 그대로 raise하지 말고 CheckResult로 감싸기)
        """
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """
        서비스 이름 반환 (로깅 및 결과 표시용).

        Returns:
            str: 사용자에게 표시될 서비스 이름

        Examples:
            "PostgreSQL", "BigQuery", "Google Cloud Storage", "MLflow" 등
        """
        pass
