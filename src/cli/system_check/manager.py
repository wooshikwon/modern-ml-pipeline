"""
Dynamic service checker manager
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, List

from .base import BaseServiceChecker
from .models import CheckResult
from .checkers.mlflow import MLflowChecker
from .checkers.postgresql import PostgreSQLChecker
from .checkers.redis import RedisChecker
from .checkers.bigquery import BigQueryChecker
from .checkers.gcs import GCSChecker
from .checkers.s3 import S3Checker
from .checkers.mysql import MySQLChecker
from .checkers.cassandra import CassandraChecker
from .checkers.mongodb import MongoDBChecker
from .checkers.elasticsearch import ElasticsearchChecker


class DynamicServiceChecker:
    """
    동적 서비스 검사 매니저.

    Config 내용을 분석해서 실제 사용하는 서비스만 자동으로 감지하고 검증합니다.
    각 서비스 체커가 can_check()로 해당 서비스 사용 여부를 판단하면,
    해당 체커의 check()를 호출해서 실제 연결 검사를 수행합니다.

    지원하는 서비스:
    - MLflow (ml_tracking.tracking_uri)
    - PostgreSQL (postgresql:// connection_uri)
    - Redis (redis:// URIs, online_store type=redis)
    - BigQuery (bigquery:// URI, offline_store type=bigquery)
    - Google Cloud Storage (gs:// URIs)
    - AWS S3 (s3:// URIs)
    - MySQL (mysql:// connection_uri)
    - Cassandra (cassandra:// URIs, type=cassandra)
    - MongoDB (mongodb:// URIs, type=mongodb)
    - Elasticsearch (elasticsearch URIs, elastic configs)
    """

    def __init__(self) -> None:
        """DynamicServiceChecker 초기화."""
        # 모든 서비스 체커 등록
        self.checkers: List[BaseServiceChecker] = [
            MLflowChecker(),
            PostgreSQLChecker(),
            RedisChecker(),
            BigQueryChecker(),
            GCSChecker(),
            S3Checker(),
            MySQLChecker(),
            CassandraChecker(),
            MongoDBChecker(),
            ElasticsearchChecker(),
        ]

    def run_checks(self, config: Dict[str, Any]) -> List[CheckResult]:
        """
        Config 내용을 분석해서 실제 사용하는 서비스만 동적 검증.

        Args:
            config: 전체 설정 딕셔너리 (config/*.yaml 파일들을 병합한 것)

        Returns:
            List[CheckResult]: 각 서비스별 검사 결과 리스트

        Notes:
            각 체커가 can_check()로 해당 서비스 사용 여부를 자동 감지합니다.
            실제 사용하는 서비스만 검사하므로 불필요한 검사를 피할 수 있습니다.
        """
        results = []
        active_checkers = []

        # 1. 각 체커가 검사 가능한지 확인
        for checker in self.checkers:
            try:
                if checker.can_check(config):
                    active_checkers.append(checker)
            except Exception as e:
                # can_check 자체에서 오류가 발생한 경우 에러 결과 생성
                results.append(self._create_error_result(checker, e, "can_check"))

        # 2. 활성 체커들 실행
        for checker in active_checkers:
            try:
                result = checker.check(config)
                results.append(result)
            except Exception as e:
                # check 메서드에서 오류가 발생한 경우 에러 결과 생성
                results.append(self._create_error_result(checker, e, "check"))

        return results

    def get_summary_stats(self, results: List[CheckResult]) -> Dict[str, Any]:
        """
        검사 결과의 요약 통계를 생성.

        Args:
            results: 검사 결과 리스트

        Returns:
            Dict[str, Any]: 요약 통계
            - total: 총 검사 수
            - healthy: 성공한 검사 수
            - unhealthy: 실패한 검사 수
            - services: 검사된 서비스 이름들
            - critical_failures: 중요한 실패 수
        """
        total = len(results)
        healthy = sum(1 for r in results if r.is_healthy)
        unhealthy = total - healthy
        services = [r.service_name for r in results]
        critical_failures = sum(
            1 for r in results if not r.is_healthy and r.severity == "critical"
        )

        return {
            "total": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "services": services,
            "critical_failures": critical_failures,
            "success_rate": round(healthy / total * 100, 1) if total > 0 else 0.0,
        }

    def get_active_services(self, config: Dict[str, Any]) -> List[str]:
        """
        Config를 분석해서 활성화된 서비스 목록을 반환.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            List[str]: 활성화된 서비스 이름들
        """
        active_services = []

        for checker in self.checkers:
            try:
                if checker.can_check(config):
                    active_services.append(checker.get_service_name())
            except Exception:
                # can_check에서 오류가 발생해도 서비스 이름은 포함
                # (오류 상태로라도 검사 대상이므로)
                active_services.append(f"{checker.get_service_name()} (오류)")

        return active_services

    def _create_error_result(
        self, checker: BaseServiceChecker, exception: Exception, phase: str
    ) -> CheckResult:
        """
        체커 실행 중 발생한 예외를 CheckResult로 변환.

        Args:
            checker: 오류가 발생한 체커
            exception: 발생한 예외
            phase: 오류 발생 단계 ("can_check" 또는 "check")

        Returns:
            CheckResult: 오류 정보를 담은 검사 결과
        """
        service_name = self._safe_get_service_name(checker)

        return CheckResult(
            is_healthy=False,
            service_name=service_name,
            message=f"{service_name} 체커 {phase} 단계에서 오류 발생: {str(exception)}",
            recommendations=[
                f"{service_name} 체커 구현을 확인하세요",
                "config 파일 형식이 올바른지 확인하세요",
                "필요한 Python 패키지가 설치되어 있는지 확인하세요",
            ],
            severity="critical",
        )

    def _safe_get_service_name(self, checker: BaseServiceChecker) -> str:
        """
        체커에서 안전하게 서비스 이름을 가져옴.

        Args:
            checker: 서비스 체커

        Returns:
            str: 서비스 이름 (오류 시 기본값)
        """
        try:
            return checker.get_service_name()
        except Exception:
            return f"Unknown Service ({checker.__class__.__name__})"
