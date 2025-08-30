"""
Health Check Orchestrator Implementation
Blueprint v17.0 - Main health check coordination and execution

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 예외 처리 및 로깅
"""

import time
from datetime import datetime
from typing import Dict, Optional

from src.health.models import (
    CheckResult, HealthCheckConfig, HealthCheckSummary,
    CheckCategory
)
from src.health.environment import EnvironmentHealthCheck
from src.health.mlflow import MLflowHealthCheck
from src.health.external import ExternalServicesHealthCheck


class HealthCheckOrchestrator:
    """
    모든 건강 검사를 조율하고 실행하는 중심 클래스.
    
    각 카테고리별 건강 검사를 실행하고 결과를 통합하여 요약을 생성합니다.
    """
    
    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """
        HealthCheckOrchestrator 인스턴스를 초기화합니다.
        
        Args:
            config: 건강 검사 설정
        """
        self.config = config or HealthCheckConfig()
        
        # 각 카테고리별 체커 인스턴스 생성
        self.environment_checker = EnvironmentHealthCheck()
        self.mlflow_checker = MLflowHealthCheck()
        self.external_checker = ExternalServicesHealthCheck(self.config)
    
    def run_all_checks(self) -> HealthCheckSummary:
        """
        모든 건강 검사를 실행하고 요약을 반환합니다.
        
        Returns:
            HealthCheckSummary: 전체 건강 검사 요약
        """
        start_time = time.time()
        
        try:
            # 각 카테고리별 검사 실행
            results: Dict[CheckCategory, CheckResult] = {}
            
            # 환경 검사
            results.update(self._run_environment_checks())
            
            # MLflow 검사
            results.update(self._run_mlflow_checks())
            
            # 외부 서비스 검사
            results.update(self._run_external_service_checks())
            
            # 실행 시간 계산
            execution_time = time.time() - start_time
            
            # 요약 정보 생성
            summary = self._generate_summary(results, execution_time)
            
            return summary
            
        except Exception as e:
            # 예상치 못한 오류 발생 시 기본 요약 반환
            execution_time = time.time() - start_time
            
            error_result = CheckResult(
                is_healthy=False,
                message=f"건강 검사 실행 중 오류 발생: {e}",
                details=[str(e)]
            )
            
            return HealthCheckSummary(
                overall_healthy=False,
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                warning_checks=0,
                categories={CheckCategory.SYSTEM: error_result},
                execution_time_seconds=execution_time,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def run_category_check(self, category: CheckCategory) -> CheckResult:
        """
        특정 카테고리의 건강 검사만 실행합니다.
        
        Args:
            category: 검사할 카테고리
            
        Returns:
            CheckResult: 해당 카테고리의 검사 결과
            
        Raises:
            ValueError: 지원되지 않는 카테고리인 경우
        """
        if category == CheckCategory.ENVIRONMENT:
            return self._run_single_environment_check()
        elif category == CheckCategory.MLFLOW:
            return self._run_single_mlflow_check()
        elif category == CheckCategory.EXTERNAL_SERVICES:
            return self._run_single_external_check()
        else:
            raise ValueError(f"지원되지 않는 카테고리: {category}")
    
    def _run_environment_checks(self) -> Dict[CheckCategory, CheckResult]:
        """환경 관련 모든 검사를 실행합니다."""
        try:
            # M04-2-2 개선: 세부 검증 기능 추가
            checks = [
                # 기본 검사
                ("Python 버전 기본", self.environment_checker.check_python_version),
                ("핵심 의존성 기본", self.environment_checker.check_core_dependencies),
                ("템플릿 접근성", self.environment_checker.check_template_accessibility),
                ("uv 가용성", self.environment_checker.check_uv_availability),
                # M04-2-2 세부 검사 추가
                ("Python 버전 세부", self.environment_checker.check_python_version_detailed),
                ("의존성 호환성 세부", self.environment_checker.check_dependencies_detailed),
                ("템플릿 내용 검증", self.environment_checker.check_template_content_validation),
                ("uv 고급 기능", self.environment_checker.check_uv_advanced_capabilities)
            ]
            
            all_healthy = True
            combined_details = []
            combined_recommendations = []
            
            for check_name, check_method in checks:
                try:
                    result = check_method()
                    
                    if not result.is_healthy:
                        all_healthy = False
                    
                    # 세부 정보 통합
                    if result.details:
                        combined_details.extend([f"{check_name}: {detail}" for detail in result.details])
                    
                    # 추천사항 통합
                    if result.recommendations:
                        combined_recommendations.extend(result.recommendations)
                        
                except Exception as e:
                    all_healthy = False
                    combined_details.append(f"{check_name}: 검사 실패 - {e}")
            
            # 통합 결과 생성
            if all_healthy:
                message = "모든 환경 검사 통과"
            else:
                failed_count = len([detail for detail in combined_details if "❌" in detail])
                message = f"환경 검사에서 {failed_count}개 문제 발견"
            
            return {
                CheckCategory.ENVIRONMENT: CheckResult(
                    is_healthy=all_healthy,
                    message=message,
                    details=combined_details,
                    recommendations=combined_recommendations
                )
            }
            
        except Exception as e:
            return {
                CheckCategory.ENVIRONMENT: CheckResult(
                    is_healthy=False,
                    message=f"환경 검사 실행 실패: {e}",
                    details=[str(e)]
                )
            }
    
    def _run_mlflow_checks(self) -> Dict[CheckCategory, CheckResult]:
        """MLflow 관련 모든 검사를 실행합니다."""
        try:
            # M04-2-3 개선: 세부 MLflow 검증 기능 추가
            checks = []
            current_mode = self.mlflow_checker.detect_current_mode()
            
            if current_mode == 'server':
                # 서버 모드 검사
                checks = [
                    ("MLflow 서버 기본", self.mlflow_checker.check_server_connectivity),
                    ("MLflow 서버 세부", self.mlflow_checker.check_server_detailed),
                    ("MLflow 추적 기능", self.mlflow_checker.check_tracking_functionality),
                ]
            elif current_mode == 'local':
                # 로컬 모드 검사
                checks = [
                    ("MLflow 로컬 기본", self.mlflow_checker.check_local_mode),
                    ("MLflow 로컬 세부", self.mlflow_checker.check_local_mode_detailed),
                    ("MLflow 추적 기능", self.mlflow_checker.check_tracking_functionality),
                ]
            else:
                return {CheckCategory.MLFLOW: CheckResult(
                    is_healthy=False,
                    message="MLflow 모드를 감지할 수 없음",
                    details=["MLflow 설정을 확인하세요"],
                    recommendations=["MLFLOW_TRACKING_URI 환경변수 설정"]
                )}
            
            # 공통 검사 추가
            checks.append(("MLflow 유연성 검사", self.mlflow_checker.check_graceful_degradation))
            
            # 모든 검사 실행 및 결과 통합
            all_healthy = True
            combined_details = []
            combined_recommendations = []
            
            for check_name, check_method in checks:
                try:
                    result = check_method()
                    
                    if not result.is_healthy:
                        all_healthy = False
                    
                    # 세부 정보 통합
                    if result.details:
                        combined_details.extend([f"{check_name}: {detail}" for detail in result.details])
                    
                    # 추천사항 통합
                    if result.recommendations:
                        combined_recommendations.extend([
                            f"{check_name} - {rec}" for rec in result.recommendations
                        ])
                        
                except Exception as e:
                    all_healthy = False
                    combined_details.append(f"{check_name}: 검사 실패 - {e}")
            
            # 통합 결과 생성
            if all_healthy:
                message = f"모든 MLflow 검사 통과 ({current_mode} 모드)"
            else:
                failed_count = len([detail for detail in combined_details if "❌" in detail or "실패" in detail])
                message = f"MLflow 검사에서 {failed_count}개 문제 발견 ({current_mode} 모드)"
            
            return {CheckCategory.MLFLOW: CheckResult(
                is_healthy=all_healthy,
                message=message,
                details=combined_details,
                recommendations=combined_recommendations
            )}
            
        except Exception as e:
            return {
                CheckCategory.MLFLOW: CheckResult(
                    is_healthy=False,
                    message=f"MLflow 검사 실행 실패: {e}",
                    details=[str(e)]
                )
            }
    
    def _run_external_service_checks(self) -> Dict[CheckCategory, CheckResult]:
        """외부 서비스 관련 모든 검사를 실행합니다."""
        try:
            # M04-2-4 Enhanced: 선택적 검증 + 세부 검증 + Docker 통합 + 호환성 검증
            checks = []
            
            # 1. 선택적 검증 (설정에 따라 서비스 스킵 가능)
            checks.append(("선택적 서비스 검증", self.external_checker.check_services_selectively))
            
            # 2. Docker 통합 검증 (mmp-local-dev 컨테이너 상태)
            if self.external_checker.config.enable_docker_integration:
                checks.append(("Docker 통합", self.external_checker.check_docker_integration))
            
            # 3. 세부 기능 검증 (실제 기능 테스트)
            if not self.external_checker.config.skip_postgresql:
                checks.append(("PostgreSQL 세부", self.external_checker.check_postgresql_detailed))
            
            if not self.external_checker.config.skip_redis:
                checks.append(("Redis 세부", self.external_checker.check_redis_detailed))
            
            if not self.external_checker.config.skip_feast:
                checks.append(("Feast 세부", self.external_checker.check_feast_detailed))
            
            # 4. mmp-local-dev 호환성 검증
            checks.append(("mmp-local-dev 호환성", self.external_checker.check_mmp_local_dev_compatibility))
            
            # 모든 검사 실행 및 결과 통합
            all_healthy = True
            combined_details = []
            combined_recommendations = []
            successful_checks = []
            failed_checks = []
            
            for check_name, check_method in checks:
                try:
                    result = check_method()
                    
                    if result.is_healthy:
                        successful_checks.append(check_name)
                    else:
                        failed_checks.append(check_name)
                        all_healthy = False
                    
                    # 세부 정보 통합
                    if result.details:
                        combined_details.extend([f"{check_name}: {detail}" for detail in result.details])
                    
                    # 추천사항 통합
                    if result.recommendations:
                        combined_recommendations.extend([
                            f"{check_name} - {rec}" for rec in result.recommendations
                        ])
                        
                except Exception as e:
                    failed_checks.append(check_name)
                    all_healthy = False
                    combined_details.append(f"{check_name}: 검사 실패 - {e}")
            
            # 통합 결과 생성
            total_checks = len(checks)
            successful_count = len(successful_checks)
            
            if all_healthy:
                message = f"모든 외부 서비스 Enhanced 검증 완료 ({successful_count}/{total_checks})"
            else:
                message = f"외부 서비스 검증에서 {len(failed_checks)}개 문제 발견 ({successful_count}/{total_checks})"
            
            # 검증 통계 추가
            if combined_details:
                combined_details.insert(0, f"📊 Enhanced 검증 통계: {successful_count}개 성공, {len(failed_checks)}개 실패")
            
            return {
                CheckCategory.EXTERNAL_SERVICES: CheckResult(
                    is_healthy=all_healthy,
                    message=message,
                    details=combined_details,
                    recommendations=combined_recommendations
                )
            }
            
        except Exception as e:
            return {
                CheckCategory.EXTERNAL_SERVICES: CheckResult(
                    is_healthy=False,
                    message=f"외부 서비스 검사 실행 실패: {e}",
                    details=[str(e)]
                )
            }
    
    def _run_single_environment_check(self) -> CheckResult:
        """단일 환경 검사를 실행합니다."""
        env_results = self._run_environment_checks()
        return env_results[CheckCategory.ENVIRONMENT]
    
    def _run_single_mlflow_check(self) -> CheckResult:
        """단일 MLflow 검사를 실행합니다."""
        mlflow_results = self._run_mlflow_checks()
        return mlflow_results[CheckCategory.MLFLOW]
    
    def _run_single_external_check(self) -> CheckResult:
        """단일 외부 서비스 검사를 실행합니다."""
        external_results = self._run_external_service_checks()
        return external_results[CheckCategory.EXTERNAL_SERVICES]
    
    def _generate_summary(self, results: Dict[CheckCategory, CheckResult], 
                         execution_time: float) -> HealthCheckSummary:
        """검사 결과를 바탕으로 요약을 생성합니다."""
        total_checks = len(results)
        passed_checks = sum(1 for result in results.values() if result.is_healthy)
        failed_checks = total_checks - passed_checks
        
        return HealthCheckSummary(
            overall_healthy=failed_checks == 0,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=0,  # 현재는 경고 수준 미구현
            categories=results,
            execution_time_seconds=execution_time,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )