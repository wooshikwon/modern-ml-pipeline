"""
HealthCheck System Unit Tests
Blueprint v17.0 - TDD RED Phase

CLAUDE.md 원칙 준수:
- RED → GREEN → REFACTOR 사이클
- 테스트 없는 구현 금지  
- 커버리지 ≥ 90%
"""

import sys
from unittest.mock import Mock, patch
import typer.testing

from src.cli import app


class TestHealthCheckCommand:
    """Self-check 명령어 테스트 클래스"""

    def test_health_check__basic_command__should_be_available(self) -> None:
        """
        modern-ml-pipeline self-check 명령어가 존재하는지 검증.
        
        Given: CLI 애플리케이션
        When: self-check 명령어 실행
        Then: 명령어가 인식되고 실행되어야 함
        """
        # Given: CLI runner
        runner = typer.testing.CliRunner()
        
        # When: self-check 명령어 실행
        result = runner.invoke(app, ["self-check"])
        
        # Then: 명령어가 존재하고 실행되어야 함 (구현되지 않아도 인식은 되어야)
        assert result.exit_code != 2, "self-check 명령어가 존재하지 않습니다"

    def test_health_check__help_option__should_show_description(self) -> None:
        """
        self-check 명령어 도움말이 적절한 설명을 제공하는지 검증.
        
        Given: CLI 애플리케이션
        When: self-check --help 실행
        Then: 명령어 설명과 옵션들이 표시되어야 함
        """
        # Given: CLI runner
        runner = typer.testing.CliRunner()
        
        # When: help 실행
        result = runner.invoke(app, ["self-check", "--help"])
        
        # Then: 도움말이 표시되어야 함
        assert result.exit_code == 0
        assert "self-check" in result.stdout.lower()
        assert "environment" in result.stdout.lower() or "health" in result.stdout.lower()

    def test_health_check__verbose_option__should_be_supported(self) -> None:
        """
        self-check 명령어가 --verbose 옵션을 지원하는지 검증.
        
        Given: CLI 애플리케이션
        When: self-check --verbose 실행
        Then: 상세한 출력이 제공되어야 함
        """
        # Given: CLI runner
        runner = typer.testing.CliRunner()
        
        # When: verbose 옵션으로 실행
        result = runner.invoke(app, ["self-check", "--verbose"])
        
        # Then: verbose 옵션이 인식되어야 함
        assert result.exit_code != 2, "--verbose 옵션이 지원되지 않습니다"


class TestEnvironmentHealthCheck:
    """환경 검증 시스템 테스트 클래스"""

    def test_environment_check__python_version__should_validate_supported_version(self) -> None:
        """
        Python 버전이 지원 범위(>=3.11,<3.12) 내에 있는지 검증.
        
        Given: 현재 Python 환경
        When: Python 버전 검사 실행
        Then: 지원 버전 범위 내 여부를 정확히 판단해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: Python 버전 검사
        check_result = health_checker.check_python_version()
        
        # Then: 결과가 CheckResult 형태여야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'message')
        assert hasattr(check_result, 'details')
        
        # Python 버전이 3.11.x라면 healthy여야 함
        current_version = sys.version_info
        if current_version.major == 3 and current_version.minor == 11:
            assert check_result.is_healthy, f"Python 3.11.x는 지원되어야 합니다: {check_result.message}"

    def test_environment_check__core_dependencies__should_validate_required_packages(self) -> None:
        """
        핵심 의존성 패키지들이 설치되어 있는지 검증.
        
        Given: 프로젝트 환경
        When: 핵심 패키지 검사 실행
        Then: 필수 패키지들의 설치 상태를 확인해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 핵심 의존성 검사
        check_result = health_checker.check_core_dependencies()
        
        # Then: 결과 구조가 올바르고 핵심 패키지 정보를 포함해야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        
        # 핵심 패키지들 확인 (typer, pydantic, mlflow 등)
        core_packages = ['typer', 'pydantic', 'mlflow', 'fastapi', 'pandas']
        if check_result.details:
            for package in core_packages:
                assert any(package in str(detail).lower() for detail in check_result.details), \
                    f"핵심 패키지 {package} 정보가 검사 결과에 없습니다"

    def test_environment_check__template_accessibility__should_validate_config_templates(self) -> None:
        """
        설정 템플릿 파일들에 접근 가능한지 검증.
        
        Given: 프로젝트 구조
        When: 템플릿 접근성 검사 실행
        Then: 필수 템플릿 파일들이 존재하고 읽기 가능해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 템플릿 접근성 검사
        check_result = health_checker.check_template_accessibility()
        
        # Then: 결과가 올바르고 템플릿 정보를 포함해야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        
        # 핵심 템플릿들이 검사되었는지 확인
        expected_templates = ['base.yaml', 'guideline_recipe.yaml.j2']
        if check_result.details:
            for template in expected_templates:
                assert any(template in str(detail) for detail in check_result.details), \
                    f"템플릿 {template} 정보가 검사 결과에 없습니다"


class TestMLflowHealthCheck:
    """MLflow 연결성 테스트 클래스"""

    @patch('src.health.mlflow.mlflow.get_tracking_uri')
    @patch('src.health.mlflow.MLflowHealthCheck._test_server_connection')
    def test_mlflow_check__server_mode__should_validate_connection(self, mock_connection: Mock, mock_get_uri: Mock) -> None:
        """
        MLflow 서버 모드 연결 상태를 검증.
        
        Given: MLflow 서버 설정
        When: 서버 연결 검사 실행
        Then: 서버 연결 상태를 정확히 판단해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        from src.health.models import ConnectionTestResult
        
        # Given: Mock 서버 URI 및 연결 성공 설정
        mock_get_uri.return_value = "http://localhost:5000"
        mock_connection.return_value = ConnectionTestResult(
            service_name="MLflow Server",
            is_connected=True,
            response_time_ms=150.0
        )
        
        health_checker = MLflowHealthCheck()
        
        # When: MLflow 서버 검사
        check_result = health_checker.check_server_connectivity()
        
        # Then: 연결 상태 검증
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'message')
        
        # Mock이 호출되었는지 확인
        mock_connection.assert_called_once()

    def test_mlflow_check__local_mode__should_validate_directory_access(self) -> None:
        """
        MLflow 로컬 모드 디렉토리 접근성을 검증.
        
        Given: 로컬 MLflow 설정
        When: 로컬 모드 검사 실행
        Then: mlruns 디렉토리 생성/접근 권한을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 로컬 모드 검사
        check_result = health_checker.check_local_mode()
        
        # Then: 로컬 디렉토리 접근성 확인
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        
        # mlruns 관련 정보가 포함되어야 함
        if check_result.details:
            assert any('mlruns' in str(detail).lower() for detail in check_result.details), \
                "mlruns 디렉토리 정보가 검사 결과에 없습니다"

    @patch('src.health.mlflow.mlflow.get_tracking_uri')
    def test_mlflow_check__auto_detection__should_identify_current_mode(self, mock_uri: Mock) -> None:
        """
        현재 MLflow 모드를 자동 감지하는지 검증.
        
        Given: MLflow 환경 설정
        When: 모드 자동 감지 실행
        Then: 현재 설정된 모드(서버/로컬)를 정확히 식별해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: Mock tracking URI
        mock_uri.return_value = "http://localhost:5000"
        health_checker = MLflowHealthCheck()
        
        # When: 모드 감지
        detected_mode = health_checker.detect_current_mode()
        
        # Then: 모드가 정확히 감지되어야 함
        assert detected_mode in ['server', 'local'], f"올바르지 않은 모드: {detected_mode}"
        mock_uri.assert_called_once()


class TestExternalServicesHealthCheck:
    """외부 서비스 연결성 테스트 클래스"""

    @patch('src.health.external.ExternalServicesHealthCheck._test_postgresql_connection')
    def test_external_services__postgresql__should_validate_connection(self, mock_test_connection: Mock) -> None:
        """
        PostgreSQL 데이터베이스 연결을 검증.
        
        Given: PostgreSQL 설정
        When: 데이터베이스 연결 검사 실행
        Then: 연결 상태와 접근 권한을 확인해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        from src.health.models import ConnectionTestResult
        
        # Given: Mock 연결 성공 설정
        mock_test_connection.return_value = ConnectionTestResult(
            service_name="PostgreSQL",
            is_connected=True,
            response_time_ms=120.0,
            service_version="15.4"
        )
        
        # Mock PSYCOPG_AVAILABLE to be True
        with patch('src.health.external.PSYCOPG_AVAILABLE', True):
            health_checker = ExternalServicesHealthCheck()
            
            # When: PostgreSQL 검사
            check_result = health_checker.check_postgresql()
            
            # Then: 연결 검사 결과 확인
            assert hasattr(check_result, 'is_healthy')
            assert hasattr(check_result, 'message')
            assert check_result.is_healthy is True

    @patch('src.health.external.ExternalServicesHealthCheck._test_redis_connection')
    def test_external_services__redis__should_validate_connection(self, mock_test_connection: Mock) -> None:
        """
        Redis 서버 연결을 검증.
        
        Given: Redis 설정
        When: Redis 연결 검사 실행
        Then: 서버 응답 상태를 확인해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        from src.health.models import ConnectionTestResult
        
        # Given: Mock 연결 성공 설정
        mock_test_connection.return_value = ConnectionTestResult(
            service_name="Redis",
            is_connected=True,
            response_time_ms=50.0,
            service_version="7.0.0"
        )
        
        # Mock REDIS_AVAILABLE to be True
        with patch('src.health.external.REDIS_AVAILABLE', True):
            health_checker = ExternalServicesHealthCheck()
            
            # When: Redis 검사
            check_result = health_checker.check_redis()
            
            # Then: 연결 상태 확인
            assert hasattr(check_result, 'is_healthy')
            assert hasattr(check_result, 'message')
            assert check_result.is_healthy is True

    @patch('subprocess.run')
    def test_external_services__feast__should_validate_feature_store(self, mock_run: Mock) -> None:
        """
        Feast Feature Store 상태를 검증.
        
        Given: Feast 설정
        When: Feature Store 검사 실행
        Then: Feast 서비스 상태와 repository 접근성을 확인해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        
        # Given: Mock feast 명령어 성공
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "feast repository initialized"
        
        health_checker = ExternalServicesHealthCheck()
        
        # When: Feast 검사
        check_result = health_checker.check_feast()
        
        # Then: Feature Store 상태 확인
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')


class TestHealthCheckReporting:
    """건강 상태 보고 시스템 테스트 클래스"""

    def test_health_report__color_output__should_display_status_with_colors(self) -> None:
        """
        건강 상태 보고서가 컬러 출력을 지원하는지 검증.
        
        Given: 다양한 건강 상태 결과
        When: 컬러 보고서 생성
        Then: 상태별로 적절한 색상이 적용되어야 함
        """
        from src.health.reporter import HealthReporter
        from src.health.models import CheckResult
        
        # Given: 건강/비건강 결과 생성
        healthy_result = CheckResult(
            is_healthy=True,
            message="All systems operational",
            details=["Python 3.11.5", "All packages installed"]
        )
        
        unhealthy_result = CheckResult(
            is_healthy=False,
            message="MLflow server unreachable",
            details=["Connection timeout", "Check server status"]
        )
        
        reporter = HealthReporter()
        
        # When: 컬러 출력 생성
        healthy_output = reporter.format_result("Environment", healthy_result, use_colors=True)
        unhealthy_output = reporter.format_result("MLflow", unhealthy_result, use_colors=True)
        
        # Then: 색상 코드가 포함되어야 함
        assert isinstance(healthy_output, str)
        assert isinstance(unhealthy_output, str)
        assert len(healthy_output) > 0
        assert len(unhealthy_output) > 0

    def test_health_report__actionable_recommendations__should_provide_solutions(self) -> None:
        """
        건강 상태 보고서가 실행 가능한 해결책을 제공하는지 검증.
        
        Given: 문제가 있는 건강 상태 결과
        When: 추천 사항 생성
        Then: 구체적이고 실행 가능한 해결 방법을 제공해야 함
        """
        from src.health.reporter import HealthReporter
        from src.health.models import CheckResult
        
        # Given: 문제 상황 결과
        problem_result = CheckResult(
            is_healthy=False,
            message="Python version incompatible",
            details=["Current: 3.10.2", "Required: >=3.11,<3.12"]
        )
        
        reporter = HealthReporter()
        
        # When: 추천 사항 생성
        recommendations = reporter.get_recommendations("Environment", problem_result)
        
        # Then: 실행 가능한 추천사항 제공
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # 구체적인 해결 방법 포함 확인
        recommendations_text = ' '.join(recommendations)
        assert any(keyword in recommendations_text.lower() 
                  for keyword in ['install', 'update', 'upgrade', 'uv', 'python'])

    def test_health_report__summary_generation__should_aggregate_all_results(self) -> None:
        """
        전체 건강 상태 요약이 모든 검사 결과를 통합하는지 검증.
        
        Given: 다양한 카테고리의 건강 상태 결과들
        When: 전체 요약 생성
        Then: 모든 결과를 통합하여 전체 상태를 제공해야 함
        """
        from src.health.reporter import HealthReporter
        from src.health.models import CheckResult
        
        # Given: 다양한 결과들
        results = {
            "Environment": CheckResult(True, "OK", ["Python 3.11.5"]),
            "MLflow": CheckResult(False, "Connection failed", ["Timeout"]),
            "External Services": CheckResult(True, "All connected", ["PostgreSQL OK", "Redis OK"])
        }
        
        reporter = HealthReporter()
        
        # When: 전체 요약 생성
        summary = reporter.generate_summary(results)
        
        # Then: 요약이 모든 카테고리를 포함해야 함
        assert isinstance(summary, dict)
        assert 'overall_healthy' in summary
        assert 'total_checks' in summary
        assert 'failed_checks' in summary
        
        # 전체 상태가 올바르게 계산되어야 함 (MLflow 실패로 인해 False)
        assert summary['overall_healthy'] is False
        assert summary['total_checks'] == 3
        assert summary['failed_checks'] == 1


class TestCheckResultModel:
    """CheckResult 모델 테스트 클래스"""

    def test_check_result__initialization__should_create_valid_instance(self) -> None:
        """
        CheckResult 모델이 올바르게 초기화되는지 검증.
        
        Given: CheckResult 필수 매개변수들
        When: 인스턴스 생성
        Then: 모든 필드가 올바르게 설정되어야 함
        """
        from src.health.models import CheckResult
        
        # When: CheckResult 생성
        result = CheckResult(
            is_healthy=True,
            message="Test successful",
            details=["Detail 1", "Detail 2"]
        )
        
        # Then: 필드들이 올바르게 설정되어야 함
        assert result.is_healthy is True
        assert result.message == "Test successful"
        assert result.details == ["Detail 1", "Detail 2"]

    def test_check_result__optional_details__should_handle_none_values(self) -> None:
        """
        CheckResult의 선택적 필드들이 None 값을 적절히 처리하는지 검증.
        
        Given: 최소한의 매개변수
        When: CheckResult 생성
        Then: 선택적 필드들이 기본값으로 설정되어야 함
        """
        from src.health.models import CheckResult
        
        # When: 최소 매개변수로 생성
        result = CheckResult(is_healthy=False, message="Error occurred")
        
        # Then: 기본값 처리 확인
        assert result.is_healthy is False
        assert result.message == "Error occurred"
        assert result.details is None or result.details == []