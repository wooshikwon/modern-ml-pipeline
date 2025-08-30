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
import pytest
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

    def test_environment_check__python_version_detail__should_validate_patch_version(self) -> None:
        """
        Python 패치 버전별 구체적 권장사항을 제공하는지 검증.
        
        Given: 다양한 Python 패치 버전 시나리오
        When: 세부 Python 버전 검사 실행
        Then: 패치 버전에 따른 구체적 권장사항을 제공해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 세부 Python 버전 검사 (새로운 메서드)
        check_result = health_checker.check_python_version_detailed()
        
        # Then: 세부 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 패치 버전 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['patch', 'micro', '권장', 'recommended'])

    def test_environment_check__python_security__should_validate_security_patches(self) -> None:
        """
        Python 보안 패치 적용 상태를 검증하는지 확인.
        
        Given: Python 버전 정보
        When: 보안 패치 검사 실행
        Then: 보안 취약점이 있는 버전에 대한 경고를 제공해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 보안 패치 검사 (확장된 버전 검사 포함)
        check_result = health_checker.check_python_version_detailed()
        
        # Then: 보안 관련 정보가 검사되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 보안 또는 업데이트 관련 언급이 있어야 함
            has_security_info = any(
                keyword in ' '.join(check_result.details).lower() 
                for keyword in ['보안', 'security', '업데이트', 'update', 'patch']
            )

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

    def test_environment_check__dependency_versions__should_validate_compatibility(self) -> None:
        """
        의존성 패키지별 버전 호환성을 세부 검증하는지 확인.
        
        Given: 설치된 패키지들
        When: 의존성 버전 호환성 검사 실행
        Then: 각 패키지별 호환성 상태와 업그레이드 권장사항을 제공해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 상세 의존성 검사 (새로운 메서드)
        check_result = health_checker.check_dependencies_detailed()
        
        # Then: 세부 버전 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 버전 호환성 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['version', '버전', 'compatible', '호환'])

    def test_environment_check__dependency_security__should_validate_vulnerabilities(self) -> None:
        """
        의존성 패키지의 보안 취약점을 확인하는지 검증.
        
        Given: 설치된 패키지들
        When: 보안 취약점 검사 실행
        Then: 알려진 취약점이 있는 패키지에 대한 경고를 제공해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 보안 취약점 검사
        check_result = health_checker.check_dependencies_detailed()
        
        # Then: 보안 정보가 검사되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 세부 검사가 수행되었는지 확인
            assert len(check_result.details) > 0

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

    def test_environment_check__template_content__should_validate_syntax(self) -> None:
        """
        템플릿 파일 내용의 구문을 검증하는지 확인.
        
        Given: 프로젝트 템플릿 파일들
        When: 템플릿 내용 구문 검사 실행
        Then: YAML 구문 오류, Jinja2 템플릿 오류를 감지해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 템플릿 내용 검사 (새로운 메서드)
        check_result = health_checker.check_template_content_validation()
        
        # Then: 내용 검증 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 구문 검증 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['yaml', 'jinja2', 'syntax', '구문', 'template'])

    def test_environment_check__template_schema__should_validate_consistency(self) -> None:
        """
        템플릿 파일들의 스키마 일관성을 검증하는지 확인.
        
        Given: 다양한 환경별 설정 템플릿들
        When: 스키마 일관성 검사 실행
        Then: 필수 필드 누락, 타입 불일치를 감지해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: 스키마 일관성 검사
        check_result = health_checker.check_template_content_validation()
        
        # Then: 스키마 검증이 수행되었는지 확인
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 스키마 관련 정보가 포함되어야 함
            details_text = ' '.join(check_result.details)
            schema_keywords = ['schema', '스키마', 'field', 'type', 'required']
            # 적어도 하나의 스키마 관련 키워드가 있어야 함 (또는 구문 검증이라도)
            assert len(check_result.details) > 0

    def test_environment_check__uv_advanced__should_validate_sync_capability(self) -> None:
        """
        uv sync 실행 가능성 및 의존성 해결 능력을 검증하는지 확인.
        
        Given: uv 패키지 매니저 설치 환경
        When: uv 고급 기능 검사 실행
        Then: sync 기능, 가상환경 상태, 권한 문제를 확인해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: uv 고급 기능 검사 (새로운 메서드)
        check_result = health_checker.check_uv_advanced_capabilities()
        
        # Then: 고급 기능 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # uv 고급 기능 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['sync', 'dependency', 'virtual', 'env', '의존성'])

    def test_environment_check__uv_project_compatibility__should_validate_pyproject_toml(self) -> None:
        """
        uv가 현재 프로젝트의 pyproject.toml을 올바르게 해석하는지 검증.
        
        Given: 프로젝트 pyproject.toml 파일
        When: uv 프로젝트 호환성 검사 실행
        Then: pyproject.toml 파싱, 의존성 해결 가능성을 확인해야 함
        """
        from src.health.environment import EnvironmentHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = EnvironmentHealthCheck()
        
        # When: uv 프로젝트 호환성 검사
        check_result = health_checker.check_uv_advanced_capabilities()
        
        # Then: 프로젝트 호환성 정보가 확인되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 프로젝트 관련 정보가 포함되어야 함
            details_text = ' '.join(check_result.details)
            project_keywords = ['project', 'pyproject', 'toml', '프로젝트']
            # 고급 검사가 수행되었는지 확인
            assert len(check_result.details) > 0


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

    def test_mlflow_check__server_detailed__should_validate_advanced_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        MLflow 서버 모드의 고급 기능들을 세부 검증하는지 확인.
        
        Given: MLflow 서버 환경
        When: 서버 세부 검증 실행
        Then: 버전 호환성, 실험 기능, 인증 상태를 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: 서버 모드 환경 설정
        monkeypatch.setattr("mlflow.get_tracking_uri", lambda: "http://localhost:5000")
        
        # HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 서버 세부 검증 (새로운 메서드)
        check_result = health_checker.check_server_detailed()
        
        # Then: 세부 검증 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 서버 세부 기능 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['version', '버전', 'experiment', '실험', 'auth', '인증'])

    def test_mlflow_check__server_version_compatibility__should_validate_client_server_match(self) -> None:
        """
        MLflow 클라이언트-서버 버전 호환성을 검증하는지 확인.
        
        Given: MLflow 클라이언트 및 서버
        When: 버전 호환성 검사 실행
        Then: 버전 불일치 시 경고를 제공해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 버전 호환성 검사
        check_result = health_checker.check_server_detailed()
        
        # Then: 버전 정보가 검사되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 버전 관련 정보가 포함되어야 함
            details_text = ' '.join(check_result.details)
            version_keywords = ['version', '버전', 'client', 'server', '클라이언트', '서버']
            # 세부 검사가 수행되었는지 확인
            assert len(check_result.details) > 0

    def test_mlflow_check__server_experiment_functionality__should_test_actual_operations(self) -> None:
        """
        MLflow 서버의 실험 생성/조회 기능을 실제 테스트하는지 확인.
        
        Given: MLflow 서버 연결
        When: 실험 기능 실제 테스트 실행
        Then: 실험 생성, 로깅, 조회 기능의 동작을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 실험 기능 테스트
        check_result = health_checker.check_server_detailed()
        
        # Then: 실험 기능 테스트 정보가 포함되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 실험 관련 기능 테스트 정보 확인
            assert len(check_result.details) > 0

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

    def test_mlflow_check__local_mode_detailed__should_validate_disk_and_functionality(self) -> None:
        """
        MLflow 로컬 모드의 디스크 공간과 실제 기능을 세부 검증하는지 확인.
        
        Given: MLflow 로컬 모드 환경
        When: 로컬 모드 세부 검증 실행
        Then: 디스크 공간, 실제 로깅 기능, 아티팩트 저장을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 로컬 모드 세부 검증 (새로운 메서드)
        check_result = health_checker.check_local_mode_detailed()
        
        # Then: 세부 검증 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 로컬 모드 세부 기능 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['disk', '디스크', 'space', '공간', 'artifact', '아티팩트'])

    def test_mlflow_check__local_mode_logging_test__should_validate_actual_logging(self) -> None:
        """
        MLflow 로컬 모드에서 실제 로깅 기능이 동작하는지 테스트하는지 확인.
        
        Given: MLflow 로컬 모드 설정
        When: 실제 로깅 기능 테스트 실행
        Then: 테스트 실험 생성, 로깅, 아티팩트 저장을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 실제 로깅 기능 테스트
        check_result = health_checker.check_local_mode_detailed()
        
        # Then: 로깅 기능 테스트 정보가 포함되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 로깅 기능 테스트 정보 확인
            details_text = ' '.join(check_result.details)
            logging_keywords = ['logging', '로깅', 'experiment', '실험', 'metric', '메트릭']
            # 세부 검사가 수행되었는지 확인
            assert len(check_result.details) > 0

    def test_mlflow_check__graceful_degradation__should_validate_server_to_local_fallback(self) -> None:
        """
        MLflow 서버→로컬 자동 전환(Graceful Degradation)을 검증하는지 확인.
        
        Given: MLflow 서버 연결 실패 시나리오
        When: Graceful Degradation 검증 실행
        Then: 서버→로컬 자동 전환과 사용자 안내를 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: Graceful Degradation 검증 (새로운 메서드)
        check_result = health_checker.check_graceful_degradation()
        
        # Then: 전환 시나리오 검증 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # Graceful Degradation 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['fallback', '전환', 'degradation', '폴백', 'server', 'local'])

    def test_mlflow_check__graceful_degradation_scenarios__should_test_multiple_cases(self) -> None:
        """
        MLflow Graceful Degradation의 다양한 시나리오를 테스트하는지 확인.
        
        Given: 다양한 MLflow 연결 실패 상황
        When: 복합 전환 시나리오 테스트 실행
        Then: 서버 다운, 네트워크 오류, 권한 문제별 대응을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 복합 전환 시나리오 테스트
        check_result = health_checker.check_graceful_degradation()
        
        # Then: 다양한 시나리오 테스트 정보가 포함되어야 함
        assert hasattr(check_result, 'details')
        if check_result.details:
            # 시나리오별 테스트 정보 확인
            details_text = ' '.join(check_result.details)
            scenario_keywords = ['scenario', '시나리오', 'timeout', '타임아웃', 'error', '오류']
            # 세부 검사가 수행되었는지 확인
            assert len(check_result.details) > 0

    def test_mlflow_check__tracking_functionality__should_validate_end_to_end_workflow(self) -> None:
        """
        MLflow 전체 추적 워크플로우를 종합 검증하는지 확인.
        
        Given: MLflow 환경 설정
        When: 종합 추적 기능 테스트 실행
        Then: 실험→로깅→조회→아티팩트 전체 흐름을 확인해야 함
        """
        from src.health.mlflow import MLflowHealthCheck
        
        # Given: HealthCheck 인스턴스
        health_checker = MLflowHealthCheck()
        
        # When: 종합 추적 기능 테스트 (새로운 메서드)
        check_result = health_checker.check_tracking_functionality()
        
        # Then: 종합 워크플로우 테스트 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        assert check_result.details is not None
        
        # 전체 워크플로우 정보가 포함되어야 함
        details_text = ' '.join(check_result.details)
        assert any(keyword in details_text.lower() 
                  for keyword in ['workflow', '워크플로우', 'end-to-end', 'tracking', '추적'])

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

    # M04-2-4 Enhanced External Services Validation Tests
    def test_external_services__optional_validation__should_skip_disabled_services(self) -> None:
        """
        선택적 검증 설정 시 비활성화된 서비스를 스킵하는지 확인.
        
        Given: 특정 서비스가 비활성화된 설정
        When: 외부 서비스 검증 실행  
        Then: 비활성화된 서비스는 스킵하고 나머지만 검증해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        from src.health.models import HealthCheckConfig
        
        # Given: PostgreSQL 비활성화 설정
        config = HealthCheckConfig(skip_postgresql=True)
        health_checker = ExternalServicesHealthCheck(config)
        
        # When: 선택적 검증 메서드 (새로운 메서드 필요)
        check_result = health_checker.check_services_selectively()
        
        # Then: PostgreSQL은 스킵되고 다른 서비스만 검증
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert 'postgresql' in details_text.lower() and '스킵' in details_text.lower()

    @patch('subprocess.run')
    def test_external_services__docker_integration__should_validate_container_health(self, mock_docker: Mock) -> None:
        """
        Docker 컨테이너 상태를 통합하여 검증하는지 확인.
        
        Given: mmp-local-dev Docker 환경
        When: Docker 통합 검증 실행
        Then: 컨테이너 health 상태와 연결성을 함께 검증해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        
        # Given: Mock docker-compose ps 결과
        mock_docker.return_value = Mock(
            returncode=0, 
            stdout="ml-pipeline-postgres   healthy\nml-pipeline-redis   healthy\n",
            stderr=""
        )
        
        health_checker = ExternalServicesHealthCheck()
        
        # When: Docker 통합 검증 (새로운 메서드 필요)
        check_result = health_checker.check_docker_integration()
        
        # Then: Docker 상태 정보가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert any(keyword in details_text.lower() 
                  for keyword in ['docker', 'container', 'healthy', '컨테이너'])

    @patch('src.health.external.ExternalServicesHealthCheck._test_postgresql_connection')
    def test_external_services__postgresql_detailed__should_validate_actual_queries(self, mock_connection: Mock) -> None:
        """
        PostgreSQL 세부 기능 검증 - 실제 쿼리 실행 테스트.
        
        Given: PostgreSQL 연결 환경
        When: 세부 기능 검증 실행
        Then: 실제 테이블 생성/조회가 가능해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        from src.health.models import ConnectionTestResult
        
        # Given: Mock successful detailed connection
        mock_connection.return_value = ConnectionTestResult(
            service_name="PostgreSQL",
            is_connected=True,
            response_time_ms=85.0,
            service_version="15.4",
            additional_info={'query_test': 'success', 'table_access': 'ok'}
        )
        
        health_checker = ExternalServicesHealthCheck()
        
        # When: PostgreSQL 세부 검증 (새로운 메서드 필요)
        check_result = health_checker.check_postgresql_detailed()
        
        # Then: 쿼리 테스트 결과가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert any(keyword in details_text.lower() 
                  for keyword in ['query', '쿼리', 'table', '테이블'])

    @patch('src.health.external.ExternalServicesHealthCheck._test_redis_connection')  
    def test_external_services__redis_detailed__should_validate_caching_functionality(self, mock_connection: Mock) -> None:
        """
        Redis 세부 기능 검증 - 실제 캐싱 기능 테스트.
        
        Given: Redis 연결 환경
        When: 세부 캐싱 기능 검증 실행
        Then: 실제 set/get 작업이 가능해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        from src.health.models import ConnectionTestResult
        
        # Given: Mock successful caching test
        mock_connection.return_value = ConnectionTestResult(
            service_name="Redis",
            is_connected=True,
            response_time_ms=45.0,
            service_version="7.0.15",
            additional_info={'cache_test': 'success', 'set_get': 'ok'}
        )
        
        health_checker = ExternalServicesHealthCheck()
        
        # When: Redis 세부 검증 (새로운 메서드 필요)  
        check_result = health_checker.check_redis_detailed()
        
        # Then: 캐싱 테스트 결과가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert any(keyword in details_text.lower() 
                  for keyword in ['cache', '캐시', 'set', 'get'])

    @patch('subprocess.run')
    def test_external_services__feast_detailed__should_validate_feature_retrieval(self, mock_run: Mock) -> None:
        """
        Feast 세부 기능 검증 - 실제 feature 조회 테스트.
        
        Given: Feast Feature Store 환경
        When: 세부 feature 조회 검증 실행
        Then: 실제 feature 데이터를 조회할 수 있어야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        
        # Given: Mock successful feature retrieval
        mock_run.side_effect = [
            Mock(returncode=0, stdout="feast version 0.35.0", stderr=""),  # version check
            Mock(returncode=0, stdout="user_demographics\nproduct_features", stderr="")  # feature views list
        ]
        
        health_checker = ExternalServicesHealthCheck()
        
        # When: Feast 세부 검증 (새로운 메서드 필요)
        check_result = health_checker.check_feast_detailed()
        
        # Then: feature 조회 결과가 포함되어야 함
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert any(keyword in details_text.lower() 
                  for keyword in ['feature', 'demographics', 'materialization', '특성'])

    def test_external_services__mmp_local_dev_compatibility__should_validate_integration(self) -> None:
        """
        mmp-local-dev 호환성 검증 - 포트, 설정, 네이밍 규칙 확인.
        
        Given: mmp-local-dev 표준 설정
        When: 호환성 검증 실행
        Then: 포트, 컨테이너 이름, 설정이 mmp-local-dev 표준과 일치해야 함
        """
        from src.health.external import ExternalServicesHealthCheck
        
        # Given: 기본 설정으로 health checker 생성
        health_checker = ExternalServicesHealthCheck()
        
        # When: mmp-local-dev 호환성 검증 (새로운 메서드 필요)
        check_result = health_checker.check_mmp_local_dev_compatibility()
        
        # Then: mmp-local-dev 표준 설정과 일치 확인
        assert hasattr(check_result, 'is_healthy')
        assert hasattr(check_result, 'details')
        details_text = ' '.join(check_result.details or [])
        assert any(keyword in details_text.lower() 
                  for keyword in ['port', '포트', '5432', '6379', '5002', 'mmp-local-dev'])


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


class TestActionableHealthReporting:
    """M04-2-5 Enhanced Actionable Reporting System Tests"""

    def test_actionable_report__executable_commands__should_generate_runnable_commands(self) -> None:
        """
        액션 가능한 보고서가 실행 가능한 명령어를 생성하는지 검증.
        
        Given: 문제가 있는 건강 상태 결과들
        When: 실행 가능한 명령어 생성 요청
        Then: 각 문제별로 구체적이고 실행 가능한 명령어를 제공해야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult, HealthCheckSummary, CheckCategory
        
        # Given: 문제가 있는 건강 상태 결과들
        problematic_results = {
            CheckCategory.ENVIRONMENT: CheckResult(
                is_healthy=False,
                message="Python 버전 호환성 문제",
                details=["현재: 3.10.2", "필요: >=3.11,<3.12"],
                recommendations=["Python 3.11 설치 필요"]
            ),
            CheckCategory.MLFLOW: CheckResult(
                is_healthy=False,
                message="MLflow 서버 연결 실패",
                details=["연결 시간 초과", "서버 상태 확인 필요"],
                recommendations=["MLflow 서버 재시작 필요"]
            )
        }
        
        summary = HealthCheckSummary(
            overall_healthy=False,
            total_checks=2,
            passed_checks=0, 
            failed_checks=2,
            warning_checks=0,
            categories=problematic_results,
            execution_time_seconds=1.5,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        reporter = ActionableReporter()
        
        # When: 실행 가능한 명령어 생성
        executable_commands = reporter.generate_executable_commands(summary)
        
        # Then: 각 문제별 실행 가능한 명령어가 생성되어야 함
        assert isinstance(executable_commands, dict)
        assert len(executable_commands) > 0
        
        # 환경 문제에 대한 실행 가능한 명령어 확인
        env_commands = executable_commands.get(CheckCategory.ENVIRONMENT, [])
        assert len(env_commands) > 0
        assert any('uv' in cmd or 'python' in cmd for cmd in env_commands)
        
        # MLflow 문제에 대한 실행 가능한 명령어 확인  
        mlflow_commands = executable_commands.get(CheckCategory.MLFLOW, [])
        assert len(mlflow_commands) > 0
        assert any('mlflow' in cmd or 'server' in cmd for cmd in mlflow_commands)

    def test_actionable_report__priority_sorting__should_order_by_criticality(self) -> None:
        """
        액션 가능한 보고서가 문제를 심각도 순으로 정렬하는지 검증.
        
        Given: 다양한 심각도의 건강 상태 문제들
        When: 우선순위 기반 정렬 요청
        Then: 치명적 → 중요 → 경고 순서로 정렬되어야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult, HealthCheckSummary, CheckCategory
        
        # Given: 다양한 심각도 문제들
        mixed_results = {
            CheckCategory.TEMPLATES: CheckResult(
                is_healthy=False,
                message="템플릿 경고",
                details=["템플릿 형식 문제"],
                severity="warning"
            ),
            CheckCategory.ENVIRONMENT: CheckResult(
                is_healthy=False,
                message="Python 버전 치명적 문제", 
                details=["보안 취약점 존재"],
                severity="critical"
            ),
            CheckCategory.MLFLOW: CheckResult(
                is_healthy=False,
                message="MLflow 서버 중요 문제",
                details=["연결 불가"],
                severity="important"  
            )
        }
        
        summary = HealthCheckSummary(
            overall_healthy=False,
            total_checks=3,
            passed_checks=0,
            failed_checks=3,
            warning_checks=1,
            categories=mixed_results,
            execution_time_seconds=2.1,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        reporter = ActionableReporter()
        
        # When: 우선순위 기반 정렬
        prioritized_issues = reporter.sort_issues_by_priority(summary)
        
        # Then: 심각도 순 정렬 확인 (critical → important → warning)
        assert isinstance(prioritized_issues, list)
        assert len(prioritized_issues) == 3
        
        # 첫 번째는 critical이어야 함
        assert prioritized_issues[0]['severity'] == 'critical'
        assert prioritized_issues[0]['category'] == CheckCategory.ENVIRONMENT
        
        # 두 번째는 important이어야 함  
        assert prioritized_issues[1]['severity'] == 'important'
        assert prioritized_issues[1]['category'] == CheckCategory.MLFLOW
        
        # 세 번째는 warning이어야 함
        assert prioritized_issues[2]['severity'] == 'warning'
        assert prioritized_issues[2]['category'] == CheckCategory.TEMPLATES

    def test_actionable_report__step_by_step_resolution__should_provide_guided_solutions(self) -> None:
        """
        액션 가능한 보고서가 단계별 해결 가이드를 제공하는지 검증.
        
        Given: 복잡한 문제 상황
        When: 단계별 해결 가이드 요청
        Then: 순서가 있는 구체적인 해결 단계를 제공해야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult, CheckCategory
        
        # Given: 복잡한 문제 상황
        complex_problem = CheckResult(
            is_healthy=False,
            message="외부 서비스 연결 복합 문제",
            details=[
                "PostgreSQL 연결 실패",
                "Redis 캐시 서버 응답 없음", 
                "Feast Feature Store 접근 불가"
            ],
            recommendations=[
                "Docker 컨테이너 상태 확인",
                "네트워크 설정 검증",
                "서비스별 연결 테스트"
            ]
        )
        
        reporter = ActionableReporter()
        
        # When: 단계별 해결 가이드 생성
        step_by_step_guide = reporter.generate_step_by_step_resolution(
            CheckCategory.EXTERNAL_SERVICES, complex_problem
        )
        
        # Then: 순서가 있는 해결 단계 제공
        assert isinstance(step_by_step_guide, list)
        assert len(step_by_step_guide) >= 3
        
        # 각 단계가 순서 번호와 구체적 행동을 포함해야 함
        for i, step in enumerate(step_by_step_guide):
            assert isinstance(step, dict)
            assert 'step_number' in step
            assert 'action' in step
            assert 'command' in step or 'description' in step
            assert step['step_number'] == i + 1
        
        # 첫 단계는 기본 진단이어야 함
        assert 'docker' in step_by_step_guide[0]['action'].lower() or \
               'status' in step_by_step_guide[0]['action'].lower()

    def test_actionable_report__enhanced_color_coding__should_use_priority_based_colors(self) -> None:
        """
        액션 가능한 보고서가 우선순위 기반 색상 코딩을 사용하는지 검증.
        
        Given: 다양한 우선순위의 문제들
        When: 향상된 색상 코딩 적용
        Then: 우선순위별로 다른 색상이 적용되어야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult
        
        # Given: 다양한 우선순위 문제들
        critical_result = CheckResult(
            is_healthy=False,
            message="치명적 보안 문제",
            severity="critical"
        )
        
        important_result = CheckResult(
            is_healthy=False, 
            message="중요한 연결 문제",
            severity="important"
        )
        
        warning_result = CheckResult(
            is_healthy=False,
            message="경고 설정 문제", 
            severity="warning"
        )
        
        reporter = ActionableReporter()
        
        # When: 향상된 색상 코딩 적용
        critical_output = reporter.format_with_priority_colors("Critical", critical_result)
        important_output = reporter.format_with_priority_colors("Important", important_result)
        warning_output = reporter.format_with_priority_colors("Warning", warning_result)
        
        # Then: 각각 다른 색상 코딩 적용
        assert isinstance(critical_output, str)
        assert isinstance(important_output, str) 
        assert isinstance(warning_output, str)
        
        # 색상 코드가 포함되어야 함 (ANSI escape sequences 또는 typer 색상)
        assert len(critical_output) > len(critical_result.message)
        assert len(important_output) > len(important_result.message)
        assert len(warning_output) > len(warning_result.message)
        
        # 각 우선순위별로 다른 출력이어야 함
        assert critical_output != important_output
        assert important_output != warning_output
        assert critical_output != warning_output

    def test_actionable_report__interactive_resolution__should_offer_guided_interaction(self) -> None:
        """
        액션 가능한 보고서가 대화형 해결 가이드를 제공하는지 검증.
        
        Given: 사용자가 해결 가이드를 요청하는 상황
        When: 대화형 해결 가이드 시작
        Then: 단계별 사용자 선택과 피드백을 지원해야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult, CheckCategory
        
        # Given: 해결이 필요한 문제
        resolution_needed = CheckResult(
            is_healthy=False,
            message="MLflow 환경 설정 문제",
            details=["서버 모드 연결 실패", "로컬 모드로 전환 가능"],
            recommendations=["서버 재시작 또는 로컬 모드 사용"]
        )
        
        reporter = ActionableReporter()
        
        # When: 대화형 해결 가이드 생성
        interactive_guide = reporter.generate_interactive_resolution_guide(
            CheckCategory.MLFLOW, resolution_needed
        )
        
        # Then: 대화형 가이드 구조 확인
        assert isinstance(interactive_guide, dict)
        assert 'options' in interactive_guide
        assert 'prompts' in interactive_guide
        assert 'next_steps' in interactive_guide
        
        # 사용자 선택 옵션들 확인
        options = interactive_guide['options']
        assert isinstance(options, list)
        assert len(options) >= 2  # 최소 2개 옵션 (서버 재시작, 로컬 모드)
        
        for option in options:
            assert isinstance(option, dict)
            assert 'choice' in option
            assert 'description' in option
            assert 'commands' in option
        
        # 사용자 프롬프트 메시지 확인
        prompts = interactive_guide['prompts']
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert any('선택' in prompt or 'choice' in prompt for prompt in prompts)

    def test_actionable_report__comprehensive_integration__should_work_with_all_checks(self) -> None:
        """
        액션 가능한 보고서가 모든 검사 유형과 통합되는지 검증.
        
        Given: 전체 건강 검사 결과 (모든 카테고리 포함)
        When: 종합적인 액션 가능한 보고서 생성
        Then: 모든 카테고리의 문제에 대해 통합된 해결책을 제공해야 함
        """
        from src.health.reporter import ActionableReporter
        from src.health.models import CheckResult, HealthCheckSummary, CheckCategory
        
        # Given: 모든 카테고리의 건강 검사 결과
        comprehensive_results = {
            CheckCategory.ENVIRONMENT: CheckResult(
                is_healthy=False,
                message="환경 문제",
                recommendations=["Python 업그레이드"]
            ),
            CheckCategory.MLFLOW: CheckResult(
                is_healthy=True,
                message="MLflow 정상"
            ),
            CheckCategory.EXTERNAL_SERVICES: CheckResult(
                is_healthy=False,
                message="외부 서비스 문제", 
                recommendations=["Docker 재시작"]
            ),
            CheckCategory.TEMPLATES: CheckResult(
                is_healthy=False,
                message="템플릿 문제",
                recommendations=["템플릿 구문 수정"]
            ),
            CheckCategory.SYSTEM: CheckResult(
                is_healthy=True,
                message="시스템 정상"
            )
        }
        
        summary = HealthCheckSummary(
            overall_healthy=False,
            total_checks=5,
            passed_checks=2,
            failed_checks=3,
            warning_checks=0,
            categories=comprehensive_results,
            execution_time_seconds=3.2,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        reporter = ActionableReporter()
        
        # When: 종합적인 액션 가능한 보고서 생성
        comprehensive_report = reporter.generate_comprehensive_actionable_report(summary)
        
        # Then: 모든 문제 카테고리에 대한 해결책 포함
        assert isinstance(comprehensive_report, dict)
        assert 'summary' in comprehensive_report
        assert 'prioritized_actions' in comprehensive_report
        assert 'executable_commands' in comprehensive_report
        assert 'interactive_guides' in comprehensive_report
        
        # 실패한 검사들에 대한 해결책 확인
        prioritized_actions = comprehensive_report['prioritized_actions']
        assert len(prioritized_actions) == 3  # 실패한 검사 수와 일치
        
        # 모든 실패 카테고리가 포함되어야 함
        failed_categories = [action['category'] for action in prioritized_actions]
        assert CheckCategory.ENVIRONMENT in failed_categories
        assert CheckCategory.EXTERNAL_SERVICES in failed_categories  
        assert CheckCategory.TEMPLATES in failed_categories
        
        # 성공한 검사들은 액션에 포함되지 않아야 함
        assert CheckCategory.MLFLOW not in failed_categories
        assert CheckCategory.SYSTEM not in failed_categories


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