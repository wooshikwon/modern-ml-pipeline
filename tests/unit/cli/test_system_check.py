"""
System Check Command Tests  
Phase 5: Config 기반 동적 시스템 체크 테스트

CLAUDE.md 원칙 준수:
- TDD 기반 테스트
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.cli.system_check.manager import DynamicServiceChecker
from src.cli.system_check.models import CheckResult
from src.cli.commands.system_check_command import _display_summary


# 이 테스트 파일은 conftest.py의 global fixture 사용을 피하기 위해 독립적으로 실행됩니다
pytestmark = pytest.mark.unit


class TestSystemCheckCommand:
    """System Check 명령어 테스트"""
    
    def setup_method(self):
        """각 테스트 전 임시 config 디렉토리 생성"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()

    def teardown_method(self):
        """각 테스트 후 임시 디렉토리 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.core
    @pytest.mark.unit
    def test_dynamic_checker_initialization(self):
        """DynamicServiceChecker 초기화 테스트"""
        # Given: DynamicServiceChecker 생성
        checker = DynamicServiceChecker()
        
        # Then: 체커가 올바르게 초기화되어야 함
        assert checker is not None
        assert len(checker.checkers) > 0  # 기본 체커들이 등록되어 있어야 함

    def test_checker_can_check_method(self):
        """DynamicServiceChecker can_check 메서드 테스트"""
        # Given: MLflow 설정이 있는 config
        config = {
            "ml_tracking": {"tracking_uri": "http://localhost:5000"}
        }
        
        # When: DynamicServiceChecker로 체크
        checker = DynamicServiceChecker()
        active_services = checker.get_active_services(config)
        
        # Then: MLflow가 활성 서비스로 감지되어야 함
        assert "MLflow" in active_services

    def test_run_dynamic_checks_with_mlflow_config(self):
        """MLflow 설정이 있는 경우 동적 체크 테스트"""
        # Given: MLflow 설정이 있는 config
        config_content = {
            "mlflow": {"tracking_uri": "http://localhost:5002"}
        }
        self._create_config_file("mlflow_test.yaml", config_content)
        
        checker = DynamicServiceChecker()
        
        # When: 동적 체크 실행 (MLflow 연결은 mock)
        with patch('src.cli.system_check.checkers.mlflow.MLflowChecker.check') as mock_check:
            mock_result = CheckResult(
                is_healthy=True,
                service_name="MLflow",
                message="MLflow Connection (mlflow_test): Connection successful"
            )
            mock_check.return_value = mock_result
            
            results = checker.run_checks(config_content)
        
        # Then: MLflow 체크가 실행되어야 함
        assert len(results) > 0
        assert any(r.service_name == "MLflow" for r in results)

    def test_run_dynamic_checks_with_no_services(self):
        """서비스 설정이 없는 경우 테스트"""
        # Given: 기본 설정만 있는 config
        config_content = {"environment": {"env_name": "test"}}
        self._create_config_file("basic.yaml", config_content)
        
        checker = DynamicServiceChecker()
        
        # When: 동적 체크 실행
        results = checker.run_checks(config_content)
        
        # Then: 체크 결과가 비어있어야 함 (서비스 설정이 없으므로)
        assert len(results) == 0

    def test_summary_stats_generation(self):
        """Summary stats 생성 테스트"""
        # Given: 테스트 결과들
        results = [
            CheckResult(is_healthy=True, service_name="MLflow", message="OK"),
            CheckResult(is_healthy=False, service_name="PostgreSQL", message="Failed", severity="critical")
        ]
        
        # When: Summary stats 생성
        stats = checker.get_summary_stats(results)
        
        # Then: 올바른 통계가 생성되어야 함
        assert stats['total'] == 2
        assert stats['healthy'] == 1
        assert stats['unhealthy'] == 1
        assert stats['critical_failures'] == 1

    def test_get_active_services(self):
        """활성 서비스 목록 가져오기 테스트"""
        # Given: 여러 서비스가 설정된 config
        config = {
            "ml_tracking": {"tracking_uri": "http://localhost:5000"},
            "data_adapters": {
                "adapters": {
                    "sql": {
                        "config": {"connection_uri": "postgresql://localhost/db"}
                    }
                }
            }
        }
        
        # When: 활성 서비스 조회
        active_services = checker.get_active_services(config)
        
        # Then: 설정된 서비스들이 반환되어야 함
        assert len(active_services) >= 2  # 최소 MLflow와 PostgreSQL

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_tracking_uri')  
    @patch('mlflow.tracking.MlflowClient')
    def test_check_mlflow_connection_success(self, mock_mlflow_client, mock_get_uri, mock_set_uri):
        """MLflow 연결 성공 테스트"""
        # Given: MLflow 설정
        config_content = {
            "mlflow": {"tracking_uri": "http://localhost:5002"}
        }
        self._create_config_file("test.yaml", config_content)
        
        checker = DynamicServiceChecker()
        
        # Mock 설정
        mock_get_uri.return_value = "original_uri"
        mock_client = Mock()
        mock_client.search_experiments.return_value = [Mock()]
        mock_mlflow_client.return_value = mock_client
        
        # When: MLflow 연결 체크
        from src.cli.system_check.checkers.mlflow import MLflowChecker
        mlflow_checker = MLflowChecker()
        if mlflow_checker.can_check(config_content):
            result = mlflow_checker.check(config_content)
        else:
            result = None
        
        # Then: 성공 결과가 반환되어야 함
        assert result is not None
        assert result.is_healthy is True
        assert "MLflow" in result.message
        assert "연결 성공" in result.message
        
        # Mock 호출 검증
        mock_mlflow_client.assert_called_once()
        mock_set_uri.assert_called()  # 2번 호출됨 (설정, 복원)

    def test_check_postgres_connection_no_config(self):
        """PostgreSQL 설정이 없는 경우 테스트"""
        # Given: PostgreSQL 설정이 없는 config
        config_content = {"environment": {"env_name": "test"}}
        self._create_config_file("test.yaml", config_content)
        
        checker = DynamicServiceChecker()
        
        # When: PostgreSQL 연결 체크  
        from src.cli.system_check.checkers.postgresql import PostgreSQLChecker
        postgres_checker = PostgreSQLChecker()
        if postgres_checker.can_check(config_content):
            result = postgres_checker.check(config_content)
        else:
            result = None
        
        # Then: None이 반환되어야 함 (설정이 없으므로)
        assert result is None

    def test_display_basic_summary_all_passed(self, capsys):
        """모든 체크가 성공한 경우 요약 표시 테스트"""
        # Given: 성공한 체크 결과들
        results = [
            CheckResult(is_healthy=True, message="Test 1: Success"),
            CheckResult(is_healthy=True, message="Test 2: Success")
        ]
        summary = {
            'results': results,
            'overall_healthy': True,
            'total_checks': 2,
            'passed_checks': 2,
            'failed_checks': 0
        }
        
        # When: 기본 요약 표시
        from rich.console import Console
        console = Console()
        _display_summary(summary, console)
        
        # Then: 성공 메시지가 출력되어야 함
        captured = capsys.readouterr()
        assert "모든 시스템이 정상적으로 연결되었습니다" in captured.out
        assert "2" in captured.out  # 숫자 확인 (Rich formatting 때문에 정확한 문자열 매칭 어려움)
        assert "0" in captured.out

    def test_display_basic_summary_with_failures(self, capsys):
        """일부 체크가 실패한 경우 요약 표시 테스트"""
        # Given: 실패한 체크 결과들
        results = [
            CheckResult(is_healthy=True, message="Test 1: Success"),
            CheckResult(is_healthy=False, message="Test 2: Connection failed", recommendations=["Fix connection"])
        ]
        summary = {
            'results': results,
            'overall_healthy': False,
            'total_checks': 2,
            'passed_checks': 1,
            'failed_checks': 1
        }
        
        # When: 기본 요약 표시
        from rich.console import Console
        console = Console()
        _display_summary(summary, console)
        
        # Then: 실패 메시지가 출력되어야 함
        captured = capsys.readouterr()
        assert "연결 문제가 발견되었습니다" in captured.out
        assert "1" in captured.out  # 성공과 실패 각각 1개
        assert "Connection failed" in captured.out  # Rich 테이블에서 메시지 일부만 매칭

    def _create_config_file(self, filename: str, content: dict) -> None:
        """테스트용 config 파일 생성 헬퍼"""
        import yaml
        config_path = self.config_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, default_flow_style=False, allow_unicode=True, indent=2)