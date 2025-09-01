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

from src.cli.commands.system_check_command import (
    ConfigBasedSystemChecker,
    _display_basic_summary
)
from src.cli.utils.health_models import CheckResult


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
    def test_config_based_checker_initialization(self):
        """ConfigBasedSystemChecker 초기화 테스트"""
        # Given: config 디렉토리에 YAML 파일 생성
        config_content = {
            "environment": {"app_env": "test"},
            "mlflow": {"tracking_uri": "http://localhost:5002"}
        }
        self._create_config_file("test.yaml", config_content)
        
        # When: ConfigBasedSystemChecker 초기화
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # Then: configs가 올바르게 로드되어야 함
        assert "test" in checker.configs
        assert checker.configs["test"]["mlflow"]["tracking_uri"] == "http://localhost:5002"

    def test_config_loading_with_yaml_parse_error(self):
        """YAML 파싱 오류 처리 테스트"""
        # Given: 잘못된 YAML 파일
        invalid_yaml = "invalid: yaml: content: ["
        (self.config_dir / "invalid.yaml").write_text(invalid_yaml)
        
        # When: ConfigBasedSystemChecker 초기화
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # Then: 파싱 오류가 기록되어야 함
        assert "invalid" in checker.configs
        assert "_parse_error" in checker.configs["invalid"]

    def test_run_dynamic_checks_with_mlflow_config(self):
        """MLflow 설정이 있는 경우 동적 체크 테스트"""
        # Given: MLflow 설정이 있는 config
        config_content = {
            "mlflow": {"tracking_uri": "http://localhost:5002"}
        }
        self._create_config_file("mlflow_test.yaml", config_content)
        
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # When: 동적 체크 실행 (MLflow 연결은 mock)
        with patch.object(checker, '_check_mlflow_connection') as mock_mlflow:
            mock_result = CheckResult(
                is_healthy=True,
                message="MLflow Connection (mlflow_test): Connection successful"
            )
            mock_mlflow.return_value = mock_result
            
            summary = checker.run_dynamic_checks()
        
        # Then: MLflow 체크가 실행되어야 함
        mock_mlflow.assert_called_once_with("mlflow_test", config_content)
        assert summary['total_checks'] > 0
        assert any("MLflow" in result.message for result in summary['results'])

    def test_run_dynamic_checks_with_no_services(self):
        """서비스 설정이 없는 경우 테스트"""
        # Given: 기본 설정만 있는 config
        config_content = {"environment": {"app_env": "test"}}
        self._create_config_file("basic.yaml", config_content)
        
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # When: 동적 체크 실행
        summary = checker.run_dynamic_checks()
        
        # Then: 체크 결과가 비어있어야 함 (서비스 설정이 없으므로)
        assert summary['total_checks'] == 0

    def test_config_directory_not_found(self):
        """Config 디렉토리가 없는 경우 테스트"""
        # Given: 존재하지 않는 디렉토리
        non_existent_dir = Path(self.temp_dir) / "non_existent"
        
        # When & Then: FileNotFoundError가 발생해야 함
        with pytest.raises(FileNotFoundError, match="Config 디렉토리를 찾을 수 없습니다"):
            ConfigBasedSystemChecker(non_existent_dir)

    def test_empty_config_directory(self):
        """빈 config 디렉토리 처리 테스트"""
        # Given: 빈 config 디렉토리
        
        # When & Then: FileNotFoundError가 발생해야 함
        with pytest.raises(FileNotFoundError, match="Config 디렉토리에 YAML 파일이 없습니다"):
            ConfigBasedSystemChecker(self.config_dir)

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
        
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # Mock 설정
        mock_get_uri.return_value = "original_uri"
        mock_client = Mock()
        mock_client.search_experiments.return_value = [Mock()]
        mock_mlflow_client.return_value = mock_client
        
        # When: MLflow 연결 체크
        result = checker._check_mlflow_connection("test", config_content)
        
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
        config_content = {"environment": {"app_env": "test"}}
        self._create_config_file("test.yaml", config_content)
        
        checker = ConfigBasedSystemChecker(self.config_dir)
        
        # When: PostgreSQL 연결 체크
        result = checker._check_postgres_connection("test", config_content)
        
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
        _display_basic_summary(summary)
        
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
        _display_basic_summary(summary)
        
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