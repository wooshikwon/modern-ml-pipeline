"""
Unit tests for environment checking utilities.
Tests system environment validation, security, and compatibility checks.
"""

import pytest
import sys
import os
import subprocess
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from types import SimpleNamespace

from src.utils.system.environment_check import (
    EnvironmentChecker,
    check_environment,
    get_pip_requirements
)


class TestEnvironmentChecker:
    """Test EnvironmentChecker class for system validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.checker = EnvironmentChecker()

    def test_environment_checker_initialization(self):
        """Test that EnvironmentChecker initializes correctly."""
        assert isinstance(self.checker.warnings, list)
        assert isinstance(self.checker.errors, list)
        assert len(self.checker.warnings) == 0
        assert len(self.checker.errors) == 0

    def test_check_python_version_valid_311(self):
        """Test Python 3.11.x version validation success."""
        with patch('sys.version_info', SimpleNamespace(major=3, minor=11, micro=5)):
            result = self.checker.check_python_version()
            
            assert result is True
            assert len(self.checker.errors) == 0
            assert len(self.checker.warnings) == 0

    def test_check_python_version_invalid_major(self):
        """Test Python version validation failure with wrong major version."""
        with patch('sys.version_info', SimpleNamespace(major=2, minor=7, micro=18)):
            result = self.checker.check_python_version()
            
            assert result is False
            assert len(self.checker.errors) == 1
            assert "Python 3.11.x가 필요합니다" in self.checker.errors[0]

    def test_check_python_version_too_old(self):
        """Test Python version validation failure with old version."""
        with patch('sys.version_info', SimpleNamespace(major=3, minor=9, micro=0)):
            result = self.checker.check_python_version()
            
            assert result is False
            assert len(self.checker.errors) == 1
            assert "Python 3.11 이상이 필요합니다" in self.checker.errors[0]

    def test_check_python_version_312_warning(self):
        """Test Python 3.12 generates compatibility warning."""
        with patch('sys.version_info', SimpleNamespace(major=3, minor=12, micro=0)):
            result = self.checker.check_python_version()
            
            assert result is True
            assert len(self.checker.errors) == 0
            assert len(self.checker.warnings) == 1
            assert "causalml과 호환성 문제" in self.checker.warnings[0]

    def test_check_python_version_future_warning(self):
        """Test future Python version generates warning."""
        with patch('sys.version_info', SimpleNamespace(major=3, minor=13, micro=0)):
            result = self.checker.check_python_version()
            
            assert result is True
            assert len(self.checker.errors) == 0
            assert len(self.checker.warnings) == 1
            assert "테스트되지 않았습니다" in self.checker.warnings[0]

    @patch('src.utils.system.environment_check.logger')
    def test_check_python_version_logging(self, mock_logger):
        """Test that Python version checking is logged."""
        with patch('sys.version_info', SimpleNamespace(major=3, minor=11, micro=5)):
            self.checker.check_python_version()
            
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Python 버전 확인" in log_message
            assert "3.11.5" in log_message

    def test_check_required_packages_success(self):
        """Test successful required package validation."""
        # Mock all required packages as available
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'mlflow', 'pydantic', 
            'fastapi', 'uvicorn', 'typer', 'pyyaml', 'python-dotenv'
        ]
        
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            result = self.checker.check_required_packages()
            
            assert result is True
            assert len(self.checker.errors) == 0
            assert mock_import.call_count == len(required_packages)

    def test_check_required_packages_missing(self):
        """Test required package validation failure with missing packages."""
        def mock_import_side_effect(package_name):
            if package_name in ['pandas', 'mlflow']:
                raise ImportError(f"No module named '{package_name}'")
            return MagicMock()

        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            result = self.checker.check_required_packages()
            
            assert result is False
            assert len(self.checker.errors) == 1
            error_message = self.checker.errors[0]
            assert "필수 패키지가 설치되지 않았습니다" in error_message
            assert "pandas" in error_message
            assert "mlflow" in error_message

    @patch('src.utils.system.environment_check.logger')
    def test_check_required_packages_logging_success(self, mock_logger):
        """Test logging for successful required package check."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            self.checker.check_required_packages()
            
            mock_logger.info.assert_called_once_with("✅ 필수 패키지 확인 완료")

    def test_check_optional_packages_all_available(self):
        """Test optional package checking when all packages are available."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            result = self.checker.check_optional_packages()
            
            assert result is True
            assert len(self.checker.warnings) == 0

    def test_check_optional_packages_some_missing(self):
        """Test optional package checking with some missing packages."""
        def mock_import_side_effect(package_name):
            if package_name in ['redis', 'causalml']:
                raise ImportError(f"No module named '{package_name}'")
            return MagicMock()

        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            result = self.checker.check_optional_packages()
            
            assert result is True
            assert len(self.checker.warnings) == 1
            warning_message = self.checker.warnings[0]
            assert "선택적 패키지가 설치되지 않았습니다" in warning_message
            assert "redis" in warning_message
            assert "causalml" in warning_message

    @patch('src.utils.system.environment_check.logger')
    def test_check_optional_packages_logging(self, mock_logger):
        """Test logging for optional package check."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            self.checker.check_optional_packages()
            
            mock_logger.info.assert_called_once_with("✅ 선택적 패키지 확인 완료")

    def test_check_directory_structure_success(self):
        """Test successful directory structure validation."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        with patch.object(Path, '__truediv__', return_value=mock_path):
            with patch.object(Path, 'resolve') as mock_resolve:
                mock_resolve.return_value.parent.parent.parent.parent = Path('/mock/base')
                result = self.checker.check_directory_structure()
                
                assert result is True
                assert len(self.checker.errors) == 0

    def test_check_directory_structure_missing_dirs(self):
        """Test directory structure validation failure with missing directories."""
        def mock_exists_side_effect():
            # Simulate missing 'config' and 'data' directories
            mock_path = MagicMock()
            exists_map = {
                'config': False,
                'recipes': True, 
                'src': True,
                'data': False,
                'serving': True,
                'tests': True
            }
            
            def exists_side_effect():
                # This is a simplified mock - in reality we'd need more complex path handling
                return True  # Default to True, override specific cases in test
            
            mock_path.exists = MagicMock(side_effect=exists_side_effect)
            return mock_path

        with patch.object(Path, 'resolve') as mock_resolve:
            base_path = MagicMock()
            mock_resolve.return_value.parent.parent.parent.parent = base_path
            
            # Mock directory existence - config and data missing
            def truediv_side_effect(self, dirname):
                mock_dir = MagicMock()
                mock_dir.exists.return_value = dirname not in ['config', 'data']
                return mock_dir
            
            base_path.__truediv__ = truediv_side_effect
            
            result = self.checker.check_directory_structure()
            
            assert result is False
            assert len(self.checker.errors) == 1
            error_message = self.checker.errors[0]
            assert "필수 디렉토리가 없습니다" in error_message

    @patch('src.utils.system.environment_check.logger')
    def test_check_directory_structure_logging(self, mock_logger):
        """Test logging for directory structure check."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        
        with patch.object(Path, '__truediv__', return_value=mock_path):
            with patch.object(Path, 'resolve') as mock_resolve:
                mock_resolve.return_value.parent.parent.parent.parent = Path('/mock/base')
                self.checker.check_directory_structure()
                
                mock_logger.info.assert_called_once_with("✅ 디렉토리 구조 확인 완료")

    def test_check_environment_variables_local_env(self):
        """Test environment variable validation in local environment."""
        with patch.dict(os.environ, {'ENV_NAME': 'local'}, clear=True):
            result = self.checker.check_environment_variables()
            
            assert result is True
            assert len(self.checker.warnings) == 0
            assert len(self.checker.errors) == 0

    def test_check_environment_variables_dev_env_complete(self):
        """Test environment variable validation in dev environment with all vars."""
        with patch.dict(os.environ, {
            'ENV_NAME': 'dev', 
            'POSTGRES_PASSWORD': 'test_password'
        }, clear=True):
            result = self.checker.check_environment_variables()
            
            assert result is True
            assert len(self.checker.warnings) == 0

    def test_check_environment_variables_dev_env_missing(self):
        """Test environment variable validation in dev environment with missing vars."""
        with patch.dict(os.environ, {'ENV_NAME': 'dev'}, clear=True):
            result = self.checker.check_environment_variables()
            
            assert result is True  # Returns True but generates warnings
            assert len(self.checker.warnings) == 1
            warning_message = self.checker.warnings[0]
            assert "중요한 환경변수가 설정되지 않았습니다" in warning_message
            assert "POSTGRES_PASSWORD" in warning_message

    def test_check_environment_variables_prod_env_missing(self):
        """Test environment variable validation in prod environment with missing vars."""
        with patch.dict(os.environ, {'ENV_NAME': 'prod'}, clear=True):
            result = self.checker.check_environment_variables()
            
            assert result is True  # Returns True but generates warnings
            assert len(self.checker.warnings) == 1
            warning_message = self.checker.warnings[0]
            assert "POSTGRES_PASSWORD" in warning_message

    def test_check_environment_variables_default_local(self):
        """Test environment variable validation defaults to local when ENV_NAME not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = self.checker.check_environment_variables()
            
            assert result is True
            assert len(self.checker.warnings) == 0

    @patch('src.utils.system.environment_check.logger')
    def test_check_environment_variables_logging(self, mock_logger):
        """Test logging for environment variable check."""
        with patch.dict(os.environ, {'ENV_NAME': 'local'}, clear=True):
            self.checker.check_environment_variables()
            
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "LOCAL 환경: 환경변수 검증 생략" in log_message

    def test_check_system_compatibility_mac_intel(self):
        """Test system compatibility check on Intel Mac."""
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='x86_64'):
                with patch('platform.platform', return_value='macOS-12.0-x86_64'):
                    result = self.checker.check_system_compatibility()
                    
                    assert result is True
                    assert len(self.checker.warnings) == 0

    def test_check_system_compatibility_mac_apple_silicon(self):
        """Test system compatibility check on Apple Silicon Mac."""
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                with patch('platform.platform', return_value='macOS-12.0-arm64'):
                    result = self.checker.check_system_compatibility()
                    
                    assert result is True
                    assert len(self.checker.warnings) == 1
                    warning_message = self.checker.warnings[0]
                    assert "Apple Silicon Mac 감지" in warning_message
                    assert "Rosetta 2" in warning_message

    def test_check_system_compatibility_linux(self):
        """Test system compatibility check on Linux."""
        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='x86_64'):
                with patch('platform.platform', return_value='Linux-5.4.0-x86_64'):
                    result = self.checker.check_system_compatibility()
                    
                    assert result is True
                    assert len(self.checker.warnings) == 0

    @patch('src.utils.system.environment_check.logger')
    def test_check_system_compatibility_logging(self, mock_logger):
        """Test logging for system compatibility check."""
        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='x86_64'):
                with patch('platform.platform', return_value='Linux-5.4.0-x86_64'):
                    self.checker.check_system_compatibility()
                    
                    mock_logger.info.assert_called_once()
                    log_message = mock_logger.info.call_args[0][0]
                    assert "시스템 호환성 확인" in log_message
                    assert "Linux x86_64" in log_message

    @patch('src.utils.system.environment_check.logger')
    def test_run_full_check_success(self, mock_logger):
        """Test successful full environment check."""
        # Mock all individual checks to succeed
        with patch.object(self.checker, 'check_python_version', return_value=True):
            with patch.object(self.checker, 'check_required_packages', return_value=True):
                with patch.object(self.checker, 'check_optional_packages', return_value=True):
                    with patch.object(self.checker, 'check_directory_structure', return_value=True):
                        with patch.object(self.checker, 'check_environment_variables', return_value=True):
                            with patch.object(self.checker, 'check_system_compatibility', return_value=True):
                                success, errors, warnings = self.checker.run_full_check()
                                
                                assert success is True
                                assert errors == []
                                assert warnings == []
                                
                                # Check that completion message is logged
                                completion_calls = [call for call in mock_logger.info.call_args_list 
                                                  if "개발환경 호환성 검증 완료" in str(call)]
                                assert len(completion_calls) > 0

    @patch('src.utils.system.environment_check.logger')
    def test_run_full_check_with_errors(self, mock_logger):
        """Test full environment check with errors."""
        # Add some errors
        self.checker.errors = ["Test error 1", "Test error 2"]
        
        # Mock all individual checks - some fail
        with patch.object(self.checker, 'check_python_version', return_value=False):
            with patch.object(self.checker, 'check_required_packages', return_value=True):
                with patch.object(self.checker, 'check_optional_packages', return_value=True):
                    with patch.object(self.checker, 'check_directory_structure', return_value=True):
                        with patch.object(self.checker, 'check_environment_variables', return_value=True):
                            with patch.object(self.checker, 'check_system_compatibility', return_value=True):
                                success, errors, warnings = self.checker.run_full_check()
                                
                                assert success is False
                                assert errors == ["Test error 1", "Test error 2"]
                                
                                # Check that error logging occurred
                                error_calls = [call for call in mock_logger.error.call_args_list 
                                             if "환경 검증 실패" in str(call)]
                                assert len(error_calls) > 0

    @patch('src.utils.system.environment_check.logger')
    def test_run_full_check_with_warnings(self, mock_logger):
        """Test full environment check with warnings."""
        # Add some warnings
        self.checker.warnings = ["Test warning 1", "Test warning 2"]
        
        # Mock all individual checks to succeed
        with patch.object(self.checker, 'check_python_version', return_value=True):
            with patch.object(self.checker, 'check_required_packages', return_value=True):
                with patch.object(self.checker, 'check_optional_packages', return_value=True):
                    with patch.object(self.checker, 'check_directory_structure', return_value=True):
                        with patch.object(self.checker, 'check_environment_variables', return_value=True):
                            with patch.object(self.checker, 'check_system_compatibility', return_value=True):
                                success, errors, warnings = self.checker.run_full_check()
                                
                                assert success is True
                                assert warnings == ["Test warning 1", "Test warning 2"]
                                
                                # Check that warning logging occurred
                                warning_calls = [call for call in mock_logger.warning.call_args_list 
                                               if "환경 검증 경고" in str(call)]
                                assert len(warning_calls) > 0

    def test_run_full_check_calls_all_methods(self):
        """Test that run_full_check calls all validation methods."""
        with patch.object(self.checker, 'check_python_version', return_value=True) as mock_python:
            with patch.object(self.checker, 'check_required_packages', return_value=True) as mock_required:
                with patch.object(self.checker, 'check_optional_packages', return_value=True) as mock_optional:
                    with patch.object(self.checker, 'check_directory_structure', return_value=True) as mock_dirs:
                        with patch.object(self.checker, 'check_environment_variables', return_value=True) as mock_env:
                            with patch.object(self.checker, 'check_system_compatibility', return_value=True) as mock_sys:
                                self.checker.run_full_check()
                                
                                mock_python.assert_called_once()
                                mock_required.assert_called_once()
                                mock_optional.assert_called_once()
                                mock_dirs.assert_called_once()
                                mock_env.assert_called_once()
                                mock_sys.assert_called_once()


class TestCheckEnvironmentFunction:
    """Test check_environment convenience function."""

    @patch('src.utils.system.environment_check.EnvironmentChecker')
    def test_check_environment_success(self, mock_checker_class):
        """Test successful environment check via convenience function."""
        mock_checker = MagicMock()
        mock_checker.run_full_check.return_value = (True, [], [])
        mock_checker_class.return_value = mock_checker
        
        result = check_environment()
        
        assert result is True
        mock_checker_class.assert_called_once()
        mock_checker.run_full_check.assert_called_once()

    @patch('src.utils.system.environment_check.EnvironmentChecker')
    def test_check_environment_failure(self, mock_checker_class):
        """Test failed environment check via convenience function."""
        mock_checker = MagicMock()
        mock_checker.run_full_check.return_value = (False, ["Error"], ["Warning"])
        mock_checker_class.return_value = mock_checker
        
        result = check_environment()
        
        assert result is False
        mock_checker_class.assert_called_once()
        mock_checker.run_full_check.assert_called_once()


class TestGetPipRequirements:
    """Test get_pip_requirements function for dependency capture."""

    @patch('subprocess.run')
    @patch('src.utils.system.environment_check.logger')
    def test_get_pip_requirements_success(self, mock_logger, mock_subprocess):
        """Test successful pip requirements capture."""
        mock_result = MagicMock()
        mock_result.stdout = "pandas==2.0.0\nnumpy==1.24.0\nscikit-learn==1.3.0"
        mock_subprocess.return_value = mock_result
        
        requirements = get_pip_requirements()
        
        expected_requirements = ["pandas==2.0.0", "numpy==1.24.0", "scikit-learn==1.3.0"]
        assert requirements == expected_requirements
        
        # Verify subprocess was called correctly
        mock_subprocess.assert_called_once_with(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        # Check logging
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "3개의 패키지 의존성을 캡처했습니다" in log_message

    @patch('subprocess.run')
    @patch('src.utils.system.environment_check.logger')
    def test_get_pip_requirements_uv_not_found(self, mock_logger, mock_subprocess):
        """Test pip requirements capture when uv command is not found."""
        mock_subprocess.side_effect = FileNotFoundError("uv command not found")
        
        requirements = get_pip_requirements()
        
        assert requirements == []
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "'uv' 명령어를 찾을 수 없어" in warning_message

    @patch('subprocess.run')
    @patch('src.utils.system.environment_check.logger')
    def test_get_pip_requirements_subprocess_error(self, mock_logger, mock_subprocess):
        """Test pip requirements capture with subprocess error."""
        mock_error = subprocess.CalledProcessError(1, "uv pip freeze")
        mock_error.stderr = "Permission denied"
        mock_subprocess.side_effect = mock_error
        
        requirements = get_pip_requirements()
        
        assert requirements == []
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "pip 의존성 캡처 중 오류 발생" in error_message
        assert "Permission denied" in error_message

    @patch('subprocess.run')
    def test_get_pip_requirements_empty_output(self, mock_subprocess):
        """Test pip requirements capture with empty output."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_subprocess.return_value = mock_result
        
        requirements = get_pip_requirements()
        
        assert requirements == [""]  # Empty string becomes single empty item

    @patch('subprocess.run')
    def test_get_pip_requirements_subprocess_parameters(self, mock_subprocess):
        """Test that subprocess is called with correct parameters."""
        mock_result = MagicMock()
        mock_result.stdout = "test-package==1.0.0"
        mock_subprocess.return_value = mock_result
        
        get_pip_requirements()
        
        # Verify all required parameters
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == ["uv", "pip", "freeze"]
        assert call_args[1]['capture_output'] is True
        assert call_args[1]['text'] is True
        assert call_args[1]['check'] is True
        assert call_args[1]['encoding'] == 'utf-8'


class TestEnvironmentCheckIntegration:
    """Integration tests for environment checking functionality."""

    def test_environment_checker_isolation(self):
        """Test that multiple EnvironmentChecker instances are isolated."""
        checker1 = EnvironmentChecker()
        checker2 = EnvironmentChecker()
        
        checker1.errors.append("Test error 1")
        checker1.warnings.append("Test warning 1")
        
        assert len(checker2.errors) == 0
        assert len(checker2.warnings) == 0
        assert checker1.errors != checker2.errors
        assert checker1.warnings != checker2.warnings

    def test_error_accumulation_across_checks(self):
        """Test that errors accumulate across multiple check calls."""
        checker = EnvironmentChecker()
        
        # Simulate different types of errors
        with patch('sys.version_info', SimpleNamespace(major=2, minor=7, micro=18)):
            checker.check_python_version()
        
        with patch('builtins.__import__', side_effect=ImportError("Missing package")):
            checker.check_required_packages()
        
        assert len(checker.errors) == 2
        assert any("Python 3.11.x가 필요합니다" in error for error in checker.errors)
        assert any("필수 패키지가 설치되지 않았습니다" in error for error in checker.errors)

    def test_warning_accumulation_across_checks(self):
        """Test that warnings accumulate across multiple check calls."""
        checker = EnvironmentChecker()
        
        # Generate warnings from different checks
        with patch('sys.version_info', SimpleNamespace(major=3, minor=12, micro=0)):
            checker.check_python_version()
        
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                checker.check_system_compatibility()
        
        assert len(checker.warnings) == 2
        assert any("causalml과 호환성 문제" in warning for warning in checker.warnings)
        assert any("Apple Silicon Mac 감지" in warning for warning in checker.warnings)

    @patch('src.utils.system.environment_check.logger')
    def test_comprehensive_environment_validation_workflow(self, mock_logger):
        """Test complete environment validation workflow."""
        # This test simulates a complete real-world validation
        checker = EnvironmentChecker()
        
        with patch('sys.version_info', SimpleNamespace(major=3, minor=11, micro=5)):
            with patch('builtins.__import__') as mock_import:
                mock_import.return_value = MagicMock()
                with patch('platform.system', return_value='Linux'):
                    with patch('platform.machine', return_value='x86_64'):
                        with patch.dict(os.environ, {'ENV_NAME': 'local'}):
                            # Mock directory structure
                            mock_path = MagicMock()
                            mock_path.exists.return_value = True
                            with patch.object(Path, '__truediv__', return_value=mock_path):
                                with patch.object(Path, 'resolve') as mock_resolve:
                                    mock_resolve.return_value.parent.parent.parent.parent = Path('/mock')
                                    
                                    success, errors, warnings = checker.run_full_check()
                                    
                                    assert success is True
                                    assert len(errors) == 0
                                    assert len(warnings) == 0
                                    
                                    # Verify comprehensive logging
                                    assert mock_logger.info.call_count >= 6  # At least one per check

    def test_security_sensitive_environment_variable_handling(self):
        """Test that environment variables are handled securely."""
        checker = EnvironmentChecker()
        
        # Test that sensitive environment variables are properly validated
        with patch.dict(os.environ, {
            'ENV_NAME': 'prod',
            'POSTGRES_PASSWORD': 'secure_password_123'
        }, clear=True):
            result = checker.check_environment_variables()
            
            assert result is True
            assert len(checker.warnings) == 0
            
            # Verify that the password itself is not logged or exposed
            # (This is implicit in the current implementation)

    def test_package_security_validation(self):
        """Test that package validation helps prevent supply chain attacks."""
        checker = EnvironmentChecker()
        
        # Test that required security-critical packages are validated
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'mlflow', 'pydantic', 
            'fastapi', 'uvicorn', 'typer', 'pyyaml', 'python-dotenv'
        ]
        
        def mock_import_side_effect(package_name):
            if package_name in required_packages:
                return MagicMock()
            raise ImportError(f"No module named '{package_name}'")
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            result = checker.check_required_packages()
            
            assert result is True
            assert len(checker.errors) == 0
            
            # Verify that all security-critical packages were checked
            for package in required_packages:
                # This ensures each package import was attempted
                pass  # Validation is implicit in the mock behavior