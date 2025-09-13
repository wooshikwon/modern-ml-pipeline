"""
Environment Check 테스트 (시스템 호환성 검증 핵심 모듈)
tests/README.md 전략 준수: 컨텍스트 기반, 퍼블릭 API, 실제 객체, 결정론적

테스트 대상 핵심 기능:
- EnvironmentChecker 클래스 - 개발환경 호환성 검증 (6가지 체크)
- check_python_version() - Python 버전 호환성 (3.11.x 권장)
- check_required_packages() - 필수 패키지 검증
- check_optional_packages() - 선택적 패키지 검증  
- check_directory_structure() - 프로젝트 구조 검증
- check_environment_variables() - 환경변수 설정 확인
- check_system_compatibility() - 시스템 호환성 (Apple Silicon 등)
- get_pip_requirements() - uv pip freeze를 통한 의존성 캡처

핵심 Edge Cases:
- Python 버전 호환성 (3.11 권장, 3.12 경고, 기타 에러)
- 패키지 누락 시나리오 (필수 vs 선택적)
- 디렉토리 구조 누락 처리
- 환경변수 설정 (LOCAL vs DEV/PROD)
- 시스템별 호환성 (Apple Silicon, Windows, Linux)
- uv 명령어 없는 환경 처리
"""
import pytest
import sys
import os
import platform
import subprocess
import collections
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.utils.core.environment_check import (
    EnvironmentChecker,
    check_environment,
    get_pip_requirements
)


class TestEnvironmentChecker:
    """EnvironmentChecker 클래스 핵심 테스트"""
    
    def test_environment_checker_initialization(self):
        """케이스 A: EnvironmentChecker 초기화"""
        # When: EnvironmentChecker 생성
        checker = EnvironmentChecker()
        
        # Then: 올바른 초기 상태
        assert checker.warnings == []
        assert checker.errors == []
    
    def test_check_python_version_success_311(self):
        """케이스 B: Python 3.11.x 버전 검증 성공"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Python 3.11.5 버전 시뮬레이션
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 11, 5)
        with patch('sys.version_info', mock_version):
            result = checker.check_python_version()
        
        # Then: 성공
        assert result is True
        assert len(checker.errors) == 0
        assert len(checker.warnings) == 0
    
    def test_check_python_version_warning_312(self):
        """케이스 C: Python 3.12 버전 시 경고"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Python 3.12.1 버전 시뮬레이션
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 12, 1)
        with patch('sys.version_info', mock_version):
            result = checker.check_python_version()
        
        # Then: 성공하지만 경고
        assert result is True
        assert len(checker.errors) == 0
        assert len(checker.warnings) == 1
        assert "Python 3.12는 causalml과 호환성 문제가 있을 수 있습니다" in checker.warnings[0]
    
    def test_check_python_version_warning_future_version(self):
        """케이스 D: Python 3.13+ 미래 버전 시 경고"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Python 3.13.0 버전 시뮬레이션
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 13, 0)
        with patch('sys.version_info', mock_version):
            result = checker.check_python_version()
        
        # Then: 성공하지만 경고
        assert result is True
        assert len(checker.errors) == 0
        assert len(checker.warnings) == 1
        assert "Python 3.13는 테스트되지 않았습니다" in checker.warnings[0]
    
    def test_check_python_version_error_old_version(self):
        """케이스 E: Python 3.10 이하 버전 시 에러"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Python 3.10.8 버전 시뮬레이션
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 10, 8)
        with patch('sys.version_info', mock_version):
            result = checker.check_python_version()
        
        # Then: 실패
        assert result is False
        assert len(checker.errors) == 1
        assert "Python 3.11 이상이 필요합니다" in checker.errors[0]
    
    def test_check_python_version_error_python2(self):
        """케이스 F: Python 2.x 버전 시 에러"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Python 2.7.18 버전 시뮬레이션
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(2, 7, 18)
        with patch('sys.version_info', mock_version):
            result = checker.check_python_version()
        
        # Then: 실패
        assert result is False
        assert len(checker.errors) == 1
        assert "Python 3.11.x가 필요합니다" in checker.errors[0]
    
    def test_check_required_packages_success(self):
        """케이스 G: 필수 패키지 검증 성공"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 모든 필수 패키지가 설치된 상황 시뮬레이션
        def mock_import(name):
            if name in ['pandas', 'numpy', 'scikit-learn', 'mlflow', 'pydantic', 
                       'fastapi', 'uvicorn', 'typer', 'pyyaml', 'python-dotenv']:
                return Mock()  # 성공적인 import 시뮬레이션
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = checker.check_required_packages()
        
        # Then: 성공
        assert result is True
        assert len(checker.errors) == 0
    
    def test_check_required_packages_missing_some(self):
        """케이스 H: 일부 필수 패키지 누락"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 일부 패키지만 설치된 상황 시뮬레이션
        def mock_import(name):
            if name in ['pandas', 'numpy']:  # 일부만 설치됨
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = checker.check_required_packages()
        
        # Then: 실패
        assert result is False
        assert len(checker.errors) == 1
        assert "필수 패키지가 설치되지 않았습니다" in checker.errors[0]
        assert "scikit-learn" in checker.errors[0]
        assert "mlflow" in checker.errors[0]
    
    def test_check_optional_packages_all_present(self):
        """케이스 I: 선택적 패키지 모두 설치"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 모든 선택적 패키지가 설치된 상황
        def mock_import(name):
            if name in ['redis', 'causalml', 'optuna', 'xgboost', 'lightgbm']:
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = checker.check_optional_packages()
        
        # Then: 성공 (경고 없음)
        assert result is True
        assert len(checker.warnings) == 0
    
    def test_check_optional_packages_some_missing(self):
        """케이스 J: 일부 선택적 패키지 누락 (경고만)"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 일부 선택적 패키지만 설치된 상황
        def mock_import(name):
            if name in ['redis', 'optuna']:  # 일부만 설치됨
                return Mock()
            raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = checker.check_optional_packages()
        
        # Then: 성공하지만 경고
        assert result is True
        assert len(checker.warnings) == 1
        assert "선택적 패키지가 설치되지 않았습니다" in checker.warnings[0]
        assert "causalml (인과추론 모델 지원)" in checker.warnings[0]
        assert "xgboost (XGBoost 모델 지원)" in checker.warnings[0]
    
    def test_check_directory_structure_success(self):
        """케이스 K: 디렉토리 구조 검증 성공"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 임시 디렉토리 구조 생성하여 테스트
        with tempfile.TemporaryDirectory() as temp_dir:
            # 필수 디렉토리들 생성
            required_dirs = ['config', 'recipes', 'src', 'data', 'serving', 'tests']
            for dir_name in required_dirs:
                (Path(temp_dir) / dir_name).mkdir()
            
            # __file__ 경로를 임시 디렉토리로 mock
            fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
            fake_file_path.parent.mkdir(parents=True)
            fake_file_path.write_text("# fake file")
            
            with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                result = checker.check_directory_structure()
        
        # Then: 성공
        assert result is True
        assert len(checker.errors) == 0
    
    def test_check_directory_structure_missing_dirs(self):
        """케이스 L: 일부 디렉토리 누락"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 일부 디렉토리만 있는 구조
        with tempfile.TemporaryDirectory() as temp_dir:
            # 일부 디렉토리만 생성
            (Path(temp_dir) / "src").mkdir()
            (Path(temp_dir) / "data").mkdir()
            # config, recipes, serving, tests 누락
            
            fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
            fake_file_path.parent.mkdir(parents=True)
            fake_file_path.write_text("# fake file")
            
            with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                result = checker.check_directory_structure()
        
        # Then: 실패
        assert result is False
        assert len(checker.errors) == 1
        assert "필수 디렉토리가 없습니다" in checker.errors[0]
        assert "config" in checker.errors[0]
        assert "recipes" in checker.errors[0]
    
    def test_check_environment_variables_local_env(self):
        """케이스 M: LOCAL 환경에서 환경변수 검증 생략"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: ENV_NAME=local 또는 미설정
        with patch.dict(os.environ, {"ENV_NAME": "local"}, clear=True):
            result = checker.check_environment_variables()
        
        # Then: 성공 (검증 생략)
        assert result is True
        assert len(checker.warnings) == 0
        
        # ENV_NAME 미설정 시에도 local로 간주
        with patch.dict(os.environ, {}, clear=True):
            result = checker.check_environment_variables()
        
        assert result is True
    
    def test_check_environment_variables_dev_env_with_vars(self):
        """케이스 N: DEV 환경에서 필요 환경변수 설정됨"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: DEV 환경에서 환경변수 설정됨
        with patch.dict(os.environ, {"ENV_NAME": "dev", "POSTGRES_PASSWORD": "secret123"}, clear=True):
            result = checker.check_environment_variables()
        
        # Then: 성공
        assert result is True
        assert len(checker.warnings) == 0
    
    def test_check_environment_variables_prod_env_missing_vars(self):
        """케이스 O: PROD 환경에서 중요 환경변수 누락"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: PROD 환경에서 POSTGRES_PASSWORD 누락
        with patch.dict(os.environ, {"ENV_NAME": "prod"}, clear=True):
            result = checker.check_environment_variables()
        
        # Then: 성공하지만 경고
        assert result is True
        assert len(checker.warnings) == 1
        assert "중요한 환경변수가 설정되지 않았습니다" in checker.warnings[0]
        assert "POSTGRES_PASSWORD" in checker.warnings[0]
    
    def test_check_system_compatibility_apple_silicon(self):
        """케이스 P: Apple Silicon Mac 호환성 체크"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Apple Silicon Mac 시뮬레이션
        with patch('platform.system', return_value='Darwin'), \
             patch('platform.machine', return_value='arm64'), \
             patch('platform.platform', return_value='macOS-14.0-arm64-arm-64bit'):
            
            result = checker.check_system_compatibility()
        
        # Then: 성공하지만 경고
        assert result is True
        assert len(checker.warnings) == 1
        assert "Apple Silicon Mac 감지" in checker.warnings[0]
        assert "Rosetta 2가 필요할 수 있습니다" in checker.warnings[0]
    
    def test_check_system_compatibility_intel_mac(self):
        """케이스 Q: Intel Mac 호환성 체크"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Intel Mac 시뮬레이션
        with patch('platform.system', return_value='Darwin'), \
             patch('platform.machine', return_value='x86_64'), \
             patch('platform.platform', return_value='macOS-14.0-x86_64-i386-64bit'):
            
            result = checker.check_system_compatibility()
        
        # Then: 성공 (경고 없음)
        assert result is True
        assert len(checker.warnings) == 0
    
    def test_check_system_compatibility_windows(self):
        """케이스 R: Windows 시스템 호환성"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Windows 시뮬레이션
        with patch('platform.system', return_value='Windows'), \
             patch('platform.machine', return_value='AMD64'), \
             patch('platform.platform', return_value='Windows-10-10.0.19041-SP0'):
            
            result = checker.check_system_compatibility()
        
        # Then: 성공 (경고 없음)
        assert result is True
        assert len(checker.warnings) == 0
    
    def test_check_system_compatibility_linux(self):
        """케이스 S: Linux 시스템 호환성"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: Linux 시뮬레이션
        with patch('platform.system', return_value='Linux'), \
             patch('platform.machine', return_value='x86_64'), \
             patch('platform.platform', return_value='Linux-5.4.0-74-generic-x86_64-with-glibc2.31'):
            
            result = checker.check_system_compatibility()
        
        # Then: 성공 (경고 없음)
        assert result is True
        assert len(checker.warnings) == 0
    
    def test_run_full_check_all_pass(self):
        """케이스 T: 전체 검증 모두 통과"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 모든 검증이 통과하도록 mock 설정
        with patch.object(checker, 'check_python_version', return_value=True), \
             patch.object(checker, 'check_required_packages', return_value=True), \
             patch.object(checker, 'check_optional_packages', return_value=True), \
             patch.object(checker, 'check_directory_structure', return_value=True), \
             patch.object(checker, 'check_environment_variables', return_value=True), \
             patch.object(checker, 'check_system_compatibility', return_value=True):
            
            success, errors, warnings = checker.run_full_check()
        
        # Then: 전체 성공
        assert success is True
        assert errors == []
        assert warnings == []
    
    def test_run_full_check_some_failures(self):
        """케이스 U: 전체 검증에서 일부 실패"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 일부 검증 실패하도록 설정
        with patch.object(checker, 'check_python_version', return_value=False), \
             patch.object(checker, 'check_required_packages', return_value=False), \
             patch.object(checker, 'check_optional_packages', return_value=True), \
             patch.object(checker, 'check_directory_structure', return_value=True), \
             patch.object(checker, 'check_environment_variables', return_value=True), \
             patch.object(checker, 'check_system_compatibility', return_value=True):
            
            # 에러와 경고 시뮬레이션
            checker.errors = ["Python version error", "Package error"]
            checker.warnings = ["System warning"]
            
            success, errors, warnings = checker.run_full_check()
        
        # Then: 전체 실패
        assert success is False
        assert len(errors) == 2
        assert len(warnings) == 1
        assert "Python version error" in errors
        assert "Package error" in errors
        assert "System warning" in warnings


class TestGetPipRequirements:
    """get_pip_requirements 함수 테스트"""
    
    def test_get_pip_requirements_success(self):
        """케이스 A: uv pip freeze 성공적 실행"""
        # Given: uv pip freeze 출력 시뮬레이션
        mock_output = """pandas==2.0.0
scikit-learn==1.3.0
numpy==1.24.0
mlflow==2.7.1
pydantic==2.4.2
fastapi==0.103.1"""
        
        mock_result = Mock()
        mock_result.stdout = mock_output
        mock_result.returncode = 0
        
        # When: uv pip freeze 실행 시뮬레이션
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            requirements = get_pip_requirements()
        
        # Then: 올바른 요구사항 목록 반환
        assert len(requirements) == 6
        assert "pandas==2.0.0" in requirements
        assert "scikit-learn==1.3.0" in requirements
        assert "mlflow==2.7.1" in requirements
        
        # subprocess.run이 올바른 명령어로 호출됨
        mock_run.assert_called_once_with(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
    
    def test_get_pip_requirements_uv_not_found(self):
        """케이스 B: uv 명령어를 찾을 수 없는 경우"""
        # When: uv 명령어 없음 시뮬레이션
        with patch('subprocess.run', side_effect=FileNotFoundError("uv command not found")):
            requirements = get_pip_requirements()
        
        # Then: 빈 리스트 반환 (에러 없이)
        assert requirements == []
    
    def test_get_pip_requirements_command_error(self):
        """케이스 C: uv pip freeze 명령어 실행 에러"""
        # Given: subprocess.CalledProcessError 시뮬레이션
        error = subprocess.CalledProcessError(1, ["uv", "pip", "freeze"], stderr="Permission denied")
        
        # When: 명령어 실행 에러
        with patch('subprocess.run', side_effect=error):
            requirements = get_pip_requirements()
        
        # Then: 빈 리스트 반환 (에러 없이)
        assert requirements == []
    
    def test_get_pip_requirements_empty_output(self):
        """케이스 D: uv pip freeze 빈 출력"""
        # Given: 빈 출력 시뮬레이션
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        
        # When: 빈 출력으로 실행
        with patch('subprocess.run', return_value=mock_result):
            requirements = get_pip_requirements()
        
        # Then: 빈 리스트 반환
        assert requirements == []
    
    def test_get_pip_requirements_multiline_output(self):
        """케이스 E: 여러 줄 출력과 특수 문자 처리"""
        # Given: 복잡한 출력 시뮬레이션
        mock_output = """pandas==2.0.0
# Comment line (should be included as-is)
scikit-learn==1.3.0
numpy==1.24.0+cu118
-e git+https://github.com/user/repo.git@branch#egg=package
torch==2.0.1+cu118
mlflow==2.7.1
"""
        
        mock_result = Mock()
        mock_result.stdout = mock_output
        mock_result.returncode = 0
        
        # When: 복잡한 출력으로 실행
        with patch('subprocess.run', return_value=mock_result):
            requirements = get_pip_requirements()
        
        # Then: 모든 라인이 포함됨 (마지막 빈 줄 제외)
        assert len(requirements) == 6  # 빈 줄 제외
        assert "pandas==2.0.0" in requirements
        assert "# Comment line (should be included as-is)" in requirements
        assert "numpy==1.24.0+cu118" in requirements
        assert "-e git+https://github.com/user/repo.git@branch#egg=package" in requirements
        assert "torch==2.0.1+cu118" in requirements
        assert "mlflow==2.7.1" in requirements


class TestCheckEnvironmentFunction:
    """check_environment 편의 함수 테스트"""
    
    def test_check_environment_success(self):
        """케이스 A: check_environment 함수 성공"""
        # When: 모든 검증 통과하도록 mock
        with patch.object(EnvironmentChecker, 'run_full_check', return_value=(True, [], [])):
            result = check_environment()
        
        # Then: 성공
        assert result is True
    
    def test_check_environment_failure(self):
        """케이스 B: check_environment 함수 실패"""
        # When: 일부 검증 실패하도록 mock
        with patch.object(EnvironmentChecker, 'run_full_check', return_value=(False, ["Error"], ["Warning"])):
            result = check_environment()
        
        # Then: 실패
        assert result is False


class TestEnvironmentCheckIntegration:
    """Environment Check 통합 테스트"""
    
    def test_complete_environment_check_scenario(self):
        """케이스 A: 완전한 환경 검증 시나리오"""
        # Given: 실제 환경에 가까운 시뮬레이션
        checker = EnvironmentChecker()
        
        # When: 현실적인 시나리오 (일부 패키지 누락, Apple Silicon 경고 등)
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 11, 5)
        with patch('sys.version_info', mock_version), \
             patch('platform.system', return_value='Darwin'), \
             patch('platform.machine', return_value='arm64'), \
             patch('platform.platform', return_value='macOS-14.0-arm64-arm-64bit'), \
             patch.dict(os.environ, {"ENV_NAME": "local"}, clear=True):
            
            # 필수 패키지는 모두 있지만 선택적 패키지 일부 누락
            def mock_import(name):
                if name in ['pandas', 'numpy', 'scikit-learn', 'mlflow', 'pydantic', 
                           'fastapi', 'uvicorn', 'typer', 'pyyaml', 'python-dotenv']:
                    return Mock()  # 필수 패키지 모두 설치됨
                elif name in ['redis', 'optuna']:
                    return Mock()  # 일부 선택적 패키지만 설치됨
                else:
                    raise ImportError(f"No module named '{name}'")  # causalml, xgboost, lightgbm 누락
            
            with patch('builtins.__import__', side_effect=mock_import):
                # 임시 디렉토리 구조로 디렉토리 검증
                with tempfile.TemporaryDirectory() as temp_dir:
                    required_dirs = ['config', 'recipes', 'src', 'data', 'serving', 'tests']
                    for dir_name in required_dirs:
                        (Path(temp_dir) / dir_name).mkdir()
                    
                    fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
                    fake_file_path.parent.mkdir(parents=True)
                    fake_file_path.write_text("# fake file")
                    
                    with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                        success, errors, warnings = checker.run_full_check()
        
        # Then: 성공하지만 경고 존재
        assert success is True  # 필수 요소는 모두 통과
        assert len(errors) == 0
        assert len(warnings) >= 2  # Apple Silicon 경고 + 선택적 패키지 누락 경고
        
        # 특정 경고 내용 확인
        warning_text = " ".join(warnings)
        assert "Apple Silicon Mac 감지" in warning_text
        assert "선택적 패키지가 설치되지 않았습니다" in warning_text
        assert "causalml" in warning_text
    
    def test_critical_failure_scenario(self):
        """케이스 B: 치명적 실패 시나리오"""
        # Given: 심각한 환경 문제 상황
        checker = EnvironmentChecker()
        
        # When: 여러 치명적 문제 발생
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 9, 7)
        with patch('sys.version_info', mock_version), \
             patch.dict(os.environ, {"ENV_NAME": "prod"}, clear=True):
            
            # 필수 패키지 대부분 누락
            def mock_import(name):
                if name in ['pandas']:  # pandas만 설치됨
                    return Mock()
                raise ImportError(f"No module named '{name}'")
            
            with patch('builtins.__import__', side_effect=mock_import):
                # 디렉토리 구조도 일부 누락
                with tempfile.TemporaryDirectory() as temp_dir:
                    # src, data만 생성 (config, recipes, serving, tests 누락)
                    (Path(temp_dir) / "src").mkdir()
                    (Path(temp_dir) / "data").mkdir()
                    
                    fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
                    fake_file_path.parent.mkdir(parents=True)
                    fake_file_path.write_text("# fake file")
                    
                    with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                        success, errors, warnings = checker.run_full_check()
        
        # Then: 치명적 실패
        assert success is False
        assert len(errors) >= 3  # Python 버전 + 패키지 누락 + 디렉토리 누락
        assert len(warnings) >= 1  # PROD 환경변수 누락
        
        # 특정 에러 내용 확인
        error_text = " ".join(errors)
        assert "Python 3.11 이상이 필요합니다" in error_text
        assert "필수 패키지가 설치되지 않았습니다" in error_text
        assert "필수 디렉토리가 없습니다" in error_text
    
    def test_mixed_success_warning_scenario(self):
        """케이스 C: 성공하지만 여러 경고 시나리오"""
        # Given: 기본적으로는 정상이지만 여러 경고 상황
        checker = EnvironmentChecker()
        
        # When: Python 3.12 + Apple Silicon + 선택적 패키지 누락 + PROD 환경변수 누락
        version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
        mock_version = version_info(3, 12, 2)
        with patch('sys.version_info', mock_version), \
             patch('platform.system', return_value='Darwin'), \
             patch('platform.machine', return_value='arm64'), \
             patch.dict(os.environ, {"ENV_NAME": "prod"}, clear=True):
            
            # 필수 패키지는 모두 있지만 선택적 패키지 모두 누락
            def mock_import(name):
                if name in ['pandas', 'numpy', 'scikit-learn', 'mlflow', 'pydantic', 
                           'fastapi', 'uvicorn', 'typer', 'pyyaml', 'python-dotenv']:
                    return Mock()  # 필수 패키지만 설치됨
                raise ImportError(f"No module named '{name}'")  # 모든 선택적 패키지 누락
            
            with patch('builtins.__import__', side_effect=mock_import):
                with tempfile.TemporaryDirectory() as temp_dir:
                    required_dirs = ['config', 'recipes', 'src', 'data', 'serving', 'tests']
                    for dir_name in required_dirs:
                        (Path(temp_dir) / dir_name).mkdir()
                    
                    fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
                    fake_file_path.parent.mkdir(parents=True)
                    fake_file_path.write_text("# fake file")
                    
                    with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                        success, errors, warnings = checker.run_full_check()
        
        # Then: 성공하지만 많은 경고
        assert success is True  # 필수 조건은 만족
        assert len(errors) == 0
        assert len(warnings) >= 3  # Python 3.12 경고 + Apple Silicon 경고 + 선택적 패키지 경고 + 환경변수 경고
        
        # 모든 경고 종류 확인
        warning_text = " ".join(warnings)
        assert "Python 3.12는 causalml과 호환성 문제가 있을 수 있습니다" in warning_text
        assert "Apple Silicon Mac 감지" in warning_text
        assert "선택적 패키지가 설치되지 않았습니다" in warning_text
        assert "중요한 환경변수가 설정되지 않았습니다" in warning_text
    
    def test_environment_check_performance(self):
        """케이스 D: 환경 검증 성능 테스트"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 환경 검증 실행 시간 측정
        import time
        start_time = time.time()
        
        # 실제 환경에서의 검증 (mocking 최소화)
        try:
            success, errors, warnings = checker.run_full_check()
        except Exception:
            # 실제 환경에서 에러 발생 가능하지만 성능 측정은 계속
            pass
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Then: 성능 기준 (3초 이내)
        assert execution_time < 3.0  # 환경 검증은 빨라야 함
    
    def test_environment_check_idempotent(self):
        """케이스 E: 환경 검증 멱등성 (여러 번 실행해도 같은 결과)"""
        # Given: EnvironmentChecker
        checker = EnvironmentChecker()
        
        # When: 동일한 환경에서 여러 번 실행
        results = []
        for i in range(3):
            # 각 실행마다 새로운 checker 인스턴스
            test_checker = EnvironmentChecker()
            
            version_info = collections.namedtuple('version_info', ['major', 'minor', 'micro'])
            mock_version = version_info(3, 11, 5)
            with patch('sys.version_info', mock_version), \
                 patch('platform.system', return_value='Linux'), \
                 patch.dict(os.environ, {"ENV_NAME": "local"}, clear=True):
                
                def mock_import(name):
                    if name in ['pandas', 'numpy', 'scikit-learn', 'mlflow']:
                        return Mock()
                    raise ImportError(f"No module named '{name}'")
                
                with patch('builtins.__import__', side_effect=mock_import), \
                     tempfile.TemporaryDirectory() as temp_dir:
                    
                    required_dirs = ['config', 'recipes', 'src', 'data', 'serving', 'tests']
                    for dir_name in required_dirs:
                        (Path(temp_dir) / dir_name).mkdir()
                    
                    fake_file_path = Path(temp_dir) / "src" / "utils" / "core" / "environment_check.py"
                    fake_file_path.parent.mkdir(parents=True)
                    fake_file_path.write_text("# fake file")
                    
                    with patch('src.utils.core.environment_check.__file__', str(fake_file_path)):
                        result = test_checker.run_full_check()
                        results.append(result)
        
        # Then: 모든 실행 결과가 동일
        assert len(set(result[0] for result in results)) == 1  # 성공/실패 일관성
        # 에러/경고 개수도 동일해야 함
        assert len(set(len(result[1]) for result in results)) == 1  # 에러 개수 일관성
        assert len(set(len(result[2]) for result in results)) == 1  # 경고 개수 일관성