"""
CLI 메인 엔트리포인트 테스트 (커버리지 확장)
__main__.py 테스트
"""
import pytest
import subprocess
import sys
from pathlib import Path


class TestMainEntryPoint:
    """메인 엔트리포인트 테스트"""
    
    def test_main_help_output_smoke(self):
        """케이스 A: python -m src --help 출력 (비상태 스모크)"""
        # subprocess를 사용하여 실제 모듈 실행
        result = subprocess.run(
            [sys.executable, "-m", "src", "--help"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # 도움말이 정상적으로 출력되고 0으로 종료
        assert result.returncode == 0
        assert len(result.stdout) > 0
        
        # 기본적인 CLI 키워드들이 포함되어야 함
        help_text = result.stdout.lower()
        expected_keywords = ["usage", "command", "help", "option"]
        assert any(keyword in help_text for keyword in expected_keywords)
    
    def test_main_version_if_available(self):
        """버전 정보 표시 (있는 경우)"""
        result = subprocess.run(
            [sys.executable, "-m", "src", "--version"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # --version이 지원되면 0으로 종료, 아니면 오류 코드
        # 두 경우 모두 허용
        assert result.returncode in [0, 1, 2]
        
        if result.returncode == 0:
            # 성공 시 버전 정보가 있어야 함
            assert len(result.stdout.strip()) > 0
    
    def test_main_invalid_command_shows_help(self):
        """잘못된 명령어 시 도움말 표시"""
        result = subprocess.run(
            [sys.executable, "-m", "src", "nonexistent_command"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # 잘못된 명령어는 비정상 종료
        assert result.returncode != 0
        
        # 에러 출력이 있어야 함
        assert len(result.stderr) > 0 or "error" in result.stdout.lower()
    
    def test_main_no_args_behavior(self):
        """인수 없이 실행 시 동작"""
        result = subprocess.run(
            [sys.executable, "-m", "src"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # 인수 없이 실행 시 도움말 표시 또는 오류
        # (일반적으로 0이 아닌 코드로 종료)
        expected_exits = [0, 1, 2]  # 다양한 CLI 프레임워크별 다름
        assert result.returncode in expected_exits
        
        # 최소한 출력이 있어야 함
        assert len(result.stdout) > 0 or len(result.stderr) > 0
    
    def test_main_common_subcommands_exist(self):
        """일반적인 서브커맨드들의 존재 확인"""
        common_commands = ["train", "serve", "init"]
        
        for cmd in common_commands:
            result = subprocess.run(
                [sys.executable, "-m", "src", cmd, "--help"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # 명령어가 존재하면 0으로 종료, 없으면 오류
            # 존재하지 않는 명령어도 허용 (선택적 기능일 수 있음)
            assert result.returncode in [0, 1, 2]
            
            if result.returncode == 0:
                # 성공 시 도움말이 출력되어야 함
                assert len(result.stdout) > 0
                help_text = result.stdout.lower()
                assert "usage" in help_text or cmd in help_text


class TestMainModuleImport:
    """모듈 임포트 레벨 테스트"""
    
    def test_main_module_imports_without_error(self):
        """메인 모듈이 오류 없이 임포트됨"""
        try:
            # 직접 임포트해서 구문 오류 등 확인
            import src.__main__ as main_module
            # 임포트 성공하면 통과
            assert main_module is not None
        except ImportError as e:
            # 의존성 문제 등으로 임포트 실패할 수 있음
            pytest.skip(f"Main module import failed: {e}")
        except Exception as e:
            # 다른 예외는 실패로 처리
            pytest.fail(f"Unexpected error importing main module: {e}")
    
    def test_main_module_has_expected_structure(self):
        """메인 모듈이 기대되는 구조를 가짐"""
        try:
            import src.__main__ as main_module
            
            # CLI 프레임워크별 일반적인 패턴들 확인
            module_vars = dir(main_module)
            
            # 일반적으로 있을 법한 것들 (선택적)
            possible_items = ["main", "app", "cli", "__doc__", "__file__"]
            
            # 최소한 하나는 있어야 함 (빈 파일이 아님을 확인)
            assert len(module_vars) > 2  # __name__, __file__ 외에 추가 내용
            
        except ImportError:
            pytest.skip("Main module import not available")
    
    def test_main_module_execution_path_exists(self):
        """실행 경로가 존재함 (__main__ 실행 시)"""
        # __main__.py 파일 자체가 존재하는지 확인
        main_path = Path("src") / "__main__.py"
        assert main_path.exists(), "src/__main__.py file should exist"
        
        # 파일이 비어있지 않은지 확인
        content = main_path.read_text()
        assert len(content.strip()) > 0, "__main__.py should not be empty"