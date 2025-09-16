"""
Test Console functionality after refactoring.
Tests the unified Console class and CLI helper functions.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.utils.core.console import (
    Console,
    get_console,
    get_rich_console,
    cli_print,
    cli_success,
    cli_error,
    cli_warning,
    cli_info,
    testing_print,
    phase_print,
    success_print,
    testing_info
)


class TestConsole:
    """Console 클래스 핵심 테스트"""

    def test_console_initialization(self):
        """Console 초기화 테스트"""
        # When: 새로운 console 생성
        console = Console()

        # Then: 올바른 초기 상태
        assert console.console is not None
        assert console.current_pipeline is None
        assert console.progress_bars == {}

    def test_console_mode_detection(self):
        """환경에 따른 모드 감지"""
        # Test mode
        with patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test'}):
            console = Console()
            assert console.mode == "test"

        # Rich mode (default)
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                console = Console()
                # Will be rich or test depending on environment
                assert console.mode in ["rich", "test"]

    def test_console_basic_logging(self):
        """기본 로깅 기능"""
        console = Console()

        # Should not raise
        console.log("Test message")
        console.info("Info message")
        console.warning("Warning message")
        console.error("Error message")

    def test_console_print_methods(self):
        """Print 메서드들"""
        console = Console()

        # Should not raise
        console.print("Direct print")
        console.success_print("Success")
        console.phase_print("Phase")
        console.testing_print("Testing")


class TestCLIHelpers:
    """CLI 헬퍼 함수 테스트"""

    def test_cli_print_functions(self):
        """CLI print 함수들"""
        # These should not raise
        cli_print("Test message")
        cli_success("Success message")
        cli_error("Error message")
        cli_warning("Warning message")
        cli_info("Info message")

    def test_testing_functions(self):
        """Testing 관련 함수들"""
        # These should not raise
        testing_print("Test")
        phase_print("Phase")
        success_print("Success")
        testing_info("Info")

    def test_get_console_functions(self):
        """Console 인스턴스 getter 함수들"""
        console1 = get_console()
        assert isinstance(console1, Console)

        console2 = get_rich_console()
        assert isinstance(console2, Console)


class TestConsoleIntegration:
    """Console 통합 테스트"""

    def test_console_with_settings(self):
        """Settings와 함께 사용"""
        from tests.conftest import SettingsBuilder

        builder = SettingsBuilder()
        settings = builder.build()

        console = Console(settings)
        assert console is not None

    def test_console_in_test_mode(self):
        """테스트 모드에서 동작"""
        # In pytest, should be in test mode
        console = Console()
        console.log("Test mode message")
        # Just verify it doesn't crash
        assert True
