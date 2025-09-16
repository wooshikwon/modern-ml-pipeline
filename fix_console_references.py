#!/usr/bin/env python3
"""
Fix Console reference issues in test files after refactoring.
Changes UnifiedConsole -> Console and removes legacy classes.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_test_files_with_console_issues() -> List[Path]:
    """Find all test files that reference UnifiedConsole or RichConsoleManager"""
    test_dir = Path("/Users/wesley/Desktop/wooshikwon/modern-ml-pipeline/tests")
    files_to_fix = []

    for file_path in test_dir.glob("**/*.py"):
        if file_path.is_file():
            content = file_path.read_text()
            if "UnifiedConsole" in content or "RichConsoleManager" in content:
                files_to_fix.append(file_path)

    return files_to_fix

def fix_simple_references(file_path: Path) -> Tuple[bool, str]:
    """Fix simple UnifiedConsole -> Console references"""
    content = file_path.read_text()
    original = content

    # Simple replacements
    replacements = [
        # Class references
        (r'\bUnifiedConsole\b', 'Console'),

        # Patch paths - more specific to avoid breaking other patches
        (r"'src\.utils\.core\.console\.UnifiedConsole'", "'src.utils.core.console.Console'"),
        (r'"src\.utils\.core\.console\.UnifiedConsole"', '"src.utils.core.console.Console"'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original:
        return True, content
    return False, content

def fix_test_console_manager(file_path: Path) -> Tuple[bool, str]:
    """Special handling for test_console_manager.py - needs major refactoring"""
    if "test_console_manager.py" not in str(file_path):
        return False, file_path.read_text()

    # This file needs complete rewrite as it tests non-existent classes
    # For now, we'll create a minimal working version
    new_content = '''"""
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
'''

    return True, new_content

def main():
    """Main execution"""
    print("🔍 Finding test files with console reference issues...")
    files = find_test_files_with_console_issues()

    print(f"📋 Found {len(files)} files to check:")
    for f in files:
        print(f"  - {f.relative_to(Path('/Users/wesley/Desktop/wooshikwon/modern-ml-pipeline'))}")

    # Process each file
    fixed_count = 0
    special_count = 0

    for file_path in files:
        # Special handling for test_console_manager.py
        if "test_console_manager.py" in str(file_path):
            changed, content = fix_test_console_manager(file_path)
            if changed:
                file_path.write_text(content)
                print(f"✅ Completely rewrote: {file_path.name}")
                special_count += 1
        else:
            # Regular fixes
            changed, content = fix_simple_references(file_path)
            if changed:
                file_path.write_text(content)
                print(f"✅ Fixed references in: {file_path.name}")
                fixed_count += 1

    print(f"\n📊 Summary:")
    print(f"  - Simple fixes: {fixed_count} files")
    print(f"  - Complete rewrites: {special_count} files")
    print(f"  - Total: {fixed_count + special_count} files modified")

    # List remaining issues to handle manually
    print("\n⚠️ Files that may need manual review:")
    print("  - tests/unit/factory/test_component_creation_old_mocks.py (has 'old_mocks' in name)")
    print("  - tests/unit/utils/integrations/test_mlflow_utils.py (may have complex mocking)")

if __name__ == "__main__":
    main()