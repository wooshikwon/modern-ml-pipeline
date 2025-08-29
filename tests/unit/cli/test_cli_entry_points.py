"""
CLI Entry Points Unit Tests
Blueprint v17.0 - TDD RED Phase

CLAUDE.md 원칙 준수:
- RED → GREEN → REFACTOR 사이클
- 테스트 없는 구현 금지
- 커버리지 ≥ 90%
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
import pytest
from unittest.mock import patch, MagicMock


class TestCLIEntryPoints:
    """CLI 진입점 테스트 클래스"""
    
    def test_cli_entry_points__modern_ml_pipeline_command__should_be_available(self) -> None:
        """
        modern-ml-pipeline 명령어가 시스템에서 사용 가능한지 검증.
        
        Given: PyPI 패키지가 설치된 환경
        When: modern-ml-pipeline --version 명령어 실행
        Then: 명령어가 정상적으로 실행되고 버전 정보 반환
        """
        # Given: CLI 명령어 실행 준비
        
        # When: modern-ml-pipeline --version 실행
        result = subprocess.run(
            ['modern-ml-pipeline', '--version'], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        # Then: 명령어 실행 성공 및 버전 정보 포함
        assert result.returncode == 0, f"명령어 실행 실패: {result.stderr}"
        assert result.stdout.strip(), "버전 정보가 비어있습니다"
    
    def test_cli_entry_points__mmp_alias_command__should_be_available(self) -> None:
        """
        mmp 축약형 명령어가 시스템에서 사용 가능한지 검증.
        
        Given: PyPI 패키지가 설치된 환경
        When: mmp --help 명령어 실행
        Then: 명령어가 정상적으로 실행되고 도움말 정보 반환
        """
        # Given: 축약형 CLI 명령어 실행 준비
        
        # When: mmp --help 실행
        result = subprocess.run(
            ['mmp', '--help'], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        # Then: 명령어 실행 성공 및 도움말 정보 포함
        assert result.returncode == 0, f"축약형 명령어 실행 실패: {result.stderr}"
        assert "Modern ML Pipeline" in result.stdout, "도움말에 프로젝트명이 없습니다"
        
    def test_cli_entry_points__help_command__should_show_available_commands(self) -> None:
        """
        --help 명령어가 사용 가능한 하위 명령어들을 표시하는지 검증.
        
        Given: PyPI 패키지가 설치된 환경  
        When: modern-ml-pipeline --help 명령어 실행
        Then: train, init, serve 등 주요 명령어들이 도움말에 표시
        """
        # Given: 도움말 명령어 실행 준비
        
        # When: modern-ml-pipeline --help 실행
        result = subprocess.run(
            ['modern-ml-pipeline', '--help'], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        # Then: 주요 명령어들이 도움말에 포함
        assert result.returncode == 0, f"도움말 명령어 실행 실패: {result.stderr}"
        help_text = result.stdout.lower()
        
        expected_commands = ['train', 'init', 'serve']
        for command in expected_commands:
            assert command in help_text, f"'{command}' 명령어가 도움말에 없습니다"

    @pytest.mark.slow
    def test_cli_entry_points__import_performance__should_be_fast(self) -> None:
        """
        CLI 모듈 import가 빠르게 수행되는지 검증.
        
        Given: 패키지 설치 환경
        When: src.cli 모듈 import
        Then: import 시간이 3초 미만
        """
        import time
        
        # Given: import 시간 측정 준비
        start_time = time.time()
        
        # When: CLI 모듈 import  
        from src.cli import app
        
        # Then: import 시간이 3초 미만
        import_time = time.time() - start_time
        assert import_time < 3.0, f"CLI import가 너무 느립니다: {import_time:.2f}초"
        assert app is not None, "app 객체가 올바르게 import되지 않았습니다"


class TestCLIModuleStructure:
    """CLI 모듈 구조 테스트 클래스"""
    
    def test_cli_module__app_export__should_be_typer_instance(self) -> None:
        """
        src.cli 모듈이 typer.Typer 인스턴스를 올바르게 export하는지 검증.
        
        Given: src.cli 모듈
        When: app 객체 import
        Then: typer.Typer 인스턴스여야 함
        """
        # Given: CLI 모듈 import 준비
        
        # When: app 객체 import
        from src.cli import app
        import typer
        
        # Then: typer.Typer 인스턴스 확인
        assert isinstance(app, typer.Typer), f"app이 typer.Typer 인스턴스가 아닙니다: {type(app)}"
        assert hasattr(app, 'callback'), "typer app에 callback 메서드가 없습니다"