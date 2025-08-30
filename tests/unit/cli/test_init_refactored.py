"""
Init Command Refactored Tests
Phase 5 Day 10: 단순화된 init 명령어 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from src.cli.commands.init_command import (
    create_project_structure,
    clone_mmp_local_dev,
)


class TestInitCommandRefactored:
    """단순화된 init 명령어 테스트"""
    
    def setup_method(self):
        """각 테스트 전 임시 디렉토리 생성"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.runner = CliRunner()

    def teardown_method(self):
        """각 테스트 후 임시 디렉토리 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_project_structure_basic(self):
        """기본 프로젝트 구조 생성 테스트"""
        # Given: 프로젝트 경로
        
        # When: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # Then: 기본 디렉토리들이 생성되어야 함
        assert self.project_path.exists()
        assert (self.project_path / "config").exists()
        assert (self.project_path / "recipes").exists()
        assert (self.project_path / "data").exists()
        assert (self.project_path / "docs").exists()
        
        # 기본 config 파일들이 생성되어야 함
        assert (self.project_path / "config" / "base.yaml").exists()
        assert (self.project_path / "config" / "local.yaml").exists()
        assert (self.project_path / "config" / "dev.yaml").exists()
        assert (self.project_path / "config" / "prod.yaml").exists()

    def test_create_project_structure_with_mmp_dev(self):
        """mmp-local-dev 호환 프로젝트 구조 생성 테스트"""
        # Given: mmp-local-dev 호환 모드
        
        # When: 프로젝트 구조 생성 (mmp-local-dev 모드)
        create_project_structure(self.project_path, with_mmp_dev=True)
        
        # Then: mmp-local-dev 호환 설정이 포함되어야 함
        dev_config_path = self.project_path / "config" / "dev.yaml"
        assert dev_config_path.exists()
        
        dev_config_content = dev_config_path.read_text()
        # mmp-local-dev 관련 설정이 포함되어야 함
        assert "postgresql" in dev_config_content.lower()
        assert "redis" in dev_config_content.lower()
        assert "localhost" in dev_config_content.lower()

    def test_create_project_structure_generates_sample_data(self):
        """샘플 데이터 생성 테스트"""
        # Given: 프로젝트 경로
        
        # When: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # Then: 샘플 데이터 파일이 생성되어야 함
        sample_data_files = list((self.project_path / "data").glob("*.csv"))
        assert len(sample_data_files) > 0
        
        # CSV 파일 내용 검증
        sample_file = sample_data_files[0]
        content = sample_file.read_text()
        assert "," in content  # CSV 형식인지 확인
        lines = content.strip().split('\n')
        assert len(lines) > 1  # 헤더 + 데이터가 있는지 확인

    def test_create_project_structure_generates_docs(self):
        """프로젝트 문서 생성 테스트"""
        # Given: 프로젝트 경로
        project_name = "test_project"
        
        # When: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # Then: 기본 문서들이 생성되어야 함
        docs_files = list((self.project_path / "docs").glob("*.md"))
        assert len(docs_files) > 0
        
        # README.md가 있는지 확인
        readme_files = [f for f in docs_files if "README" in f.name.upper()]
        assert len(readme_files) > 0

    @patch('subprocess.run')
    def test_clone_mmp_local_dev_success(self, mock_subprocess):
        """mmp-local-dev 성공적 clone 테스트"""
        # Given: subprocess.run이 성공하도록 모킹
        mock_subprocess.return_value.returncode = 0
        parent_dir = Path(self.temp_dir)
        
        # When: mmp-local-dev clone
        clone_mmp_local_dev(parent_dir)
        
        # Then: git clone 명령이 올바른 인수로 호출되어야 함
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "git" in call_args
        assert "clone" in call_args
        assert "mmp-local-dev" in " ".join(call_args)

    @patch('subprocess.run')
    def test_clone_mmp_local_dev_failure(self, mock_subprocess):
        """mmp-local-dev clone 실패 테스트"""
        # Given: subprocess.run이 실패하도록 모킹
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.side_effect = Exception("Git clone failed")
        parent_dir = Path(self.temp_dir)
        
        # When & Then: clone 실패 시 예외가 발생해야 함
        with pytest.raises(Exception, match="Git clone failed"):
            clone_mmp_local_dev(parent_dir)

    def test_create_project_structure_with_existing_directory(self):
        """기존 디렉토리가 있을 때의 처리 테스트"""
        # Given: 이미 존재하는 프로젝트 디렉토리
        self.project_path.mkdir(parents=True, exist_ok=True)
        (self.project_path / "existing_file.txt").write_text("existing content")
        
        # When: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # Then: 기존 파일은 유지되고 새로운 구조가 추가되어야 함
        assert (self.project_path / "existing_file.txt").exists()
        assert (self.project_path / "config").exists()
        assert (self.project_path / "recipes").exists()

    def test_config_files_contain_valid_yaml(self):
        """생성된 config 파일들이 유효한 YAML인지 테스트"""
        # Given: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # When & Then: 각 config 파일이 유효한 YAML인지 확인
        import yaml
        
        config_files = [
            "base.yaml", "local.yaml", "dev.yaml", "prod.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.project_path / "config" / config_file
            assert config_path.exists(), f"{config_file} should exist"
            
            # YAML 파싱 테스트
            content = config_path.read_text()
            parsed_yaml = yaml.safe_load(content)
            assert isinstance(parsed_yaml, dict), f"{config_file} should contain valid YAML dict"

    def test_project_structure_permissions(self):
        """생성된 디렉토리와 파일의 권한 테스트"""
        # Given: 프로젝트 구조 생성
        create_project_structure(self.project_path, with_mmp_dev=False)
        
        # When & Then: 디렉토리들이 접근 가능한지 확인
        directories = ["config", "recipes", "data", "docs"]
        for directory in directories:
            dir_path = self.project_path / directory
            assert dir_path.exists()
            assert dir_path.is_dir()
            # 읽기/쓰기 권한 확인
            test_file = dir_path / "test_permission.tmp"
            test_file.write_text("permission test")
            assert test_file.exists()
            test_file.unlink()  # 정리