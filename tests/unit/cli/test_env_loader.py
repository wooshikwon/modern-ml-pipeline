"""
Test Environment Loader
Phase 2: 환경 로더 단위 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.cli.utils.env_loader import (
    load_environment,
    get_config_path,
    resolve_env_variables,
    load_config_with_env
)


class TestLoadEnvironment:
    """load_environment 함수 테스트."""
    
    def test_load_environment_success(self):
        """환경변수 파일 로드 성공 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # .env.test 파일 생성
            env_file = tmpdir_path / ".env.test"
            env_file.write_text("TEST_VAR=test_value\nDB_PASSWORD=secret123\n")
            
            # 환경변수 로드
            load_environment("test", base_path=tmpdir_path)
            
            # 검증
            assert os.getenv("TEST_VAR") == "test_value"
            assert os.getenv("DB_PASSWORD") == "secret123"
            assert os.getenv("ENV_NAME") == "test"
    
    def test_load_environment_missing_file(self):
        """환경변수 파일 없을 때 에러 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            with pytest.raises(FileNotFoundError) as exc_info:
                load_environment("nonexistent", base_path=tmpdir_path)
            
            assert "mmp get-config" in str(exc_info.value)
            assert ".env.nonexistent" in str(exc_info.value)


class TestGetConfigPath:
    """get_config_path 함수 테스트."""
    
    def test_get_config_path_configs_dir(self):
        """configs 디렉토리에서 config 파일 찾기."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # configs/test.yaml 생성
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            config_file = configs_dir / "test.yaml"
            config_file.write_text("test: config")
            
            # 경로 가져오기
            path = get_config_path("test", base_path=tmpdir_path)
            
            assert path == config_file
            assert path.exists()
    
    def test_get_config_path_config_dir_fallback(self):
        """config 디렉토리 fallback 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # config/test.yaml 생성 (configs 없을 때)
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            config_file = config_dir / "test.yaml"
            config_file.write_text("test: config")
            
            # 경로 가져오기
            path = get_config_path("test", base_path=tmpdir_path)
            
            assert path == config_file
            assert path.exists()
    
    def test_get_config_path_not_found(self):
        """config 파일 없을 때 에러 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            with pytest.raises(FileNotFoundError) as exc_info:
                get_config_path("missing", base_path=tmpdir_path)
            
            assert "configs/missing.yaml" in str(exc_info.value)
            assert "mmp get-config" in str(exc_info.value)


class TestResolveEnvVariables:
    """resolve_env_variables 함수 테스트."""
    
    def test_resolve_string_with_env_var(self):
        """문자열 환경변수 치환 테스트."""
        os.environ["TEST_VAR"] = "replaced"
        
        result = resolve_env_variables("${TEST_VAR}")
        assert result == "replaced"
        
        result = resolve_env_variables("prefix_${TEST_VAR}_suffix")
        assert result == "prefix_replaced_suffix"
    
    def test_resolve_with_default_value(self):
        """기본값 사용 테스트."""
        # 환경변수 제거
        os.environ.pop("MISSING_VAR", None)
        
        result = resolve_env_variables("${MISSING_VAR:default_value}")
        assert result == "default_value"
        
        # 환경변수 있을 때는 환경변수 사용
        os.environ["EXISTING_VAR"] = "actual_value"
        result = resolve_env_variables("${EXISTING_VAR:default_value}")
        assert result == "actual_value"
    
    def test_resolve_type_conversion(self):
        """타입 변환 테스트."""
        os.environ["TEST_INT"] = "8080"
        os.environ["TEST_FLOAT"] = "3.14"
        os.environ["TEST_BOOL_TRUE"] = "true"
        os.environ["TEST_BOOL_FALSE"] = "false"
        
        # 완전히 환경변수인 경우만 타입 변환
        assert resolve_env_variables("${TEST_INT}") == 8080
        assert resolve_env_variables("${TEST_FLOAT}") == 3.14
        assert resolve_env_variables("${TEST_BOOL_TRUE}") is True
        assert resolve_env_variables("${TEST_BOOL_FALSE}") is False
        
        # 일부만 환경변수인 경우 문자열 유지
        assert resolve_env_variables("port:${TEST_INT}") == "port:8080"
    
    def test_resolve_nested_structures(self):
        """중첩 구조 치환 테스트."""
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PORT"] = "5432"
        os.environ["DB_NAME"] = "testdb"
        
        config = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "name": "${DB_NAME}",
                "options": ["${DB_HOST}", "${DB_PORT}"]
            }
        }
        
        resolved = resolve_env_variables(config)
        
        assert resolved["database"]["host"] == "localhost"
        assert resolved["database"]["port"] == 5432  # 타입 변환됨
        assert resolved["database"]["name"] == "testdb"
        assert resolved["database"]["options"] == ["localhost", 5432]
    
    def test_resolve_no_replacement(self):
        """환경변수 없을 때 원본 유지 테스트."""
        os.environ.pop("NO_SUCH_VAR", None)
        
        result = resolve_env_variables("${NO_SUCH_VAR}")
        assert result == "${NO_SUCH_VAR}"


class TestLoadConfigWithEnv:
    """load_config_with_env 함수 테스트."""
    
    def test_load_config_with_env_full_flow(self):
        """전체 플로우 테스트: 환경변수 로드 → config 로드 → 치환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # .env.dev 파일 생성
            env_file = tmpdir_path / ".env.dev"
            env_file.write_text("DB_HOST=localhost\nDB_PORT=5432\n")
            
            # configs/dev.yaml 파일 생성
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            config_file = configs_dir / "dev.yaml"
            config_content = {
                "database": {
                    "host": "${DB_HOST}",
                    "port": "${DB_PORT:3306}"
                },
                "app": {
                    "env": "dev"
                }
            }
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f)
            
            # 로드 및 치환
            result = load_config_with_env("dev", base_path=tmpdir_path)
            
            # 검증
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == 5432
            assert result["app"]["env"] == "dev"
            assert os.getenv("ENV_NAME") == "dev"


