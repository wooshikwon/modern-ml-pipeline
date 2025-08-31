"""
Phase 0+1+2 Full Integration Test

Verifies complete workflow:
1. Phase 1: get-config로 환경 설정 생성
2. Phase 2: --env-name으로 환경별 실행
3. Phase 0: Settings 로더 호환성
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

import pytest

from src.cli.commands.get_config_command import get_config_command
from src.cli.utils.env_loader import load_config_with_env
from src.settings._builder import load_config_for_env as phase0_load_config


class TestFullPhaseIntegration:
    """Phase 0, 1, 2 전체 통합 테스트."""
    
    def test_full_workflow_local_environment(self):
        """전체 워크플로우 테스트: local 환경."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Step 1: Phase 1 - get-config로 local 환경 생성
                with patch('src.cli.commands.get_config_command.console'):
                    get_config_command(
                        env_name="local",
                        non_interactive=True,
                        template="local"
                    )
                
                # 생성된 파일 확인
                config_path = Path(tmpdir) / "configs" / "local.yaml"
                env_template_path = Path(tmpdir) / ".env.local.template"
                assert config_path.exists(), "Config file should be created"
                assert env_template_path.exists(), "Env template should be created"
                
                # .env 파일 생성 (템플릿에서 복사)
                env_file = Path(tmpdir) / ".env.local"
                env_content = env_template_path.read_text()
                env_content += "\nDB_PASSWORD=localpass123\n"
                env_file.write_text(env_content)
                
                # Step 2: Phase 2 - env_loader로 설정 로드
                config_from_phase2 = load_config_with_env("local", base_path=Path(tmpdir))
                
                # 검증
                assert config_from_phase2['environment']['app_env'] == "local"
                assert config_from_phase2['mlflow']['tracking_uri'] == "./mlruns"
                assert os.getenv("ENV_NAME") == "local"
                
                # Step 3: Phase 0 - Settings 로더 호환성 확인
                with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                    config_from_phase0 = phase0_load_config("local")
                    
                    # Phase 0과 Phase 2 결과 동일성 확인
                    assert config_from_phase0['environment']['app_env'] == config_from_phase2['environment']['app_env']
                    assert config_from_phase0['mlflow']['tracking_uri'] == config_from_phase2['mlflow']['tracking_uri']
                
            finally:
                os.chdir(original_cwd)
    
    def test_dev_to_prod_workflow(self):
        """개발 → 운영 환경 전환 워크플로우 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                with patch('src.cli.commands.get_config_command.console'):
                    # Dev 환경 생성
                    get_config_command(
                        env_name="dev",
                        non_interactive=True,
                        template="dev"
                    )
                    
                    # Prod 환경 생성
                    get_config_command(
                        env_name="prod",
                        non_interactive=True,
                        template="prod"
                    )
                
                # .env 파일들 생성
                dev_env = Path(tmpdir) / ".env.dev"
                dev_env.write_text(
                    "ENV_NAME=dev\n"
                    "DB_HOST=dev.db.example.com\n"
                    "DB_PASSWORD=devpass\n"
                    "MLFLOW_TRACKING_URI=http://mlflow.dev:5000\n"
                )
                
                prod_env = Path(tmpdir) / ".env.prod"
                prod_env.write_text(
                    "ENV_NAME=prod\n"
                    "DB_HOST=prod.db.example.com\n"
                    "DB_PASSWORD=prodpass\n"
                    "MLFLOW_TRACKING_URI=http://mlflow.prod:5000\n"
                )
                
                # Dev 환경 로드
                dev_config = load_config_with_env("dev", base_path=Path(tmpdir))
                assert dev_config['environment']['app_env'] == "dev"
                assert os.getenv("DB_HOST") == "dev.db.example.com"
                
                # Prod 환경으로 전환
                prod_config = load_config_with_env("prod", base_path=Path(tmpdir))
                assert prod_config['environment']['app_env'] == "prod"
                assert os.getenv("DB_HOST") == "prod.db.example.com"  # 환경변수 덮어씌워짐
                assert os.getenv("ENV_NAME") == "prod"
                
            finally:
                os.chdir(original_cwd)
    
    def test_environment_variable_substitution(self):
        """환경변수 치환 통합 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Custom config 생성
                configs_dir = Path(tmpdir) / "configs"
                configs_dir.mkdir()
                
                custom_config = {
                    "environment": {
                        "app_env": "custom"
                    },
                    "database": {
                        "host": "${DB_HOST:localhost}",
                        "port": "${DB_PORT:5432}",
                        "connection_string": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
                    },
                    "features": {
                        "enabled": "${FEATURE_FLAG:false}",
                        "max_workers": "${MAX_WORKERS:4}"
                    }
                }
                
                config_file = configs_dir / "custom.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(custom_config, f)
                
                # .env 파일 생성
                env_file = Path(tmpdir) / ".env.custom"
                env_file.write_text(
                    "DB_HOST=custom.db.com\n"
                    "DB_PORT=3306\n"
                    "DB_USER=customuser\n"
                    "DB_PASSWORD=custompass\n"
                    "DB_NAME=customdb\n"
                    "FEATURE_FLAG=true\n"
                    "MAX_WORKERS=8\n"
                )
                
                # 로드 및 치환 확인
                config = load_config_with_env("custom", base_path=Path(tmpdir))
                
                # 타입 변환 확인
                assert config['database']['host'] == "custom.db.com"
                assert config['database']['port'] == 3306  # int로 변환됨
                assert config['features']['enabled'] is True  # bool로 변환됨
                assert config['features']['max_workers'] == 8  # int로 변환됨
                
                # 복합 문자열 치환 확인
                expected_conn = "postgresql://customuser:custompass@custom.db.com:3306/customdb"
                assert config['database']['connection_string'] == expected_conn
                
            finally:
                os.chdir(original_cwd)
    
    def test_backward_compatibility(self):
        """Phase 0 하위 호환성 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 기존 방식 config 생성 (Phase 0 스타일)
            config_dir = Path(tmpdir) / "config"  # configs가 아닌 config
            config_dir.mkdir()
            
            base_config = config_dir / "base.yaml"
            base_config.write_text("base_setting: true\n")
            
            test_config = config_dir / "test.yaml"
            test_config.write_text("environment:\n  app_env: test\n")
            
            # .env.test 파일
            env_file = Path(tmpdir) / ".env.test"
            env_file.write_text("TEST_VAR=test_value\n")
            
            # Phase 0 방식으로 로드
            with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                config_phase0 = phase0_load_config("test")
                assert 'base_setting' in config_phase0  # base.yaml 병합됨
                assert config_phase0['environment']['app_env'] == 'test'
            
            # Phase 2 방식으로도 로드 가능 (config 디렉토리 fallback)
            config_phase2 = load_config_with_env("test", base_path=Path(tmpdir))
            assert config_phase2['environment']['app_env'] == 'test'
            assert os.getenv("TEST_VAR") == "test_value"