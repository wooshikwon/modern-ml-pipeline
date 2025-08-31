"""
Phase 0 + Phase 1 Integration Test

Verifies that:
1. get-config command generates valid configuration files
2. Settings loader can load the generated configurations using env_name
3. Environment variable substitution works correctly
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

import pytest

from src.cli.commands.get_config_command import get_config_command
from src.settings._builder import load_config_for_env, load_config_files
from src.settings import Settings


class TestPhase0Phase1Integration:
    """Phase 0 과 Phase 1 통합 테스트."""
    
    def test_get_config_generates_loadable_settings(self):
        """get-config로 생성한 설정을 Settings가 로드할 수 있는지 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Phase 1: get-config으로 설정 생성
                with patch('src.cli.commands.get_config_command.console'):
                    get_config_command(
                        env_name="test",
                        non_interactive=True,
                        template="local"
                    )
                
                # 생성된 파일 확인
                config_path = Path(tmpdir) / "configs" / "test.yaml"
                env_path = Path(tmpdir) / ".env.test.template"
                assert config_path.exists(), "Config file should be created"
                assert env_path.exists(), "Env template should be created"
                
                # Phase 0: Settings 로더로 설정 로드
                with patch.dict(os.environ, {
                    'ENV_NAME': 'test',
                    'GCP_PROJECT': 'test-project',
                    'DB_USER': 'test_user',
                    'DB_PASSWORD': 'test_pass'
                }):
                    # BASE_DIR을 패치하여 load_config_for_env 사용
                    with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                        config_dict = load_config_for_env("test")
                        
                        # 설정 검증
                        assert config_dict['environment']['app_env'] == "test"
                        assert config_dict['mlflow']['tracking_uri'] == "./mlruns"
                        assert config_dict['mlflow']['experiment_name'] == "test-experiment"
                    
            finally:
                os.chdir(original_cwd)
    
    def test_multiple_environments_isolation(self):
        """여러 환경 설정이 독립적으로 동작하는지 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                with patch('src.cli.commands.get_config_command.console'):
                    # dev 환경 생성
                    get_config_command(
                        env_name="dev",
                        non_interactive=True,
                        template="dev"
                    )
                    
                    # prod 환경 생성
                    get_config_command(
                        env_name="prod",
                        non_interactive=True,
                        template="prod"
                    )
                
                # 각 환경 파일 확인
                dev_config = Path(tmpdir) / "configs" / "dev.yaml"
                prod_config = Path(tmpdir) / "configs" / "prod.yaml"
                assert dev_config.exists()
                assert prod_config.exists()
                
                # 각 환경별 설정 로드 및 검증
                
                # dev 환경
                with patch.dict(os.environ, {'ENV_NAME': 'dev'}):
                    with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                        dev_config = load_config_for_env("dev")
                        assert dev_config['environment']['app_env'] == "dev"
                
                # prod 환경
                with patch.dict(os.environ, {'ENV_NAME': 'prod'}):
                    with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                        prod_config = load_config_for_env("prod")
                        assert prod_config['environment']['app_env'] == "prod"
                
                # 환경이 서로 독립적인지 확인
                assert dev_config['environment']['app_env'] != prod_config['environment']['app_env']
                
            finally:
                os.chdir(original_cwd)
    
    def test_interactive_config_loadable_by_settings(self):
        """대화형으로 생성한 설정도 Settings가 로드할 수 있는지 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # 대화형 설정 생성 모의
                with patch('src.cli.commands.get_config_command.console'):
                    with patch('src.cli.core.config_builder.InteractiveConfigBuilder') as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder_class.return_value = mock_builder
                        
                        # 대화형 선택 결과 설정
                        mock_builder.run_interactive_flow.return_value = {
                            'env_name': 'custom',
                            'project_name': 'my-project',
                            'data_source': 'postgresql',
                            'db_host': 'localhost',
                            'db_port': '5432',
                            'db_name': 'custom_db',
                            'db_user': 'custom_user',
                            'db_connection_uri': 'postgresql://${DB_USER}:${DB_PASSWORD}@localhost:5432/custom_db',
                            'mlflow_type': 'local',
                            'mlflow_uri': './custom_mlruns',
                            'feature_store_enabled': False,
                            'storage_type': 'local',
                            'storage_path': './custom_data'
                        }
                        
                        # 실제 파일 생성
                        config_path = Path(tmpdir) / "configs" / "custom.yaml"
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        config_content = {
                            'environment': {
                                'app_env': 'custom',
                                'gcp_project_id': '${GCP_PROJECT:}'
                            },
                            'mlflow': {
                                'tracking_uri': './custom_mlruns',
                                'experiment_name': 'custom-experiment'
                            },
                            'data_adapters': {
                                'default_loader': 'sql',
                                'adapters': {
                                    'sql': {
                                        'class_name': 'SqlAdapter',
                                        'config': {
                                            'connection_uri': 'postgresql://${DB_USER}:${DB_PASSWORD}@localhost:5432/custom_db'
                                        }
                                    }
                                }
                            }
                        }
                        
                        with open(config_path, 'w') as f:
                            yaml.dump(config_content, f)
                        
                        mock_builder.generate_config_file.return_value = config_path
                        mock_builder.generate_env_template.return_value = Path(tmpdir) / ".env.custom.template"
                        
                        # get-config 실행
                        get_config_command(
                            env_name="custom",
                            non_interactive=False,
                            template=None
                        )
                
                # Settings로 로드
                with patch.dict(os.environ, {
                    'ENV_NAME': 'custom',
                    'DB_USER': 'custom_user',
                    'DB_PASSWORD': 'custom_pass'
                }):
                    with patch('src.settings._builder.BASE_DIR', Path(tmpdir)):
                        config_dict = load_config_for_env("custom")
                        
                        # 검증
                        assert config_dict['environment']['app_env'] == "custom"
                        assert config_dict['mlflow']['tracking_uri'] == "./custom_mlruns"
                        assert "sql" in config_dict['data_adapters']['adapters']
                    
            finally:
                os.chdir(original_cwd)