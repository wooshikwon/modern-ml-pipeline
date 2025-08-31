"""
Test Get-Config Command
Phase 1: 대화형 설정 생성 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import yaml
import tempfile

from src.cli.commands.get_config_command import get_config_command, _create_from_template
from src.cli.core.config_builder import InteractiveConfigBuilder
from src.cli.ui.interactive_selector import InteractiveSelector


class TestGetConfigCommand:
    """get-config 명령어 테스트."""
    
    def test_get_config_command_with_template(self):
        """템플릿 기반 설정 생성 테스트."""
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # 현재 디렉토리를 임시 디렉토리로 변경
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                with patch('src.cli.commands.get_config_command.console'):
                    # 템플릿 기반 생성
                    get_config_command(
                        env_name="test",
                        non_interactive=True,
                        template="local"
                    )
                    
                    # 파일 생성 확인
                    config_path = Path(tmpdir) / "configs" / "test.yaml"
                    env_template_path = Path(tmpdir) / ".env.test.template"
                    
                    assert config_path.exists()
                    assert env_template_path.exists()
                    
                    # 내용 검증
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        assert config['environment']['app_env'] == 'test'
            finally:
                # 원래 디렉토리로 복원
                os.chdir(original_cwd)
    
    def test_get_config_command_interactive(self):
        """대화형 설정 생성 테스트."""
        with patch('src.cli.commands.get_config_command.console'):
            with patch('src.cli.core.config_builder.InteractiveConfigBuilder') as mock_builder_class:
                # Mock 설정
                mock_builder = MagicMock()
                mock_builder_class.return_value = mock_builder
                
                mock_builder.run_interactive_flow.return_value = {
                    'env_name': 'dev',
                    'project_name': 'test-project',
                    'data_source': 'postgresql',
                    'db_host': 'localhost',
                    'db_port': '5432',
                    'db_name': 'test_db',
                    'db_user': 'test_user'
                }
                
                config_path = Path("configs/dev.yaml")
                env_path = Path(".env.dev.template")
                mock_builder.generate_config_file.return_value = config_path
                mock_builder.generate_env_template.return_value = env_path
                
                # 실행 (non_interactive=False, template=None)
                get_config_command(env_name="dev", non_interactive=False, template=None)
                
                # 검증
                mock_builder.run_interactive_flow.assert_called_once_with("dev")
                mock_builder.generate_config_file.assert_called_once()
                mock_builder.generate_env_template.assert_called_once()
    
    def test_create_from_template_invalid_template(self):
        """잘못된 템플릿 이름 테스트."""
        import click
        with patch('src.cli.commands.get_config_command.console'):
            with pytest.raises(click.exceptions.Exit):
                _create_from_template("test", "invalid_template")


class TestInteractiveConfigBuilder:
    """InteractiveConfigBuilder 테스트."""
    
    def test_generate_config_file(self):
        """Config 파일 생성 테스트."""
        builder = InteractiveConfigBuilder()
        
        selections = {
            'env_name': 'test',
            'project_name': 'test-project',
            'data_source': 'postgresql',
            'db_host': 'localhost',
            'db_port': '5432',
            'db_name': 'test_db',
            'db_user': 'test_user',
            'db_connection_uri': 'postgresql://${DB_USER}:${DB_PASSWORD}@localhost:5432/test_db',
            'mlflow_type': 'local',
            'mlflow_uri': './mlruns',
            'feature_store_enabled': False,
            'storage_type': 'local',
            'storage_path': './data'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.cli.core.config_builder.Path.cwd', return_value=Path(tmpdir)):
                config_path = builder.generate_config_file('test', selections)
                
                assert config_path.exists()
                assert config_path.name == 'test.yaml'
                
                # Config 내용 검증
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                assert config['environment']['app_env'] == 'test'
                assert config['mlflow']['tracking_uri'] == './mlruns'
                assert 'sql' in config['data_adapters']['adapters']
    
    def test_generate_env_template(self):
        """환경 변수 템플릿 생성 테스트."""
        builder = InteractiveConfigBuilder()
        
        selections = {
            'env_name': 'test',
            'project_name': 'test-project',
            'data_source': 'postgresql',
            'db_user': 'test_user',
            'mlflow_type': 'remote',
            'mlflow_uri': 'http://localhost:5000',
            'online_store_type': 'redis',
            'redis_host': 'localhost',
            'redis_port': '6379',
            'storage_type': 'gcs',
            'gcs_bucket': 'test-bucket'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.cli.core.config_builder.Path.cwd', return_value=Path(tmpdir)):
                env_path = builder.generate_env_template('test', selections)
                
                assert env_path.exists()
                assert env_path.name == '.env.test.template'
                
                # 내용 검증
                content = env_path.read_text()
                assert 'ENV_NAME=test' in content
                assert 'PROJECT_NAME=test-project' in content
                assert 'DB_USER=test_user' in content
                assert 'DB_PASSWORD=' in content
                assert 'MLFLOW_TRACKING_URI=' in content
                assert 'REDIS_HOST=' in content
                assert 'GCS_BUCKET=' in content
    
    @patch('src.cli.core.config_builder.Prompt')
    @patch('src.cli.core.config_builder.Confirm')
    def test_run_interactive_flow(self, mock_confirm, mock_prompt):
        """대화형 플로우 테스트."""
        builder = InteractiveConfigBuilder()
        
        # Mock 설정
        mock_prompt.ask.side_effect = [
            'dev',  # env_name
            'my-project',  # project_name
            '1',  # PostgreSQL 선택
            'localhost',  # db_host
            '5432',  # db_port
            'mlflow',  # db_name
            'postgres',  # db_user
            '1',  # MLflow local
            './mlruns',  # mlflow_uri
            'dev-experiment',  # experiment
            '1',  # Local storage
            './data'  # storage_path
        ]
        
        mock_confirm.ask.side_effect = [
            False,  # Feature Store 사용 안함
            False   # 고급 설정 안함
        ]
        
        # Mock selector
        with patch.object(builder.selector, 'select') as mock_select:
            mock_select.side_effect = [
                'postgresql',  # data_source
                'local',  # mlflow_type
                'local'   # storage_type
            ]
            
            # 실행
            selections = builder.run_interactive_flow()
            
            # 검증
            assert selections['env_name'] == 'dev'
            assert selections['project_name'] == 'my-project'
            assert selections['data_source'] == 'postgresql'
            assert selections['mlflow_type'] == 'local'
            assert selections['storage_type'] == 'local'


class TestInteractiveSelector:
    """InteractiveSelector 테스트."""
    
    @patch('src.cli.ui.interactive_selector.Prompt')
    def test_select_single_option(self, mock_prompt):
        """단일 선택 테스트."""
        selector = InteractiveSelector()
        
        options = [
            ("Option 1", "value1"),
            ("Option 2", "value2"),
            ("Option 3", "value3")
        ]
        
        mock_prompt.ask.return_value = "2"
        
        with patch.object(selector.console, 'print'):
            result = selector.select("Choose one", options)
        
        assert result == "value2"
    
    @patch('src.cli.ui.interactive_selector.Prompt')
    def test_multi_select(self, mock_prompt):
        """다중 선택 테스트."""
        selector = InteractiveSelector()
        
        options = [
            ("Option 1", "value1"),
            ("Option 2", "value2"),
            ("Option 3", "value3")
        ]
        
        mock_prompt.ask.return_value = "1,3"
        
        with patch.object(selector.console, 'print'):
            results = selector.multi_select("Choose multiple", options)
        
        assert results == ["value1", "value3"]
    
    def test_select_empty_options(self):
        """빈 옵션 리스트 테스트."""
        selector = InteractiveSelector()
        
        with pytest.raises(ValueError, match="옵션 목록이 비어있습니다"):
            selector.select("Choose", [])