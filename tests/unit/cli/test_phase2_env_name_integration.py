"""
Phase 2 Integration Test: --env-name Parameter
모든 명령어의 --env-name 파라미터 통합 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import yaml
import json

import pytest
from typer.testing import CliRunner

from src.cli.main_commands import app


class TestEnvNameIntegration:
    """--env-name 파라미터 통합 테스트."""
    
    @pytest.fixture
    def cli_runner(self):
        """CLI 테스트 러너."""
        return CliRunner()
    
    @pytest.fixture
    def test_environment(self):
        """테스트 환경 설정."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # .env.test 파일 생성
            env_file = tmpdir_path / ".env.test"
            env_file.write_text(
                "DB_HOST=localhost\n"
                "DB_PORT=5432\n"
                "DB_USER=testuser\n"
                "DB_PASSWORD=testpass\n"
                "MLFLOW_TRACKING_URI=./mlruns\n"
            )
            
            # configs/test.yaml 파일 생성
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            config_file = configs_dir / "test.yaml"
            config_content = {
                "environment": {
                    "app_env": "test",
                    "gcp_project_id": "${GCP_PROJECT:}"
                },
                "mlflow": {
                    "tracking_uri": "${MLFLOW_TRACKING_URI}",
                    "experiment_name": "test-experiment"
                },
                "data_adapters": {
                    "default_loader": "sql",
                    "adapters": {
                        "sql": {
                            "class_name": "SqlAdapter",
                            "config": {
                                "connection_uri": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/testdb"
                            }
                        }
                    }
                }
            }
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f)
            
            # recipes/test.yaml 파일 생성
            recipes_dir = tmpdir_path / "recipes"
            recipes_dir.mkdir()
            recipe_file = recipes_dir / "test.yaml"
            recipe_content = {
                "model": {
                    "type": "sklearn",
                    "class_path": "sklearn.linear_model.LogisticRegression",
                    "params": {}
                },
                "training": {
                    "epochs": 10
                }
            }
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_content, f)
            
            yield tmpdir_path
    
    def test_train_with_env_name(self, cli_runner, test_environment):
        """train 명령어 --env-name 테스트."""
        recipe_file = test_environment / "recipes" / "test.yaml"
        
        with patch('src.cli.main_commands.load_environment') as mock_load_env:
            with patch('src.cli.main_commands.load_settings_by_file') as mock_load_settings:
                with patch('src.cli.main_commands.run_training') as mock_run_training:
                    with patch('src.cli.main_commands.setup_logging'):
                        mock_settings = MagicMock()
                        mock_settings.recipe.model.computed = {"run_name": "test_run"}
                        mock_load_settings.return_value = mock_settings
                        
                        # 명령어 실행
                        result = cli_runner.invoke(app, [
                            "train",
                            "--recipe-file", str(recipe_file),
                            "--env-name", "test"
                        ])
                        
                        # 검증
                        assert result.exit_code == 0
                        mock_load_env.assert_called_once_with("test")
                        mock_load_settings.assert_called_once_with(
                            str(recipe_file),
                            context_params=None,
                            env_name="test"
                        )
                        mock_run_training.assert_called_once()
    
    def test_batch_inference_with_env_name(self, cli_runner):
        """batch-inference 명령어 --env-name 테스트."""
        with patch('src.cli.main_commands.load_environment') as mock_load_env:
            with patch('src.cli.main_commands.load_config_files') as mock_load_config:
                with patch('src.cli.main_commands.create_settings_for_inference') as mock_create_settings:
                    with patch('src.cli.main_commands.run_batch_inference') as mock_run_inference:
                        with patch('src.cli.main_commands.setup_logging'):
                            mock_config = {"test": "config"}
                            mock_load_config.return_value = mock_config
                            mock_settings = MagicMock()
                            mock_create_settings.return_value = mock_settings
                            
                            # 명령어 실행
                            result = cli_runner.invoke(app, [
                                "batch-inference",
                                "--run-id", "test_run_id",
                                "--env-name", "prod"
                            ])
                            
                            # 검증
                            assert result.exit_code == 0
                            mock_load_env.assert_called_once_with("prod")
                            mock_load_config.assert_called_once_with(env_name="prod")
                            mock_create_settings.assert_called_once_with(mock_config)
                            mock_run_inference.assert_called_once_with(
                                settings=mock_settings,
                                run_id="test_run_id",
                                context_params={}
                            )
    
    def test_serve_api_with_env_name(self, cli_runner):
        """serve-api 명령어 --env-name 테스트."""
        with patch('src.cli.main_commands.load_environment') as mock_load_env:
            with patch('src.cli.main_commands.load_config_files') as mock_load_config:
                with patch('src.cli.main_commands.create_settings_for_inference') as mock_create_settings:
                    with patch('src.cli.main_commands.run_api_server') as mock_run_server:
                        with patch('src.cli.main_commands.setup_logging'):
                            mock_config = {"test": "config"}
                            mock_load_config.return_value = mock_config
                            mock_settings = MagicMock()
                            mock_create_settings.return_value = mock_settings
                            
                            # 명령어 실행
                            result = cli_runner.invoke(app, [
                                "serve-api",
                                "--run-id", "test_run_id",
                                "--env-name", "dev",
                                "--host", "localhost",
                                "--port", "8080"
                            ])
                            
                            # 검증
                            assert result.exit_code == 0
                            mock_load_env.assert_called_once_with("dev")
                            mock_load_config.assert_called_once_with(env_name="dev")
                            mock_create_settings.assert_called_once_with(mock_config)
                            mock_run_server.assert_called_once_with(
                                settings=mock_settings,
                                run_id="test_run_id",
                                host="localhost",
                                port=8080
                            )
    
    def test_system_check_with_env_name(self, cli_runner, test_environment):
        """system-check 명령어 --env-name 테스트."""
        with patch('src.cli.commands.system_check_command.load_config_with_env') as mock_load_config:
            with patch('src.cli.commands.system_check_command.DynamicServiceChecker') as mock_checker_class:
                mock_config = {"test": "config"}
                mock_load_config.return_value = mock_config
                
                mock_checker = MagicMock()
                mock_checker_class.return_value = mock_checker
                mock_checker.check_single_environment.return_value = {
                    'overall_healthy': True,
                    'passed_checks': 4,
                    'failed_checks': 0,
                    'total_checks': 4,
                    'results': []
                }
                
                # 명령어 실행
                result = cli_runner.invoke(app, [
                    "system-check",
                    "--env-name", "test"
                ])
                
                # 검증
                assert result.exit_code == 0
                mock_load_config.assert_called_once_with("test")
                mock_checker.check_single_environment.assert_called_once_with(
                    "test", mock_config, actionable=False
                )
    
    def test_env_name_fallback_to_env_var(self, cli_runner):
        """ENV_NAME 환경변수 fallback 테스트."""
        os.environ["ENV_NAME"] = "env_from_var"
        
        with patch('src.cli.main_commands.load_environment') as mock_load_env:
            with patch('src.cli.main_commands.load_config_files') as mock_load_config:
                with patch('src.cli.main_commands.create_settings_for_inference'):
                    with patch('src.cli.main_commands.run_batch_inference'):
                        with patch('src.cli.main_commands.setup_logging'):
                            # env_name 파라미터 없이 실행
                            result = cli_runner.invoke(app, [
                                "batch-inference",
                                "--run-id", "test_run_id"
                            ])
                            
                            # ENV_NAME 환경변수 사용 확인
                            assert result.exit_code == 0
                            mock_load_env.assert_called_once_with("env_from_var")
                            mock_load_config.assert_called_once_with(env_name="env_from_var")
    
    def test_error_when_no_env_name(self, cli_runner):
        """env_name 없을 때 에러 테스트."""
        os.environ.pop("ENV_NAME", None)
        
        # env_name 파라미터와 ENV_NAME 환경변수 둘 다 없을 때
        result = cli_runner.invoke(app, [
            "batch-inference",
            "--run-id", "test_run_id"
        ])
        
        # 에러 발생 확인 (exit code만 확인)
        assert result.exit_code != 0
        # 로거에 에러가 기록되었는지는 위 exit_code로 충분히 검증됨
    
    def test_train_with_context_params(self, cli_runner, test_environment):
        """train 명령어 context_params 테스트."""
        recipe_file = test_environment / "recipes" / "test.yaml"
        context_params = {"date": "2024-01-01", "version": "v1"}
        
        with patch('src.cli.main_commands.load_environment'):
            with patch('src.cli.main_commands.load_settings_by_file') as mock_load_settings:
                with patch('src.cli.main_commands.run_training') as mock_run_training:
                    with patch('src.cli.main_commands.setup_logging'):
                        mock_settings = MagicMock()
                        mock_settings.recipe.model.computed = {}
                        mock_load_settings.return_value = mock_settings
                        
                        # 명령어 실행
                        result = cli_runner.invoke(app, [
                            "train",
                            "--recipe-file", str(recipe_file),
                            "--env-name", "test",
                            "--params", json.dumps(context_params)
                        ])
                        
                        # 검증
                        assert result.exit_code == 0
                        mock_load_settings.assert_called_once_with(
                            str(recipe_file),
                            context_params=context_params,
                            env_name="test"
                        )
                        mock_run_training.assert_called_once_with(
                            settings=mock_settings,
                            context_params=context_params
                        )