"""
E2E Test: Full CLI Workflow
Phase 3: 전체 5단계 CLI 플로우 통합 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
import subprocess
import os
import tempfile
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner


class TestFullCLIFlow:
    """5단계 CLI 플로우 전체 테스트."""
    
    @pytest.fixture
    def cli_runner(self):
        """CLI 테스트 러너."""
        return CliRunner()
    
    @pytest.fixture
    def test_project_dir(self):
        """테스트용 프로젝트 디렉토리 생성."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            
            # 기본 디렉토리 구조 생성
            (project_dir / "recipes").mkdir()
            (project_dir / "configs").mkdir()
            (project_dir / "sql").mkdir()
            (project_dir / "data").mkdir()
            
            # SQL 파일 생성
            sql_file = project_dir / "sql" / "train.sql"
            sql_file.write_text("SELECT user_id, feature_1, target, created_at FROM test_table LIMIT 100")
            
            # 원래 디렉토리 저장
            original_cwd = os.getcwd()
            os.chdir(project_dir)
            
            yield project_dir
            
            # 원래 디렉토리로 복원
            os.chdir(original_cwd)
    
    def test_phase1_get_config(self, test_project_dir, cli_runner):
        """Phase 1: Config 생성 테스트."""
        from src.cli.main_commands import app
        
        # 비대화형 모드로 테스트
        result = cli_runner.invoke(app, [
            "get-config",
            "--env-name", "test",
            "--template", "local",
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert (test_project_dir / "configs" / "test.yaml").exists()
        assert (test_project_dir / ".env.test.template").exists()
        
        # .env 파일 생성
        env_template = test_project_dir / ".env.test.template"
        env_file = test_project_dir / ".env.test"
        env_content = env_template.read_text()
        env_content += "\nDB_PASSWORD=testpass123\n"
        env_file.write_text(env_content)
    
    def test_phase2_system_check(self, test_project_dir, cli_runner):
        """Phase 2: 시스템 체크 테스트."""
        from src.cli.main_commands import app
        
        # 먼저 config 생성
        self.test_phase1_get_config(test_project_dir, cli_runner)
        
        # Mock 서비스 설정
        self._setup_mock_services(test_project_dir)
        
        with patch('src.cli.commands.system_check_command.DynamicServiceChecker') as mock_checker:
            mock_instance = MagicMock()
            mock_checker.return_value = mock_instance
            mock_instance.check_single_environment.return_value = {
                'overall_healthy': True,
                'passed_checks': 3,
                'failed_checks': 0,
                'total_checks': 3,
                'results': []
            }
            
            result = cli_runner.invoke(app, [
                "system-check",
                "--env-name", "test"
            ])
            
            assert result.exit_code == 0
            mock_instance.check_single_environment.assert_called_once()
    
    def test_phase3_create_recipe(self, test_project_dir):
        """Phase 3: Recipe 생성 테스트."""
        # SQL 파일 생성 (Recipe에서 참조하는 파일)
        sql_file = test_project_dir / "sql" / "train.sql"
        sql_file.parent.mkdir(exist_ok=True)
        sql_file.write_text("SELECT user_id, feature_1, target, created_at FROM test_table")
        
        # Recipe 파일 직접 생성 (대화형 테스트 대신)
        recipe = {
            "name": "test_model",
            "model": {
                "class_path": "sklearn.linear_model.LogisticRegression",
                "loader": {
                    "adapter": "sql",
                    "source_uri": "sql/train.sql",
                    "entity_schema": {
                        "entity_columns": ["user_id"],
                        "timestamp_column": "created_at"
                    }
                },
                "data_interface": {
                    "task_type": "classification",
                    "target_column": "target"
                },
                "hyperparameters": {},
                "computed": {
                    "run_name": "test_model_run"
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision_weighted", "recall_weighted"],
                "validation": {"method": "train_test_split"}
            }
        }
        
        recipe_file = test_project_dir / "recipes" / "test.yaml"
        with open(recipe_file, 'w') as f:
            yaml.dump(recipe, f)
        
        assert recipe_file.exists()
        
        # Recipe 내용 검증
        with open(recipe_file, 'r') as f:
            loaded_recipe = yaml.safe_load(f)
        
        assert loaded_recipe['name'] == 'test_model'
        assert loaded_recipe['model']['class_path'] == 'sklearn.linear_model.LogisticRegression'
    
    def test_phase4_train_command(self, test_project_dir, cli_runner):
        """Phase 4: 학습 실행 테스트."""
        from src.cli.main_commands import app
        
        # 준비: config와 recipe 생성
        self.test_phase1_get_config(test_project_dir, cli_runner)
        self.test_phase3_create_recipe(test_project_dir)
        
        # Mock 데이터 준비
        self._prepare_mock_data(test_project_dir)
        
        with patch('src.cli.main_commands.run_training') as mock_run_training:
            with patch('src.cli.main_commands.load_settings_by_file') as mock_load_settings:
                mock_settings = MagicMock()
                mock_settings.recipe.model.computed = {"run_name": "test_run"}
                mock_load_settings.return_value = mock_settings
                
                result = cli_runner.invoke(app, [
                    "train",
                    "--recipe-file", "recipes/test.yaml",
                    "--env-name", "test"
                ])
                
                assert result.exit_code == 0
                mock_load_settings.assert_called_once()
                mock_run_training.assert_called_once()
    
    def test_full_flow_integration(self, test_project_dir, cli_runner):
        """전체 플로우 통합 테스트."""
        from src.cli.main_commands import app
        
        # Step 1: Config 생성
        self.test_phase1_get_config(test_project_dir, cli_runner)
        
        # Step 2: Recipe 생성
        self.test_phase3_create_recipe(test_project_dir)
        
        # Step 3: 환경변수 설정
        os.environ['ENV_NAME'] = 'test'
        
        # Step 4: Settings 로드 테스트
        with patch('src.settings._builder.BASE_DIR', test_project_dir):
            with patch('src.settings.loaders.BASE_DIR', test_project_dir):
                with patch('src.settings._utils.BASE_DIR', test_project_dir):
                    from src.settings import load_settings_by_file
                    
                    with patch('src.settings.loaders.load_config_files') as mock_load_config:
                        mock_config = {
                            'environment': {'app_env': 'test', 'gcp_project_id': 'test-project'},
                            'mlflow': {'tracking_uri': './mlruns', 'experiment_name': 'test_experiment'},
                            'data_adapters': {'adapters': {}},
                            'serving': {
                                'model_registry': {'type': 'local'},
                                'model_stage': 'production',
                                'realtime_feature_store': {
                                    'enabled': False,
                                    'store_type': 'none',
                                    'connection': {'host': 'localhost', 'port': 6379}
                                }
                            },
                            'artifact_stores': {
                                'default': {
                                    'type': 'local',
                                    'path': './artifacts',
                                    'enabled': True,
                                    'base_uri': './artifacts'
                                }
                            }
                        }
                        mock_load_config.return_value = mock_config
                        
                        settings = load_settings_by_file(
                            str(test_project_dir / "recipes" / "test.yaml"),
                            env_name="test"
                        )
                        
                        assert settings is not None
    
    def test_environment_switching(self, test_project_dir, cli_runner):
        """환경 전환 테스트."""
        from src.cli.main_commands import app
        
        # Dev 환경 생성
        result = cli_runner.invoke(app, [
            "get-config",
            "--env-name", "dev",
            "--template", "dev",
            "--non-interactive"
        ])
        assert result.exit_code == 0
        
        # Prod 환경 생성
        result = cli_runner.invoke(app, [
            "get-config",
            "--env-name", "prod",
            "--template", "prod",
            "--non-interactive"
        ])
        assert result.exit_code == 0
        
        # 두 환경의 config 파일이 모두 존재하는지 확인
        assert (test_project_dir / "configs" / "dev.yaml").exists()
        assert (test_project_dir / "configs" / "prod.yaml").exists()
        
        # 환경별로 다른 설정 확인
        with open(test_project_dir / "configs" / "dev.yaml", 'r') as f:
            dev_config = yaml.safe_load(f)
        with open(test_project_dir / "configs" / "prod.yaml", 'r') as f:
            prod_config = yaml.safe_load(f)
        
        assert dev_config['environment']['app_env'] == 'dev'
        assert prod_config['environment']['app_env'] == 'prod'
    
    def test_batch_inference_flow(self, test_project_dir, cli_runner):
        """배치 추론 플로우 테스트."""
        from src.cli.main_commands import app
        
        # 준비
        self.test_phase1_get_config(test_project_dir, cli_runner)
        
        with patch('src.cli.main_commands.run_batch_inference') as mock_run_inference:
            with patch('src.cli.main_commands.create_settings_for_inference') as mock_create_settings:
                mock_settings = MagicMock()
                mock_create_settings.return_value = mock_settings
                
                result = cli_runner.invoke(app, [
                    "batch-inference",
                    "--run-id", "test_run_123",
                    "--env-name", "test"
                ])
                
                assert result.exit_code == 0
                mock_run_inference.assert_called_once_with(
                    settings=mock_settings,
                    run_id="test_run_123",
                    context_params={}
                )
    
    def test_serve_api_flow(self, test_project_dir, cli_runner):
        """API 서빙 플로우 테스트."""
        from src.cli.main_commands import app
        
        # 준비
        self.test_phase1_get_config(test_project_dir, cli_runner)
        
        with patch('src.cli.main_commands.run_api_server') as mock_run_server:
            with patch('src.cli.main_commands.create_settings_for_inference') as mock_create_settings:
                mock_settings = MagicMock()
                mock_create_settings.return_value = mock_settings
                
                result = cli_runner.invoke(app, [
                    "serve-api",
                    "--run-id", "test_run_123",
                    "--env-name", "test",
                    "--port", "8080"
                ])
                
                assert result.exit_code == 0
                mock_run_server.assert_called_once_with(
                    settings=mock_settings,
                    run_id="test_run_123",
                    host="0.0.0.0",
                    port=8080
                )
    
    def _setup_mock_services(self, project_dir: Path) -> None:
        """Mock 서비스 설정."""
        os.environ['DB_CONNECTION_URI'] = 'postgresql://test:test@localhost:5432/test'
        os.environ['MLFLOW_TRACKING_URI'] = str(project_dir / 'mlruns')
    
    def _prepare_mock_data(self, project_dir: Path) -> None:
        """Mock 데이터 준비."""
        # 테스트용 CSV 생성
        data_content = """user_id,feature_1,target,created_at
1,0.5,0,2024-01-01
2,0.7,1,2024-01-02
3,0.3,0,2024-01-03
4,0.9,1,2024-01-04
5,0.1,0,2024-01-05"""
        
        data_file = project_dir / "data" / "test_data.csv"
        data_file.write_text(data_content)