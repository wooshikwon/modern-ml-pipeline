"""
Backward Compatibility Test
Phase 3: 기존 시스템과의 호환성 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

import pytest

from src.settings import load_settings_by_file, load_config_files
from src.cli.utils.env_loader import get_env_name_with_fallback


class TestBackwardCompatibility:
    """기존 시스템과의 후방 호환성 테스트."""
    
    def test_legacy_mode_without_env_name(self):
        """env_name 없이 기존 방식으로 동작 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # 기존 config 디렉토리 구조 생성 (config/base.yaml + config/config.yaml)
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            
            # base.yaml
            base_config = {
                "base_setting": True,
                "environment": {
                    "app_env": "default"
                }
            }
            with open(config_dir / "base.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # config.yaml
            main_config = {
                "mlflow": {
                    "tracking_uri": "./mlruns"
                }
            }
            with open(config_dir / "config.yaml", 'w') as f:
                yaml.dump(main_config, f)
            
            # SQL 파일 생성
            sql_dir = tmpdir_path / "sql"
            sql_dir.mkdir()
            sql_file = sql_dir / "query.sql"
            sql_file.write_text("SELECT id, feature_1, feature_2, target FROM legacy_table")
            
            # recipe 파일
            recipes_dir = tmpdir_path / "recipes"
            recipes_dir.mkdir()
            recipe_file = recipes_dir / "legacy.yaml"
            recipe_content = {
                "name": "legacy_model",
                "model": {
                    "class_path": "sklearn.linear_model.LogisticRegression",
                    "loader": {
                        "adapter": "sql",
                        "source_uri": "sql/query.sql",
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "created_at"
                        }
                    },
                    "data_interface": {
                        "task_type": "classification",
                        "target_column": "target"
                    },
                    "hyperparameters": {}
                },
                "evaluation": {
                    "metrics": ["accuracy"],
                    "validation": {"method": "train_test_split"}
                }
            }
            with open(recipe_file, 'w') as f:
                yaml.dump(recipe_content, f)
            
            # env_name 없이 호출
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                with patch('src.settings.loaders.BASE_DIR', tmpdir_path):
                    with patch('src.settings._utils.BASE_DIR', tmpdir_path):
                        with patch('src.settings.loaders.load_config_files') as mock_load_config:
                            mock_config = {
                                'base_setting': True,
                                'environment': {'app_env': 'default', 'gcp_project_id': 'test-project'},
                                'mlflow': {'tracking_uri': './mlruns', 'experiment_name': 'default_experiment'},
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
                            
                            # env_name 없이 Settings 로드
                            settings = load_settings_by_file(str(recipe_file))
                            
                            assert settings is not None
                            # 기본 병합된 config 사용 확인
                            mock_load_config.assert_called_once()
    
    def test_env_name_from_environment_variable(self):
        """ENV_NAME 환경변수로 환경 지정 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # configs/prod.yaml 생성
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            
            prod_config = {
                "environment": {
                    "app_env": "prod"
                },
                "mlflow": {
                    "tracking_uri": "http://mlflow.prod:5000"
                }
            }
            with open(configs_dir / "prod.yaml", 'w') as f:
                yaml.dump(prod_config, f)
            
            # .env.prod 파일 생성
            env_file = tmpdir_path / ".env.prod"
            env_file.write_text("DB_HOST=prod.db.com\n")
            
            # ENV_NAME 환경변수 설정
            os.environ['ENV_NAME'] = 'prod'
            
            # env_name 파라미터 없이 환경 이름 가져오기
            env_name = get_env_name_with_fallback(None)
            assert env_name == 'prod'
            
            # 환경변수 정리
            os.environ.pop('ENV_NAME', None)
    
    def test_mixed_config_and_configs_directory(self):
        """config/ 와 configs/ 디렉토리 공존 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # 기존 config/ 디렉토리
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            old_config = config_dir / "legacy.yaml"
            old_config.write_text("environment:\n  app_env: legacy\n")
            
            # 새로운 configs/ 디렉토리
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            new_config = configs_dir / "new.yaml"
            new_config.write_text("environment:\n  app_env: new\n")
            
            # configs/ 디렉토리가 우선됨
            from src.cli.utils.env_loader import get_config_path
            
            config_path = get_config_path("new", base_path=tmpdir_path)
            assert "configs" in str(config_path)
            assert config_path.name == "new.yaml"
            
            # config/ 디렉토리 fallback
            config_path = get_config_path("legacy", base_path=tmpdir_path)
            assert "config" in str(config_path)
            assert config_path.name == "legacy.yaml"
    
    def test_recipe_compatibility(self):
        """기존 Recipe 파일 호환성 테스트."""
        # 기존 Recipe 구조 (환경 무관)
        old_recipe = {
            "name": "existing_model",
            "model": {
                "loader": {
                    "adapter": "sql",
                    "source_uri": "sql/query.sql",
                    "entity_schema": {
                        "entity_columns": ["user_id"],
                        "timestamp_column": "created_at"
                    }
                },
                "hyperparameters": {},
                "class_path": "xgboost.XGBClassifier",
                "hyperparameters": {
                    "n_estimators": 100
                }
            }
        }
        
        # 새로운 시스템에서도 동일하게 동작
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_file = Path(tmpdir) / "recipe.yaml"
            with open(recipe_file, 'w') as f:
                yaml.dump(old_recipe, f)
            
            # Recipe 로드 및 검증
            with open(recipe_file, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert loaded['name'] == 'existing_model'
            assert loaded['model']['loader']['adapter'] == 'sql'
            assert loaded['model']['hyperparameters']['n_estimators'] == 100
    
    def test_environment_variable_substitution_compatibility(self):
        """환경변수 치환 호환성 테스트."""
        # 기존 ${VAR} 형식
        old_format = "${DB_HOST}"
        
        # 새로운 ${VAR:default} 형식
        new_format = "${DB_HOST:localhost}"
        
        from src.cli.utils.env_loader import resolve_env_variables
        
        # 환경변수 설정
        os.environ['DB_HOST'] = 'test.db.com'
        
        # 둘 다 동작
        assert resolve_env_variables(old_format) == 'test.db.com'
        assert resolve_env_variables(new_format) == 'test.db.com'
        
        # 환경변수 제거
        os.environ.pop('DB_HOST', None)
        
        # 기존 형식은 그대로, 새 형식은 기본값 사용
        assert resolve_env_variables(old_format) == '${DB_HOST}'
        assert resolve_env_variables(new_format) == 'localhost'
    
    def test_cli_command_compatibility(self):
        """CLI 명령어 호환성 테스트."""
        from typer.testing import CliRunner
        from src.cli.main_commands import app
        
        runner = CliRunner()
        
        # 기존 명령어들이 여전히 동작하는지 확인
        commands_to_test = [
            ["--help"],
            ["list", "models"],
            ["list", "adapters"],
            ["list", "evaluators"],
            ["list", "preprocessors"],
        ]
        
        for cmd in commands_to_test:
            result = runner.invoke(app, cmd)
            assert result.exit_code == 0
    
    def test_settings_schema_compatibility(self):
        """Settings 스키마 호환성 테스트."""
        # 기존 Settings 구조가 유지되는지 확인
        from src.settings.schema import Settings
        
        # Settings 클래스의 필드 확인 (클래스 속성이 아닌 인스턴스 필드)
        # Settings는 Pydantic BaseModel이므로 __fields__ 또는 model_fields 사용
        fields = Settings.model_fields if hasattr(Settings, 'model_fields') else Settings.__fields__
        field_names = set(fields.keys())
        
        assert 'environment' in field_names
        assert 'mlflow' in field_names
        assert 'data_adapters' in field_names
        assert 'artifact_stores' in field_names
        assert 'recipe' in field_names
    
    def test_phase0_load_config_for_env_compatibility(self):
        """Phase 0 load_config_for_env 함수 호환성 테스트."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # configs 디렉토리 생성
            configs_dir = tmpdir_path / "configs"
            configs_dir.mkdir()
            
            # base.yaml
            base_config = configs_dir / "base.yaml"
            base_config.write_text("base: true\n")
            
            # test.yaml
            test_config = configs_dir / "test.yaml"
            test_config.write_text("environment:\n  app_env: test\n")
            
            # Phase 0 함수 사용
            from src.settings._builder import load_config_for_env
            
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                config = load_config_for_env('test')
                
                # base.yaml과 test.yaml이 병합됨
                assert 'base' in config
                assert config['environment']['app_env'] == 'test'