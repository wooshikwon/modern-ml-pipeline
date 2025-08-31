"""
Phase 0: env_name 파라미터 지원 테스트 (Simplified)

Settings 로더의 env_name 파라미터 지원과 하위 호환성을 테스트합니다.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.settings._builder import load_config_files, load_config_for_env


class TestEnvNameParameterSupport:
    """env_name 파라미터 지원 단순 테스트"""
    
    def test_load_config_files_with_env_name(self):
        """load_config_files가 env_name을 받을 수 있는지 테스트"""
        with patch('src.settings._builder.load_config_for_env') as mock_load:
            mock_load.return_value = {'test': 'config'}
            
            # env_name이 지정되면 load_config_for_env 호출
            result = load_config_files(env_name='dev')
            
            mock_load.assert_called_once_with('dev')
            assert result == {'test': 'config'}
    
    def test_load_config_files_requires_env_name(self):
        """load_config_files가 v2.0에서 env_name을 필수로 요구하는지 테스트"""
        # v2.0: env_name은 필수 파라미터
        with pytest.raises(TypeError) as exc_info:
            load_config_files()  # env_name 없이 호출
        
        assert "missing 1 required positional argument: 'env_name'" in str(exc_info.value)
    
    def test_load_settings_by_file_accepts_env_name(self):
        """load_settings_by_file가 env_name 파라미터를 받는지 테스트"""
        from src.settings import load_settings_by_file
        from tests.factories.settings_factory import SettingsFactory
        
        with patch('src.settings.loaders.load_config_files') as mock_load_config:
            with patch('src.settings.loaders.load_recipe_file') as mock_load_recipe:
                # SettingsFactory를 사용해서 완전한 config 생성
                base_settings = SettingsFactory.create_base_settings()
                # recipe 부분은 제거 (별도로 로드됨)
                base_settings.pop('recipe', None)
                mock_load_config.return_value = base_settings
                
                mock_load_recipe.return_value = {
                    'name': 'test',
                    'model': {
                        'class_path': 'test.Model',
                        'loader': {
                            'name': 'test',
                            'source_uri': 'test.csv',
                            'adapter': 'storage',
                            'entity_schema': {
                                'entity_columns': ['id'],
                                'timestamp_column': 'ts'
                            }
                        },
                        'data_interface': {
                            'task_type': 'classification',
                            'target_column': 'target'
                        },
                        'hyperparameters': {}
                    },
                    'evaluation': {
                        'metrics': ['accuracy'],
                        'validation': {'method': 'train_test_split'}
                    }
                }
                
                # env_name 파라미터로 호출
                result = load_settings_by_file('test.yaml', env_name='prod')
                
                # load_config_files가 env_name과 함께 호출되었는지 확인
                mock_load_config.assert_called_once_with(env_name='prod')
                assert result is not None
                assert result.recipe.name == 'test'


class TestEnvironmentSpecificLoading:
    """환경별 로딩 기능 테스트"""
    
    def test_load_config_for_env_basic(self):
        """load_config_for_env 기본 동작 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # configs 디렉토리 생성
            configs_dir = tmpdir_path / 'configs'
            configs_dir.mkdir()
            
            # base.yaml 생성
            base_yaml = configs_dir / 'base.yaml'
            base_yaml.write_text('base: config\ncommon: value')
            
            # dev.yaml 생성
            dev_yaml = configs_dir / 'dev.yaml'
            dev_yaml.write_text('env: dev\ncommon: overridden')
            
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                result = load_config_for_env('dev')
                
                assert result['base'] == 'config'
                assert result['env'] == 'dev'
                assert result['common'] == 'overridden'  # dev가 base를 덮어씀
                assert os.environ.get('ENV_NAME') == 'dev'
    
    def test_env_file_loading(self):
        """환경별 .env 파일 로딩 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # configs 디렉토리 생성
            configs_dir = tmpdir_path / 'configs'
            configs_dir.mkdir()
            
            # 기본 yaml 파일들 생성
            (configs_dir / 'base.yaml').write_text('config: base')
            (configs_dir / 'test.yaml').write_text('config: test')
            
            # .env.test 파일 생성
            env_file = tmpdir_path / '.env.test'
            env_file.write_text('TEST_VAR=test_value\n')
            
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                with patch('src.settings._builder.load_dotenv') as mock_dotenv:
                    result = load_config_for_env('test')
                    
                    # .env.test 파일이 로드되었는지 확인
                    mock_dotenv.assert_called_once()
                    call_args = mock_dotenv.call_args[0]
                    assert str(call_args[0]).endswith('.env.test')
    
    def test_configs_directory_preference(self):
        """configs/ 디렉토리가 config/보다 우선되는지 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # 두 디렉토리 모두 생성
            config_dir = tmpdir_path / 'config'
            config_dir.mkdir()
            configs_dir = tmpdir_path / 'configs'
            configs_dir.mkdir()
            
            # 각 디렉토리에 다른 내용의 base.yaml 생성
            (config_dir / 'base.yaml').write_text('source: config')
            (configs_dir / 'base.yaml').write_text('source: configs')
            
            # 환경별 yaml
            (configs_dir / 'dev.yaml').write_text('env: dev')
            
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                result = load_config_for_env('dev')
                
                # configs/ 디렉토리의 파일이 사용되었는지 확인
                assert result['source'] == 'configs'
    
    def test_configs_directory_required(self):
        """v2.0에서 configs/ 디렉토리가 필수인지 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # config/ 디렉토리만 생성 (레거시, 지원 안 함)
            config_dir = tmpdir_path / 'config'
            config_dir.mkdir()
            
            # base.yaml 생성
            (config_dir / 'base.yaml').write_text('source: legacy')
            (config_dir / 'dev.yaml').write_text('env: dev')
            
            with patch('src.settings._builder.BASE_DIR', tmpdir_path):
                # v2.0: configs/ 디렉토리가 없으면 에러
                with pytest.raises(FileNotFoundError) as exc_info:
                    load_config_for_env('dev')
                
                assert 'configs/ directory not found' in str(exc_info.value)
                assert "Run 'mmp init'" in str(exc_info.value)