"""
Config Loader 테스트 (커버리지 확장)
config_loader.py 테스트 - 실제 존재하는 함수들만 테스트
"""
import pytest
import yaml
import os
from pathlib import Path

from src.cli.utils.config_loader import (
    resolve_env_variables,
    get_config_path,
    load_environment
)


class TestConfigLoaderBasic:
    """Config Loader 기본 테스트"""
    
    def test_resolve_env_variables_with_string(self):
        """환경변수 치환 - 문자열"""
        # 환경변수 설정
        os.environ['TEST_VAR'] = 'test_value'
        
        # 단순 문자열
        result = resolve_env_variables('hello world')
        assert result == 'hello world'
        
        # 환경변수 참조
        result = resolve_env_variables('${TEST_VAR}')
        assert result == 'test_value'
        
        # 기본값이 있는 환경변수
        result = resolve_env_variables('${NONEXISTENT_VAR:default}')
        assert result == 'default'
        
        # 정리
        del os.environ['TEST_VAR']
    
    def test_resolve_env_variables_with_dict(self):
        """환경변수 치환 - 딕셔너리"""
        os.environ['TEST_KEY'] = 'test_value'
        
        test_dict = {
            'key1': '${TEST_KEY}',
            'key2': 'static_value',
            'key3': '${MISSING:fallback}'
        }
        
        result = resolve_env_variables(test_dict)
        
        assert result['key1'] == 'test_value'
        assert result['key2'] == 'static_value'
        assert result['key3'] == 'fallback'
        
        del os.environ['TEST_KEY']
    
    def test_resolve_env_variables_with_list(self):
        """환경변수 치환 - 리스트"""
        os.environ['LIST_VAR'] = 'item'
        
        test_list = ['${LIST_VAR}', 'static', '${MISSING:default}']
        result = resolve_env_variables(test_list)
        
        assert result == ['item', 'static', 'default']
        
        del os.environ['LIST_VAR']
    
    def test_resolve_env_variables_type_conversion(self):
        """환경변수 치환 - 타입 변환"""
        os.environ['BOOL_VAR'] = 'true'
        os.environ['INT_VAR'] = '42'
        os.environ['FLOAT_VAR'] = '3.14'
        
        # 불린 변환
        result = resolve_env_variables('${BOOL_VAR}')
        assert result is True
        
        # 정수 변환
        result = resolve_env_variables('${INT_VAR}')
        assert result == 42
        
        # 실수 변환
        result = resolve_env_variables('${FLOAT_VAR}')
        assert result == 3.14
        
        # 정리
        del os.environ['BOOL_VAR']
        del os.environ['INT_VAR']
        del os.environ['FLOAT_VAR']
    
    def test_get_config_path_existing_file(self, tmp_path):
        """설정 파일 경로 찾기 - 존재하는 파일"""
        # configs 디렉토리 생성
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        
        # 설정 파일 생성
        config_file = configs_dir / "test.yaml"
        config_file.write_text("task: test")
        
        # 경로 찾기
        result = get_config_path("test", base_path=tmp_path)
        assert result == config_file
        assert result.exists()
    
    def test_get_config_path_nonexistent_file(self, tmp_path):
        """설정 파일 경로 찾기 - 존재하지 않는 파일"""
        with pytest.raises(FileNotFoundError) as exc_info:
            get_config_path("nonexistent", base_path=tmp_path)
        
        error_msg = str(exc_info.value)
        assert "nonexistent.yaml" in error_msg
    
    def test_load_environment_existing_file(self, tmp_path):
        """환경변수 파일 로드 - 존재하는 파일"""
        # .env.test 파일 생성
        env_file = tmp_path / ".env.test"
        env_file.write_text("TEST_ENV_VAR=test_value\nANOTHER_VAR=another")
        
        # 환경변수 로드
        load_environment("test", base_path=tmp_path)
        
        # 환경변수 확인
        assert os.environ.get("TEST_ENV_VAR") == "test_value"
        assert os.environ.get("ANOTHER_VAR") == "another"
        assert os.environ.get("ENV_NAME") == "test"
        
        # 정리
        del os.environ["TEST_ENV_VAR"]
        del os.environ["ANOTHER_VAR"]
        del os.environ["ENV_NAME"]
    
    def test_load_environment_nonexistent_file(self, tmp_path):
        """환경변수 파일 로드 - 존재하지 않는 파일"""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_environment("nonexistent", base_path=tmp_path)
        
        error_msg = str(exc_info.value)
        assert ".env.nonexistent" in error_msg


class TestConfigLoaderEdgeCases:
    """Config Loader 엣지 케이스"""
    
    def test_resolve_env_variables_none_input(self):
        """None 입력 처리"""
        result = resolve_env_variables(None)
        assert result is None
    
    def test_resolve_env_variables_numeric_input(self):
        """숫자 입력 처리"""
        result = resolve_env_variables(42)
        assert result == 42
        
        result = resolve_env_variables(3.14)
        assert result == 3.14
    
    def test_resolve_env_variables_empty_dict(self):
        """빈 딕셔너리 처리"""
        result = resolve_env_variables({})
        assert result == {}
    
    def test_resolve_env_variables_empty_list(self):
        """빈 리스트 처리"""
        result = resolve_env_variables([])
        assert result == []
    
    def test_resolve_env_variables_nested_structure(self):
        """중첩 구조 처리"""
        os.environ['NESTED_VAR'] = 'nested_value'
        
        nested_data = {
            'level1': {
                'level2': ['${NESTED_VAR}', 'static'],
                'simple': '${NESTED_VAR:default}'
            }
        }
        
        result = resolve_env_variables(nested_data)
        
        assert result['level1']['level2'][0] == 'nested_value'
        assert result['level1']['level2'][1] == 'static'
        assert result['level1']['simple'] == 'nested_value'
        
        del os.environ['NESTED_VAR']
    
    def test_resolve_env_variables_malformed_pattern(self):
        """잘못된 형식의 환경변수 패턴"""
        # 잘못된 형식은 그대로 반환되어야 함
        result = resolve_env_variables('${MISSING_CLOSE')
        assert result == '${MISSING_CLOSE'
        
        result = resolve_env_variables('MISSING_OPEN}')
        assert result == 'MISSING_OPEN}'
    
    def test_get_config_path_configs_dir_missing(self, tmp_path):
        """configs 디렉토리가 없는 경우"""
        # configs 디렉토리를 생성하지 않음
        with pytest.raises(FileNotFoundError):
            get_config_path("test", base_path=tmp_path)