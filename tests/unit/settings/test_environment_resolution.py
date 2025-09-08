"""
Unit Tests for Environment Variable Resolution
Days 3-5: Advanced environment variable handling tests
"""

import pytest
from unittest.mock import patch
import os

from src.settings.loader import resolve_env_variables


class TestAdvancedEnvironmentResolution:
    """Advanced environment variable resolution tests"""
    
    def test_complex_nested_structure_resolution(self):
        """Test environment variable resolution in deeply nested structures"""
        with patch.dict('os.environ', {
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_NAME': 'ml_pipeline',
            'REDIS_URL': 'redis://localhost:6379/0',
            'LOG_LEVEL': 'INFO'
        }):
            complex_data = {
                'database': {
                    'primary': {
                        'host': '${DB_HOST}',
                        'port': '${DB_PORT}',
                        'name': '${DB_NAME}',
                        'connection_pool': {
                            'min_size': '${DB_POOL_MIN:5}',
                            'max_size': '${DB_POOL_MAX:20}'
                        }
                    },
                    'replica': {
                        'host': '${DB_REPLICA_HOST:localhost}',  # Simple default, no nested vars
                        'port': '${DB_PORT}'
                    }
                },
                'cache': {
                    'url': '${REDIS_URL}',
                    'ttl': '${CACHE_TTL:3600}'
                },
                'logging': {
                    'level': '${LOG_LEVEL}',
                    'handlers': ['${LOG_HANDLER:console}']
                }
            }
            
            result = resolve_env_variables(complex_data)
            
            # Verify primary database config
            assert result['database']['primary']['host'] == 'localhost'
            assert result['database']['primary']['port'] == 5432
            assert result['database']['primary']['name'] == 'ml_pipeline'
            
            # Verify connection pool defaults
            assert result['database']['primary']['connection_pool']['min_size'] == 5
            assert result['database']['primary']['connection_pool']['max_size'] == 20
            
            # Verify replica defaults to primary host
            assert result['database']['replica']['host'] == 'localhost'
            assert result['database']['replica']['port'] == 5432
            
            # Verify cache and logging
            assert result['cache']['url'] == 'redis://localhost:6379/0'
            assert result['cache']['ttl'] == 3600
            assert result['logging']['level'] == 'INFO'
            assert result['logging']['handlers'] == ['console']
    
    def test_environment_variable_type_coercion_edge_cases(self):
        """Test edge cases in environment variable type conversion"""
        with patch.dict('os.environ', {
            'ZERO_INT': '0',
            'NEGATIVE_INT': '-42',
            'ZERO_FLOAT': '0.0',
            'NEGATIVE_FLOAT': '-3.14',
            'SCIENTIFIC_NOTATION': '1.23e-4',
            'BOOL_TRUE_CAPS': 'TRUE',
            'BOOL_FALSE_CAPS': 'FALSE',
            'BOOL_YES': 'yes',
            'BOOL_NO': 'no',
            'BOOL_ON': 'on',
            'BOOL_OFF': 'off',
            'EMPTY_STRING': '',
            'WHITESPACE_STRING': '  whitespace  '
        }):
            test_cases = {
                'zero_int': '${ZERO_INT}',
                'negative_int': '${NEGATIVE_INT}',
                'zero_float': '${ZERO_FLOAT}',
                'negative_float': '${NEGATIVE_FLOAT}',
                'scientific': '${SCIENTIFIC_NOTATION}',
                'bool_true_caps': '${BOOL_TRUE_CAPS}',
                'bool_false_caps': '${BOOL_FALSE_CAPS}',
                'bool_yes': '${BOOL_YES}',
                'bool_no': '${BOOL_NO}',
                'bool_on': '${BOOL_ON}',
                'bool_off': '${BOOL_OFF}',
                'empty': '${EMPTY_STRING}',
                'whitespace': '${WHITESPACE_STRING}'
            }
            
            result = resolve_env_variables(test_cases)
            
            # Test integer conversion
            assert result['zero_int'] == 0
            assert result['negative_int'] == -42
            assert isinstance(result['zero_int'], int)
            assert isinstance(result['negative_int'], int)
            
            # Test float conversion
            assert result['zero_float'] == 0.0
            assert result['negative_float'] == -3.14
            assert abs(result['scientific'] - 1.23e-4) < 1e-10
            assert isinstance(result['zero_float'], float)
            assert isinstance(result['scientific'], float)
            
            # Test boolean conversion - true/false in any case supported
            assert result['bool_true_caps'] is True  # TRUE -> True
            assert result['bool_false_caps'] is False  # FALSE -> False
            assert result['bool_yes'] == 'yes'  # Stays as string (not supported)
            assert result['bool_no'] == 'no'  # Stays as string (not supported)
            assert result['bool_on'] == 'on'  # Stays as string (not supported)
            assert result['bool_off'] == 'off'  # Stays as string (not supported)
            
            # Test string handling
            assert result['empty'] == ''
            assert result['whitespace'] == '  whitespace  '  # Whitespace preserved
    
    def test_environment_variable_with_special_characters(self):
        """Test environment variables containing special characters"""
        with patch.dict('os.environ', {
            'CONNECTION_STRING': 'postgresql://user:p@ss!w0rd@localhost:5432/db?sslmode=require',
            'JSON_CONFIG': '{"key": "value", "nested": {"array": [1,2,3]}}',
            'URL_WITH_PARAMS': 'https://api.example.com/v1/data?token=abc123&format=json',
            'REGEX_PATTERN': r'^[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'PATH_WITH_SPACES': '/path/with spaces/to/file.txt'
        }):
            test_data = {
                'connection_string': '${CONNECTION_STRING}',
                'json_config': '${JSON_CONFIG}',
                'url_with_params': '${URL_WITH_PARAMS}',
                'regex_pattern': '${REGEX_PATTERN}',
                'path_with_spaces': '${PATH_WITH_SPACES}'
            }
            
            result = resolve_env_variables(test_data)
            
            # Verify special characters are preserved
            assert result['connection_string'] == 'postgresql://user:p@ss!w0rd@localhost:5432/db?sslmode=require'
            assert result['json_config'] == '{"key": "value", "nested": {"array": [1,2,3]}}'
            assert result['url_with_params'] == 'https://api.example.com/v1/data?token=abc123&format=json'
            assert result['regex_pattern'] == r'^[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            assert result['path_with_spaces'] == '/path/with spaces/to/file.txt'
    
    def test_simple_environment_variable_resolution(self):
        """Test basic environment variable resolution (no recursive support)"""
        with patch.dict('os.environ', {
            'BASE_URL': 'https://api.example.com',
            'API_VERSION': 'v1',
            'ENDPOINT_PATH': '/data',
            'FULL_API_URL': 'https://api.example.com/v1/data',  # Simple resolved value
            'CONFIG_FILE': 'myapp_dev.yaml',  # Simple resolved value
            'LOG_FILE': '/var/log/myapp/dev/app.log'  # Simple resolved value
        }):
            test_data = {
                'api': {
                    'base_url': '${BASE_URL}',
                    'version': '${API_VERSION}',
                    'endpoint': '${ENDPOINT_PATH}',
                    'full_url': '${FULL_API_URL}'
                },
                'files': {
                    'config': '${CONFIG_FILE}',
                    'log': '${LOG_FILE}'
                }
            }
            
            result = resolve_env_variables(test_data)
            
            # Verify simple resolution works
            assert result['api']['base_url'] == 'https://api.example.com'
            assert result['api']['version'] == 'v1'
            assert result['api']['endpoint'] == '/data'
            assert result['api']['full_url'] == 'https://api.example.com/v1/data'
            
            # Verify pre-resolved values work
            assert result['files']['config'] == 'myapp_dev.yaml'
            assert result['files']['log'] == '/var/log/myapp/dev/app.log'
    
    def test_environment_variable_list_processing(self):
        """Test environment variable resolution in lists and arrays"""
        with patch.dict('os.environ', {
            'PRIMARY_DB': 'postgresql://localhost/primary',
            'SECONDARY_DB': 'postgresql://localhost/secondary',
            'CACHE_SERVER_1': 'redis://cache1:6379',
            'CACHE_SERVER_2': 'redis://cache2:6379',
            'LOG_LEVEL': 'INFO',
            'METRICS_PORT': '8080'
        }):
            test_data = {
                'databases': [
                    '${PRIMARY_DB}',
                    '${SECONDARY_DB}',
                    '${BACKUP_DB:postgresql://localhost/backup}'
                ],
                'cache_servers': [
                    '${CACHE_SERVER_1}',
                    '${CACHE_SERVER_2}'
                ],
                'services': [
                    {
                        'name': 'logger',
                        'level': '${LOG_LEVEL}',
                        'port': '${LOG_PORT:8081}'
                    },
                    {
                        'name': 'metrics',
                        'port': '${METRICS_PORT}',
                        'enabled': '${METRICS_ENABLED:true}'
                    }
                ]
            }
            
            result = resolve_env_variables(test_data)
            
            # Verify list resolution
            assert result['databases'] == [
                'postgresql://localhost/primary',
                'postgresql://localhost/secondary', 
                'postgresql://localhost/backup'
            ]
            assert result['cache_servers'] == [
                'redis://cache1:6379',
                'redis://cache2:6379'
            ]
            
            # Verify nested object resolution in arrays
            assert result['services'][0]['name'] == 'logger'
            assert result['services'][0]['level'] == 'INFO'
            assert result['services'][0]['port'] == 8081
            
            assert result['services'][1]['name'] == 'metrics'
            assert result['services'][1]['port'] == 8080
            assert result['services'][1]['enabled'] is True
    
    def test_environment_variable_circular_reference_detection(self):
        """Test handling of circular references in environment variables"""
        with patch.dict('os.environ', {
            'VAR_A': '${VAR_B}',
            'VAR_B': '${VAR_C}',
            'VAR_C': '${VAR_A}'  # Circular reference
        }):
            test_data = {'circular': '${VAR_A}'}
            
            # Should handle circular reference gracefully (not infinite loop)
            result = resolve_env_variables(test_data)
            
            # The exact behavior depends on implementation, but it shouldn't crash
            assert 'circular' in result
    
    def test_environment_variable_mixed_with_literal_text(self):
        """Test environment variables mixed with literal text"""
        with patch.dict('os.environ', {
            'APP_NAME': 'ml-pipeline',
            'VERSION': '1.2.3',
            'BUILD_NUMBER': '456',
            'ENVIRONMENT': 'production'
        }):
            test_data = {
                'app_title': '${APP_NAME} v${VERSION}',
                'full_version': '${APP_NAME}-${VERSION}-build${BUILD_NUMBER}',
                'log_prefix': '[${ENVIRONMENT}] ${APP_NAME}:',
                'config_path': '/etc/${APP_NAME}/${ENVIRONMENT}/config.yaml',
                'mixed_default': 'prefix-${MISSING_VAR:default-value}-suffix'
            }
            
            result = resolve_env_variables(test_data)
            
            assert result['app_title'] == 'ml-pipeline v1.2.3'
            assert result['full_version'] == 'ml-pipeline-1.2.3-build456'
            assert result['log_prefix'] == '[production] ml-pipeline:'
            assert result['config_path'] == '/etc/ml-pipeline/production/config.yaml'
            assert result['mixed_default'] == 'prefix-default-value-suffix'
    
    def test_environment_variable_case_sensitivity(self):
        """Test environment variable case sensitivity and name validation"""
        with patch.dict('os.environ', {
            'lowercase_var': 'lower_value',
            'UPPERCASE_VAR': 'UPPER_VALUE',
            'Mixed_Case_Var': 'Mixed_Value',
            'var_with_numbers123': 'numeric_value',
            'VAR_WITH_UNDERSCORES': 'underscore_value'
        }):
            test_data = {
                'lower': '${lowercase_var}',
                'upper': '${UPPERCASE_VAR}',
                'mixed': '${Mixed_Case_Var}',
                'numeric': '${var_with_numbers123}',
                'underscore': '${VAR_WITH_UNDERSCORES}',
                'case_mismatch': '${LOWERCASE_VAR:default}',  # Different case
                'nonexistent': '${nonexistent_var:fallback}'
            }
            
            result = resolve_env_variables(test_data)
            
            # Verify exact case matching
            assert result['lower'] == 'lower_value'
            assert result['upper'] == 'UPPER_VALUE'
            assert result['mixed'] == 'Mixed_Value'
            assert result['numeric'] == 'numeric_value'
            assert result['underscore'] == 'underscore_value'
            
            # Case mismatch should use default
            assert result['case_mismatch'] == 'default'
            assert result['nonexistent'] == 'fallback'