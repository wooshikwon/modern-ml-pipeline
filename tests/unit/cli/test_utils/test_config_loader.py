"""
Unit tests for config_loader.
Tests environment loading and configuration utilities.
"""

import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

from src.cli.utils.config_loader import load_environment, get_config_path


class TestLoadEnvironment:
    """Test load_environment function."""
    
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_success(self, mock_path_class, mock_load_dotenv):
        """Test successful environment loading."""
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_base_path.__truediv__.return_value = mock_env_file
        
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act
        load_environment("test")
        
        # Assert
        mock_load_dotenv.assert_called_once_with(mock_env_file, override=True)
        assert os.environ.get('ENV_NAME') == "test"
    
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_file_not_found(self, mock_path_class):
        """Test FileNotFoundError when env file doesn't exist."""
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_base_path.__truediv__.return_value = mock_env_file
        
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            load_environment("missing")
        
        assert ".env.missing 파일을 찾을 수 없습니다" in str(exc_info.value)
        assert "mmp get-config --env-name missing" in str(exc_info.value)
    
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_with_custom_base_path(self, mock_path_class, mock_load_dotenv):
        """Test load_environment with custom base path."""
        # Arrange
        custom_base_path = Path("/custom/path")
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        
        with patch.object(custom_base_path, '__truediv__', return_value=mock_env_file):
            # Act
            load_environment("test", base_path=custom_base_path)
            
            # Assert
            mock_load_dotenv.assert_called_once_with(mock_env_file, override=True)
            assert os.environ.get('ENV_NAME') == "test"
    
    @patch('src.cli.utils.config_loader.logger')
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_logging(self, mock_path_class, mock_load_dotenv, mock_logger):
        """Test that environment loading is properly logged."""
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_base_path.__truediv__.return_value = mock_env_file
        
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act
        load_environment("dev")
        
        # Assert
        mock_logger.info.assert_called_once_with(f"환경변수 파일 로드됨: {mock_env_file}")


class TestGetConfigPath:
    """Test get_config_path function."""
    
    @patch('src.cli.utils.config_loader.Path')
    def test_get_config_path_success(self, mock_path_class):
        """Test successful config path retrieval."""
        # Arrange
        mock_base_path = Mock()
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        mock_base_path.__truediv__.return_value = mock_configs_dir
        mock_configs_dir.__truediv__.return_value = mock_config_file
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act
        result = get_config_path("test")
        
        # Assert
        assert result == mock_config_file
        mock_base_path.__truediv__.assert_called_with("configs")
        mock_configs_dir.__truediv__.assert_called_with("test.yaml")
    
    @patch('src.cli.utils.config_loader.Path')
    def test_get_config_path_file_not_found(self, mock_path_class):
        """Test FileNotFoundError when config file doesn't exist."""
        # Arrange
        mock_base_path = Mock()
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = False
        
        mock_base_path.__truediv__.return_value = mock_configs_dir
        mock_configs_dir.__truediv__.return_value = mock_config_file
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            get_config_path("missing")
        
        assert "configs/missing.yaml 파일을 찾을 수 없습니다" in str(exc_info.value)
        assert "mmp get-config --env-name missing" in str(exc_info.value)
    
    @patch('src.cli.utils.config_loader.Path')
    def test_get_config_path_with_custom_base_path(self, mock_path_class):
        """Test get_config_path with custom base path."""
        # Arrange
        custom_base_path = Path("/custom/project")
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        with patch.object(custom_base_path, '__truediv__', return_value=mock_configs_dir):
            with patch.object(mock_configs_dir, '__truediv__', return_value=mock_config_file):
                # Act
                result = get_config_path("prod", base_path=custom_base_path)
                
                # Assert
                assert result == mock_config_file


class TestConfigLoaderIntegration:
    """Test config loader integration scenarios."""
    
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_full_environment_setup_workflow(self, mock_path_class, mock_load_dotenv):
        """Test complete workflow of environment setup."""
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def truediv_side_effect(path):
            if path == ".env.production":
                return mock_env_file
            elif path == "configs":
                return mock_configs_dir
            return Mock()
        
        mock_base_path.__truediv__.side_effect = truediv_side_effect
        mock_configs_dir.__truediv__.return_value = mock_config_file
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act
        load_environment("production")
        config_path = get_config_path("production")
        
        # Assert
        mock_load_dotenv.assert_called_once_with(mock_env_file, override=True)
        assert os.environ.get('ENV_NAME') == "production"
        assert config_path == mock_config_file


class TestConfigLoaderEdgeCases:
    """Test config loader edge cases."""
    
    def test_load_environment_env_name_none(self):
        """Test load_environment with None env name."""
        with pytest.raises(TypeError):
            load_environment(None)
    
    def test_load_environment_empty_env_name(self):
        """Test load_environment with empty env name."""
        with patch('src.cli.utils.config_loader.Path') as mock_path_class:
            mock_base_path = Mock()
            mock_env_file = Mock()
            mock_env_file.exists.return_value = False
            mock_base_path.__truediv__.return_value = mock_env_file
            mock_path_class.cwd.return_value = mock_base_path
            
            with pytest.raises(FileNotFoundError):
                load_environment("")
    
    @patch('src.cli.utils.config_loader.Path')
    def test_get_config_path_special_characters(self, mock_path_class):
        """Test get_config_path with special characters in env name."""
        # Arrange
        mock_base_path = Mock()
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        mock_base_path.__truediv__.return_value = mock_configs_dir
        mock_configs_dir.__truediv__.return_value = mock_config_file
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act
        result = get_config_path("test-env_123")
        
        # Assert
        assert result == mock_config_file
        mock_configs_dir.__truediv__.assert_called_with("test-env_123.yaml")


class TestConfigLoaderErrorHandling:
    """Test config loader error handling."""
    
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_dotenv_exception(self, mock_path_class, mock_load_dotenv):
        """Test handling of dotenv loading exceptions."""
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_base_path.__truediv__.return_value = mock_env_file
        mock_path_class.cwd.return_value = mock_base_path
        
        mock_load_dotenv.side_effect = Exception("Failed to load .env file")
        
        # Act & Assert
        with pytest.raises(Exception, match="Failed to load .env file"):
            load_environment("test")
    
    @patch('src.cli.utils.config_loader.Path')
    def test_load_environment_path_permission_error(self, mock_path_class):
        """Test handling of path permission errors.""" 
        # Arrange
        mock_base_path = Mock()
        mock_env_file = Mock()
        mock_env_file.exists.side_effect = PermissionError("Permission denied")
        mock_base_path.__truediv__.return_value = mock_env_file
        mock_path_class.cwd.return_value = mock_base_path
        
        # Act & Assert
        with pytest.raises(PermissionError, match="Permission denied"):
            load_environment("test")


class TestConfigLoaderRealWorldScenarios:
    """Test config loader with real-world scenarios."""
    
    @patch('src.cli.utils.config_loader.load_dotenv')
    @patch('src.cli.utils.config_loader.Path')
    def test_multiple_environment_loading(self, mock_path_class, mock_load_dotenv):
        """Test loading multiple environments sequentially."""
        # Arrange
        mock_base_path = Mock()
        mock_path_class.cwd.return_value = mock_base_path
        
        def create_mock_env_file(env_name):
            mock_file = Mock()
            mock_file.exists.return_value = True
            return mock_file
        
        def truediv_side_effect(path):
            return create_mock_env_file(path)
        
        mock_base_path.__truediv__.side_effect = truediv_side_effect
        
        # Act
        environments = ["dev", "staging", "prod"]
        for env in environments:
            load_environment(env)
            
        # Assert
        assert mock_load_dotenv.call_count == 3
        assert os.environ.get('ENV_NAME') == "prod"  # Last loaded
    
    @patch('src.cli.utils.config_loader.Path')
    def test_nested_project_structure(self, mock_path_class):
        """Test config loading with nested project structure."""
        # Arrange
        nested_base_path = Path("/projects/ml/deep-learning-project")
        mock_configs_dir = Mock()
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        with patch.object(nested_base_path, '__truediv__', return_value=mock_configs_dir):
            with patch.object(mock_configs_dir, '__truediv__', return_value=mock_config_file):
                # Act
                result = get_config_path("research", base_path=nested_base_path)
                
                # Assert
                assert result == mock_config_file