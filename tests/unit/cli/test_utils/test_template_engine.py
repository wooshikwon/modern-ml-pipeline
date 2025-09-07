"""
Unit tests for template_engine.
Tests Jinja2-based template rendering functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from jinja2 import TemplateNotFound

from src.cli.utils.template_engine import TemplateEngine


class TestTemplateEngineInitialization:
    """Test TemplateEngine initialization."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_template_engine_initialization_success(self, mock_file_loader, mock_environment):
        """Test successful template engine initialization."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/path/to/templates")
        
        mock_loader_instance = Mock()
        mock_env_instance = Mock()
        mock_file_loader.return_value = mock_loader_instance
        mock_environment.return_value = mock_env_instance
        
        # Act
        engine = TemplateEngine(mock_template_dir)
        
        # Assert
        assert engine.template_dir == mock_template_dir
        assert engine.env == mock_env_instance
        mock_file_loader.assert_called_once_with("/path/to/templates")
        mock_environment.assert_called_once_with(
            loader=mock_loader_instance,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
    
    def test_template_engine_initialization_directory_not_found(self):
        """Test FileNotFoundError when template directory doesn't exist."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = False
        mock_template_dir.__str__ = Mock(return_value="/nonexistent/path")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Template directory not found: /nonexistent/path"):
            TemplateEngine(mock_template_dir)


class TestTemplateEngineRenderTemplate:
    """Test TemplateEngine render_template method."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_template_success(self, mock_file_loader, mock_environment):
        """Test successful template rendering."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "rendered content"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act
        result = engine.render_template("test_template.j2", {"key": "value"})
        
        # Assert
        assert result == "rendered content"
        mock_env_instance.get_template.assert_called_once_with("test_template.j2")
        mock_template.render.assert_called_once_with({"key": "value"})
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_template_not_found(self, mock_file_loader, mock_environment):
        """Test TemplateNotFound exception handling."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_env_instance = Mock()
        mock_env_instance.get_template.side_effect = TemplateNotFound("nonexistent.j2")
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert
        with pytest.raises(TemplateNotFound, match="Template 'nonexistent.j2' not found"):
            engine.render_template("nonexistent.j2", {"key": "value"})
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_template_empty_context(self, mock_file_loader, mock_environment):
        """Test template rendering with empty context."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "static content"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act
        result = engine.render_template("static_template.j2", {})
        
        # Assert
        assert result == "static content"
        mock_template.render.assert_called_once_with({})


class TestTemplateEngineRenderString:
    """Test TemplateEngine render_string method."""
    
    @patch('src.cli.utils.template_engine.Template')
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_string_success(self, mock_file_loader, mock_environment, mock_template_class):
        """Test successful string template rendering."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_env_instance = Mock()
        mock_environment.return_value = mock_env_instance
        
        mock_template = Mock()
        mock_template.render.return_value = "Hello, World!"
        mock_template_class.return_value = mock_template
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act
        result = engine.render_string("Hello, {{ name }}!", {"name": "World"})
        
        # Assert
        assert result == "Hello, World!"
        mock_template_class.assert_called_once_with("Hello, {{ name }}!", environment=mock_env_instance)
        mock_template.render.assert_called_once_with({"name": "World"})
    
    @patch('src.cli.utils.template_engine.Template')
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_string_template_syntax_error(self, mock_file_loader, mock_environment, mock_template_class):
        """Test handling of template syntax errors in string rendering."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_env_instance = Mock()
        mock_environment.return_value = mock_env_instance
        
        from jinja2 import TemplateSyntaxError
        mock_template_class.side_effect = TemplateSyntaxError("Invalid syntax", lineno=1)
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert
        with pytest.raises(TemplateSyntaxError):
            engine.render_string("{{ invalid syntax }}", {})


class TestTemplateEngineWriteToFile:
    """Test TemplateEngine write_to_file method."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_write_to_file_success(self, mock_file_loader, mock_environment):
        """Test successful template rendering to file."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "file content"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        mock_output_path = Mock(spec=Path)
        mock_output_path.parent.mkdir = Mock()
        mock_output_path.write_text = Mock()
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act
        engine.write_to_file("template.j2", {"key": "value"}, mock_output_path)
        
        # Assert
        mock_output_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_output_path.write_text.assert_called_once_with("file content", encoding='utf-8')
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_write_to_file_directory_creation_failure(self, mock_file_loader, mock_environment):
        """Test handling of directory creation failures."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "content"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        mock_output_path = Mock(spec=Path)
        mock_output_path.parent.mkdir.side_effect = PermissionError("Permission denied")
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert
        with pytest.raises(PermissionError, match="Permission denied"):
            engine.write_to_file("template.j2", {"key": "value"}, mock_output_path)


class TestTemplateEngineIntegration:
    """Test TemplateEngine integration scenarios."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_multiple_template_rendering_workflow(self, mock_file_loader, mock_environment):
        """Test rendering multiple templates in sequence."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        # Create different mock templates
        mock_template1 = Mock()
        mock_template1.render.return_value = "config content"
        mock_template2 = Mock()
        mock_template2.render.return_value = "recipe content"
        
        mock_env_instance = Mock()
        mock_env_instance.get_template.side_effect = [mock_template1, mock_template2]
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act
        config_result = engine.render_template("config.yaml.j2", {"env": "dev"})
        recipe_result = engine.render_template("recipe.yaml.j2", {"model": "lstm"})
        
        # Assert
        assert config_result == "config content"
        assert recipe_result == "recipe content"
        assert mock_env_instance.get_template.call_count == 2
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_complex_context_rendering(self, mock_file_loader, mock_environment):
        """Test template rendering with complex context data."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "complex rendered content"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        complex_context = {
            "project": {
                "name": "ml_project",
                "version": "1.0.0",
                "dependencies": ["numpy", "pandas", "scikit-learn"]
            },
            "environment": {
                "name": "production",
                "database": {
                    "host": "prod-db.example.com",
                    "port": 5432
                }
            },
            "features": {
                "logging": True,
                "monitoring": True,
                "auto_scaling": False
            }
        }
        
        # Act
        result = engine.render_template("complex_template.j2", complex_context)
        
        # Assert
        assert result == "complex rendered content"
        mock_template.render.assert_called_once_with(complex_context)


class TestTemplateEngineErrorHandling:
    """Test TemplateEngine error handling scenarios."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_template_rendering_exception(self, mock_file_loader, mock_environment):
        """Test handling of template rendering exceptions."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.side_effect = Exception("Rendering failed")
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert
        with pytest.raises(Exception, match="Rendering failed"):
            engine.render_template("failing_template.j2", {"key": "value"})
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_file_system_loader_exception(self, mock_file_loader, mock_environment):
        """Test handling of FileSystemLoader exceptions."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_file_loader.side_effect = Exception("Failed to initialize loader")
        
        # Act & Assert
        with pytest.raises(Exception, match="Failed to initialize loader"):
            TemplateEngine(mock_template_dir)


class TestTemplateEngineEdgeCases:
    """Test TemplateEngine edge cases."""
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_template_none_context(self, mock_file_loader, mock_environment):
        """Test template rendering with None context."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_template = Mock()
        mock_template.render.return_value = "content without context"
        mock_env_instance = Mock()
        mock_env_instance.get_template.return_value = mock_template
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert - Should handle None gracefully or raise appropriate error
        with pytest.raises(TypeError):
            engine.render_template("template.j2", None)
    
    @patch('src.cli.utils.template_engine.Environment')
    @patch('src.cli.utils.template_engine.FileSystemLoader')
    def test_render_template_empty_template_name(self, mock_file_loader, mock_environment):
        """Test template rendering with empty template name."""
        # Arrange
        mock_template_dir = Mock(spec=Path)
        mock_template_dir.exists.return_value = True
        mock_template_dir.__str__ = Mock(return_value="/templates")
        
        mock_env_instance = Mock()
        mock_env_instance.get_template.side_effect = TemplateNotFound("")
        mock_environment.return_value = mock_env_instance
        
        engine = TemplateEngine(mock_template_dir)
        
        # Act & Assert
        with pytest.raises(TemplateNotFound):
            engine.render_template("", {"key": "value"})