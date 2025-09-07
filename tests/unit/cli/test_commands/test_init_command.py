"""
Unit tests for init_command.
Tests project initialization command functionality with typer and CLI integration.
"""

import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.cli.commands.init_command import init_command, create_project_structure


class TestInitCommandInitialization:
    """Test init command initialization and basic functionality."""
    
    def test_init_command_exists_and_callable(self):
        """Test that init_command is a callable function."""
        assert callable(init_command)
        assert hasattr(init_command, '__call__')
    
    def test_create_project_structure_exists_and_callable(self):
        """Test that create_project_structure helper function exists."""
        assert callable(create_project_structure)
        assert hasattr(create_project_structure, '__call__')


class TestInitCommandParameterHandling:
    """Test init command parameter processing and validation."""
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.create_project_structure')
    def test_init_command_with_project_name(self, mock_create_structure, mock_console, mock_ui_class):
        """Test init command with project name provided as argument."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance
        mock_ui = Mock()
        mock_ui.show_info = Mock()
        mock_ui.show_success = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        with patch('src.cli.commands.init_command.Path') as mock_path:
            # Mock Path behavior
            mock_project_path = Mock()
            mock_project_path.exists.return_value = False
            mock_project_path.absolute.return_value = "/test/my-project"
            
            mock_path.cwd.return_value = Mock()
            mock_path.cwd.return_value.__truediv__ = Mock(return_value=mock_project_path)
            
            # Act
            result = runner.invoke(app, ["my-project"])
            
            # Assert
            assert result.exit_code == 0
            mock_create_structure.assert_called_once()
            mock_ui.show_success.assert_called_once()
            mock_ui.show_panel.assert_called_once()
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.create_project_structure')
    def test_init_command_without_project_name_interactive(self, mock_create_structure, mock_console, mock_ui_class):
        """Test init command without project name (interactive mode)."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance with interactive input
        mock_ui = Mock()
        mock_ui.text_input.return_value = "interactive-project"
        mock_ui.show_info = Mock()
        mock_ui.show_success = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        with patch('src.cli.commands.init_command.Path') as mock_path:
            # Mock Path behavior
            mock_project_path = Mock()
            mock_project_path.exists.return_value = False
            mock_project_path.absolute.return_value = "/test/interactive-project"
            
            mock_path.cwd.return_value = Mock()
            mock_path.cwd.return_value.__truediv__ = Mock(return_value=mock_project_path)
            
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            mock_ui.text_input.assert_called_once()
            mock_create_structure.assert_called_once()
            mock_ui.show_success.assert_called_once()


class TestInitCommandDirectoryHandling:
    """Test init command directory handling scenarios."""
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.create_project_structure')
    def test_init_command_existing_directory_confirm(self, mock_create_structure, mock_console, mock_ui_class):
        """Test init command with existing directory - user confirms."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance - user confirms continuation
        mock_ui = Mock()
        mock_ui.confirm.return_value = True
        mock_ui.show_info = Mock()
        mock_ui.show_success = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        with patch('src.cli.commands.init_command.Path') as mock_path:
            # Mock Path behavior - directory exists
            mock_project_path = Mock()
            mock_project_path.exists.return_value = True
            mock_project_path.absolute.return_value = "/test/existing-project"
            
            mock_path.cwd.return_value = Mock()
            mock_path.cwd.return_value.__truediv__ = Mock(return_value=mock_project_path)
            
            # Act
            result = runner.invoke(app, ["existing-project"])
            
            # Assert
            assert result.exit_code == 0
            mock_ui.confirm.assert_called_once()
            mock_create_structure.assert_called_once()
            mock_ui.show_success.assert_called_once()
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    def test_init_command_existing_directory_cancel(self, mock_console, mock_ui_class):
        """Test init command with existing directory - user cancels."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance - user cancels
        mock_ui = Mock()
        mock_ui.confirm.return_value = False
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        with patch('src.cli.commands.init_command.Path') as mock_path:
            # Mock Path behavior - directory exists
            mock_project_path = Mock()
            mock_project_path.exists.return_value = True
            
            mock_path.cwd.return_value = Mock()
            mock_path.cwd.return_value.__truediv__ = Mock(return_value=mock_project_path)
            
            # Act
            result = runner.invoke(app, ["existing-project"])
            
            # Assert
            assert result.exit_code == 0  # Clean exit when user cancels
            mock_ui.confirm.assert_called_once()
            mock_console_instance.print.assert_called()


class TestInitCommandFileGeneration:
    """Test init command file and directory generation."""
    
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_create_project_structure_creates_directories(self, mock_path_class, mock_template_engine_class):
        """Test that create_project_structure creates expected directories."""
        # Mock project path
        mock_project_path = Mock()
        mock_dir_paths = {}
        
        def create_dir_mock(dir_name):
            mock_dir = Mock()
            mock_dir.mkdir = Mock()
            mock_dir_paths[dir_name] = mock_dir
            return mock_dir
        
        mock_project_path.__truediv__ = Mock(side_effect=create_dir_mock)
        mock_project_path.name = "test-project"
        
        # Mock TemplateEngine
        mock_template_engine = Mock()
        mock_template_engine_class.return_value = mock_template_engine
        
        # Mock Path for templates directory
        mock_path_class.return_value = Mock()
        
        with patch('src.cli.commands.init_command.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01 12:00:00"
            
            # Act
            create_project_structure(mock_project_path)
            
            # Assert - check that all expected directories are created
            expected_directories = ["data", "configs", "recipes", "sql"]
            for dir_name in expected_directories:
                assert dir_name in mock_dir_paths
                mock_dir_paths[dir_name].mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_create_project_structure_generates_template_files(self, mock_path_class, mock_template_engine_class):
        """Test that create_project_structure generates template files."""
        # Mock project path
        mock_project_path = Mock()
        mock_project_path.name = "test-project"
        mock_project_path.__truediv__ = Mock(return_value=Mock())
        
        # Mock TemplateEngine
        mock_template_engine = Mock()
        mock_template_engine_class.return_value = mock_template_engine
        
        # Mock Path for templates directory
        mock_templates_dir = Mock()
        mock_path_class.return_value = mock_templates_dir
        
        with patch('src.cli.commands.init_command.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01 12:00:00"
            
            # Act
            create_project_structure(mock_project_path)
            
            # Assert - check template file generation
            expected_template_calls = [
                ("docker/docker-compose.yml.j2", "docker-compose.yml"),
                ("docker/Dockerfile.j2", "Dockerfile"),
                ("project/pyproject.toml.j2", "pyproject.toml"),
                ("project/README.md.j2", "README.md"),
            ]
            
            assert mock_template_engine.write_rendered_file.call_count == len(expected_template_calls)
            
            # Verify context passed to template engine
            expected_context = {
                "project_name": "test-project",
                "timestamp": "2024-01-01 12:00:00"
            }
            
            # Check that context was passed correctly (check any call for context)
            call_args = mock_template_engine.write_rendered_file.call_args_list
            for call in call_args:
                args, kwargs = call
                assert args[2] == expected_context  # Context is the 3rd argument
    
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_create_project_structure_copies_static_files(self, mock_path_class, mock_template_engine_class):
        """Test that create_project_structure copies static files."""
        # Mock project path
        mock_project_path = Mock()
        mock_project_path.name = "test-project"
        mock_project_path.__truediv__ = Mock(return_value=Mock())
        
        # Mock TemplateEngine
        mock_template_engine = Mock()
        mock_template_engine_class.return_value = mock_template_engine
        
        with patch('src.cli.commands.init_command.datetime'):
            # Act
            create_project_structure(mock_project_path)
            
            # Assert - check static file copying
            mock_template_engine.copy_static_file.assert_called_once_with(
                "project/.gitignore",
                mock_project_path / ".gitignore"
            )


class TestInitCommandErrorHandling:
    """Test init command error scenarios."""
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    def test_init_command_keyboard_interrupt(self, mock_ui_class):
        """Test handling of KeyboardInterrupt during interactive input."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI to raise KeyboardInterrupt
        mock_ui = Mock()
        mock_ui.text_input.side_effect = KeyboardInterrupt("User interrupted")
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_ui.show_error.assert_called_once()
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.create_project_structure')
    def test_init_command_general_exception(self, mock_create_structure, mock_console, mock_ui_class):
        """Test handling of general exceptions during project creation."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance
        mock_ui = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock create_project_structure to raise exception
        mock_create_structure.side_effect = RuntimeError("Project creation failed")
        
        with patch('src.cli.commands.init_command.Path') as mock_path:
            mock_project_path = Mock()
            mock_project_path.exists.return_value = False
            mock_path.cwd.return_value = Mock()
            mock_path.cwd.return_value.__truediv__ = Mock(return_value=mock_project_path)
            
            # Act
            result = runner.invoke(app, ["test-project"])
            
            # Assert
            assert result.exit_code == 1
            mock_ui.show_error.assert_called_once()
    
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_create_project_structure_template_engine_failure(self, mock_path_class, mock_template_engine_class):
        """Test create_project_structure when TemplateEngine fails."""
        # Mock project path
        mock_project_path = Mock()
        mock_project_path.name = "test-project"
        mock_project_path.__truediv__ = Mock(return_value=Mock())
        
        # Mock TemplateEngine to raise exception
        mock_template_engine_class.side_effect = Exception("Template engine initialization failed")
        
        # Act & Assert
        with pytest.raises(Exception, match="Template engine initialization failed"):
            create_project_structure(mock_project_path)


class TestInitCommandIntegration:
    """Test init command integration scenarios."""
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_init_command_complete_workflow(self, mock_path_class, mock_template_engine_class, mock_console, mock_ui_class):
        """Test complete init workflow from CLI invocation to project creation."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance
        mock_ui = Mock()
        mock_ui.show_info = Mock()
        mock_ui.show_success = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        # Mock Path behavior
        mock_project_path = Mock()
        mock_project_path.exists.return_value = False
        mock_project_path.absolute.return_value = "/test/complete-project"
        mock_project_path.name = "complete-project"
        
        mock_cwd = Mock()
        mock_cwd.__truediv__ = Mock(return_value=mock_project_path)
        mock_path_class.cwd.return_value = mock_cwd
        
        # Mock directory creation
        def create_mock_dir(dir_name):
            mock_dir = Mock()
            mock_dir.mkdir = Mock()
            return mock_dir
        
        mock_project_path.__truediv__ = Mock(side_effect=create_mock_dir)
        
        # Mock TemplateEngine
        mock_template_engine = Mock()
        mock_template_engine.write_rendered_file = Mock()
        mock_template_engine.copy_static_file = Mock()
        mock_template_engine_class.return_value = mock_template_engine
        
        with patch('src.cli.commands.init_command.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01 12:00:00"
            
            # Act
            result = runner.invoke(app, ["complete-project"])
            
            # Assert - verify entire workflow
            assert result.exit_code == 0
            
            # Verify UI interactions
            mock_ui.show_info.assert_called_once()
            mock_ui.show_success.assert_called_once()
            mock_ui.show_panel.assert_called_once()
            
            # Verify template engine usage
            assert mock_template_engine.write_rendered_file.call_count == 4
            mock_template_engine.copy_static_file.assert_called_once()
            
            # Verify console output
            mock_console_instance.print.assert_called()
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    @patch('src.cli.commands.init_command.Console')
    @patch('src.cli.commands.init_command.TemplateEngine')
    @patch('src.cli.commands.init_command.Path')
    def test_init_command_interactive_workflow(self, mock_path_class, mock_template_engine_class, mock_console, mock_ui_class):
        """Test interactive workflow when no project name is provided."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock UI instance with interactive input
        mock_ui = Mock()
        mock_ui.text_input.return_value = "interactive-test-project"
        mock_ui.show_info = Mock()
        mock_ui.show_success = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock Console
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        
        # Mock Path behavior
        mock_project_path = Mock()
        mock_project_path.exists.return_value = False
        mock_project_path.absolute.return_value = "/test/interactive-test-project"
        mock_project_path.name = "interactive-test-project"
        mock_project_path.__truediv__ = Mock(return_value=Mock())
        
        mock_cwd = Mock()
        mock_cwd.__truediv__ = Mock(return_value=mock_project_path)
        mock_path_class.cwd.return_value = mock_cwd
        
        # Mock TemplateEngine
        mock_template_engine = Mock()
        mock_template_engine_class.return_value = mock_template_engine
        
        with patch('src.cli.commands.init_command.datetime'):
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            mock_ui.text_input.assert_called_once()
            mock_ui.show_success.assert_called_once()


class TestInitCommandValidation:
    """Test init command input validation scenarios."""
    
    @patch('src.cli.commands.init_command.InteractiveUI')
    def test_init_command_project_name_validation(self, mock_ui_class):
        """Test project name validation logic."""
        # Mock UI instance
        mock_ui = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Test the validator function indirectly by checking UI setup
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)
        
        # Mock to trigger interactive input but then raise exception to avoid full execution
        mock_ui.text_input.side_effect = KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            runner.invoke(app, [])
        
        # Verify text_input was called with validator
        mock_ui.text_input.assert_called_once()
        call_args = mock_ui.text_input.call_args
        
        # Extract the validator function and test it
        validator = call_args[1]['validator'] if len(call_args) > 1 and 'validator' in call_args[1] else None
        
        if validator:
            # Test valid project names
            assert validator("valid-project") == True
            assert validator("valid_project") == True
            assert validator("validproject123") == True
            
            # Test invalid project names
            assert validator("") == False
            assert validator("invalid project") == False  # spaces
            assert validator("invalid@project") == False  # special chars