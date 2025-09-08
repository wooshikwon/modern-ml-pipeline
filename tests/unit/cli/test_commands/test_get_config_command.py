"""
Unit tests for get_config_command.
Tests configuration generation command functionality with typer and CLI integration.
"""

import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.cli.commands.get_config_command import get_config_command, _show_completion_message


class TestGetConfigCommandInitialization:
    """Test get_config command initialization and basic functionality."""
    
    def test_get_config_command_exists_and_callable(self):
        """Test that get_config_command is a callable function."""
        assert callable(get_config_command)
        assert hasattr(get_config_command, '__call__')
    
    def test_show_completion_message_exists_and_callable(self):
        """Test that _show_completion_message helper function exists."""
        assert callable(_show_completion_message)
        assert hasattr(_show_completion_message, '__call__')


class TestGetConfigCommandParameterHandling:
    """Test get_config command parameter processing and validation."""
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_with_env_name(self, mock_console, mock_builder_class):
        """Test get_config command with env_name provided."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'development',
            'mlflow': {'enabled': True},
            'data_source': 'PostgreSQL'
        }
        mock_builder.generate_config_file.return_value = Path("configs/development.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.development.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "development"])
        
        # Assert
        assert result.exit_code == 0
        mock_builder.run_interactive_flow.assert_called_once_with("development")
        mock_builder.generate_config_file.assert_called_once()
        mock_builder.generate_env_template.assert_called_once()
        mock_console.print.assert_called()  # Completion message displayed
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_without_env_name_interactive(self, mock_console, mock_builder_class):
        """Test get_config command without env_name (interactive mode)."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'interactive-env',
            'mlflow': {'enabled': True},
            'data_source': 'Local Files'
        }
        mock_builder.generate_config_file.return_value = Path("configs/interactive-env.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.interactive-env.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_builder.run_interactive_flow.assert_called_once_with(None)
        mock_builder.generate_config_file.assert_called_once()
        mock_builder.generate_env_template.assert_called_once()
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_short_option(self, mock_console, mock_builder_class):
        """Test get_config command with short option -e."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'prod',
            'mlflow': {'enabled': True},
            'data_source': 'BigQuery'
        }
        mock_builder.generate_config_file.return_value = Path("configs/prod.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.prod.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["-e", "prod"])
        
        # Assert
        assert result.exit_code == 0
        mock_builder.run_interactive_flow.assert_called_once_with("prod")


class TestGetConfigCommandConfigBuilderIntegration:
    """Test get_config command integration with InteractiveConfigBuilder."""
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_config_builder_workflow(self, mock_console, mock_builder_class):
        """Test complete workflow with InteractiveConfigBuilder."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock complex selections from interactive flow
        mock_selections = {
            'env_name': 'staging',
            'mlflow': {'enabled': True, 'tracking_uri': 'http://mlflow:5000'},
            'data_source': 'PostgreSQL',
            'feature_store': {'provider': 'feast'},
            'serving': {'enabled': True},
            'artifact_storage': 'S3'
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_config_file.return_value = Path("configs/staging.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.staging.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "staging"])
        
        # Assert
        assert result.exit_code == 0
        
        # Verify builder method calls with correct parameters
        mock_builder.run_interactive_flow.assert_called_once_with("staging")
        mock_builder.generate_config_file.assert_called_once_with("staging", mock_selections)
        mock_builder.generate_env_template.assert_called_once_with("staging", mock_selections)
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_verifies_file_paths(self, mock_console, mock_builder_class):
        """Test that generated file paths are correctly passed to completion message."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock with specific file paths
        config_path = Path("configs/test-env.yaml")
        env_template_path = Path(".env.test-env.template")
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'env_name': 'test-env'}
        mock_builder.generate_config_file.return_value = config_path
        mock_builder.generate_env_template.return_value = env_template_path
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_config_command._show_completion_message') as mock_show_completion:
            # Act
            result = runner.invoke(app, ["--env-name", "test-env"])
            
            # Assert
            assert result.exit_code == 0
            mock_show_completion.assert_called_once_with("test-env", config_path, env_template_path)


class TestGetConfigCommandErrorHandling:
    """Test get_config command error scenarios."""
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_keyboard_interrupt(self, mock_console, mock_builder_class):
        """Test handling of KeyboardInterrupt during interactive flow."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder to raise KeyboardInterrupt
        mock_builder = Mock()
        mock_builder.run_interactive_flow.side_effect = KeyboardInterrupt("User interrupted")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_console.print.assert_called_with("\n❌ 설정 생성이 취소되었습니다.", style="red")
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_config_builder_initialization_error(self, mock_console, mock_builder_class):
        """Test handling of InteractiveConfigBuilder initialization error."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder to raise exception during initialization
        mock_builder_class.side_effect = RuntimeError("Config builder initialization failed")
        
        # Act
        result = runner.invoke(app, ["--env-name", "test"])
        
        # Assert
        assert result.exit_code == 1
        mock_console.print.assert_called_with("❌ 오류 발생: Config builder initialization failed", style="red")
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_file_generation_error(self, mock_console, mock_builder_class):
        """Test handling of file generation errors."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder with file generation failure
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'env_name': 'test'}
        mock_builder.generate_config_file.side_effect = IOError("Cannot write config file")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "test"])
        
        # Assert
        assert result.exit_code == 1
        mock_console.print.assert_called_with("❌ 오류 발생: Cannot write config file", style="red")
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_env_template_generation_error(self, mock_console, mock_builder_class):
        """Test handling of env template generation errors."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock InteractiveConfigBuilder with env template generation failure
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'env_name': 'test'}
        mock_builder.generate_config_file.return_value = Path("configs/test.yaml")
        mock_builder.generate_env_template.side_effect = IOError("Cannot write env template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "test"])
        
        # Assert
        assert result.exit_code == 1
        mock_console.print.assert_called_with("❌ 오류 발생: Cannot write env template", style="red")


class TestGetConfigCommandCompletionMessage:
    """Test get_config command completion message functionality."""
    
    @patch('src.cli.commands.get_config_command.console')
    @patch('src.cli.commands.get_config_command.Panel')
    def test_show_completion_message_displays_correct_info(self, mock_panel, mock_console):
        """Test _show_completion_message displays correct information."""
        config_path = Path("configs/dev.yaml")
        env_template_path = Path(".env.dev.template")
        
        # Mock Panel.fit to return a mock panel
        mock_panel_instance = Mock()
        mock_panel.fit.return_value = mock_panel_instance
        
        # Act
        _show_completion_message("dev", config_path, env_template_path)
        
        # Assert
        # Check success message
        mock_console.print.assert_any_call("\n✅ [bold green]설정 파일 생성 완료![/bold green]")
        mock_console.print.assert_any_call(f"  📄 Config: {config_path}")
        mock_console.print.assert_any_call(f"  📄 Env Template: {env_template_path}")
        mock_console.print.assert_any_call(mock_panel_instance)
        
        # Check that panel was created with correct content
        mock_panel.fit.assert_called_once()
        panel_args = mock_panel.fit.call_args[0]
        panel_content = panel_args[0]
        
        # Verify next steps content includes environment name
        assert "dev" in panel_content
        assert ".env.dev" in panel_content
        assert "mmp system-check --env-name dev" in panel_content
    
    @patch('src.cli.commands.get_config_command.console')
    def test_show_completion_message_handles_different_env_names(self, mock_console):
        """Test _show_completion_message works with different environment names."""
        test_cases = [
            ("production", "configs/production.yaml", ".env.production.template"),
            ("staging", "configs/staging.yaml", ".env.staging.template"),
            ("local-dev", "configs/local-dev.yaml", ".env.local-dev.template")
        ]
        
        for env_name, config_file, env_file in test_cases:
            mock_console.reset_mock()
            
            # Act
            _show_completion_message(env_name, Path(config_file), Path(env_file))
            
            # Assert
            mock_console.print.assert_any_call(f"  📄 Config: {config_file}")
            mock_console.print.assert_any_call(f"  📄 Env Template: {env_file}")


class TestGetConfigCommandIntegration:
    """Test get_config command integration scenarios."""
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    @patch('src.cli.commands.get_config_command._show_completion_message')
    def test_get_config_command_complete_workflow(self, mock_show_completion, mock_console, mock_builder_class):
        """Test complete workflow from CLI invocation to completion message."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock comprehensive workflow
        mock_selections = {
            'env_name': 'integration-test',
            'mlflow': {'enabled': True, 'tracking_uri': 'http://mlflow:5000'},
            'data_source': 'PostgreSQL',
            'feature_store': {'provider': 'feast', 'feast_config': {'registry': 'file'}},
            'serving': {'enabled': True, 'host': '0.0.0.0', 'port': 8000},
            'artifact_storage': 'Local',
            'monitoring': {'enabled': False}
        }
        
        config_path = Path("configs/integration-test.yaml")
        env_template_path = Path(".env.integration-test.template")
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_config_file.return_value = config_path
        mock_builder.generate_env_template.return_value = env_template_path
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "integration-test"])
        
        # Assert - verify entire workflow
        assert result.exit_code == 0
        
        # Verify builder workflow
        mock_builder.run_interactive_flow.assert_called_once_with("integration-test")
        mock_builder.generate_config_file.assert_called_once_with("integration-test", mock_selections)
        mock_builder.generate_env_template.assert_called_once_with("integration-test", mock_selections)
        
        # Verify completion message
        mock_show_completion.assert_called_once_with("integration-test", config_path, env_template_path)
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_interactive_workflow(self, mock_console, mock_builder_class):
        """Test interactive workflow without env_name parameter."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock interactive workflow where user selects environment name
        mock_selections = {
            'env_name': 'user-selected-env',
            'mlflow': {'enabled': False},
            'data_source': 'Local Files',
            'feature_store': {'provider': 'none'},
            'serving': {'enabled': False}
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_config_file.return_value = Path("configs/user-selected-env.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.user-selected-env.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_builder.run_interactive_flow.assert_called_once_with(None)
        
        # Verify files generated with user-selected environment name
        mock_builder.generate_config_file.assert_called_once_with("user-selected-env", mock_selections)
        mock_builder.generate_env_template.assert_called_once_with("user-selected-env", mock_selections)


class TestGetConfigCommandEdgeCases:
    """Test get_config command edge cases and boundary conditions."""
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_empty_selections(self, mock_console, mock_builder_class):
        """Test get_config command with minimal/empty selections."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        # Mock minimal selections
        minimal_selections = {
            'env_name': 'minimal'
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = minimal_selections
        mock_builder.generate_config_file.return_value = Path("configs/minimal.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.minimal.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", "minimal"])
        
        # Assert
        assert result.exit_code == 0
        mock_builder.generate_config_file.assert_called_once_with("minimal", minimal_selections)
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    @patch('src.cli.commands.get_config_command.console')
    def test_get_config_command_long_env_name(self, mock_console, mock_builder_class):
        """Test get_config command with very long environment name."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_config_command)
        
        long_env_name = "very_long_environment_name_for_testing_edge_cases_and_boundary_conditions"
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'env_name': long_env_name}
        mock_builder.generate_config_file.return_value = Path(f"configs/{long_env_name}.yaml")
        mock_builder.generate_env_template.return_value = Path(f".env.{long_env_name}.template")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, ["--env-name", long_env_name])
        
        # Assert - should handle long environment names without issues
        assert result.exit_code == 0
        mock_builder.run_interactive_flow.assert_called_once_with(long_env_name)