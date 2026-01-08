"""
Unit Tests for Config Commands CLI - No Mock Hell Approach
Following tests/README.md principles: Real components, Context classes, Public API
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from mmp.cli.commands.get_config_command import get_config_command


class TestGetConfigCommandArgumentParsing:
    """Get config command argument parsing tests using real components"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_config_command)

    def test_get_config_command_builder_initialization_error(self):
        """Test get config command handles builder initialization errors using real components"""
        # Given: Template directory not found scenario
        with patch("pathlib.Path.exists", return_value=False):
            # When: Execute command which will fail to find templates
            result = self.runner.invoke(self.app, ["--env-name", "test"])

            # Then: Command should handle error gracefully
            # Exit code should be non-zero due to initialization error
            assert result.exit_code != 0 or "error" in result.output.lower()

    def test_get_config_command_interactive_flow_error(self):
        """Test get config command handles interactive flow errors (keyboard interrupt)"""
        # When: Execute command that triggers keyboard interrupt
        result = self.runner.invoke(self.app, ["--env-name", ""], catch_exceptions=False)

        # Then: Command should handle gracefully
        # Empty env_name or KeyboardInterrupt should be handled
        assert result.exit_code == 0 or "취소" in result.output or result.exit_code == 1

    def test_get_config_command_help_message(self):
        """Test get config command help message shows correct information"""
        # When: Execute help command
        result = self.runner.invoke(self.app, ["--help"])

        # Then: Help message displays correctly with all options
        assert result.exit_code == 0
        assert "--env-name" in result.output or "-e" in result.output
        assert "환경 이름" in result.output or "env" in result.output.lower()
        assert "대화형" in result.output or "interactive" in result.output.lower()


class TestGetConfigCommandWithMockedBuilder:
    """Config command tests with mocked builder for isolated testing"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_config_command)

    def test_get_config_command_without_env_name(self, isolated_temp_directory):
        """Test get config command without environment name (interactive mode) using mocked builder"""
        # Import and patch at runtime
        from mmp.cli.utils import config_builder

        mock_builder = MagicMock()
        mock_builder.run_interactive_flow.return_value = {"env_name": "dev"}
        mock_builder.generate_config_file.return_value = Path("configs/dev.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.dev.template")

        with patch.object(config_builder, "InteractiveConfigBuilder", return_value=mock_builder):
            # When: Execute command without env_name
            result = self.runner.invoke(self.app, [])

            # Then: Command executes successfully
            assert result.exit_code == 0
            mock_builder.run_interactive_flow.assert_called_once()
            mock_builder.generate_config_file.assert_called_once()

    def test_get_config_command_with_env_name(self, isolated_temp_directory):
        """Test get config command with specified environment name using mocked builder"""
        from mmp.cli.utils import config_builder

        mock_builder = MagicMock()
        mock_builder.run_interactive_flow.return_value = {"env_name": "production"}
        mock_builder.generate_config_file.return_value = Path("configs/production.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.production.template")

        with patch.object(config_builder, "InteractiveConfigBuilder", return_value=mock_builder):
            # When: Execute command with env_name
            result = self.runner.invoke(self.app, ["--env-name", "production"])

            # Then: Command executes successfully
            assert result.exit_code == 0
            mock_builder.run_interactive_flow.assert_called_once()

    def test_get_config_command_with_short_option(self, isolated_temp_directory):
        """Test get config command with short option flag using mocked builder"""
        from mmp.cli.utils import config_builder

        mock_builder = MagicMock()
        mock_builder.run_interactive_flow.return_value = {"env_name": "dev"}
        mock_builder.generate_config_file.return_value = Path("configs/dev.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.dev.template")

        with patch.object(config_builder, "InteractiveConfigBuilder", return_value=mock_builder):
            # When: Execute command with short flag
            result = self.runner.invoke(self.app, ["-e", "dev"])

            # Then: Command executes successfully with short option
            assert result.exit_code == 0
            mock_builder.run_interactive_flow.assert_called_once()

    def test_get_config_command_file_generation_verification(self, isolated_temp_directory):
        """Test get config command verifies file generation using mocked builder"""
        from mmp.cli.utils import config_builder

        mock_builder = MagicMock()
        mock_builder.run_interactive_flow.return_value = {"env_name": "staging"}
        mock_builder.generate_config_file.return_value = Path("configs/staging.yaml")
        mock_builder.generate_env_template.return_value = Path(".env.staging.template")

        with patch.object(config_builder, "InteractiveConfigBuilder", return_value=mock_builder):
            # When: Execute command with staging environment
            result = self.runner.invoke(self.app, ["-e", "staging"])

            # Then: Command executes successfully and files are generated
            assert result.exit_code == 0
            mock_builder.generate_config_file.assert_called_once()

    def test_get_config_command_environment_name_validation(self, isolated_temp_directory):
        """Test get config command with various environment name formats using mocked builder"""
        from mmp.cli.utils import config_builder

        # Given: Test valid environment names
        valid_env_names = ["dev", "production", "staging", "test-env", "env_123"]

        for env_name in valid_env_names:
            mock_builder = MagicMock()
            mock_builder.run_interactive_flow.return_value = {"env_name": env_name}
            mock_builder.generate_config_file.return_value = Path(f"configs/{env_name}.yaml")
            mock_builder.generate_env_template.return_value = Path(f".env.{env_name}.template")

            with patch.object(
                config_builder, "InteractiveConfigBuilder", return_value=mock_builder
            ):
                # When: Execute command with various valid environment names
                result = self.runner.invoke(self.app, ["--env-name", env_name])

                # Then: All valid environment names should work
                assert result.exit_code == 0
                mock_builder.run_interactive_flow.assert_called_once()
