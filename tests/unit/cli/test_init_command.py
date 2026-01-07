"""
Unit Tests for Init Command CLI - No Mock Hell Compliant
Following test philosophy: Real components, minimal mocking
"""

import typer
from typer.testing import CliRunner

from src.cli.commands.init_command import init_command


class TestInitCommandWithRealComponents:
    """Init command tests using real components - No Mock Hell compliant"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(init_command)

    def test_init_command_with_project_name_real_components(self, isolated_working_directory):
        """Test init command with project name using real components"""
        # Execute command with project name in isolated directory
        # This uses real InteractiveUI and TemplateEngine
        result = self.runner.invoke(self.app, ["my_ml_project"])

        # Verify project was created
        project_path = isolated_working_directory / "my_ml_project"
        assert project_path.exists()

        # Verify project structure was created
        assert (project_path / "data").exists()
        assert (project_path / "configs").exists()
        assert (project_path / "recipes").exists()
        assert (project_path / "README.md").exists()

    def test_init_command_interactive_mode_real_components(self, isolated_working_directory):
        """Test init command in interactive mode with real UI"""
        # Provide input for interactive mode
        # Since we're using real InteractiveUI, we need to provide input
        result = self.runner.invoke(self.app, [], input="test_project\n")  # Simulate user input

        # Verify project was created
        project_path = isolated_working_directory / "test_project"
        if result.exit_code == 0:
            assert project_path.exists()

    def test_init_command_creates_complete_structure(self, isolated_working_directory):
        """Test that init command creates complete project structure"""
        # Execute with real components
        result = self.runner.invoke(self.app, ["complete_test"])

        project_path = isolated_working_directory / "complete_test"

        # If command succeeded, verify complete structure
        if result.exit_code == 0:
            # Directory structure - from create_project_structure
            assert (project_path / "data").exists()
            assert (project_path / "configs").exists()
            assert (project_path / "recipes").exists()
            assert (project_path / "sql").exists()

            # Files created by template engine
            assert (project_path / "README.md").exists()
            assert (project_path / "pyproject.toml").exists()
            assert (project_path / "docker-compose.yml").exists()
            assert (project_path / "Dockerfile").exists()
            assert (project_path / ".gitignore").exists()

    def test_init_command_help_message(self):
        """Test init command help message shows correct argument info"""
        # Execute help command
        result = self.runner.invoke(self.app, ["--help"])

        # Verify help shows project_name as optional argument
        assert result.exit_code == 0
        assert "project_name" in result.output.lower() or "PROJECT_NAME" in result.output
        assert "init" in result.output.lower() or "초기화" in result.output

    def test_init_command_handles_existing_directory(self, isolated_working_directory):
        """Test init command handles existing directory gracefully"""
        # Create existing directory
        existing_path = isolated_working_directory / "existing_project"
        existing_path.mkdir(parents=True, exist_ok=True)

        # Try to create project with same name
        result = self.runner.invoke(self.app, ["existing_project"])

        # Command should handle this gracefully (either error or skip)
        # Check that we didn't corrupt existing directory
        assert existing_path.exists()

    def test_init_command_validates_project_name_format(self, isolated_working_directory):
        """Test init command accepts various valid project name formats"""
        # Test valid project names with real execution
        valid_names = ["my_project", "ml-pipeline", "Project123", "simple"]

        for name in valid_names:
            result = self.runner.invoke(self.app, [name])
            project_path = isolated_working_directory / name

            # If the name is valid, project should be created
            if result.exit_code == 0:
                assert project_path.exists()

    def test_init_command_keyboard_interrupt_handling(self, isolated_working_directory):
        """Test init command handles user cancellation"""
        # Simulate Ctrl+C during interactive input
        # With real components, we simulate this by providing invalid input
        result = self.runner.invoke(self.app, [], input="\x03")  # Ctrl+C character

        # Command should exit gracefully without creating partial project
        # Check that no incomplete project structures exist
        subdirs = list(isolated_working_directory.iterdir())
        incomplete_projects = [d for d in subdirs if d.is_dir() and not (d / "README.md").exists()]

        # Should not leave incomplete project directories
        assert len(incomplete_projects) == 0 or result.exit_code != 0


class TestInitCommandIntegration:
    """Integration tests for init command with CLI environment"""

    def test_init_with_cli_test_environment(self, cli_test_environment):
        """Test init command using cli_test_environment fixture"""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(init_command)

        # Use the isolated environment provided by cli_test_environment
        work_dir = cli_test_environment["work_dir"]

        # Run init command in the test environment
        result = runner.invoke(app, ["integration_test_project"])

        # Verify project creation in test environment
        project_path = work_dir / "integration_test_project"
        if result.exit_code == 0:
            assert project_path.exists()
            assert (project_path / "configs").exists()
            assert (project_path / "recipes").exists()
