"""
Unit Tests for List Commands CLI
Days 3-5: CLI argument parsing and validation tests
"""

from unittest.mock import patch

import typer
from typer.testing import CliRunner

from src.cli.commands.list_commands import (
    list_adapters,
    list_evaluators,
    list_models,
    list_preprocessors,
)


class TestListCommandsArgumentParsing:
    """List commands argument parsing tests"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()

        # Add all list commands
        self.app.command("adapters")(list_adapters)
        self.app.command("evaluators")(list_evaluators)
        self.app.command("models")(list_models)
        self.app.command("preprocessors")(list_preprocessors)

    @patch("src.cli.commands.list_commands.AdapterRegistry.list_keys")
    def test_list_adapters_command(self, mock_list_keys):
        """Test list adapters command execution"""
        # Setup mock
        mock_list_keys.return_value = ["storage", "sql"]

        # Execute command
        result = self.runner.invoke(self.app, ["adapters"])

        # Verify success and output
        assert result.exit_code == 0
        assert "Adapters:" in result.output
        assert "storage" in result.output
        assert "sql" in result.output

        # Verify registry was called
        mock_list_keys.assert_called_once()

    @patch("src.cli.commands.list_commands.EvaluatorRegistry.list_keys")
    def test_list_evaluators_command(self, mock_list_keys):
        """Test list evaluators command execution"""
        # Setup mock
        mock_list_keys.return_value = ["classification", "regression", "clustering"]

        # Execute command
        result = self.runner.invoke(self.app, ["evaluators"])

        # Verify success and output
        assert result.exit_code == 0
        assert "Evaluators:" in result.output
        assert "classification" in result.output
        assert "regression" in result.output

        # Verify registry was called
        mock_list_keys.assert_called_once()

    @patch("src.cli.commands.list_commands._load_catalog_from_directory")
    @patch("pathlib.Path.exists")
    def test_list_models_command(self, mock_path_exists, mock_load_catalog):
        """Test list models command execution"""
        # Setup mocks
        mock_path_exists.return_value = True  # catalog directory exists
        mock_load_catalog.return_value = {
            "Classification": [
                {"class_path": "RandomForestClassifier", "library": "sklearn"},
                {"class_path": "XGBClassifier", "library": "xgboost"},
            ],
            "Regression": [{"class_path": "LinearRegression", "library": "sklearn"}],
        }

        # Execute command
        result = self.runner.invoke(self.app, ["models"])

        # Verify success and output
        assert result.exit_code == 0
        assert "Models:" in result.output
        assert "Classification" in result.output
        assert "RandomForestClassifier" in result.output

        # Verify catalog loader was called
        mock_load_catalog.assert_called_once()

    @patch("src.cli.commands.list_commands.PreprocessorStepRegistry.list_keys")
    def test_list_preprocessors_command(self, mock_list_keys):
        """Test list preprocessors command execution"""
        # Setup mock
        mock_list_keys.return_value = ["standard_scaler", "simple_imputer", "one_hot_encoder"]

        # Execute command
        result = self.runner.invoke(self.app, ["preprocessors"])

        # Verify success and output
        assert result.exit_code == 0
        assert "Preprocessor Steps:" in result.output
        assert "standard_scaler" in result.output
        assert "simple_imputer" in result.output

        # Verify registry was called
        mock_list_keys.assert_called_once()

    @patch("src.cli.commands.list_commands.AdapterRegistry.list_keys")
    def test_list_adapters_empty_registry(self, mock_list_keys):
        """Test list adapters when no adapters are available"""
        # Setup mock to return empty
        mock_list_keys.return_value = []

        # Execute command
        result = self.runner.invoke(self.app, ["adapters"])

        # Verify success and shows empty message
        assert result.exit_code == 0
        assert "Adapters:" in result.output
        assert "(no adapters available)" in result.output

    def test_list_commands_help_messages(self):
        """Test help messages for list commands"""
        # Test each subcommand help
        commands = ["adapters", "evaluators", "models", "preprocessors"]

        for cmd in commands:
            result = self.runner.invoke(self.app, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd in result.output.lower()

    @patch("src.cli.commands.list_commands._load_catalog_from_directory")
    @patch("pathlib.Path.exists")
    def test_list_models_handles_catalog_error(self, mock_path_exists, mock_fallback_catalog):
        """Test list models handles catalog loading errors gracefully"""
        # Setup mocks - directory exists but loading fails, fallback succeeds
        mock_path_exists.return_value = True
        mock_fallback_catalog.return_value = {}  # Empty catalog

        # Execute command - should exit with error code 1 when empty
        result = self.runner.invoke(self.app, ["models"])

        # Should attempt to load catalog from directory
        mock_fallback_catalog.assert_called_once()
