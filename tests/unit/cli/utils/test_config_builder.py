"""
Test Suite for Config Builder
Following Real Object Testing philosophy - minimal mocking

Tests cover:
- SimpleInteractiveUI user input handling
- InteractiveConfigBuilder configuration flow
- Template rendering integration
- Error handling and validation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
from datetime import datetime

from src.cli.utils.config_builder import SimpleInteractiveUI, InteractiveConfigBuilder
from src.cli.utils.template_engine import TemplateEngine


class TestSimpleInteractiveUI:
    """Test SimpleInteractiveUI class - user interaction methods."""

    def test_text_input_with_default(self):
        """Test text input returns default when user provides no input."""
        ui = SimpleInteractiveUI()

        with patch('builtins.input', return_value=""):
            result = ui.text_input("Enter name", default="default_name")
            assert result == "default_name"

    def test_text_input_with_user_value(self):
        """Test text input returns user provided value."""
        ui = SimpleInteractiveUI()

        with patch('builtins.input', return_value="custom_value"):
            result = ui.text_input("Enter name", default="default_name")
            assert result == "custom_value"

    def test_text_input_with_validation_success(self):
        """Test text input with validator - successful validation."""
        ui = SimpleInteractiveUI()

        def validator(value):
            return len(value) > 3

        with patch('builtins.input', return_value="valid_input"):
            result = ui.text_input("Enter name", validator=validator)
            assert result == "valid_input"

    def test_text_input_with_validation_retry(self):
        """Test text input retries on validation failure."""
        ui = SimpleInteractiveUI()

        def validator(value):
            return len(value) > 3

        # First invalid, then valid
        with patch('builtins.input', side_effect=["no", "valid_input"]):
            with patch('src.cli.utils.config_builder.cli_print') as mock_print:
                result = ui.text_input("Enter name", validator=validator)
                assert result == "valid_input"
                mock_print.assert_called_with("❌ 유효하지 않은 입력입니다. 다시 시도해주세요.", style="red")

    def test_confirm_yes_variations(self):
        """Test confirm accepts various yes inputs."""
        ui = SimpleInteractiveUI()

        yes_inputs = ['y', 'yes', '1', 'true', 'Y', 'YES']
        for yes_input in yes_inputs:
            with patch('builtins.input', return_value=yes_input):
                result = ui.confirm("Confirm?")
                assert result is True

    def test_confirm_no_variations(self):
        """Test confirm accepts various no inputs."""
        ui = SimpleInteractiveUI()

        no_inputs = ['n', 'no', '0', 'false', 'N', 'NO']
        for no_input in no_inputs:
            with patch('builtins.input', return_value=no_input):
                result = ui.confirm("Confirm?")
                assert result is False

    def test_confirm_default_true(self):
        """Test confirm returns default True when empty."""
        ui = SimpleInteractiveUI()

        with patch('builtins.input', return_value=""):
            result = ui.confirm("Confirm?", default=True)
            assert result is True

    def test_confirm_default_false(self):
        """Test confirm returns default False when empty."""
        ui = SimpleInteractiveUI()

        with patch('builtins.input', return_value=""):
            result = ui.confirm("Confirm?", default=False)
            assert result is False

    def test_confirm_invalid_input_retry(self):
        """Test confirm retries on invalid input."""
        ui = SimpleInteractiveUI()

        with patch('builtins.input', side_effect=["invalid", "y"]):
            with patch('src.cli.utils.config_builder.cli_print') as mock_print:
                result = ui.confirm("Confirm?")
                assert result is True
                mock_print.assert_called_with("❌ y/n으로 답해주세요.", style="red")

    def test_select_from_list(self):
        """Test selecting from list of options."""
        ui = SimpleInteractiveUI()
        options = ["Option 1", "Option 2", "Option 3"]

        with patch('builtins.input', return_value="2"):
            with patch('src.cli.utils.config_builder.cli_print'):
                result = ui.select_from_list("Select option", options)
                assert result == "Option 2"

    def test_select_from_list_invalid_number(self):
        """Test select from list with invalid number retries."""
        ui = SimpleInteractiveUI()
        options = ["Option 1", "Option 2"]

        with patch('builtins.input', side_effect=["3", "1"]):
            with patch('src.cli.utils.config_builder.cli_print') as mock_print:
                result = ui.select_from_list("Select option", options)
                assert result == "Option 1"
                assert any("1-2" in str(call) for call in mock_print.call_args_list)

    def test_select_from_list_non_numeric(self):
        """Test select from list with non-numeric input."""
        ui = SimpleInteractiveUI()
        options = ["Option 1", "Option 2"]

        with patch('builtins.input', side_effect=["abc", "2"]):
            with patch('src.cli.utils.config_builder.cli_print') as mock_print:
                result = ui.select_from_list("Select option", options)
                assert result == "Option 2"
                assert any("숫자를 입력해주세요" in str(call) for call in mock_print.call_args_list)

    def test_show_info(self):
        """Test show_info displays message."""
        ui = SimpleInteractiveUI()

        with patch('src.cli.utils.config_builder.cli_info') as mock_info:
            ui.show_info("Test message")
            mock_info.assert_called_once_with("Test message")

    def test_show_warning(self):
        """Test show_warning displays warning message."""
        ui = SimpleInteractiveUI()

        with patch('src.cli.utils.config_builder.cli_print') as mock_print:
            ui.show_warning("Warning message")
            mock_print.assert_called_once_with("⚠️ Warning message", style="yellow")

    def test_print_divider(self):
        """Test print_divider outputs divider line."""
        ui = SimpleInteractiveUI()

        with patch('src.cli.utils.config_builder.cli_print') as mock_print:
            ui.print_divider()
            mock_print.assert_called_once_with("\n" + "="*50 + "\n")


class TestInteractiveConfigBuilder:
    """Test InteractiveConfigBuilder - configuration flow and template rendering."""

    def test_initialization(self):
        """Test InteractiveConfigBuilder initialization."""
        builder = InteractiveConfigBuilder()

        assert builder.ui is not None
        assert isinstance(builder.ui, SimpleInteractiveUI)
        assert builder.template_engine is not None
        assert isinstance(builder.template_engine, TemplateEngine)

    def test_run_interactive_flow_basic(self, monkeypatch):
        """Test basic interactive flow for local environment."""
        builder = InteractiveConfigBuilder()

        # Mock user inputs following the actual flow
        inputs = [
            "local",  # Environment name
            "y",  # Use MLflow
            "./mlruns",  # MLflow tracking URI
            "3",  # Data source: Local Files (option 3)
            "1",  # Feature store: None (option 1)
            "1",  # Artifact storage: Local (option 1)
            "y",  # Enable inference output
            "3",  # Inference output source: Local Files (option 3)
            "y"  # Final confirmation
        ]

        with patch('builtins.input', side_effect=inputs):
            selections = builder.run_interactive_flow()

            assert selections['env_name'] == "local"
            assert selections['use_mlflow'] == True
            assert selections['mlflow_tracking_uri'] == "./mlruns"
            assert selections['data_source'] == "Local Files"
            assert selections['feature_store'] == "없음"
            assert selections['artifact_storage'] == "Local"
            assert selections['inference_output_enabled'] == True
            assert selections['inference_output_source'] == "Local Files"

    def test_run_interactive_flow_with_sql(self):
        """Test interactive flow with SQL adapter."""
        builder = InteractiveConfigBuilder()

        inputs = [
            "prod",  # Environment name
            "y",  # Use MLflow
            "http://mlflow-server:5000",  # MLflow tracking URI
            "1",  # Data source: PostgreSQL (option 1)
            "1",  # Feature store: None (option 1)
            "2",  # Artifact storage: S3 (option 2)
            "y",  # Enable inference output
            "1",  # Inference output source: PostgreSQL (option 1)
            "y"  # Final confirmation
        ]

        with patch('builtins.input', side_effect=inputs):
            selections = builder.run_interactive_flow()

            assert selections['env_name'] == "prod"
            assert selections['use_mlflow'] == True
            assert selections['mlflow_tracking_uri'] == "http://mlflow-server:5000"
            assert selections['data_source'] == "PostgreSQL"
            assert selections['feature_store'] == "없음"
            assert selections['artifact_storage'] == "S3"
            assert selections['inference_output_enabled'] == True
            assert selections['inference_output_source'] == "PostgreSQL"

    def test_run_interactive_flow_with_feature_store(self):
        """Test interactive flow with Feast feature store."""
        builder = InteractiveConfigBuilder()

        inputs = [
            "staging",  # Environment name
            "y",  # Use MLflow
            "./mlruns",  # MLflow tracking URI
            "3",  # Data source: Local Files (option 3)
            "2",  # Feature store: Feast (option 2)
            "1",  # Feast registry location: Local (option 1)
            "n",  # No online store
            "1",  # Artifact storage: Local (option 1)
            "y",  # Enable inference output
            "3",  # Inference output source: Local Files (option 3)
            "y"  # Final confirmation
        ]

        with patch('builtins.input', side_effect=inputs):
            selections = builder.run_interactive_flow()

            assert selections['env_name'] == "staging"
            assert selections['use_mlflow'] == True
            assert selections['data_source'] == "Local Files"
            assert selections['feature_store'] == "Feast"
            assert selections['feast_registry_location'] == "로컬"
            assert selections['feast_online_store'] == "SQLite"
            assert selections['feast_needs_offline_path'] == True

    def test_prepare_template_context(self, monkeypatch):
        """Test preparing template context from selections."""
        builder = InteractiveConfigBuilder()

        selections = {
            'env_name': 'test',
            'use_mlflow': True,
            'mlflow_tracking_uri': './mlruns',
            'data_source': 'Local Files',
            'feature_store': '없음',
            'artifact_storage': 'Local',
            'inference_output_enabled': True,
            'inference_output_source': 'Local Files'
        }

        context = builder._prepare_template_context(selections)

        # Verify context structure
        assert isinstance(context, dict)
        assert 'env_name' in context
        assert context['env_name'] == 'test'
        assert 'timestamp' in context

    def test_generate_config_file(self, tmp_path, monkeypatch):
        """Test generating config file from selections."""
        builder = InteractiveConfigBuilder()

        # Change to temp directory to avoid creating files in project
        monkeypatch.chdir(tmp_path)

        selections = {
            'env_name': 'test',
            'use_mlflow': True,
            'mlflow_tracking_uri': './mlruns',
            'data_source': 'Local Files',
            'feature_store': '없음',
            'artifact_storage': 'Local',
            'inference_output_enabled': True,
            'inference_output_source': 'Local Files'
        }

        with patch.object(builder.template_engine, 'write_rendered_file') as mock_write:
            result_path = builder.generate_config_file('test', selections)

            # Verify the path returned
            assert result_path == Path('configs/test.yaml')
            # Verify template engine was called
            mock_write.assert_called_once()

    def test_generate_env_template(self, tmp_path, monkeypatch):
        """Test generating environment template file."""
        builder = InteractiveConfigBuilder()

        # Change to temp directory to avoid creating files in project
        monkeypatch.chdir(tmp_path)

        selections = {
            'env_name': 'test',
            'use_mlflow': True,
            'mlflow_tracking_uri': './mlruns',
            'data_source': 'PostgreSQL',
            'feature_store': 'Feast',
            'feast_registry_location': '로컬',
            'feast_online_store': 'Redis',
            'artifact_storage': 'S3',
            'inference_output_enabled': True,
            'inference_output_source': 'PostgreSQL'
        }

        result_path = builder.generate_env_template('test', selections)

        # Check file created
        assert result_path.exists()
        assert result_path == Path('.env.test.template')

    def test_build_config_integration(self, tmp_path, monkeypatch):
        """Test complete config building flow."""
        builder = InteractiveConfigBuilder()

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Mock user inputs for complete flow
        inputs = [
            "dev",  # Environment name
            "n",  # No MLflow
            "3",  # Data source: Local Files
            "1",  # Feature store: None
            "n",  # Inference output disabled
            "y"  # Final confirmation
        ]

        with patch('builtins.input', side_effect=inputs):
            with patch.object(builder.template_engine, 'write_rendered_file'):
                selections = builder.run_interactive_flow()
                result_path = builder.generate_config_file('dev', selections)

                assert result_path == Path('configs/dev.yaml')
                assert selections['env_name'] == 'dev'
                assert selections['use_mlflow'] == False

    def test_error_handling_template_not_found(self, tmp_path, monkeypatch):
        """Test error handling when template is not found."""
        builder = InteractiveConfigBuilder()

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        selections = {
            'env_name': 'test',
            'use_mlflow': True
        }

        with patch.object(builder.template_engine, 'write_rendered_file',
                         side_effect=FileNotFoundError("Template not found")):
            with pytest.raises(FileNotFoundError):
                builder.generate_config_file('test', selections)

    def test_interactive_flow_with_validation(self):
        """Test interactive flow with input validation."""
        builder = InteractiveConfigBuilder()

        # Test environment name validation (non-empty)
        inputs = [
            "",  # Empty environment name (should use default)
            "y",  # Use MLflow
            "./mlruns",  # MLflow tracking URI
            "3",  # Data source: Local Files
            "1",  # Feature store: None
            "1",  # Artifact storage: Local
            "y",  # Enable inference output
            "3",  # Inference output source: Local Files
            "y"  # Final confirmation
        ]

        with patch('builtins.input', side_effect=inputs):
            with patch('src.cli.utils.config_builder.cli_print'):
                selections = builder.run_interactive_flow()

                # Empty input should use default 'local'
                assert selections['env_name'] == "local"