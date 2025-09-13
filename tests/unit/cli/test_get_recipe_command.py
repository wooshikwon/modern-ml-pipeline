"""
Unit Tests for Get-Recipe Command CLI
Phase 3: Interactive recipe generation command tests

This module follows the tests/README.md architecture principles:
- Real object testing over mock hell
- Rich console integration patterns
- Interactive UI component testing
- Comprehensive error scenario coverage
- Architecture compliance with existing CLI test patterns
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import typer
from typer.testing import CliRunner
from pathlib import Path
import json
import tempfile
import yaml

from src.cli.commands.get_recipe_command import get_recipe_command, _show_success_message


class TestGetRecipeCommandBasicFunctionality:
    """Get-recipe command basic functionality and argument parsing tests"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_step_complete')
    @patch('src.cli.commands.get_recipe_command.cli_info')
    def test_get_recipe_command_successful_flow(
        self, mock_cli_info, mock_cli_step, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test complete successful recipe generation flow"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock user selections
        mock_selections = {
            'recipe_name': 'test_classification_model',
            'task': 'classification',
            'model_class': 'RandomForestClassifier',
            'model_library': 'sklearn',
            'target_column': 'target',
            'entity_schema': ['feature1', 'feature2', 'feature3']
        }
        mock_builder.run_interactive_flow.return_value = mock_selections

        # Mock recipe file generation
        mock_recipe_path = Path('recipes/test_classification_model.yaml')
        mock_builder.generate_recipe_file.return_value = mock_recipe_path

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify success
        assert result.exit_code == 0

        # Verify Rich Console sequence
        mock_cli_start.assert_called_once_with("Get Recipe", "대화형 Recipe 파일 생성")

        # Verify UI initialization and welcome panel
        mock_ui_class.assert_called_once()
        mock_ui.show_panel.assert_called_once_with(
            """🚀 환경 독립적인 Recipe 생성을 시작합니다!

        Recipe는 환경 설정과 분리되어 있어,
        다양한 환경에서 재사용할 수 있습니다.""",
            title="Recipe Generator",
            style="green"
        )

        # Verify RecipeBuilder initialization and workflow
        mock_builder_class.assert_called_once()
        mock_cli_step.assert_any_call("초기화", "RecipeBuilder 준비 완료")
        mock_cli_info.assert_any_call("대화형 Recipe 생성을 시작합니다...")

        # Verify interactive flow execution
        mock_builder.run_interactive_flow.assert_called_once()

        # Verify recipe file generation
        mock_cli_step.assert_any_call(
            "대화형 플로우",
            "Task: classification, Model: RandomForestClassifier"
        )
        mock_cli_info.assert_any_call("Recipe 파일을 생성하는 중...")
        mock_builder.generate_recipe_file.assert_called_once_with(mock_selections)

        # Verify completion
        mock_cli_step.assert_any_call(
            "파일 생성",
            f"Recipe: {mock_recipe_path}"
        )

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_command_error')
    def test_get_recipe_command_keyboard_interrupt(
        self, mock_cli_error, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test get-recipe command handles KeyboardInterrupt (user cancellation)"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock KeyboardInterrupt during interactive flow
        mock_builder.run_interactive_flow.side_effect = KeyboardInterrupt()

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify graceful exit
        assert result.exit_code == 0  # Keyboard interrupt exits with 0

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Get Recipe", "대화형 Recipe 파일 생성")
        mock_cli_error.assert_called_once_with("Get Recipe", "Recipe 생성이 취소되었습니다")

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_command_error')
    def test_get_recipe_command_file_not_found_error(
        self, mock_cli_error, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test get-recipe command handles FileNotFoundError"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock FileNotFoundError during recipe generation
        mock_builder.run_interactive_flow.return_value = {'task': 'classification'}
        mock_builder.generate_recipe_file.side_effect = FileNotFoundError("Template 파일을 찾을 수 없습니다")

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Get Recipe", "대화형 Recipe 파일 생성")
        mock_cli_error.assert_called_once_with(
            "Get Recipe",
            "파일을 찾을 수 없습니다: Template 파일을 찾을 수 없습니다"
        )

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_command_error')
    def test_get_recipe_command_value_error(
        self, mock_cli_error, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test get-recipe command handles ValueError (validation errors)"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock ValueError during validation
        mock_builder.run_interactive_flow.side_effect = ValueError("선택한 모델이 Task와 호환되지 않습니다")

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Get Recipe", "대화형 Recipe 파일 생성")
        mock_cli_error.assert_called_once_with(
            "Get Recipe",
            "잘못된 값: 선택한 모델이 Task와 호환되지 않습니다"
        )

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_command_error')
    def test_get_recipe_command_general_exception(
        self, mock_cli_error, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test get-recipe command handles unexpected exceptions"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock general Exception
        mock_builder.run_interactive_flow.side_effect = Exception("예상치 못한 Registry 오류")

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Get Recipe", "대화형 Recipe 파일 생성")
        mock_cli_error.assert_called_once_with(
            "Get Recipe",
            "Recipe 생성 중 오류가 발생했습니다: 예상치 못한 Registry 오류",
            "자세한 오류 정보는 로그를 확인하세요"
        )

    def test_get_recipe_command_help_message(self):
        """Test get-recipe command shows proper help message"""
        result = self.runner.invoke(self.app, ['--help'])

        assert result.exit_code == 0
        assert "환경 독립적인 모델 Recipe 생성" in result.output
        assert "대화형 인터페이스" in result.output
        assert "Recipe 파일 생성" in result.output


class TestGetRecipeCommandInteractiveFlow:
    """Interactive flow integration tests for get-recipe command"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_step_complete')
    @patch('src.cli.commands.get_recipe_command.cli_info')
    @patch('src.cli.commands.get_recipe_command._show_success_message')
    def test_get_recipe_command_with_regression_task(
        self, mock_success_msg, mock_cli_info, mock_cli_step, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test recipe generation for regression task"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock regression selections
        mock_selections = {
            'recipe_name': 'house_price_prediction',
            'task': 'regression',
            'model_class': 'XGBRegressor',
            'model_library': 'xgboost',
            'target_column': 'price',
            'entity_schema': ['bedrooms', 'bathrooms', 'sqft_living'],
            'preprocessor_steps': ['scaler', 'imputer']
        }
        mock_builder.run_interactive_flow.return_value = mock_selections

        # Mock recipe file generation
        mock_recipe_path = Path('recipes/house_price_prediction.yaml')
        mock_builder.generate_recipe_file.return_value = mock_recipe_path

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify success
        assert result.exit_code == 0

        # Verify step completion includes correct task and model
        mock_cli_step.assert_any_call(
            "대화형 플로우",
            "Task: regression, Model: XGBRegressor"
        )

        # Verify success message is called with correct parameters
        mock_success_msg.assert_called_once_with(mock_recipe_path, mock_selections)

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_step_complete')
    def test_get_recipe_command_clustering_task(
        self, mock_cli_step, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test recipe generation for clustering task"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock clustering selections (no target_column for unsupervised)
        mock_selections = {
            'recipe_name': 'customer_segmentation',
            'task': 'clustering',
            'model_class': 'KMeans',
            'model_library': 'sklearn',
            'entity_schema': ['annual_spending', 'frequency', 'recency'],
            'n_clusters': 5
        }
        mock_builder.run_interactive_flow.return_value = mock_selections

        mock_recipe_path = Path('recipes/customer_segmentation.yaml')
        mock_builder.generate_recipe_file.return_value = mock_recipe_path

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify success
        assert result.exit_code == 0

        # Verify clustering-specific step completion
        mock_cli_step.assert_any_call(
            "대화형 플로우",
            "Task: clustering, Model: KMeans"
        )

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    def test_get_recipe_command_recipe_builder_integration(
        self, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test integration between command and RecipeBuilder components"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock complex selections with preprocessing
        mock_selections = {
            'recipe_name': 'advanced_classification',
            'task': 'classification',
            'model_class': 'LGBMClassifier',
            'model_library': 'lightgbm',
            'target_column': 'churn',
            'entity_schema': ['age', 'tenure', 'monthly_charges'],
            'preprocessor_steps': ['imputer', 'scaler', 'encoder'],
            'feature_engineering': True,
            'hyperparameter_tuning': True
        }
        mock_builder.run_interactive_flow.return_value = mock_selections

        mock_recipe_path = Path('recipes/advanced_classification.yaml')
        mock_builder.generate_recipe_file.return_value = mock_recipe_path

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify success
        assert result.exit_code == 0

        # Verify RecipeBuilder was properly initialized
        mock_builder_class.assert_called_once()

        # Verify interactive flow was called without parameters (no pre-selections)
        mock_builder.run_interactive_flow.assert_called_once()

        # Verify recipe generation received the selections
        mock_builder.generate_recipe_file.assert_called_once_with(mock_selections)

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    def test_get_recipe_command_ui_panel_display(self, mock_ui_class, mock_builder_class):
        """Test UI panel display functionality"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {'task': 'classification'}
        mock_builder.generate_recipe_file.return_value = Path('recipes/test.yaml')

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify welcome panel was displayed with correct content
        expected_panel_content = """🚀 환경 독립적인 Recipe 생성을 시작합니다!

        Recipe는 환경 설정과 분리되어 있어,
        다양한 환경에서 재사용할 수 있습니다."""

        mock_ui.show_panel.assert_called_once_with(
            expected_panel_content,
            title="Recipe Generator",
            style="green"
        )


class TestShowSuccessMessage:
    """Test the success message display function"""

    @patch('src.cli.commands.get_recipe_command.cli_success_panel')
    def test_show_success_message_classification(self, mock_success_panel):
        """Test success message display for classification task"""
        recipe_path = Path('recipes/fraud_detection.yaml')
        selections = {
            'task': 'classification',
            'model_class': 'RandomForestClassifier',
            'model_library': 'sklearn',
            'recipe_name': 'fraud_detection'
        }

        _show_success_message(recipe_path, selections)

        # Verify success panel was called
        mock_success_panel.assert_called_once()

        # Get the content passed to success panel
        call_args = mock_success_panel.call_args
        content = call_args[0][0]  # First positional argument
        title = call_args[0][1]    # Second positional argument

        # Verify content includes key information
        assert "✅ Recipe가 성공적으로 생성되었습니다!" in content
        assert str(recipe_path) in content
        assert "classification" in content
        assert "RandomForestClassifier" in content
        assert "sklearn" in content
        assert "mmp train --recipe-file" in content
        assert "Recipe 생성 완료" == title

    @patch('src.cli.commands.get_recipe_command.cli_success_panel')
    def test_show_success_message_regression(self, mock_success_panel):
        """Test success message display for regression task"""
        recipe_path = Path('recipes/price_prediction.yaml')
        selections = {
            'task': 'regression',
            'model_class': 'XGBRegressor',
            'model_library': 'xgboost',
            'recipe_name': 'price_prediction'
        }

        _show_success_message(recipe_path, selections)

        # Verify success panel was called with regression-specific content
        mock_success_panel.assert_called_once()

        call_args = mock_success_panel.call_args
        content = call_args[0][0]

        assert "regression" in content
        assert "XGBRegressor" in content
        assert "xgboost" in content
        assert "price_prediction.yaml" in content

    @patch('src.cli.commands.get_recipe_command.cli_success_panel')
    def test_show_success_message_includes_usage_examples(self, mock_success_panel):
        """Test success message includes usage examples and next steps"""
        recipe_path = Path('recipes/test_model.yaml')
        selections = {
            'task': 'clustering',
            'model_class': 'KMeans',
            'model_library': 'sklearn'
        }

        _show_success_message(recipe_path, selections)

        call_args = mock_success_panel.call_args
        content = call_args[0][0]

        # Verify usage examples are included
        assert "💡 다음 단계:" in content
        assert "cat recipes/test_model.yaml" in content
        assert "mmp train -r recipes/test_model.yaml -e local" in content
        assert "mmp train -r recipes/test_model.yaml -e dev" in content
        assert "mmp train -r recipes/test_model.yaml -e prod" in content
        assert "Recipe는 환경과 독립적이므로" in content

    @patch('src.cli.commands.get_recipe_command.cli_success_panel')
    def test_show_success_message_with_long_path(self, mock_success_panel):
        """Test success message handles long file paths properly"""
        recipe_path = Path('recipes/very/deep/nested/directory/structure/complex_model_name.yaml')
        selections = {
            'task': 'classification',
            'model_class': 'GradientBoostingClassifier',
            'model_library': 'sklearn'
        }

        _show_success_message(recipe_path, selections)

        call_args = mock_success_panel.call_args
        content = call_args[0][0]

        # Verify long path is properly included
        assert str(recipe_path) in content
        assert "complex_model_name.yaml" in content


class TestGetRecipeCommandErrorHandling:
    """Comprehensive error handling tests for edge cases"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_command_error')
    def test_get_recipe_command_permission_error(
        self, mock_cli_error, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test recipe command handles permission errors during file writing"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {'task': 'classification'}

        # Mock PermissionError during file writing
        mock_builder.generate_recipe_file.side_effect = PermissionError("권한이 없습니다: recipes/ 디렉토리")

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify error handling
        assert result.exit_code == 1
        mock_cli_error.assert_called_once_with(
            "Get Recipe",
            "Recipe 생성 중 오류가 발생했습니다: 권한이 없습니다: recipes/ 디렉토리",
            "자세한 오류 정보는 로그를 확인하세요"
        )

    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.cli_command_start')
    @patch('src.cli.commands.get_recipe_command.cli_step_complete')
    def test_get_recipe_command_empty_selections(
        self, mock_cli_step, mock_cli_start, mock_ui_class, mock_builder_class
    ):
        """Test recipe command handles empty or minimal selections"""
        # Setup mocks
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        # Mock minimal selections that should cause the command to succeed with N/A values
        mock_selections = {
            'task': 'classification',
            'model_class': None,
            'model_library': 'sklearn'
        }  # Minimal selection with required fields
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_recipe_file.return_value = Path('recipes/minimal.yaml')

        # Execute command
        result = self.runner.invoke(self.app, [])

        # Verify it handles minimal selections
        assert result.exit_code == 0

        # Verify step completion with None for missing model
        mock_cli_step.assert_any_call(
            "대화형 플로우",
            "Task: classification, Model: None"
        )