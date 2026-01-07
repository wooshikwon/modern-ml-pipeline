"""
Unit Tests for Get-Recipe Command CLI

get_recipe_command의 대화형 Recipe 생성 기능을 테스트합니다.
현재 구현은 _print_header, _print_step, _print_error 함수를 사용하여 stdout으로 출력합니다.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from src.cli.commands.get_recipe_command import _show_success_message, get_recipe_command


class TestGetRecipeCommandBasicFunctionality:
    """Get-recipe command 기본 기능 테스트"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_successful_flow(self, mock_ui_class, mock_builder_class):
        """성공적인 Recipe 생성 플로우 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "name": "test_classification_model",
            "task_choice": "classification",
            "model": {
                "class_path": "RandomForestClassifier",
                "library": "sklearn",
                "hyperparameters": {"tuning_enabled": False, "values": {}},
            },
            "data": {
                "data_interface": {
                    "target_column": "target",
                    "entity_columns": ["id"],
                },
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
            "evaluation": {"metrics": ["accuracy"]},
            "metadata": {"author": "CLI", "created_at": "2024-01-01", "description": ""},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data

        mock_recipe_path = Path("recipes/test_classification_model.yaml")
        mock_builder.create_recipe_file.return_value = mock_recipe_path

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
        assert "Recipe 생성 완료" in result.output
        mock_builder.build_recipe_interactively.assert_called_once()
        mock_builder.create_recipe_file.assert_called_once_with(mock_recipe_data)

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_keyboard_interrupt(self, mock_ui_class, mock_builder_class):
        """KeyboardInterrupt 처리 테스트 (사용자 취소)"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_recipe_interactively.side_effect = KeyboardInterrupt()

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
        assert "취소" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_file_not_found_error(self, mock_ui_class, mock_builder_class):
        """FileNotFoundError 처리 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "task_choice": "classification",
            "model": {"class_path": "Test", "library": "sklearn"},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.side_effect = FileNotFoundError("Template not found")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "파일 없음" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_value_error(self, mock_ui_class, mock_builder_class):
        """ValueError 처리 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_recipe_interactively.side_effect = ValueError("Invalid task")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "잘못된 값" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_general_exception(self, mock_ui_class, mock_builder_class):
        """일반 예외 처리 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_recipe_interactively.side_effect = Exception("Unexpected error")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "Recipe 생성 실패" in result.output

    def test_get_recipe_command_help_message(self):
        """도움말 메시지 테스트"""
        result = self.runner.invoke(self.app, ["--help"])

        assert result.exit_code == 0
        assert "환경 독립적인 모델 Recipe 생성" in result.output


class TestGetRecipeCommandInteractiveFlow:
    """대화형 플로우 테스트"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_with_regression_task(self, mock_ui_class, mock_builder_class):
        """Regression task Recipe 생성 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "name": "house_price_prediction",
            "task_choice": "regression",
            "model": {
                "class_path": "XGBRegressor",
                "library": "xgboost",
                "hyperparameters": {"tuning_enabled": False, "values": {}},
            },
            "data": {
                "data_interface": {
                    "target_column": "price",
                    "entity_columns": ["id"],
                },
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
            "evaluation": {"metrics": ["r2_score"]},
            "metadata": {"author": "CLI", "created_at": "2024-01-01", "description": ""},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.return_value = Path("recipes/house_price_prediction.yaml")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
        assert "regression" in result.output
        assert "XGBRegressor" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_clustering_task(self, mock_ui_class, mock_builder_class):
        """Clustering task Recipe 생성 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "name": "customer_segmentation",
            "task_choice": "clustering",
            "model": {
                "class_path": "KMeans",
                "library": "sklearn",
                "hyperparameters": {"tuning_enabled": False, "values": {"n_clusters": 5}},
            },
            "data": {
                "data_interface": {"entity_columns": ["customer_id"]},
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
            "evaluation": {"metrics": ["silhouette_score"]},
            "metadata": {"author": "CLI", "created_at": "2024-01-01", "description": ""},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.return_value = Path("recipes/customer_segmentation.yaml")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
        assert "clustering" in result.output
        assert "KMeans" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_recipe_builder_integration(self, mock_ui_class, mock_builder_class):
        """RecipeBuilder 통합 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "name": "advanced_classification",
            "task_choice": "classification",
            "model": {
                "class_path": "LGBMClassifier",
                "library": "lightgbm",
                "hyperparameters": {"tuning_enabled": True, "n_trials": 10},
            },
            "data": {
                "data_interface": {
                    "target_column": "churn",
                    "entity_columns": ["user_id"],
                },
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
            "evaluation": {"metrics": ["accuracy"]},
            "metadata": {"author": "CLI", "created_at": "2024-01-01", "description": ""},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.return_value = Path("recipes/advanced_classification.yaml")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
        mock_builder_class.assert_called_once()
        mock_builder.build_recipe_interactively.assert_called_once()
        mock_builder.create_recipe_file.assert_called_once_with(mock_recipe_data)

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_ui_panel_display(self, mock_ui_class, mock_builder_class):
        """UI 패널 표시 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_recipe_data = {
            "name": "test",
            "task_choice": "classification",
            "model": {"class_path": "Test", "library": "sklearn"},
            "data": {"data_interface": {"entity_columns": []}, "split": {}},
            "evaluation": {"metrics": []},
            "metadata": {},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.return_value = Path("recipes/test.yaml")

        result = self.runner.invoke(self.app, [])

        mock_ui.show_panel.assert_called_once()
        call_args = mock_ui.show_panel.call_args
        assert "Recipe Generator" in str(call_args)


class TestShowSuccessMessage:
    """성공 메시지 표시 함수 테스트"""

    def test_show_success_message_classification(self, capsys):
        """Classification 성공 메시지 테스트"""
        recipe_path = Path("recipes/fraud_detection.yaml")
        recipe_data = {
            "name": "fraud_detection",
            "task_choice": "classification",
            "model": {"class_path": "RandomForestClassifier", "library": "sklearn"},
        }

        _show_success_message(recipe_path, recipe_data)

        captured = capsys.readouterr()
        assert "Recipe 생성 완료" in captured.out
        assert str(recipe_path) in captured.out
        assert "classification" in captured.out
        assert "RandomForestClassifier" in captured.out

    def test_show_success_message_regression(self, capsys):
        """Regression 성공 메시지 테스트"""
        recipe_path = Path("recipes/price_prediction.yaml")
        recipe_data = {
            "name": "price_prediction",
            "task_choice": "regression",
            "model": {"class_path": "XGBRegressor", "library": "xgboost"},
        }

        _show_success_message(recipe_path, recipe_data)

        captured = capsys.readouterr()
        assert "regression" in captured.out
        assert "XGBRegressor" in captured.out
        assert "xgboost" in captured.out

    def test_show_success_message_includes_next_steps(self, capsys):
        """다음 단계 안내 포함 테스트"""
        recipe_path = Path("recipes/test_model.yaml")
        recipe_data = {
            "name": "test_model",
            "task_choice": "clustering",
            "model": {"class_path": "KMeans", "library": "sklearn"},
        }

        _show_success_message(recipe_path, recipe_data)

        captured = capsys.readouterr()
        assert "다음 단계" in captured.out
        assert "mmp train" in captured.out

    def test_show_success_message_ml_extras_hint(self, capsys):
        """ml-extras 힌트 표시 테스트 (xgboost는 core이므로 lightgbm 사용)"""
        recipe_path = Path("recipes/lgb_model.yaml")
        recipe_data = {
            "name": "lgb_model",
            "task_choice": "classification",
            "model": {"class_path": "LGBMClassifier", "library": "lightgbm"},
        }

        _show_success_message(recipe_path, recipe_data)

        captured = capsys.readouterr()
        assert "ml-extras" in captured.out

    def test_show_success_message_torch_extras_hint(self, capsys):
        """torch-extras 힌트 표시 테스트"""
        recipe_path = Path("recipes/lstm_model.yaml")
        recipe_data = {
            "name": "lstm_model",
            "task_choice": "timeseries",
            "model": {"class_path": "LSTMTimeSeries", "library": "torch"},
        }

        _show_success_message(recipe_path, recipe_data)

        captured = capsys.readouterr()
        assert "torch-extras" in captured.out


class TestGetRecipeCommandErrorHandling:
    """에러 처리 테스트"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_recipe_command)

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_permission_error(self, mock_ui_class, mock_builder_class):
        """권한 오류 처리 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_recipe_data = {
            "task_choice": "classification",
            "model": {"class_path": "Test", "library": "sklearn"},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.side_effect = PermissionError("Permission denied")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 1
        assert "FAIL" in result.output

    @patch("src.cli.utils.recipe_builder.RecipeBuilder")
    @patch("src.cli.utils.interactive_ui.InteractiveUI")
    def test_get_recipe_command_empty_selections(self, mock_ui_class, mock_builder_class):
        """최소 선택 처리 테스트"""
        mock_ui = MagicMock()
        mock_ui_class.return_value = mock_ui

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        mock_recipe_data = {
            "name": "minimal",
            "task_choice": "classification",
            "model": {"class_path": "LogisticRegression", "library": "sklearn"},
            "data": {"data_interface": {"entity_columns": []}, "split": {}},
            "evaluation": {"metrics": []},
            "metadata": {},
        }
        mock_builder.build_recipe_interactively.return_value = mock_recipe_data
        mock_builder.create_recipe_file.return_value = Path("recipes/minimal.yaml")

        result = self.runner.invoke(self.app, [])

        assert result.exit_code == 0
