"""
Unit tests for get_recipe_command.
Tests recipe generation command functionality with typer and CLI integration.
"""

import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.cli.commands.get_recipe_command import get_recipe_command, _show_success_message


class TestGetRecipeCommandInitialization:
    """Test get_recipe command initialization and basic functionality."""
    
    def test_get_recipe_command_exists_and_callable(self):
        """Test that get_recipe_command is a callable function."""
        assert callable(get_recipe_command)
        assert hasattr(get_recipe_command, '__call__')
    
    def test_show_success_message_exists_and_callable(self):
        """Test that _show_success_message helper function exists."""
        assert callable(_show_success_message)
        assert hasattr(_show_success_message, '__call__')


class TestGetRecipeCommandRecipeBuilderIntegration:
    """Test get_recipe command integration with RecipeBuilder."""
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_successful_workflow(self, mock_console, mock_ui_class, mock_builder_class):
        """Test successful recipe generation workflow."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder with successful workflow
        mock_selections = {
            'recipe_name': 'my_model',
            'task': 'Classification',
            'model_class': 'RandomForestClassifier',
            'model_library': 'scikit-learn',
            'data_config': {'source_uri': 'data.csv'},
            'preprocessor_steps': ['scaling', 'encoding']
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_recipe_file.return_value = Path("recipes/my_model.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message') as mock_show_success:
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            mock_ui.show_panel.assert_called_once()  # Welcome message
            mock_builder.run_interactive_flow.assert_called_once()
            mock_builder.generate_recipe_file.assert_called_once_with(mock_selections)
            mock_show_success.assert_called_once_with(Path("recipes/my_model.yaml"), mock_selections)
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_recipe_builder_initialization(self, mock_console, mock_ui_class, mock_builder_class):
        """Test that RecipeBuilder is properly initialized."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'recipe_name': 'test'}
        mock_builder.generate_recipe_file.return_value = Path("recipes/test.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message'):
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            mock_builder_class.assert_called_once()  # RecipeBuilder() called
            mock_builder.run_interactive_flow.assert_called_once()
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_console_output(self, mock_console, mock_ui_class, mock_builder_class):
        """Test that appropriate console messages are displayed."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'recipe_name': 'test'}
        mock_builder.generate_recipe_file.return_value = Path("recipes/test.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message'):
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            # Check that progress messages were displayed
            mock_console.print.assert_any_call("\n[dim]대화형 Recipe 생성을 시작합니다...[/dim]\n")
            mock_console.print.assert_any_call("\n[dim]Recipe 파일을 생성하는 중...[/dim]")


class TestGetRecipeCommandErrorHandling:
    """Test get_recipe command error scenarios."""
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    def test_get_recipe_command_keyboard_interrupt(self, mock_ui_class, mock_builder_class):
        """Test handling of KeyboardInterrupt during interactive flow."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder to raise KeyboardInterrupt
        mock_builder = Mock()
        mock_builder.run_interactive_flow.side_effect = KeyboardInterrupt("User interrupted")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0  # KeyboardInterrupt exits with code 0
        mock_ui.show_error.assert_called_once_with("Recipe 생성이 취소되었습니다.")
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_file_not_found_error(self, mock_console, mock_ui_class, mock_builder_class):
        """Test handling of FileNotFoundError during recipe generation."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder to raise FileNotFoundError
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'recipe_name': 'test'}
        mock_builder.generate_recipe_file.side_effect = FileNotFoundError("Template file not found")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_ui.show_error.assert_called_once_with("파일을 찾을 수 없습니다: Template file not found")
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_value_error(self, mock_console, mock_ui_class, mock_builder_class):
        """Test handling of ValueError during recipe generation."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder to raise ValueError
        mock_builder = Mock()
        mock_builder.run_interactive_flow.side_effect = ValueError("Invalid model selection")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_ui.show_error.assert_called_once_with("잘못된 값: Invalid model selection")
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_general_exception(self, mock_console, mock_ui_class, mock_builder_class):
        """Test handling of general exceptions during recipe generation."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder to raise general exception
        mock_builder = Mock()
        mock_builder.run_interactive_flow.side_effect = RuntimeError("Unexpected error")
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_ui.show_error.assert_called_once_with("Recipe 생성 중 오류가 발생했습니다: Unexpected error")
        mock_console.print.assert_called_with("[dim]자세한 오류 정보는 로그를 확인하세요.[/dim]")
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    def test_get_recipe_command_recipe_builder_initialization_error(self, mock_ui_class, mock_builder_class):
        """Test handling of RecipeBuilder initialization error."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui.show_error = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder to raise exception during initialization
        mock_builder_class.side_effect = RuntimeError("RecipeBuilder initialization failed")
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1
        mock_ui.show_error.assert_called_with("Recipe 생성 중 오류가 발생했습니다: RecipeBuilder initialization failed")


class TestGetRecipeCommandSuccessMessage:
    """Test get_recipe command success message functionality."""
    
    @patch('src.cli.commands.get_recipe_command.console')
    @patch('src.cli.commands.get_recipe_command.Panel')
    def test_show_success_message_displays_correct_info(self, mock_panel, mock_console):
        """Test _show_success_message displays correct recipe information."""
        recipe_path = Path("recipes/classification_model.yaml")
        selections = {
            'task': 'Classification',
            'model_class': 'RandomForestClassifier',
            'model_library': 'scikit-learn',
            'recipe_name': 'classification_model'
        }
        
        # Mock Panel constructor
        mock_panel_instance = Mock()
        mock_panel.return_value = mock_panel_instance
        
        # Act
        _show_success_message(recipe_path, selections)
        
        # Assert
        mock_panel.assert_called_once()
        panel_args = mock_panel.call_args
        
        # Check panel content includes recipe information
        panel_content = panel_args[0][0]  # First positional argument (content)
        assert "Classification" in panel_content
        assert "RandomForestClassifier" in panel_content
        assert "scikit-learn" in panel_content
        assert str(recipe_path) in panel_content
        
        # Check panel styling
        panel_kwargs = panel_args[1]  # Keyword arguments
        assert panel_kwargs['title'] == "Recipe Generation Complete"
        assert panel_kwargs['border_style'] == "green"
        
        # Check that panel is printed
        mock_console.print.assert_called_once_with(mock_panel_instance)
    
    @patch('src.cli.commands.get_recipe_command.console')
    def test_show_success_message_includes_usage_examples(self, mock_console):
        """Test _show_success_message includes usage examples."""
        recipe_path = Path("recipes/regression_model.yaml")
        selections = {
            'task': 'Regression',
            'model_class': 'LinearRegression',
            'model_library': 'scikit-learn'
        }
        
        with patch('src.cli.commands.get_recipe_command.Panel') as mock_panel:
            # Act
            _show_success_message(recipe_path, selections)
            
            # Assert
            panel_content = mock_panel.call_args[0][0]
            
            # Check that usage examples are included
            assert "mmp train --recipe-file" in panel_content
            assert str(recipe_path) in panel_content
            assert "mmp train -r" in panel_content
            assert "-e local" in panel_content
            assert "-e dev" in panel_content
            assert "-e prod" in panel_content
    
    @patch('src.cli.commands.get_recipe_command.console')
    def test_show_success_message_different_models(self, mock_console):
        """Test _show_success_message works with different model types."""
        test_cases = [
            {
                'recipe_path': Path("recipes/clustering.yaml"),
                'selections': {
                    'task': 'Clustering',
                    'model_class': 'KMeans',
                    'model_library': 'scikit-learn'
                }
            },
            {
                'recipe_path': Path("recipes/causal.yaml"),
                'selections': {
                    'task': 'Causal',
                    'model_class': 'CausalForest',
                    'model_library': 'econml'
                }
            }
        ]
        
        for case in test_cases:
            mock_console.reset_mock()
            
            with patch('src.cli.commands.get_recipe_command.Panel') as mock_panel:
                # Act
                _show_success_message(case['recipe_path'], case['selections'])
                
                # Assert
                panel_content = mock_panel.call_args[0][0]
                assert case['selections']['task'] in panel_content
                assert case['selections']['model_class'] in panel_content
                assert case['selections']['model_library'] in panel_content


class TestGetRecipeCommandIntegration:
    """Test get_recipe command integration scenarios."""
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    @patch('src.cli.commands.get_recipe_command._show_success_message')
    def test_get_recipe_command_complete_workflow(self, mock_show_success, mock_console, mock_ui_class, mock_builder_class):
        """Test complete workflow from CLI invocation to success message."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock comprehensive recipe selections
        mock_selections = {
            'recipe_name': 'advanced_classification',
            'task': 'Classification',
            'model_class': 'XGBClassifier',
            'model_library': 'xgboost',
            'data_config': {
                'source_uri': 'data/train.parquet',
                'target_column': 'target',
                'entity_schema': ['user_id', 'timestamp']
            },
            'preprocessor_steps': [
                {'name': 'scaling', 'type': 'StandardScaler'},
                {'name': 'encoding', 'type': 'OneHotEncoder'}
            ],
            'evaluation_config': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'cv_folds': 5
            }
        }
        
        recipe_path = Path("recipes/advanced_classification.yaml")
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = mock_selections
        mock_builder.generate_recipe_file.return_value = recipe_path
        mock_builder_class.return_value = mock_builder
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert - verify entire workflow
        assert result.exit_code == 0
        
        # Verify UI interactions
        mock_ui.show_panel.assert_called_once()
        
        # Verify console progress messages
        mock_console.print.assert_any_call("\n[dim]대화형 Recipe 생성을 시작합니다...[/dim]\n")
        mock_console.print.assert_any_call("\n[dim]Recipe 파일을 생성하는 중...[/dim]")
        
        # Verify RecipeBuilder workflow
        mock_builder.run_interactive_flow.assert_called_once()
        mock_builder.generate_recipe_file.assert_called_once_with(mock_selections)
        
        # Verify success message
        mock_show_success.assert_called_once_with(recipe_path, mock_selections)
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_welcome_message_display(self, mock_console, mock_ui_class, mock_builder_class):
        """Test that welcome message is properly displayed."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock RecipeBuilder
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = {'recipe_name': 'test'}
        mock_builder.generate_recipe_file.return_value = Path("recipes/test.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message'):
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            
            # Verify welcome message call
            mock_ui.show_panel.assert_called_once()
            call_args = mock_ui.show_panel.call_args
            
            # Check welcome message content
            welcome_content = call_args[0][0]  # First positional argument
            assert "환경 독립적인 Recipe 생성" in welcome_content
            assert "Recipe는 환경 설정과 분리" in welcome_content
            
            # Check welcome message styling
            call_kwargs = call_args[1]  # Keyword arguments
            assert call_kwargs['title'] == "Recipe Generator"
            assert call_kwargs['style'] == "green"


class TestGetRecipeCommandEdgeCases:
    """Test get_recipe command edge cases and boundary conditions."""
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_minimal_selections(self, mock_console, mock_ui_class, mock_builder_class):
        """Test get_recipe command with minimal recipe selections."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        # Mock minimal selections
        minimal_selections = {
            'recipe_name': 'minimal',
            'task': 'Classification',
            'model_class': 'LogisticRegression',
            'model_library': 'scikit-learn'
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = minimal_selections
        mock_builder.generate_recipe_file.return_value = Path("recipes/minimal.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message') as mock_show_success:
            # Act
            result = runner.invoke(app, [])
            
            # Assert
            assert result.exit_code == 0
            mock_builder.generate_recipe_file.assert_called_once_with(minimal_selections)
            mock_show_success.assert_called_once_with(Path("recipes/minimal.yaml"), minimal_selections)
    
    @patch('src.cli.utils.recipe_builder.RecipeBuilder')
    @patch('src.cli.utils.interactive_ui.InteractiveUI')
    @patch('src.cli.commands.get_recipe_command.console')
    def test_get_recipe_command_long_recipe_name(self, mock_console, mock_ui_class, mock_builder_class):
        """Test get_recipe command with very long recipe name."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(get_recipe_command)
        
        # Mock InteractiveUI
        mock_ui = Mock()
        mock_ui.show_panel = Mock()
        mock_ui_class.return_value = mock_ui
        
        long_recipe_name = "very_long_recipe_name_for_testing_edge_cases_and_boundary_conditions"
        long_selections = {
            'recipe_name': long_recipe_name,
            'task': 'Regression',
            'model_class': 'RandomForestRegressor',
            'model_library': 'scikit-learn'
        }
        
        mock_builder = Mock()
        mock_builder.run_interactive_flow.return_value = long_selections
        mock_builder.generate_recipe_file.return_value = Path(f"recipes/{long_recipe_name}.yaml")
        mock_builder_class.return_value = mock_builder
        
        with patch('src.cli.commands.get_recipe_command._show_success_message'):
            # Act
            result = runner.invoke(app, [])
            
            # Assert - should handle long recipe names without issues
            assert result.exit_code == 0
            mock_builder.generate_recipe_file.assert_called_once_with(long_selections)