"""
Test Suite for Interactive UI
Following Real Object Testing philosophy

Tests cover:
- User input methods (text, number, selection, confirmation)
- Display methods (panels, tables, info messages)
- Input validation
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock, call
from typing import Any, List, Optional

from src.cli.utils.interactive_ui import InteractiveUI


class TestInteractiveUI:
    """Test InteractiveUI class - comprehensive user interaction methods."""

    def setup_method(self):
        """Setup test environment before each test."""
        self.ui = InteractiveUI()

    # Text Input Tests

    def test_text_input_basic(self):
        """Test basic text input functionality."""
        with patch('rich.prompt.Prompt.ask', return_value="test_value"):
            result = self.ui.text_input("Enter text")
            assert result == "test_value"

    def test_text_input_with_default(self):
        """Test text input with default value."""
        with patch('rich.prompt.Prompt.ask', return_value="custom"):
            result = self.ui.text_input("Enter text", default="default_value")
            assert result == "custom"

        # Test when user accepts default (returns empty string)
        with patch('rich.prompt.Prompt.ask', return_value="default_value"):
            result = self.ui.text_input("Enter text", default="default_value")
            assert result == "default_value"

    def test_text_input_password_mode(self):
        """Test text input in password mode."""
        with patch('rich.prompt.Prompt.ask', return_value="secret123") as mock_ask:
            result = self.ui.text_input("Enter password", password=True)
            assert result == "secret123"
            # Verify password parameter was passed
            mock_ask.assert_called_once()

    def test_text_input_with_validation(self):
        """Test text input with custom validator."""
        def validator(value):
            return len(value) >= 5

        # Mock Prompt.ask to simulate validation
        with patch('rich.prompt.Prompt.ask', return_value="valid_input"):
            result = self.ui.text_input("Enter text", validator=validator)
            assert result == "valid_input"

    def test_text_input_no_default_display(self):
        """Test text input without showing default."""
        with patch('rich.prompt.Prompt.ask', return_value="value") as mock_ask:
            result = self.ui.text_input("Enter text", default="default", show_default=False)
            assert result == "value"

    # Number Input Tests

    def test_number_input_integer(self):
        """Test number input for integer values."""
        with patch('rich.prompt.IntPrompt.ask', return_value=42):
            result = self.ui.number_input("Enter number")
            assert result == 42

    def test_number_input_with_default(self):
        """Test number input with default value."""
        with patch('rich.prompt.IntPrompt.ask', return_value=100):
            result = self.ui.number_input("Enter number", default=50)
            assert result == 100

    def test_number_input_with_range(self):
        """Test number input with min/max constraints."""
        with patch('rich.prompt.IntPrompt.ask', return_value=75):
            result = self.ui.number_input("Enter number", min_value=0, max_value=100)
            assert result == 75

    def test_number_input_validation(self):
        """Test number input with range validation."""
        # This would typically validate internally in rich.prompt
        with patch('rich.prompt.IntPrompt.ask', return_value=50):
            result = self.ui.number_input("Enter number", min_value=10, max_value=100)
            assert result == 50

    # Selection Tests

    def test_select_from_list(self):
        """Test selecting from a list of options."""
        options = ["Option 1", "Option 2", "Option 3"]

        with patch('rich.prompt.Prompt.ask', return_value="2"):
            result = self.ui.select_from_list("Select option", options)
            assert result == "Option 2"

    def test_select_from_list_with_cancel(self):
        """Test selecting from list with cancel option."""
        options = ["Option 1", "Option 2"]

        with patch('rich.prompt.Prompt.ask', return_value="0"):
            result = self.ui.select_from_list("Select option", options, allow_cancel=True)
            assert result is None

    def test_select_from_list_no_numbers(self):
        """Test selecting from list without showing numbers."""
        options = ["Option 1", "Option 2"]

        with patch('rich.prompt.Prompt.ask', return_value="1"):
            with patch.object(self.ui.console, 'print') as mock_print:
                result = self.ui.select_from_list("Select", options, show_numbers=False)
                assert result == "Option 1"

    def test_select_from_list_invalid_then_valid(self):
        """Test selection retry on invalid input."""
        options = ["Option 1", "Option 2"]

        # Simulate invalid then valid input
        with patch('rich.prompt.Prompt.ask', side_effect=["5", "1"]):
            with patch.object(self.ui.console, 'print'):
                result = self.ui.select_from_list("Select", options)
                assert result == "Option 1"

    # Confirmation Tests

    def test_confirm_yes(self):
        """Test confirmation with yes answer."""
        with patch('rich.prompt.Confirm.ask', return_value=True):
            result = self.ui.confirm("Continue?")
            assert result is True

    def test_confirm_no(self):
        """Test confirmation with no answer."""
        with patch('rich.prompt.Confirm.ask', return_value=False):
            result = self.ui.confirm("Continue?")
            assert result is False

    def test_confirm_with_default_true(self):
        """Test confirmation with default True."""
        with patch('rich.prompt.Confirm.ask', return_value=True):
            result = self.ui.confirm("Continue?", default=True)
            assert result is True

    def test_confirm_no_default_display(self):
        """Test confirmation without showing default."""
        with patch('rich.prompt.Confirm.ask', return_value=True) as mock_ask:
            result = self.ui.confirm("Continue?", default=True, show_default=False)
            assert result is True

    # Display Method Tests

    def test_show_panel(self):
        """Test showing a panel with title."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_panel("Panel content", "Panel Title")
            mock_print.assert_called_once()

    def test_show_panel_with_style(self):
        """Test showing a panel with custom style."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_panel("Content", "Title", style="bold blue")
            mock_print.assert_called_once()

    def test_show_info(self):
        """Test showing info message."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_info("Information message")
            mock_print.assert_called_once()
            # Check if info style is applied
            call_args = mock_print.call_args
            assert "[cyan]ℹ️  Information message[/cyan]" in str(call_args)

    def test_show_success(self):
        """Test showing success message."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_success("Success message")
            mock_print.assert_called_once()
            call_args = mock_print.call_args
            assert "[green]✅ Success message[/green]" in str(call_args)

    def test_show_warning(self):
        """Test showing warning message."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_warning("Warning message")
            mock_print.assert_called_once()
            call_args = mock_print.call_args
            assert "[yellow]⚠️  Warning message[/yellow]" in str(call_args)

    def test_show_error(self):
        """Test showing error message."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_error("Error message")
            mock_print.assert_called_once()
            call_args = mock_print.call_args
            assert "[red]❌ Error message[/red]" in str(call_args)

    def test_show_table_basic(self):
        """Test showing basic table."""
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "Los Angeles"]
        ]

        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_table(headers, rows)
            mock_print.assert_called_once()

    def test_show_table_with_title(self):
        """Test showing table with title."""
        headers = ["Column1", "Column2"]
        rows = [["Data1", "Data2"]]

        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_table(headers, rows, title="Test Table")
            mock_print.assert_called_once()

    def test_show_table_empty_rows(self):
        """Test showing table with no rows."""
        headers = ["Column1", "Column2"]
        rows = []

        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.show_table(headers, rows)
            mock_print.assert_called_once()

    def test_print_divider(self):
        """Test printing divider."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.print_divider()
            mock_print.assert_called_once()
            call_args = mock_print.call_args
            assert "─" in str(call_args)

    def test_print_divider_with_text(self):
        """Test printing divider with text."""
        with patch.object(self.ui.console, 'print') as mock_print:
            self.ui.print_divider("Section Title")
            mock_print.assert_called_once()

    # Validation Helper Tests

    def test_non_empty_validator(self):
        """Test non-empty string validator."""
        validator = self.ui.non_empty_validator()

        assert validator("valid") is True
        assert validator("  valid  ") is True
        assert validator("") is False
        assert validator("   ") is False

    def test_non_empty_validator_with_text_input(self):
        """Test non-empty validator integrated with text input."""
        validator = self.ui.non_empty_validator()

        with patch('rich.prompt.Prompt.ask', return_value="valid_text"):
            result = self.ui.text_input("Enter text", validator=validator)
            assert result == "valid_text"

    # Edge Cases and Error Handling

    def test_select_from_empty_list(self):
        """Test selecting from an empty list."""
        options = []

        with pytest.raises(ValueError):
            self.ui.select_from_list("Select option", options)

    def test_number_input_invalid_range(self):
        """Test number input with invalid range (min > max)."""
        with pytest.raises(ValueError):
            self.ui.number_input("Enter number", min_value=100, max_value=10)

    def test_text_input_interrupt(self):
        """Test handling keyboard interrupt during text input."""
        with patch('rich.prompt.Prompt.ask', side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                self.ui.text_input("Enter text")

    def test_confirm_interrupt(self):
        """Test handling keyboard interrupt during confirmation."""
        with patch('rich.prompt.Confirm.ask', side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                self.ui.confirm("Continue?")

    # Progress and Status Display Tests

    def test_show_progress_start(self):
        """Test starting progress display."""
        with patch.object(self.ui.console, 'status') as mock_status:
            self.ui.show_progress("Processing...")
            mock_status.assert_called_once()

    def test_show_spinner(self):
        """Test showing spinner for long operations."""
        with patch.object(self.ui.console, 'status') as mock_status:
            with self.ui.show_spinner("Loading..."):
                # Simulate work being done
                pass
            mock_status.assert_called_once()

    # Multi-select Tests

    def test_multi_select_basic(self):
        """Test multi-selection from options."""
        options = ["Option 1", "Option 2", "Option 3"]

        # Simulate selecting options 1 and 3
        with patch('rich.prompt.Prompt.ask', side_effect=["1,3", ""]):
            result = self.ui.multi_select("Select multiple", options)
            assert result == ["Option 1", "Option 3"]

    def test_multi_select_none_selected(self):
        """Test multi-selection with no selections."""
        options = ["Option 1", "Option 2"]

        with patch('rich.prompt.Prompt.ask', return_value=""):
            result = self.ui.multi_select("Select multiple", options)
            assert result == []

    def test_multi_select_all(self):
        """Test selecting all options."""
        options = ["Option 1", "Option 2", "Option 3"]

        with patch('rich.prompt.Prompt.ask', return_value="1,2,3"):
            result = self.ui.multi_select("Select multiple", options)
            assert result == options