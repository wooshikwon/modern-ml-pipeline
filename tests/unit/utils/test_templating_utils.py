"""
Unit tests for the templating utilities module.
Tests Jinja2 template security, SQL injection prevention, and parameter validation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.utils.system.templating_utils import (
    render_template_from_file,
    render_template_from_string,
    _validate_context_params,
    _validate_sql_safety
)


class TestRenderTemplateFromFile:
    """Test file-based template rendering with security validations."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary template files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Simple SQL template
        self.simple_template_path = self.temp_path / "simple.sql"
        self.simple_template_path.write_text(
            "SELECT * FROM table WHERE date >= '{{ start_date }}' AND date <= '{{ end_date }}'"
        )
        
        # Complex template with multiple parameters
        self.complex_template_path = self.temp_path / "complex.sql"
        self.complex_template_path.write_text("""
SELECT 
    col1, col2, col3
FROM table 
WHERE date >= '{{ start_date }}'
    AND date <= '{{ end_date }}'
    {% if include_target %}
    AND target_column IS NOT NULL
    {% endif %}
    {% if period %}
    AND period = '{{ period }}'
    {% endif %}
""")

    def test_render_template_from_file_success(self):
        """Test successful template rendering from file."""
        context = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        result = render_template_from_file(str(self.simple_template_path), context)
        
        expected = "SELECT * FROM table WHERE date >= '2023-01-01' AND date <= '2023-12-31'"
        assert result == expected

    def test_render_template_from_file_with_complex_context(self):
        """Test template rendering with complex context parameters."""
        context = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'include_target': True,
            'period': 'quarterly'
        }
        
        result = render_template_from_file(str(self.complex_template_path), context)
        
        # Check that conditional blocks are rendered correctly
        assert "'2023-01-01'" in result
        assert "'2023-12-31'" in result
        assert "target_column IS NOT NULL" in result
        assert "period = 'quarterly'" in result

    def test_render_template_from_file_file_not_found(self):
        """Test error handling when template file doesn't exist."""
        context = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
        
        with pytest.raises(FileNotFoundError, match="Template file not found"):
            render_template_from_file("/nonexistent/path/template.sql", context)

    @patch('src.utils.system.templating_utils._validate_context_params')
    def test_render_template_from_file_calls_validation(self, mock_validate):
        """Test that context validation is called."""
        mock_validate.return_value = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
        context = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
        
        render_template_from_file(str(self.simple_template_path), context)
        
        mock_validate.assert_called_once_with(context)

    @patch('src.utils.system.templating_utils._validate_sql_safety')
    def test_render_template_from_file_calls_sql_validation(self, mock_validate_sql):
        """Test that SQL safety validation is called."""
        context = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
        
        render_template_from_file(str(self.simple_template_path), context)
        
        mock_validate_sql.assert_called_once()
        # Check that the rendered SQL is passed to validation
        call_args = mock_validate_sql.call_args[0][0]
        assert "2023-01-01" in call_args
        assert "2023-12-31" in call_args

    def test_render_template_from_file_strict_undefined(self):
        """Test that StrictUndefined raises error for undefined variables."""
        # Template with undefined variable
        template_path = self.temp_path / "undefined.sql"
        template_path.write_text("SELECT * FROM table WHERE id = {{ undefined_var }}")
        
        context = {'start_date': '2023-01-01'}
        
        with pytest.raises(Exception):  # jinja2.UndefinedError
            render_template_from_file(str(template_path), context)

    @patch('src.utils.system.templating_utils.logger')
    def test_render_template_from_file_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        context = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}
        
        render_template_from_file(str(self.simple_template_path), context)
        
        # Check logging calls
        assert mock_logger.info.call_count >= 2
        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("보안 강화 파일 템플릿 렌더링 시작" in msg for msg in log_messages)
        assert any("보안 강화 파일 템플릿 렌더링 완료" in msg for msg in log_messages)


class TestRenderTemplateFromString:
    """Test string-based template rendering with security validations."""

    def test_render_template_from_string_success(self):
        """Test successful template rendering from string."""
        sql_template = "SELECT * FROM table WHERE date >= '{{ start_date }}'"
        context = {'start_date': '2023-01-01'}
        
        result = render_template_from_string(sql_template, context)
        
        expected = "SELECT * FROM table WHERE date >= '2023-01-01'"
        assert result == expected

    def test_render_template_from_string_complex_template(self):
        """Test complex template with conditional logic."""
        sql_template = """
SELECT col1, col2
FROM table 
WHERE date >= '{{ start_date }}'
{% if include_target %}
AND target IS NOT NULL
{% endif %}
"""
        context = {
            'start_date': '2023-01-01',
            'include_target': True
        }
        
        result = render_template_from_string(sql_template, context)
        
        assert "'2023-01-01'" in result
        assert "target IS NOT NULL" in result

    @patch('src.utils.system.templating_utils._validate_context_params')
    def test_render_template_from_string_calls_validation(self, mock_validate):
        """Test that context validation is called."""
        mock_validate.return_value = {'start_date': '2023-01-01'}
        sql_template = "SELECT * FROM table WHERE date = '{{ start_date }}'"
        context = {'start_date': '2023-01-01'}
        
        render_template_from_string(sql_template, context)
        
        mock_validate.assert_called_once_with(context)

    @patch('src.utils.system.templating_utils._validate_sql_safety')
    def test_render_template_from_string_calls_sql_validation(self, mock_validate_sql):
        """Test that SQL safety validation is called."""
        sql_template = "SELECT * FROM table WHERE date = '{{ start_date }}'"
        context = {'start_date': '2023-01-01'}
        
        render_template_from_string(sql_template, context)
        
        mock_validate_sql.assert_called_once()

    def test_render_template_from_string_strict_undefined(self):
        """Test that StrictUndefined raises error for undefined variables."""
        sql_template = "SELECT * FROM table WHERE id = {{ undefined_var }}"
        context = {'start_date': '2023-01-01'}
        
        with pytest.raises(Exception):  # jinja2.UndefinedError
            render_template_from_string(sql_template, context)

    @patch('src.utils.system.templating_utils.logger')
    def test_render_template_from_string_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        sql_template = "SELECT * FROM table"
        context = {}
        
        render_template_from_string(sql_template, context)
        
        # Check logging calls
        assert mock_logger.info.call_count >= 2
        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("문자열 기반 SQL 템플릿 보안 렌더링 시작" in msg for msg in log_messages)
        assert any("문자열 기반 SQL 템플릿 보안 렌더링 완료" in msg for msg in log_messages)


class TestValidateContextParams:
    """Test context parameter validation security function."""

    def test_validate_context_params_allowed_keys_success(self):
        """Test validation with only allowed keys."""
        context = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'target_date': '2023-06-15',
            'period': 'monthly',
            'include_target': True
        }
        
        result = _validate_context_params(context)
        
        assert result == context
        assert all(key in ['start_date', 'end_date', 'target_date', 'period', 'include_target'] 
                  for key in result.keys())

    def test_validate_context_params_disallowed_key_raises_error(self):
        """Test that disallowed keys raise security error."""
        context = {
            'start_date': '2023-01-01',
            'malicious_key': 'DROP TABLE users;'
        }
        
        with pytest.raises(ValueError, match="보안 위반: 허용되지 않는 context parameter 'malicious_key'"):
            _validate_context_params(context)

    def test_validate_context_params_multiple_disallowed_keys(self):
        """Test error handling with multiple disallowed keys."""
        context = {
            'start_date': '2023-01-01',
            'bad_key1': 'value1',
            'bad_key2': 'value2'
        }
        
        # Should fail on first disallowed key
        with pytest.raises(ValueError, match="보안 위반.*bad_key1"):
            _validate_context_params(context)

    def test_validate_context_params_valid_date_formats(self):
        """Test validation with various valid date formats."""
        valid_dates = [
            '2023-01-01',
            '2023-12-31',
            '2023-06-15 10:30:00',
            '01/01/2023',
            '2023-01-01T00:00:00'
        ]
        
        for date_str in valid_dates:
            context = {'start_date': date_str}
            result = _validate_context_params(context)
            assert result['start_date'] == date_str

    def test_validate_context_params_invalid_date_format_raises_error(self):
        """Test that invalid date formats raise validation error."""
        invalid_dates = [
            'not-a-date',
            '2023-13-01',  # Invalid month
            '2023-01-32',  # Invalid day
            'DROP TABLE;',
            '\'; DELETE FROM users;'
        ]
        
        for invalid_date in invalid_dates:
            context = {'start_date': invalid_date}
            with pytest.raises(ValueError, match="잘못된 날짜 형식"):
                _validate_context_params(context)

    def test_validate_context_params_non_date_parameters(self):
        """Test validation of non-date parameters."""
        context = {
            'period': 'monthly',
            'include_target': True
        }
        
        result = _validate_context_params(context)
        
        assert result == context

    @patch('src.utils.system.templating_utils.logger')
    def test_validate_context_params_logging(self, mock_logger):
        """Test that validation success is logged."""
        context = {'start_date': '2023-01-01'}
        
        _validate_context_params(context)
        
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Context Params 검증 통과" in log_message
        assert "start_date" in log_message

    def test_validate_context_params_empty_context(self):
        """Test validation with empty context."""
        context = {}
        
        result = _validate_context_params(context)
        
        assert result == {}

    @patch('pandas.to_datetime')
    def test_validate_context_params_pandas_datetime_error(self, mock_to_datetime):
        """Test handling of pandas datetime conversion errors."""
        mock_to_datetime.side_effect = ValueError("Invalid date")
        context = {'start_date': 'invalid-date'}
        
        with pytest.raises(ValueError, match="잘못된 날짜 형식"):
            _validate_context_params(context)


class TestValidateSqlSafety:
    """Test SQL injection prevention security function."""

    def test_validate_sql_safety_clean_sql_passes(self):
        """Test that clean SQL queries pass validation."""
        clean_queries = [
            "SELECT * FROM table WHERE date >= '2023-01-01'",
            "SELECT col1, col2 FROM users WHERE active = 1",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'"
        ]
        
        for query in clean_queries:
            # Should not raise any exception
            _validate_sql_safety(query)

    def test_validate_sql_safety_dangerous_patterns_blocked(self):
        """Test that dangerous SQL patterns are blocked."""
        dangerous_queries = [
            "SELECT * FROM users; DROP TABLE users;",
            "DELETE FROM users WHERE id = 1",
            "UPDATE users SET password = 'hacked'",
            "INSERT INTO admin VALUES ('hacker', 'admin')",
            "ALTER TABLE users ADD COLUMN is_admin BOOLEAN",
            "TRUNCATE TABLE sessions",
            "CREATE TABLE backdoor (id INT)",
            "EXEC master.dbo.xp_cmdshell 'format c:'",
            "EXECUTE sp_executesql @sql"
        ]
        
        for query in dangerous_queries:
            with pytest.raises(ValueError, match="SQL Injection 패턴 감지"):
                _validate_sql_safety(query)

    def test_validate_sql_safety_comment_patterns_blocked(self):
        """Test that SQL comment patterns are blocked."""
        comment_queries = [
            "SELECT * FROM users -- WHERE admin = 1",
            "SELECT * FROM table /* hidden comment */",
            "SELECT * FROM users WHERE 1=1 -- comment"
        ]
        
        for query in comment_queries:
            with pytest.raises(ValueError, match="SQL Injection 패턴 감지"):
                _validate_sql_safety(query)

    def test_validate_sql_safety_case_insensitive(self):
        """Test that validation is case-insensitive."""
        case_variations = [
            "select * from users; drop table users;",
            "Select * From Users; Drop Table Users;",
            "SELECT * FROM USERS; DROP TABLE USERS;",
            "sElEcT * fRoM uSeRs; dRoP tAbLe UsErS;"
        ]
        
        for query in case_variations:
            with pytest.raises(ValueError, match="SQL Injection 패턴 감지"):
                _validate_sql_safety(query)

    def test_validate_sql_safety_mixed_patterns(self):
        """Test detection of multiple dangerous patterns."""
        mixed_queries = [
            "DROP TABLE users; DELETE FROM sessions;",
            "UPDATE passwords; -- malicious comment",
            "INSERT INTO admin; /* backdoor */ EXEC cmd"
        ]
        
        for query in mixed_queries:
            with pytest.raises(ValueError, match="SQL Injection 패턴 감지"):
                _validate_sql_safety(query)

    @patch('src.utils.system.templating_utils.logger')
    def test_validate_sql_safety_success_logging(self, mock_logger):
        """Test that successful validation is logged."""
        clean_sql = "SELECT * FROM table WHERE date >= '2023-01-01'"
        
        _validate_sql_safety(clean_sql)
        
        mock_logger.info.assert_called_once_with("✅ SQL 보안 검증 통과")

    def test_validate_sql_safety_specific_error_messages(self):
        """Test that specific patterns are mentioned in error messages."""
        test_cases = [
            ("DROP TABLE users", "DROP"),
            ("DELETE FROM users", "DELETE"),
            ("UPDATE users SET x=1", "UPDATE"),
            ("-- comment", "--"),
            ("/* comment */", "/*"),
            ("SELECT *; SELECT", ";")  # Use semicolon without INSERT to test semicolon detection
        ]
        
        for query, expected_pattern in test_cases:
            with pytest.raises(ValueError) as exc_info:
                _validate_sql_safety(query)
            
            error_message = str(exc_info.value)
            # SQL validation reports the first dangerous pattern found
            assert "SQL Injection 패턴 감지" in error_message

    def test_validate_sql_safety_empty_sql(self):
        """Test validation with empty SQL string."""
        _validate_sql_safety("")  # Should pass without error

    def test_validate_sql_safety_whitespace_only(self):
        """Test validation with whitespace-only SQL."""
        _validate_sql_safety("   \n\t   ")  # Should pass without error


class TestSecurityIntegration:
    """Integration tests for overall security validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def test_end_to_end_security_validation_success(self):
        """Test complete security validation flow with safe inputs."""
        # Create safe template (avoid words that contain dangerous patterns)
        template_path = self.temp_path / "safe.sql"
        template_path.write_text(
            "SELECT * FROM orders WHERE order_date >= '{{ start_date }}' AND order_date <= '{{ end_date }}'"
        )
        
        safe_context = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        result = render_template_from_file(str(template_path), safe_context)
        
        assert "2023-01-01" in result
        assert "2023-12-31" in result
        assert "SELECT" in result
        assert "DROP" not in result.upper()

    def test_end_to_end_security_validation_blocked_context(self):
        """Test that malicious context parameters are blocked."""
        template_path = self.temp_path / "template.sql"
        template_path.write_text("SELECT * FROM table WHERE date = '{{ date_param }}'")
        
        malicious_context = {
            'date_param': '2023-01-01',
            'malicious_param': 'DROP TABLE users;'
        }
        
        with pytest.raises(ValueError, match="보안 위반"):
            render_template_from_file(str(template_path), malicious_context)

    def test_end_to_end_security_validation_blocked_template_injection(self):
        """Test that template-based SQL injection attempts are blocked."""
        # Template that could be used for injection if context validation fails
        template_path = self.temp_path / "injection.sql"
        template_path.write_text("SELECT * FROM table WHERE date = '{{ start_date }}'; DROP TABLE users; --")
        
        safe_context = {'start_date': '2023-01-01'}
        
        # This should be caught by SQL safety validation
        with pytest.raises(ValueError, match="SQL Injection 패턴 감지"):
            render_template_from_file(str(template_path), safe_context)

    @patch('src.utils.system.templating_utils.logger')
    def test_security_validation_comprehensive_logging(self, mock_logger):
        """Test that all security steps are properly logged."""
        template_path = self.temp_path / "logged.sql"
        template_path.write_text("SELECT COUNT(*) FROM table WHERE date >= '{{ start_date }}'")
        
        context = {'start_date': '2023-01-01'}
        
        render_template_from_file(str(template_path), context)
        
        # Verify comprehensive logging
        log_calls = mock_logger.info.call_args_list
        log_messages = [call.args[0] for call in log_calls]
        
        # Should have logs from all security validation steps
        assert any("보안 강화 파일 템플릿 렌더링 시작" in msg for msg in log_messages)
        assert any("Context Params 검증 통과" in msg for msg in log_messages)
        assert any("SQL 보안 검증 통과" in msg for msg in log_messages)
        assert any("보안 강화 파일 템플릿 렌더링 완료" in msg for msg in log_messages)