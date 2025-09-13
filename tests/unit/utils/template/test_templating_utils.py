"""
Templating utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/template/templating_utils.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.utils.template.templating_utils import (
    render_template_from_file,
    render_template_from_string,
    _validate_context_params,
    _validate_sql_safety,
    is_jinja_template
)


class TestTemplateRendering:
    """템플릿 렌더링 핵심 기능 테스트 - Context 클래스 기반"""

    def test_render_template_from_file_success(self, component_test_context):
        """파일 기반 템플릿 렌더링 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create template file
            template_content = """
            SELECT * FROM users
            WHERE created_date >= '{{ start_date }}'
            AND created_date <= '{{ end_date }}'
            """

            template_path = ctx.temp_dir / "test_template.sql.j2"
            template_path.write_text(template_content)

            context_params = {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            }

            # Test template rendering
            result = render_template_from_file(str(template_path), context_params)

            # Verify rendered result
            assert "'2023-01-01'" in result
            assert "'2023-12-31'" in result
            assert "SELECT * FROM users" in result

    def test_render_template_from_file_not_found(self, component_test_context):
        """존재하지 않는 템플릿 파일 오류 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {'start_date': '2023-01-01'}

            with pytest.raises(FileNotFoundError):
                render_template_from_file("nonexistent_template.sql.j2", context_params)

    def test_render_template_from_string_success(self, component_test_context):
        """문자열 기반 템플릿 렌더링 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            sql_template = """
            SELECT product_id, sales_amount
            FROM sales
            WHERE date = '{{ target_date }}'
            AND period = {{ period }}
            """

            context_params = {
                'target_date': '2023-06-15',
                'period': 12
            }

            result = render_template_from_string(sql_template, context_params)

            # Verify rendered result
            assert "'2023-06-15'" in result
            assert "period = 12" in result
            assert "SELECT product_id, sales_amount" in result

    def test_render_template_strict_undefined_error(self, component_test_context):
        """정의되지 않은 변수 사용 시 오류 테스트"""
        with component_test_context.classification_stack() as ctx:
            sql_template = "SELECT * FROM table WHERE date = '{{ undefined_variable }}'"
            context_params = {'start_date': '2023-01-01'}  # undefined_variable not provided

            # Should raise jinja2.UndefinedError due to StrictUndefined
            with pytest.raises(Exception):  # jinja2.UndefinedError
                render_template_from_string(sql_template, context_params)


class TestContextParameterValidation:
    """컨텍스트 파라미터 검증 테스트"""

    def test_validate_context_params_success(self, component_test_context):
        """허용된 파라미터 검증 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'target_date': '2023-06-15',
                'period': 30,
                'include_target': True
            }

            result = _validate_context_params(context_params)

            # All parameters should pass validation
            assert result == context_params

    def test_validate_context_params_invalid_key(self, component_test_context):
        """허용되지 않는 파라미터 키 오류 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {
                'start_date': '2023-01-01',
                'malicious_param': 'DROP TABLE users;'  # Not in whitelist
            }

            with pytest.raises(ValueError) as exc_info:
                _validate_context_params(context_params)

            assert "보안 위반" in str(exc_info.value)
            assert "malicious_param" in str(exc_info.value)
            assert "허용된 파라미터" in str(exc_info.value)

    def test_validate_context_params_invalid_date_format(self, component_test_context):
        """잘못된 날짜 형식 오류 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {
                'start_date': 'invalid-date-format',  # Invalid date
                'end_date': '2023-12-31'
            }

            with pytest.raises(ValueError) as exc_info:
                _validate_context_params(context_params)

            assert "잘못된 날짜 형식" in str(exc_info.value)
            assert "start_date" in str(exc_info.value)

    def test_validate_context_params_valid_date_formats(self, component_test_context):
        """다양한 유효한 날짜 형식 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31 23:59:59',  # With time
                'target_date': '2023/06/15'  # Different format
            }

            result = _validate_context_params(context_params)

            # All dates should be valid
            assert result == context_params

    def test_validate_context_params_non_date_parameters(self, component_test_context):
        """날짜가 아닌 파라미터 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            context_params = {
                'period': 30,
                'include_target': False
            }

            result = _validate_context_params(context_params)

            # Non-date parameters should pass through without date validation
            assert result == context_params


class TestSQLSafetyValidation:
    """SQL 보안 검증 테스트"""

    def test_validate_sql_safety_safe_query(self, component_test_context):
        """안전한 SQL 쿼리 검증 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            safe_sql = """
            SELECT product_id, sales_amount, created_date
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.created_date >= '2023-01-01'
            ORDER BY s.created_date DESC
            """

            # Should not raise exception
            _validate_sql_safety(safe_sql)

    def test_validate_sql_safety_dangerous_patterns(self, component_test_context):
        """위험한 SQL 패턴 감지 테스트"""
        with component_test_context.classification_stack() as ctx:
            dangerous_patterns = [
                "DROP TABLE users;",
                "DELETE FROM products;",
                "UPDATE users SET password = 'hacked';",
                "INSERT INTO admin VALUES ('hacker');",
                "ALTER TABLE users ADD COLUMN backdoor TEXT;",
                "TRUNCATE TABLE logs;",
                "CREATE TABLE backdoor (id INT);",
                "EXEC sp_executesql @malicious;",
                "SELECT * FROM users; -- comment",
                "SELECT * FROM users /* comment */",
            ]

            for dangerous_sql in dangerous_patterns:
                with pytest.raises(ValueError) as exc_info:
                    _validate_sql_safety(dangerous_sql)

                assert "SQL Injection 패턴 감지" in str(exc_info.value)
                assert "허용되지 않는 SQL 명령어" in str(exc_info.value)

    def test_validate_sql_safety_case_insensitive(self, component_test_context):
        """대소문자 무관 SQL 패턴 감지 테스트"""
        with component_test_context.classification_stack() as ctx:
            mixed_case_dangerous = [
                "drop table users;",
                "Delete FROM products;",
                "uPdAtE users SET password = 'hacked';",
                "Insert Into admin VALUES ('hacker');"
            ]

            for dangerous_sql in mixed_case_dangerous:
                with pytest.raises(ValueError):
                    _validate_sql_safety(dangerous_sql)

    def test_validate_sql_safety_semicolon_detection(self, component_test_context):
        """세미콜론 패턴 감지 테스트 (SQL injection 방지)"""
        with component_test_context.classification_stack() as ctx:
            sql_with_semicolon = "SELECT * FROM users WHERE id = 1; DROP TABLE admin;"

            with pytest.raises(ValueError) as exc_info:
                _validate_sql_safety(sql_with_semicolon)

            assert "SQL Injection 패턴 감지" in str(exc_info.value)
            assert ";" in str(exc_info.value)


class TestJinjaTemplateDetection:
    """Jinja 템플릿 감지 테스트"""

    def test_is_jinja_template_positive_cases(self, component_test_context):
        """Jinja 템플릿 패턴 감지 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            jinja_templates = [
                "SELECT * FROM table WHERE date = '{{ start_date }}'",
                "{% if condition %}SELECT * FROM users{% endif %}",
                "SELECT {{ column_name }} FROM table",
                "{% for item in items %}{{ item }}{% endfor %}",
                "Mix of {{ variable }} and {% statement %}"
            ]

            for template in jinja_templates:
                assert is_jinja_template(template) is True

    def test_is_jinja_template_negative_cases(self, component_test_context):
        """일반 텍스트 (비-Jinja) 감지 테스트"""
        with component_test_context.classification_stack() as ctx:
            non_jinja_texts = [
                "SELECT * FROM table WHERE date = '2023-01-01'",
                "This is plain text without templates",
                "SELECT COUNT(*) FROM users",
                "",  # Empty string
                "Some text with {single braces} but not jinja"
            ]

            for text in non_jinja_texts:
                assert is_jinja_template(text) is False

    def test_is_jinja_template_edge_cases(self, component_test_context):
        """Edge case 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # None input
            assert is_jinja_template(None) is False

            # Empty string
            assert is_jinja_template("") is False

            # Only partial patterns
            assert is_jinja_template("{{") is False  # Incomplete pattern
            assert is_jinja_template("}}") is False  # Incomplete pattern
            assert is_jinja_template("{%") is False  # Incomplete pattern

            # Complete minimal patterns
            assert is_jinja_template("{{}}") is True
            assert is_jinja_template("{%%}") is True


class TestTemplateIntegrationScenarios:
    """템플릿 통합 시나리오 테스트"""

    def test_end_to_end_secure_template_processing(self, component_test_context):
        """End-to-end 보안 템플릿 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create secure template
            template_content = """
            SELECT user_id, order_amount, order_date
            FROM orders
            WHERE order_date >= '{{ start_date }}'
            AND order_date <= '{{ end_date }}'
            {% if include_target %}
            AND target_column IS NOT NULL
            {% endif %}
            ORDER BY order_date DESC
            """

            template_path = ctx.temp_dir / "secure_template.sql.j2"
            template_path.write_text(template_content)

            # Secure context parameters
            context_params = {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'include_target': True
            }

            # Process template
            result = render_template_from_file(str(template_path), context_params)

            # Verify complete processing
            assert "'2023-01-01'" in result
            assert "'2023-12-31'" in result
            assert "target_column IS NOT NULL" in result
            assert "ORDER BY order_date DESC" in result

    def test_security_violation_comprehensive_blocking(self, component_test_context):
        """보안 위반 종합 차단 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Template with injection attempt
            malicious_template = """
            SELECT * FROM users
            WHERE date = '{{ start_date }}'
            ; DROP TABLE admin; --
            """

            template_path = ctx.temp_dir / "malicious_template.sql.j2"
            template_path.write_text(malicious_template)

            context_params = {'start_date': '2023-01-01'}

            # Should block due to SQL injection pattern
            with pytest.raises(ValueError) as exc_info:
                render_template_from_file(str(template_path), context_params)

            assert "SQL Injection 패턴 감지" in str(exc_info.value)

    def test_context_and_sql_validation_chain(self, component_test_context):
        """컨텍스트 검증과 SQL 검증 체인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Valid template but invalid context parameter
            template = "SELECT * FROM table WHERE date = '{{ start_date }}'"

            # Invalid context parameter (not in whitelist)
            invalid_context = {
                'start_date': '2023-01-01',
                'malicious_param': 'DROP TABLE'
            }

            # Should fail at context validation stage
            with pytest.raises(ValueError) as exc_info:
                render_template_from_string(template, invalid_context)

            assert "보안 위반" in str(exc_info.value)
            assert "malicious_param" in str(exc_info.value)

    def test_template_detection_and_processing_workflow(self, component_test_context):
        """템플릿 감지 및 처리 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Text that looks like SQL but has Jinja patterns
            sql_with_jinja = """
            SELECT orders.*, customers.name
            FROM orders
            JOIN customers ON orders.customer_id = customers.id
            WHERE orders.date >= '{{ start_date }}'
            """

            # 1. Detect it's a template
            assert is_jinja_template(sql_with_jinja) is True

            # 2. Process as template
            context = {'start_date': '2023-01-01'}
            result = render_template_from_string(sql_with_jinja, context)

            # 3. Verify processing
            assert "'2023-01-01'" in result
            assert "JOIN customers" in result

            # Text without Jinja patterns should not be processed as template
            plain_sql = "SELECT * FROM orders WHERE date = '2023-01-01'"
            assert is_jinja_template(plain_sql) is False