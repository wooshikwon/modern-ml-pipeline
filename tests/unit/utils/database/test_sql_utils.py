"""
SQL utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/database/sql_utils.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
from unittest.mock import patch

from src.utils.database.sql_utils import (
    prevent_select_star,
    get_selected_columns,
    parse_select_columns,
    parse_feature_columns
)


class TestSelectStarPrevention:
    """SELECT * 방지 기능 테스트 - Context 클래스 기반"""

    def test_prevent_select_star_valid_queries(self, component_test_context):
        """유효한 SQL 쿼리 (SELECT * 없음) 테스트"""
        with component_test_context.classification_stack() as ctx:
            valid_queries = [
                "SELECT user_id, name, email FROM users",
                "SELECT u.id, u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id",
                "SELECT COUNT(*) as count FROM users",  # COUNT(*) should be allowed
                "SELECT MAX(score) as max_score FROM tests",
                "SELECT a, b, c FROM table WHERE condition = 1"
            ]

            for sql in valid_queries:
                # Should not raise exception
                prevent_select_star(sql)

    def test_prevent_select_star_invalid_queries(self, component_test_context):
        """SELECT * 포함된 무효한 SQL 쿼리 테스트"""
        with component_test_context.classification_stack() as ctx:
            invalid_queries = [
                "SELECT * FROM users",
                "SELECT * FROM users WHERE id = 1",
                "SELECT   *   FROM products",  # With extra whitespace
                "select * from orders",  # Lowercase
                "SELECT u.*, p.title FROM users u JOIN posts p ON u.id = p.user_id"
            ]

            for sql in invalid_queries:
                with pytest.raises(ValueError) as exc_info:
                    prevent_select_star(sql)

                assert "SELECT *" in str(exc_info.value)
                assert "재현성을 위해" in str(exc_info.value)

    def test_prevent_select_star_complex_queries(self, component_test_context):
        """복잡한 SQL 쿼리 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Valid complex query
            valid_complex = """
            SELECT
                u.user_id,
                u.name,
                COUNT(o.order_id) as order_count,
                SUM(o.amount) as total_amount
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id
            WHERE u.created_date >= '2023-01-01'
            GROUP BY u.user_id, u.name
            HAVING COUNT(o.order_id) > 0
            ORDER BY total_amount DESC
            """

            # Should not raise exception
            prevent_select_star(valid_complex)

            # Invalid complex query with SELECT *
            invalid_complex = """
            SELECT * FROM (
                SELECT user_id, name FROM users
                WHERE status = 'active'
            ) active_users
            """

            with pytest.raises(ValueError):
                prevent_select_star(invalid_complex)


class TestColumnExtraction:
    """SQL 컬럼 추출 기능 테스트"""

    def test_get_selected_columns_simple_queries(self, component_test_context):
        """단순한 SQL에서 컬럼 추출 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_cases = [
                ("SELECT user_id, name, email FROM users", ["user_id", "name", "email"]),
                ("SELECT id FROM products", ["id"]),
                ("select a, b, c from table", ["a", "b", "c"]),  # Lowercase
                ("SELECT COUNT(*) as count FROM users", ["count"]),  # Aggregate with alias
            ]

            for sql, expected_columns in test_cases:
                result = get_selected_columns(sql)
                assert result == expected_columns

    def test_get_selected_columns_with_aliases(self, component_test_context):
        """Alias 포함된 SQL에서 컬럼 추출 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_cases = [
                ("SELECT user_id as id, name as user_name FROM users", ["id", "user_name"]),
                ("SELECT u.id as user_id, u.name as full_name FROM users u", ["user_id", "full_name"]),
                ("SELECT price * 1.2 as price_with_tax FROM products", ["price_with_tax"]),
                ("SELECT first_name || ' ' || last_name as full_name FROM users", ["full_name"]),
            ]

            for sql, expected_columns in test_cases:
                result = get_selected_columns(sql)
                assert result == expected_columns

    def test_get_selected_columns_with_table_prefixes(self, component_test_context):
        """테이블 prefix가 있는 컬럼 추출 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_cases = [
                ("SELECT u.user_id, u.name FROM users u", ["user_id", "name"]),
                ("SELECT users.id, posts.title FROM users JOIN posts", ["id", "title"]),
                ("SELECT t1.a, t2.b, t3.c FROM table1 t1, table2 t2, table3 t3", ["a", "b", "c"]),
            ]

            for sql, expected_columns in test_cases:
                result = get_selected_columns(sql)
                assert result == expected_columns

    def test_get_selected_columns_complex_queries(self, component_test_context):
        """복잡한 SQL에서 컬럼 추출 테스트"""
        with component_test_context.classification_stack() as ctx:
            complex_sql = """
            SELECT
                u.user_id as id,
                u.name,
                COUNT(o.order_id) as order_count,
                SUM(o.amount) as total_spent,
                AVG(r.rating) as avg_rating
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id
            LEFT JOIN reviews r ON u.user_id = r.user_id
            WHERE u.created_date >= '2023-01-01'
            GROUP BY u.user_id, u.name
            """

            expected_columns = ["id", "name", "order_count", "total_spent", "avg_rating"]
            result = get_selected_columns(complex_sql)
            assert result == expected_columns

    def test_get_selected_columns_empty_result(self, component_test_context):
        """컬럼을 추출할 수 없는 SQL 테스트"""
        with component_test_context.classification_stack() as ctx:
            invalid_queries = [
                "",  # Empty string
                "INSERT INTO users VALUES (1, 'John')",  # Not a SELECT
                "UPDATE users SET name = 'John'",  # Not a SELECT
                "DELETE FROM users WHERE id = 1",  # Not a SELECT
            ]

            for sql in invalid_queries:
                result = get_selected_columns(sql)
                assert result == []


class TestAPISchemaColumnParsing:
    """API 스키마용 컬럼 파싱 테스트"""

    def test_parse_select_columns_api_schema(self, component_test_context):
        """API 스키마용 컬럼 추출 테스트 (시간 컬럼 제외)"""
        with component_test_context.classification_stack() as ctx:
            test_cases = [
                (
                    "SELECT user_id, product_id, event_timestamp FROM events",
                    ["user_id", "product_id"]  # event_timestamp excluded
                ),
                (
                    "SELECT session_id, user_id, created_at, updated_at FROM sessions",
                    ["session_id", "user_id"]  # timestamps excluded
                ),
                (
                    "SELECT order_id, customer_id, order_date FROM orders",
                    ["order_id", "customer_id", "order_date"]  # order_date not excluded
                ),
                (
                    "SELECT id, name, timestamp FROM users",
                    ["id", "name"]  # timestamp excluded
                ),
            ]

            for sql, expected_columns in test_cases:
                result = parse_select_columns(sql)
                assert result == expected_columns

    def test_parse_select_columns_case_insensitive(self, component_test_context):
        """대소문자 무관 시간 컬럼 제외 테스트"""
        with component_test_context.classification_stack() as ctx:
            sql_with_mixed_case = """
            SELECT
                user_id,
                EVENT_TIMESTAMP as event_ts,
                CREATED_AT,
                Updated_At as last_updated
            FROM events
            """

            result = parse_select_columns(sql_with_mixed_case)
            # Only user_id should remain (all timestamp columns excluded)
            assert result == ["user_id"]

    def test_parse_select_columns_error_handling(self, component_test_context):
        """SQL 파싱 오류 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            invalid_sql = "INVALID SQL SYNTAX HERE"

            with patch('src.utils.database.sql_utils.logger') as mock_logger:
                result = parse_select_columns(invalid_sql)

                # Should return empty list on parsing error
                assert result == []
                mock_logger.warning.assert_called_once()


class TestFeatureColumnParsing:
    """피처 컬럼 파싱 테스트"""

    def test_parse_feature_columns_with_common_join_keys(self, component_test_context):
        """일반적인 JOIN 키가 있는 피처 컬럼 파싱 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_cases = [
                (
                    "SELECT user_id, age, income, location FROM user_features",
                    (["user_id", "age", "income", "location"], "user_id")
                ),
                (
                    "SELECT product_id, category, price, rating FROM product_features",
                    (["product_id", "category", "price", "rating"], "product_id")
                ),
                (
                    "SELECT customer_id, segment, lifetime_value FROM customer_features",
                    (["customer_id", "segment", "lifetime_value"], "customer_id")
                ),
                (
                    "SELECT session_id, device_type, duration FROM session_features",
                    (["session_id", "device_type", "duration"], "session_id")
                ),
            ]

            for sql, (expected_columns, expected_join_key) in test_cases:
                columns, join_key = parse_feature_columns(sql)
                assert columns == expected_columns
                assert join_key == expected_join_key

    def test_parse_feature_columns_fallback_join_key(self, component_test_context):
        """일반적인 JOIN 키가 없을 때 첫 번째 컬럼 사용 테스트"""
        with component_test_context.classification_stack() as ctx:
            sql = "SELECT custom_id, feature_a, feature_b, feature_c FROM custom_features"

            columns, join_key = parse_feature_columns(sql)

            assert columns == ["custom_id", "feature_a", "feature_b", "feature_c"]
            assert join_key == "custom_id"  # First column used as fallback

    def test_parse_feature_columns_no_columns(self, component_test_context):
        """컬럼이 없는 경우 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock get_selected_columns to return empty list
            with patch('src.utils.database.sql_utils.get_selected_columns', return_value=[]):
                columns, join_key = parse_feature_columns("SELECT ... FROM table")

                assert columns == []
                assert join_key == ""

    def test_parse_feature_columns_multiple_join_key_patterns(self, component_test_context):
        """여러 JOIN 키 패턴이 있을 때 우선순위 테스트"""
        with component_test_context.classification_stack() as ctx:
            # user_id comes first in the pattern list, so it should be selected
            sql = "SELECT product_id, user_id, member_id FROM mixed_features"

            columns, join_key = parse_feature_columns(sql)

            assert columns == ["product_id", "user_id", "member_id"]
            assert join_key == "user_id"  # First pattern match in priority order

    def test_parse_feature_columns_error_handling(self, component_test_context):
        """SQL 파싱 오류 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            invalid_sql = "COMPLETELY BROKEN SQL"

            with patch('src.utils.database.sql_utils.logger') as mock_logger:
                columns, join_key = parse_feature_columns(invalid_sql)

                # Should return empty results on parsing error
                assert columns == []
                assert join_key == ""
                mock_logger.warning.assert_called_once()


class TestSQLUtilsIntegration:
    """SQL 유틸리티 통합 시나리오 테스트"""

    def test_end_to_end_sql_processing_workflow(self, component_test_context):
        """End-to-end SQL 처리 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Realistic SQL query for ML pipeline
            ml_query = """
            SELECT
                u.user_id,
                u.age,
                u.income,
                u.region,
                f.purchase_count,
                f.avg_order_value,
                f.last_purchase_days_ago,
                u.created_at
            FROM users u
            JOIN user_features f ON u.user_id = f.user_id
            WHERE u.status = 'active'
            """

            # 1. Verify no SELECT * (security check)
            prevent_select_star(ml_query)  # Should not raise

            # 2. Extract all columns
            all_columns = get_selected_columns(ml_query)
            expected_all = [
                "user_id", "age", "income", "region",
                "purchase_count", "avg_order_value", "last_purchase_days_ago", "created_at"
            ]
            assert all_columns == expected_all

            # 3. Parse for API schema (exclude timestamps)
            api_columns = parse_select_columns(ml_query)
            expected_api = [
                "user_id", "age", "income", "region",
                "purchase_count", "avg_order_value", "last_purchase_days_ago"
            ]
            assert api_columns == expected_api

            # 4. Parse for feature columns (get JOIN key)
            feature_columns, join_key = parse_feature_columns(ml_query)
            assert feature_columns == expected_all
            assert join_key == "user_id"

    def test_sql_security_and_parsing_consistency(self, component_test_context):
        """SQL 보안 검사와 파싱 일관성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Secure SQL that should pass all checks
            secure_sql = """
            SELECT
                customer_id as id,
                segment,
                lifetime_value as ltv,
                recency_score,
                frequency_score,
                monetary_score
            FROM customer_segments
            WHERE segment IN ('high_value', 'loyal')
            """

            # 1. Security check should pass
            prevent_select_star(secure_sql)

            # 2. Parsing should work consistently
            columns = get_selected_columns(secure_sql)
            api_columns = parse_select_columns(secure_sql)
            feature_columns, join_key = parse_feature_columns(secure_sql)

            # All parsing should succeed and return consistent results
            expected_columns = ["id", "segment", "ltv", "recency_score", "frequency_score", "monetary_score"]
            assert columns == expected_columns
            assert api_columns == expected_columns  # No timestamp columns to exclude
            assert feature_columns == expected_columns
            assert join_key == "customer_id"  # Should detect customer_id pattern

    def test_sql_processing_error_resilience(self, component_test_context):
        """SQL 처리 과정에서 오류 복원력 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Malformed SQL that might cause parsing issues
            problematic_sqls = [
                "",  # Empty
                "SELECT",  # Incomplete
                "SELECT FROM table",  # Missing columns
                "INVALID SYNTAX",  # Completely invalid
            ]

            for sql in problematic_sqls:
                # Security check might fail, but parsing should handle gracefully
                columns = get_selected_columns(sql)
                api_columns = parse_select_columns(sql)
                feature_columns, join_key = parse_feature_columns(sql)

                # All should return empty/safe results
                assert isinstance(columns, list)
                assert isinstance(api_columns, list)
                assert isinstance(feature_columns, list)
                assert isinstance(join_key, str)