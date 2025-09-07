"""
Unit tests for SQL utilities module.
Tests SQL parsing, security validation, and column extraction functionality.
"""

import pytest
from unittest.mock import patch
from typing import List

from src.utils.system.sql_utils import (
    prevent_select_star,
    get_selected_columns, 
    parse_select_columns,
    parse_feature_columns
)


class TestPreventSelectStar:
    """Test prevent_select_star function for SQL security."""

    def test_prevent_select_star_valid_queries(self):
        """Test that valid queries without SELECT * pass validation."""
        valid_queries = [
            "SELECT id, name, email FROM users",
            "SELECT u.id, u.name FROM users u",
            "SELECT id FROM products WHERE active = 1",
            "SELECT COUNT(*) FROM orders",  # COUNT(*) is allowed
            "SELECT id, name, price * quantity AS total FROM items",  # Arithmetic * is allowed
            "SELECT user_id, product_id FROM purchases",
            """
            SELECT 
                id, 
                name,
                email
            FROM users
            WHERE active = 1
            """
        ]
        
        for query in valid_queries:
            # Should not raise any exception
            prevent_select_star(query)

    def test_prevent_select_star_blocked_queries(self):
        """Test that SELECT * queries are blocked."""
        blocked_queries = [
            "SELECT * FROM users",
            "select * from products",
            "SELECT   *   FROM orders",  # With extra spaces
            """
            SELECT *
            FROM customers
            """,
            "SELECT * FROM users WHERE id = 1"
        ]
        
        for query in blocked_queries:
            with pytest.raises(ValueError, match="SQL loader에서.*SELECT.*사용은 금지됩니다"):
                prevent_select_star(query)

    def test_prevent_select_star_error_message_details(self):
        """Test that error message contains helpful information."""
        with pytest.raises(ValueError) as exc_info:
            prevent_select_star("SELECT * FROM test_table")
        
        error_message = str(exc_info.value)
        assert "SELECT *" in error_message
        assert "재현성을 위해" in error_message
        assert "명시적으로 지정" in error_message

    def test_prevent_select_star_complex_queries(self):
        """Test complex queries with various SQL constructs."""
        complex_valid_queries = [
            "SELECT u.id, p.name FROM users u JOIN profiles p ON u.id = p.user_id",
            "SELECT id, (price * quantity) AS total FROM orders",
            "SELECT DISTINCT category FROM products",
            "SELECT id FROM users UNION SELECT id FROM customers",
            "SELECT COUNT(id), AVG(price) FROM products GROUP BY category"
        ]
        
        for query in complex_valid_queries:
            prevent_select_star(query)  # Should not raise

    def test_prevent_select_star_case_insensitive(self):
        """Test that validation is case insensitive."""
        case_variations = [
            "select * from users",
            "Select * From users", 
            "SELECT * FROM USERS",
            "sElEcT * fRoM users"
        ]
        
        for query in case_variations:
            with pytest.raises(ValueError):
                prevent_select_star(query)


class TestGetSelectedColumns:
    """Test get_selected_columns function for column extraction."""

    def test_get_selected_columns_simple_select(self):
        """Test column extraction from simple SELECT statements."""
        test_cases = [
            ("SELECT id, name, email FROM users", ["id", "name", "email"]),
            ("SELECT user_id, product_id FROM orders", ["user_id", "product_id"]),
            ("SELECT id FROM customers", ["id"])
        ]
        
        for query, expected_columns in test_cases:
            result = get_selected_columns(query)
            assert result == expected_columns

    def test_get_selected_columns_with_aliases(self):
        """Test column extraction with aliases."""
        test_cases = [
            ("SELECT id, name AS full_name FROM users", ["id", "full_name"]),
            ("SELECT u.id, u.name AS user_name FROM users u", ["id", "user_name"]),
            ("SELECT price * quantity AS total FROM orders", ["total"]),
            ("SELECT id AS user_id, name AS full_name FROM users", ["user_id", "full_name"])
        ]
        
        for query, expected_columns in test_cases:
            result = get_selected_columns(query)
            assert result == expected_columns

    def test_get_selected_columns_with_table_prefixes(self):
        """Test column extraction with table prefixes."""
        test_cases = [
            ("SELECT u.id, u.name FROM users u", ["id", "name"]),
            ("SELECT users.id, profiles.bio FROM users JOIN profiles", ["id", "bio"]),
            ("SELECT t1.a, t2.b FROM table1 t1 JOIN table2 t2", ["a", "b"])
        ]
        
        for query, expected_columns in test_cases:
            result = get_selected_columns(query)
            assert result == expected_columns

    def test_get_selected_columns_complex_expressions(self):
        """Test column extraction with complex expressions."""
        test_cases = [
            ("SELECT DISTINCT category FROM products", ["category"]),
            ("SELECT name FROM users", ["name"])
        ]
        
        for query, expected_columns in test_cases:
            result = get_selected_columns(query)
            assert result == expected_columns

    def test_get_selected_columns_multiline_queries(self):
        """Test column extraction from multiline SQL queries."""
        multiline_query = """
        SELECT 
            id,
            name,
            email,
            created_at
        FROM users
        WHERE active = 1
        """
        
        expected_columns = ["id", "name", "email", "created_at"]
        result = get_selected_columns(multiline_query)
        assert result == expected_columns

    @patch('src.utils.system.sql_utils.logger')
    def test_get_selected_columns_logging(self, mock_logger):
        """Test that column extraction is logged."""
        query = "SELECT id, name FROM users"
        result = get_selected_columns(query)
        
        mock_logger.info.assert_called_with("SQL에서 2개 컬럼 추출: ['id', 'name']")

    def test_get_selected_columns_empty_or_invalid(self):
        """Test column extraction with empty or invalid queries."""
        non_select_queries = [
            "INSERT INTO users VALUES (1, 'test')",  # Not a SELECT
            "UPDATE users SET name = 'test'",  # Not a SELECT
            "DELETE FROM users WHERE id = 1"  # Not a SELECT
        ]
        
        for query in non_select_queries:
            result = get_selected_columns(query)
            assert result == []
        
        # Empty query causes IndexError in current implementation
        with pytest.raises(IndexError):
            get_selected_columns("")


class TestParseSelectColumns:
    """Test parse_select_columns function for API schema column extraction."""

    def test_parse_select_columns_basic_functionality(self):
        """Test basic API column extraction functionality."""
        test_cases = [
            ("SELECT user_id, product_id, session_id FROM table", ["user_id", "product_id", "session_id"]),
            ("SELECT id, name, email FROM users", ["id", "name", "email"]),
            ("SELECT customer_id, order_id FROM orders", ["customer_id", "order_id"])
        ]
        
        for query, expected_columns in test_cases:
            result = parse_select_columns(query)
            assert result == expected_columns

    def test_parse_select_columns_excludes_timestamp_columns(self):
        """Test that timestamp-related columns are excluded from API schema."""
        timestamp_queries = [
            ("SELECT user_id, product_id, event_timestamp FROM events", ["user_id", "product_id"]),
            ("SELECT id, name, created_at FROM users", ["id", "name"]),
            ("SELECT order_id, customer_id, timestamp FROM orders", ["order_id", "customer_id"]),
            ("SELECT id, updated_at, email FROM profiles", ["id", "email"])
        ]
        
        for query, expected_columns in timestamp_queries:
            result = parse_select_columns(query)
            assert result == expected_columns

    def test_parse_select_columns_case_insensitive_exclusion(self):
        """Test that timestamp exclusion is case insensitive."""
        case_variations = [
            "SELECT user_id, EVENT_TIMESTAMP FROM events",
            "SELECT id, Created_At FROM users", 
            "SELECT order_id, UPDATED_AT FROM orders"
        ]
        
        for query in case_variations:
            result = parse_select_columns(query)
            # Should exclude timestamp columns regardless of case
            assert "EVENT_TIMESTAMP" not in [col.upper() for col in result]

    def test_parse_select_columns_with_aliases(self):
        """Test API column extraction with column aliases."""
        aliased_queries = [
            ("SELECT user_id AS uid, product_id AS pid, timestamp FROM events", ["uid", "pid"]),
            ("SELECT id, name AS full_name, created_at FROM users", ["id", "full_name"])
        ]
        
        for query, expected_columns in aliased_queries:
            result = parse_select_columns(query)
            assert result == expected_columns

    @patch('src.utils.system.sql_utils.logger')
    def test_parse_select_columns_success_logging(self, mock_logger):
        """Test that successful parsing is logged."""
        query = "SELECT user_id, product_id FROM table"
        result = parse_select_columns(query)
        
        mock_logger.info.assert_called_with("API 스키마용 컬럼 추출 완료: ['user_id', 'product_id']")

    @patch('src.utils.system.sql_utils.logger')
    @patch('src.utils.system.sql_utils.get_selected_columns')
    def test_parse_select_columns_error_handling(self, mock_get_columns, mock_logger):
        """Test error handling when SQL parsing fails."""
        mock_get_columns.side_effect = Exception("SQL parsing failed")
        
        result = parse_select_columns("INVALID SQL")
        
        assert result == []
        mock_logger.warning.assert_called_with("SQL 파싱 실패, 빈 목록 반환: SQL parsing failed")

    def test_parse_select_columns_empty_result(self):
        """Test handling when no valid columns are found."""
        queries_with_only_timestamps = [
            "SELECT event_timestamp FROM events",
            "SELECT created_at, updated_at FROM users"
        ]
        
        for query in queries_with_only_timestamps:
            result = parse_select_columns(query)
            assert result == []


class TestParseFeatureColumns:
    """Test parse_feature_columns function for Feature Store analysis."""

    def test_parse_feature_columns_basic_functionality(self):
        """Test basic feature column and JOIN key extraction."""
        test_cases = [
            ("SELECT user_id, feature1, feature2 FROM features", (["user_id", "feature1", "feature2"], "user_id")),
            ("SELECT product_id, price, category FROM products", (["product_id", "price", "category"], "product_id")),
            ("SELECT customer_id, age, income FROM demographics", (["customer_id", "age", "income"], "customer_id"))
        ]
        
        for query, expected_result in test_cases:
            result = parse_feature_columns(query)
            assert result == expected_result

    def test_parse_feature_columns_join_key_patterns(self):
        """Test JOIN key identification with common patterns."""
        join_key_patterns = [
            ("SELECT user_id, feature1 FROM table", "user_id"),
            ("SELECT member_id, feature2 FROM table", "member_id"), 
            ("SELECT customer_id, feature3 FROM table", "customer_id"),
            ("SELECT product_id, feature4 FROM table", "product_id"),
            ("SELECT session_id, feature5 FROM table", "session_id")
        ]
        
        for query, expected_join_key in join_key_patterns:
            columns, join_key = parse_feature_columns(query)
            assert join_key == expected_join_key

    def test_parse_feature_columns_fallback_join_key(self):
        """Test fallback to first column as JOIN key when no pattern matches."""
        query = "SELECT account_id, transaction_id, amount FROM transactions"
        columns, join_key = parse_feature_columns(query)
        
        assert columns == ["account_id", "transaction_id", "amount"]
        assert join_key == "account_id"  # First column as fallback

    def test_parse_feature_columns_multiple_join_keys(self):
        """Test behavior when multiple JOIN key patterns are present."""
        query = "SELECT user_id, customer_id, product_id, feature1 FROM multi_key_table"
        columns, join_key = parse_feature_columns(query)
        
        # Should pick the first matching pattern (user_id)
        assert join_key == "user_id"
        assert len(columns) == 4

    def test_parse_feature_columns_with_aliases(self):
        """Test feature column parsing with aliases."""
        query = "SELECT user_id AS uid, feature1 AS f1, feature2 FROM features"
        columns, join_key = parse_feature_columns(query)
        
        assert columns == ["uid", "f1", "feature2"]
        assert join_key == "uid"  # Alias should be used for JOIN key

    def test_parse_feature_columns_complex_query(self):
        """Test feature parsing with complex SQL queries."""
        complex_query = """
        SELECT 
            u.user_id,
            u.age,
            p.category,
            AVG(o.amount) AS avg_order
        FROM users u
        JOIN products p ON u.id = p.user_id
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.user_id, u.age, p.category
        """
        
        columns, join_key = parse_feature_columns(complex_query)
        
        assert "user_id" in columns
        assert join_key == "user_id"

    @patch('src.utils.system.sql_utils.logger')
    def test_parse_feature_columns_success_logging(self, mock_logger):
        """Test that successful feature parsing is logged."""
        query = "SELECT user_id, feature1, feature2 FROM features"
        columns, join_key = parse_feature_columns(query)
        
        mock_logger.info.assert_called_with("피처 컬럼 분석 완료: 3개, JOIN 키: user_id")

    @patch('src.utils.system.sql_utils.logger')
    @patch('src.utils.system.sql_utils.get_selected_columns')
    def test_parse_feature_columns_error_handling(self, mock_get_columns, mock_logger):
        """Test error handling when fetcher SQL parsing fails."""
        mock_get_columns.side_effect = Exception("Fetcher SQL parsing failed")
        
        columns, join_key = parse_feature_columns("INVALID FETCHER SQL")
        
        assert columns == []
        assert join_key == ""
        mock_logger.warning.assert_called_with("fetcher SQL 파싱 실패: Fetcher SQL parsing failed")

    def test_parse_feature_columns_empty_columns(self):
        """Test handling when no columns are extracted."""
        # Mock a scenario where get_selected_columns returns empty list
        with patch('src.utils.system.sql_utils.get_selected_columns') as mock_get_columns:
            mock_get_columns.return_value = []
            
            columns, join_key = parse_feature_columns("SELECT FROM invalid")
            
            assert columns == []
            assert join_key == ""

    def test_parse_feature_columns_no_matching_join_key_pattern(self):
        """Test JOIN key selection when no standard patterns match."""
        query = "SELECT account_num, balance, status FROM accounts"
        columns, join_key = parse_feature_columns(query)
        
        assert columns == ["account_num", "balance", "status"]
        assert join_key == "account_num"  # Falls back to first column


class TestSqlUtilsIntegration:
    """Integration tests for SQL utilities."""

    def test_sql_security_workflow(self):
        """Test complete SQL security validation workflow."""
        # Test secure query processing pipeline
        secure_query = "SELECT id, name, email FROM users WHERE active = 1"
        
        # 1. Validate no SELECT *
        prevent_select_star(secure_query)  # Should not raise
        
        # 2. Extract columns
        columns = get_selected_columns(secure_query)
        assert columns == ["id", "name", "email"]
        
        # 3. Parse for API schema
        api_columns = parse_select_columns(secure_query)
        assert api_columns == ["id", "name", "email"]

    def test_feature_store_integration_workflow(self):
        """Test Feature Store SQL analysis workflow."""
        fetcher_query = """
        SELECT 
            user_id,
            age,
            income,
            purchase_frequency,
            last_login_days
        FROM user_features
        WHERE user_id IN (SELECT id FROM active_users)
        """
        
        # 1. Validate security
        prevent_select_star(fetcher_query)
        
        # 2. Parse for Feature Store
        columns, join_key = parse_feature_columns(fetcher_query)
        
        assert "user_id" in columns
        assert join_key == "user_id"
        assert len(columns) == 5

    def test_api_schema_generation_workflow(self):
        """Test API schema generation from SQL snapshot."""
        loader_query = """
        SELECT 
            user_id,
            product_id, 
            session_id,
            event_timestamp,
            created_at
        FROM events
        """
        
        # 1. Security validation
        prevent_select_star(loader_query)
        
        # 2. API schema extraction (excluding timestamps)
        api_columns = parse_select_columns(loader_query)
        
        expected_api_columns = ["user_id", "product_id", "session_id"]
        assert api_columns == expected_api_columns

    def test_error_handling_integration(self):
        """Test integrated error handling across all SQL utilities."""
        # Test invalid SQL handling across all functions
        invalid_sql = "INVALID SQL QUERY"
        
        # parse_select_columns should handle errors gracefully
        api_columns = parse_select_columns(invalid_sql)
        assert api_columns == []
        
        # parse_feature_columns should handle errors gracefully
        columns, join_key = parse_feature_columns(invalid_sql)
        assert columns == []
        assert join_key == ""

    @patch('src.utils.system.sql_utils.logger')
    def test_comprehensive_logging_integration(self, mock_logger):
        """Test logging across all SQL utility functions."""
        query = "SELECT user_id, feature1, feature2 FROM features"
        
        # Test logging from all functions
        get_selected_columns(query)
        parse_select_columns(query) 
        parse_feature_columns(query)
        
        # Verify logging occurred from all functions
        log_calls = mock_logger.info.call_args_list
        log_messages = [call.args[0] for call in log_calls]
        
        assert any("컬럼 추출" in msg for msg in log_messages)
        assert any("API 스키마용 컬럼 추출 완료" in msg for msg in log_messages)
        assert any("피처 컬럼 분석 완료" in msg for msg in log_messages)

    def test_blueprint_v17_features(self):
        """Test    specific functionality."""
        # Test    API schema extraction
        blueprint_query = """
        SELECT 
            user_id,
            product_id,
            session_id,
            event_timestamp
        FROM blueprint_events
        """
        
        api_columns = parse_select_columns(blueprint_query)
        assert api_columns == ["user_id", "product_id", "session_id"]
        
        # Test    feature analysis
        feature_columns, join_key = parse_feature_columns(blueprint_query)
        assert join_key == "user_id"
        assert len(feature_columns) == 4