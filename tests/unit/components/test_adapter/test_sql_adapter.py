"""
Unit tests for SqlAdapter.
Tests SQL adapter functionality with multiple database support and security features.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.components.adapter.modules.sql_adapter import SqlAdapter
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings


class TestSqlAdapterInitialization:
    """Test SqlAdapter initialization and engine creation."""
    
    def test_sql_adapter_inherits_base_adapter(self):
        """Test that SqlAdapter properly inherits from BaseAdapter."""
        # Arrange & Act
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        # Assert
        assert isinstance(adapter, BaseAdapter)
        assert isinstance(adapter, SqlAdapter)
    
    @patch('sqlalchemy.create_engine')
    def test_init_creates_engine_successfully(self, mock_create_engine):
        """Test successful engine creation during initialization."""
        # Arrange
        mock_engine = Mock()
        mock_engine.connect.return_value.__enter__ = Mock()
        mock_engine.connect.return_value.__exit__ = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'sql': Mock(config={'connection_uri': 'sqlite:///test.db'})
        }
        
        # Act
        adapter = SqlAdapter(mock_settings)
        
        # Assert
        assert adapter.settings == mock_settings
        assert adapter.engine == mock_engine
        mock_create_engine.assert_called_once()
    
    def test_init_missing_sql_config_raises_error(self):
        """Test initialization fails when SQL config is missing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {}  # No 'sql' config
        
        # Act & Assert
        with pytest.raises(ValueError, match="SQL 어댑터 설정이 누락되었습니다"):
            SqlAdapter(mock_settings)


class TestSqlAdapterConnectionUriParsing:
    """Test connection URI parsing for different database types."""
    
    def test_parse_bigquery_uri(self):
        """Test BigQuery URI parsing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        uri = "bigquery://project/dataset"
        
        # Act
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(uri)
        
        # Assert
        assert db_type == 'bigquery'
        assert processed_uri == uri
        assert 'pool_pre_ping' in engine_kwargs
        assert engine_kwargs['pool_size'] == 5
    
    def test_parse_postgresql_uri(self):
        """Test PostgreSQL URI parsing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        uri = "postgresql://user:pass@localhost/db"
        
        # Act
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(uri)
        
        # Assert
        assert db_type == 'postgresql'
        assert processed_uri == uri
        assert engine_kwargs['pool_size'] == 10
        assert engine_kwargs['max_overflow'] == 20
        assert engine_kwargs['pool_pre_ping'] is True
    
    def test_parse_mysql_uri(self):
        """Test MySQL URI parsing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        uri = "mysql://user:pass@localhost/db"
        
        # Act
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(uri)
        
        # Assert
        assert db_type == 'mysql'
        assert processed_uri == uri
        assert engine_kwargs['pool_size'] == 10
        assert engine_kwargs['pool_recycle'] == 3600
    
    def test_parse_sqlite_uri(self):
        """Test SQLite URI parsing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        uri = "sqlite:///test.db"
        
        # Act
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(uri)
        
        # Assert
        assert db_type == 'sqlite'
        assert processed_uri == uri
        assert 'poolclass' in engine_kwargs
        assert engine_kwargs['connect_args']['check_same_thread'] is False
    
    def test_parse_unknown_uri_scheme(self):
        """Test parsing URI with unknown scheme."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        uri = "unknown://localhost/db"
        
        # Act
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(uri)
        
        # Assert
        assert db_type == 'generic'
        assert processed_uri == uri
        assert engine_kwargs == {}


class TestSqlAdapterSecurityGuards:
    """Test SQL security guard mechanisms."""
    
    def test_enforce_sql_guards_blocks_select_star(self):
        """Test that SELECT * queries are blocked."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        sql_query = "SELECT * FROM users"
        
        # Act & Assert
        with pytest.raises(ValueError, match="SQL loader에서 `SELECT \\*` 사용은 금지됩니다"):
            adapter._enforce_sql_guards(sql_query)
    
    def test_enforce_sql_guards_blocks_dangerous_keywords(self):
        """Test that dangerous SQL keywords are blocked."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        dangerous_queries = [
            "DROP TABLE users",
            "DELETE FROM users",
            "UPDATE users SET password = 'hacked'",
            "INSERT INTO users VALUES ('hacker')",
            "ALTER TABLE users ADD COLUMN hacked INT",
            "TRUNCATE TABLE users",
        ]
        
        # Act & Assert
        for query in dangerous_queries:
            with pytest.raises(ValueError, match="보안 위반: 금지된 SQL 키워드 포함"):
                adapter._enforce_sql_guards(query)
    
    def test_enforce_sql_guards_warns_missing_limit(self):
        """Test warning for queries without LIMIT clause."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        sql_query = "SELECT id, name FROM users WHERE active = 1"
        
        # Act & Assert (should not raise, but should warn)
        with patch('src.components.adapter.modules.sql_adapter.logger') as mock_logger:
            adapter._enforce_sql_guards(sql_query)
            mock_logger.warning.assert_called_with(
                "SQL LIMIT 가드: LIMIT 절이 없습니다. 대용량 쿼리일 수 있습니다."
            )
    
    def test_enforce_sql_guards_allows_safe_query(self):
        """Test that safe queries with LIMIT pass all guards."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        safe_query = "SELECT id, name, email FROM users WHERE active = 1 LIMIT 100"
        
        # Act & Assert (should not raise any exceptions)
        adapter._enforce_sql_guards(safe_query)


class TestSqlAdapterReadOperations:
    """Test SQL read operations."""
    
    @patch('pandas.read_sql_query')
    def test_read_sql_string_success(self, mock_read_sql):
        """Test successful SQL string execution."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        expected_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_read_sql.return_value = expected_df
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        sql_query = "SELECT id, name FROM users LIMIT 10"
        
        # Act
        result = adapter.read(sql_query)
        
        # Assert
        assert result.equals(expected_df)
        mock_read_sql.assert_called_once_with(sql_query, mock_engine)
    
    @patch('pandas.read_sql_query')
    def test_read_sql_file_success(self, mock_read_sql):
        """Test successful SQL file execution."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        expected_df = pd.DataFrame({'count': [42]})
        mock_read_sql.return_value = expected_df
        
        sql_content = "SELECT COUNT(*) as count FROM users LIMIT 1"
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=sql_content):
            
            # Act
            result = adapter.read("query.sql")
        
        # Assert
        assert result.equals(expected_df)
        mock_read_sql.assert_called_once_with(sql_content, mock_engine)
    
    def test_read_sql_file_not_found(self):
        """Test reading non-existent SQL file raises error."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        with patch('pathlib.Path.exists', return_value=False):
            # Act & Assert
            with pytest.raises(FileNotFoundError, match="SQL 파일을 찾을 수 없습니다"):
                adapter.read("nonexistent.sql")
    
    @patch('pandas.read_sql_query')
    def test_read_sql_execution_error(self, mock_read_sql):
        """Test SQL execution error handling."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        mock_read_sql.side_effect = SQLAlchemyError("Syntax error")
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        sql_query = "SELECT id, name FROM users LIMIT 10"
        
        # Act & Assert
        with pytest.raises(SQLAlchemyError, match="Syntax error"):
            adapter.read(sql_query)
    
    def test_read_dangerous_sql_blocked(self):
        """Test that dangerous SQL queries are blocked in read operations."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        dangerous_query = "DROP TABLE users; SELECT * FROM admin"
        
        # Act & Assert
        with pytest.raises(ValueError, match="보안 위반"):
            adapter.read(dangerous_query)


class TestSqlAdapterWriteOperations:
    """Test SQL write operations."""
    
    def test_write_dataframe_success(self):
        """Test successful DataFrame write to database."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        table_name = "test_table"
        
        # Act
        with patch.object(df, 'to_sql') as mock_to_sql:
            adapter.write(df, table_name, if_exists='replace')
        
        # Assert
        mock_to_sql.assert_called_once_with(table_name, mock_engine, if_exists='replace')
    
    def test_write_dataframe_error(self):
        """Test DataFrame write error handling."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        df = pd.DataFrame({'id': [1, 2]})
        
        # Act & Assert
        with patch.object(df, 'to_sql', side_effect=SQLAlchemyError("Write failed")):
            with pytest.raises(SQLAlchemyError, match="Write failed"):
                adapter.write(df, "test_table")


class TestSqlAdapterEngineCreation:
    """Test engine creation with various configurations."""
    
    @patch('sqlalchemy.create_engine')
    def test_create_engine_with_bigquery_credentials(self, mock_create_engine):
        """Test engine creation with BigQuery credentials configuration."""
        # Arrange
        mock_engine = Mock()
        mock_engine.connect.return_value.__enter__ = Mock()
        mock_engine.connect.return_value.__exit__ = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'sql': Mock(config={
                'connection_uri': 'bigquery://project/dataset',
                'credentials_path': '/path/to/credentials.json'
            })
        }
        
        # Act
        with patch.dict('os.environ', {}, clear=True):
            adapter = SqlAdapter(mock_settings)
        
        # Assert
        assert adapter.engine == mock_engine
        mock_create_engine.assert_called_once()
    
    @patch('sqlalchemy.create_engine')
    def test_create_engine_connection_test_failure(self, mock_create_engine):
        """Test engine creation when connection test fails."""
        # Arrange
        mock_engine = Mock()
        mock_engine.connect.side_effect = SQLAlchemyError("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'sql': Mock(config={'connection_uri': 'postgresql://localhost/test'})
        }
        
        # Act & Assert
        with pytest.raises(SQLAlchemyError, match="Connection failed"):
            SqlAdapter(mock_settings)


class TestSqlAdapterIntegration:
    """Test SqlAdapter integration scenarios."""
    
    def test_sql_adapter_with_various_kwargs(self):
        """Test SqlAdapter handles various initialization kwargs."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            # Act
            adapter = SqlAdapter(
                mock_settings, 
                custom_param="test_value",
                timeout=30,
                pool_size=15
            )
        
        # Assert
        assert adapter.settings == mock_settings
        assert hasattr(adapter, 'engine')
    
    @patch('pandas.read_sql_query')
    def test_read_with_additional_pandas_params(self, mock_read_sql):
        """Test read method passes through additional pandas parameters."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_engine = Mock()
        expected_df = pd.DataFrame({'data': [1, 2, 3]})
        mock_read_sql.return_value = expected_df
        
        with patch.object(SqlAdapter, '_create_engine', return_value=mock_engine):
            adapter = SqlAdapter(mock_settings)
        
        sql_query = "SELECT id, data FROM test LIMIT 3"
        
        # Act
        result = adapter.read(
            sql_query, 
            parse_dates=['created_at'],
            chunksize=1000,
            dtype={'id': 'int64'}
        )
        
        # Assert
        assert result.equals(expected_df)
        mock_read_sql.assert_called_once_with(
            sql_query, 
            mock_engine,
            parse_dates=['created_at'],
            chunksize=1000,
            dtype={'id': 'int64'}
        )
    
    def test_sql_adapter_logging_behavior(self):
        """Test that SqlAdapter properly logs operations."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            adapter = SqlAdapter(mock_settings)
        
        with patch('src.components.adapter.modules.sql_adapter.logger') as mock_logger:
            sql_query = "SELECT id FROM users LIMIT 10"
            
            with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
                # Act
                adapter.read(sql_query)
        
        # Assert
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Executing SQL query" in log for log in log_calls)


class TestSqlAdapterSelfRegistration:
    """Test SqlAdapter self-registration mechanism."""
    
    def test_sql_adapter_self_registration(self):
        """Test that SqlAdapter registers itself in AdapterRegistry."""
        # Act - Import triggers self-registration
        from src.components.adapter.modules import sql_adapter
        from src.components.adapter.registry import AdapterRegistry
        
        # Assert
        assert "sql" in AdapterRegistry.adapters
        assert AdapterRegistry.adapters["sql"] == SqlAdapter
        
        # Verify can create instance through registry
        mock_settings = Mock(spec=Settings)
        with patch.object(SqlAdapter, '_create_engine', return_value=Mock()):
            instance = AdapterRegistry.create("sql", mock_settings)
            assert isinstance(instance, SqlAdapter)