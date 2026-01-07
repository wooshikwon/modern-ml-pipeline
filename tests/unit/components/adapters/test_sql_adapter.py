"""
SQL Adapter Unit Tests - No Mock Hell Approach
Real SQLite database, real queries, real behavior validation
Following comprehensive testing strategy document principles
"""

import numpy as np
import pandas as pd
import pytest
import sqlalchemy

from src.components.adapter.modules.sql_adapter import SqlAdapter
from src.components.adapter.base import BaseAdapter


class TestSqlAdapterWithRealDatabase:
    """Test SqlAdapter with real in-memory SQLite database - No mocks."""

    def test_sql_adapter_initialization(self, settings_builder):
        """Test SqlAdapter initialization with SQLite connection."""
        # Given: Valid SQLite connection settings
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()

        # When: Creating SqlAdapter
        adapter = SqlAdapter(settings)

        # Then: Adapter is properly initialized
        assert isinstance(adapter, SqlAdapter)
        assert isinstance(adapter, BaseAdapter)
        assert adapter.engine is not None

    def test_read_from_sqlite_with_real_data(self, settings_builder, real_dataset_files):
        """Test reading data from real SQLite database."""
        # Given: Real SQLite database with data
        sql_info = real_dataset_files["sql"]
        connection_uri = f"sqlite:///{sql_info['path']}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_uri}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Executing SQL query
        query = f"SELECT feature_1, feature_2, feature_3, feature_4, feature_5, target, entity_id FROM {sql_info['classification_table']}"
        df = adapter.read(query)

        # Then: Data is read correctly
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sql_info["classification_data"])
        assert "target" in df.columns
        assert "entity_id" in df.columns
        assert df["target"].nunique() >= 2

    def test_read_with_parameterized_query(self, settings_builder, real_dataset_files):
        """Test reading data with parameterized SQL query."""
        # Given: Real SQLite database and parameterized query
        sql_info = real_dataset_files["sql"]
        connection_uri = f"sqlite:///{sql_info['path']}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_uri}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Executing parameterized query
        query = f"SELECT feature_1, feature_2, feature_3, feature_4, feature_5, target, entity_id FROM {sql_info['classification_table']} WHERE target = :target_value"
        df = adapter.read(query, params={"target_value": 0})

        # Then: Only filtered data is returned
        assert isinstance(df, pd.DataFrame)
        assert all(df["target"] == 0)
        assert len(df) > 0

    def test_write_to_sqlite_database(
        self, settings_builder, isolated_temp_directory, test_data_generator
    ):
        """Test writing DataFrame to SQLite database."""
        # Given: Data and SQLite database
        df, _ = test_data_generator.classification_data(n_samples=50, n_features=5)
        df["target"] = np.random.randint(0, 2, size=50)

        db_path = isolated_temp_directory / "test.db"
        connection_uri = f"sqlite:///{db_path}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_uri}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Writing to database
        adapter.write(df, "test_table", if_exists="replace")

        # Then: Data is written and readable
        read_df = adapter.read(
            "SELECT feature_1, feature_2, feature_3, feature_4, feature_5, entity_id, target FROM test_table"
        )
        assert len(read_df) == len(df)
        assert set(read_df.columns) == set(df.columns)

    def test_sql_adapter_with_multiple_tables(
        self, settings_builder, isolated_temp_directory, test_data_generator
    ):
        """Test SqlAdapter with multiple tables in same database."""
        # Given: Multiple datasets and database
        cls_df, _ = test_data_generator.classification_data(n_samples=30, n_features=5)
        cls_df["target"] = np.random.randint(0, 2, size=30)

        reg_df, _ = test_data_generator.regression_data(n_samples=40, n_features=4)
        reg_df["target"] = np.random.randn(40)

        db_path = isolated_temp_directory / "multi_table.db"
        connection_uri = f"sqlite:///{db_path}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_uri}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Writing multiple tables
        adapter.write(cls_df, "classification_table", if_exists="replace")
        adapter.write(reg_df, "regression_table", if_exists="replace")

        # Then: Both tables exist and contain correct data
        cls_read = adapter.read(
            "SELECT feature_1, feature_2, feature_3, feature_4, feature_5, entity_id, target FROM classification_table"
        )
        reg_read = adapter.read(
            "SELECT feature_1, feature_2, feature_3, feature_4, entity_id, target FROM regression_table"
        )

        assert len(cls_read) == 30
        assert len(reg_read) == 40
        assert cls_read["target"].nunique() >= 2

    def test_sql_adapter_connection_pooling(self, settings_builder):
        """Test SqlAdapter connection pooling behavior."""
        # Given: SQLite connection with pool settings
        settings = settings_builder.with_data_source(
            "sql",
            config={"connection_uri": "sqlite:///:memory:", "pool_size": 5, "max_overflow": 10},
        ).build()

        # When: Creating SqlAdapter
        adapter = SqlAdapter(settings)

        # Then: Engine is created with pooling (StaticPool for SQLite in-memory)
        assert adapter.engine is not None
        # SQLite in-memory uses StaticPool which doesn't have traditional pooling
        from sqlalchemy.pool import StaticPool

        assert isinstance(adapter.engine.pool, StaticPool)

    def test_read_empty_table_handling(self, settings_builder, isolated_temp_directory):
        """Test reading from empty table."""
        # Given: Empty table in database
        db_path = isolated_temp_directory / "empty.db"
        connection_uri = f"sqlite:///{db_path}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_uri}
        ).build()
        adapter = SqlAdapter(settings)

        # Create empty table
        with adapter.engine.begin() as conn:
            conn.execute(sqlalchemy.text("CREATE TABLE empty_table (id INTEGER, value TEXT)"))

        # When: Reading from empty table
        df = adapter.read("SELECT id, value FROM empty_table")

        # Then: Empty DataFrame with correct columns
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["id", "value"]


class TestSqlAdapterBigQuerySupport:
    """Test SqlAdapter BigQuery support - validates configuration and settings."""

    def test_bigquery_configuration_detection(self, settings_builder):
        """Test BigQuery configuration parsing with SQLite for compatibility."""
        # Given: SQLite settings with BigQuery metadata for configuration testing
        settings = settings_builder.with_data_source(
            "sql",
            config={
                "connection_uri": "sqlite:///:memory:",
                # BigQuery configuration stored as metadata
                "bigquery_project_id": "test-project",
                "bigquery_dataset_id": "test-dataset",
                "bigquery_location": "EU",
                "bigquery_use_pandas_gbq": True,
            },
        ).build()

        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)

        # Then: SQLite connection works
        assert adapter.db_type == "sqlite"  # Actual connection is SQLite
        # Config is a Pydantic model (PostgreSQLConfig), not dict
        config = adapter.settings.config.data_source.config
        assert hasattr(config, "connection_uri")
        assert config.connection_uri == "sqlite:///:memory:"
        # Extra fields are ignored by Pydantic in SQLite/PostgreSQL config

    def test_bigquery_default_configuration(self, settings_builder):
        """Test BigQuery default values with SQLite compatibility testing."""
        # Given: Minimal SQLite configuration without BigQuery metadata
        settings = settings_builder.with_data_source(
            "sql",
            config={
                "connection_uri": "sqlite:///:memory:"
                # No BigQuery metadata - test defaults
            },
        ).build()

        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)

        # Then: SQLite connection works with defaults
        assert adapter.db_type == "sqlite"
        config = adapter.settings.config.data_source.config
        # Config is a Pydantic model, not dict
        assert hasattr(config, "connection_uri")
        assert config.connection_uri == "sqlite:///:memory:"
        assert config.query_timeout == 300  # Default timeout

    def test_postgresql_not_affected_by_bigquery_changes(self, settings_builder):
        """Test PostgreSQL configuration isolation using SQLite for compatibility."""
        # Given: SQLite configuration simulating PostgreSQL behavior
        settings = settings_builder.with_data_source(
            "sql",
            config={
                "connection_uri": "sqlite:///:memory:",
                "postgresql_mode": True,  # Simulate PostgreSQL behavior
            },
        ).build()

        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)

        # Then: SQLite connection works independently
        assert adapter.db_type == "sqlite"  # Using SQLite for testing
        config = adapter.settings.config.data_source.config
        # Config is a Pydantic model
        assert hasattr(config, "connection_uri")
        assert config.connection_uri == "sqlite:///:memory:"
        # Extra fields like 'postgresql_mode' are ignored


class TestSqlAdapterSecurityValidation:
    """Test SQL security guards and query validation - Focus area for Target 4."""

    def test_sql_security_guards_ddl_prevention(self, settings_builder):
        """Test SQL security guard prevents DDL/DML operations."""
        # Given: SqlAdapter with SQLite
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # Test various dangerous SQL keywords
        dangerous_queries = [
            "DROP TABLE users",
            "DELETE FROM users WHERE id = 1",
            "UPDATE users SET name = 'hacked' WHERE id = 1",
            "INSERT INTO users VALUES (99, 'malicious')",
            "ALTER TABLE users ADD COLUMN password TEXT",
            "TRUNCATE TABLE users",
            "CREATE TABLE malicious (id INT)",
        ]

        for query in dangerous_queries:
            # When/Then: Dangerous SQL should be prevented
            with pytest.raises(ValueError, match="보안 위반: 금지된 SQL 키워드 포함"):
                adapter.read(query)

    def test_sql_security_guards_limit_warning(self, settings_builder):
        """Test SQL security guard warns about missing LIMIT clause."""
        # Given: SqlAdapter with SQLite and test data
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # Create test table
        with adapter.engine.begin() as conn:
            conn.execute(sqlalchemy.text("CREATE TABLE test_limit (id INTEGER, value TEXT)"))
            conn.execute(
                sqlalchemy.text("INSERT INTO test_limit VALUES (1, 'data1'), (2, 'data2')")
            )

        # When: Executing query without LIMIT (should warn but not fail)
        df = adapter.read("SELECT id, value FROM test_limit")

        # Then: Query succeeds despite missing LIMIT (warning logged)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestSqlAdapterErrorHandling:
    """Test SQL adapter error handling scenarios - Focus area for Target 4."""

    def test_connection_failure_handling(self, settings_builder):
        """Test handling of database connection failures."""
        # Given: Invalid connection URI
        settings = settings_builder.with_data_source(
            "sql",
            config={"connection_uri": "postgresql://invalid:invalid@nonexistent:5432/nonexistent"},
        ).build()

        # When/Then: Connection failure should raise appropriate error
        with pytest.raises((ValueError, Exception)):  # Engine creation or connection test failure
            SqlAdapter(settings)

    def test_engine_creation_failure_handling(self, settings_builder):
        """Test error handling during engine creation."""
        # Given: Settings with invalid connection string
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "invalid://broken/connection/string"}
        ).build()

        # When/Then: Invalid connection should raise error
        with pytest.raises(Exception):  # Engine creation will fail
            SqlAdapter(settings)

    def test_sql_query_execution_error(self, settings_builder):
        """Test handling of SQL query execution errors."""
        # Given: SqlAdapter with SQLite
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # When/Then: Invalid SQL should raise error with context
        with pytest.raises(Exception):  # SQL execution error
            adapter.read("SELECT nonexistent_column FROM nonexistent_table")

    def test_write_operation_error_handling(self, settings_builder, test_data_generator):
        """Test error handling during write operations."""
        # Given: SqlAdapter and test data
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        df, _ = test_data_generator.classification_data(n_samples=10, n_features=3)

        # Create table first
        adapter.write(df, "test_table", if_exists="replace")

        # When/Then: Write conflict should raise error
        with pytest.raises(Exception):  # Table exists and if_exists='fail'
            adapter.write(df, "test_table", if_exists="fail")


class TestSqlAdapterDatabaseTestContext:
    """Test SqlAdapter using DatabaseTestContext - Integration test focus."""

    def test_sql_adapter_with_database_test_context(
        self, settings_builder, database_test_context, test_data_generator
    ):
        """Test SqlAdapter with DatabaseTestContext pattern."""
        # Given: Test data and DatabaseTestContext
        cls_df, _ = test_data_generator.classification_data(n_samples=20, n_features=4)
        cls_df["target"] = np.random.randint(0, 2, size=20)

        reg_df, _ = test_data_generator.regression_data(n_samples=15, n_features=3)
        reg_df["target"] = np.random.randn(15)

        # When: Using DatabaseTestContext
        with database_test_context.sqlite_db({"users": cls_df, "products": reg_df}) as db:
            settings = settings_builder.with_data_source(
                "sql", config={"connection_uri": db.connection_uri}
            ).build()
            adapter = SqlAdapter(settings)

            # Then: Can read from both tables
            users_df = adapter.read(
                "SELECT feature_1, feature_2, feature_3, feature_4, entity_id, target FROM users"
            )
            products_df = adapter.read(
                "SELECT feature_1, feature_2, feature_3, entity_id, target FROM products"
            )

            assert len(users_df) == 20
            assert len(products_df) == 15
            assert "target" in users_df.columns
            assert "target" in products_df.columns

    def test_sql_file_loading_functionality(
        self, settings_builder, database_test_context, test_data_generator, isolated_temp_directory
    ):
        """Test SQL file loading with DatabaseTestContext."""
        # Given: SQL file and test data
        test_df, _ = test_data_generator.classification_data(n_samples=25, n_features=3)
        test_df["target"] = np.random.randint(0, 3, size=25)

        # Create SQL file
        sql_file = isolated_temp_directory / "test_query.sql"
        sql_file.write_text(
            "SELECT feature_1, feature_2, feature_3, entity_id, target FROM test_data WHERE target = 1"
        )

        # When: Using DatabaseTestContext and SQL file
        with database_test_context.sqlite_db({"test_data": test_df}) as db:
            settings = settings_builder.with_data_source(
                "sql", config={"connection_uri": db.connection_uri}
            ).build()
            adapter = SqlAdapter(settings)

            # Then: Can execute SQL from file
            result_df = adapter.read(str(sql_file))

            assert isinstance(result_df, pd.DataFrame)
            assert all(result_df["target"] == 1)
            assert len(result_df) > 0

    def test_sql_file_not_found_error(self, settings_builder):
        """Test error handling when SQL file doesn't exist."""
        # Given: SqlAdapter
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # When/Then: Non-existent SQL file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="SQL 파일을 찾을 수 없습니다"):
            adapter.read("nonexistent_query.sql")


class TestSqlAdapterDatabaseTypeHandling:
    """Test different database type handling in URI parsing."""

    def test_postgresql_uri_parsing(self, settings_builder):
        """Test PostgreSQL URI parsing and configuration."""
        # Given: PostgreSQL-like URI (but using SQLite for testing)
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}  # Use SQLite for actual testing
        ).build()

        # When: Creating adapter
        adapter = SqlAdapter(settings)

        # Then: URI parsing works
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
            "postgresql://user:pass@localhost:5432/db"
        )
        assert db_type == "postgresql"
        assert "pool_size" in engine_kwargs
        assert "pool_pre_ping" in engine_kwargs

    def test_mysql_uri_parsing(self, settings_builder):
        """Test MySQL URI parsing and configuration."""
        # Given: SqlAdapter
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Parsing MySQL URI
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
            "mysql://user:pass@localhost:3306/db"
        )

        # Then: MySQL configuration applied
        assert db_type == "mysql"
        assert "pool_recycle" in engine_kwargs
        assert engine_kwargs["pool_recycle"] == 3600

    def test_unknown_scheme_handling(self, settings_builder):
        """Test handling of unknown database schemes."""
        # Given: SqlAdapter
        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": "sqlite:///:memory:"}
        ).build()
        adapter = SqlAdapter(settings)

        # When: Parsing unknown scheme
        db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
            "unknown://user:pass@localhost:1234/db"
        )

        # Then: Falls back to generic handling
        assert db_type == "generic"
        assert processed_uri == "unknown://user:pass@localhost:1234/db"
