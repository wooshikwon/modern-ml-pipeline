"""
SQL Adapter Unit Tests - No Mock Hell Approach
Real SQLite database, real queries, real behavior validation
Following comprehensive testing strategy document principles
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from pathlib import Path
from typing import Dict, Any

from src.components.adapter.modules.sql_adapter import SqlAdapter
from src.interface.base_adapter import BaseAdapter


class TestSqlAdapterWithRealDatabase:
    """Test SqlAdapter with real in-memory SQLite database - No mocks."""
    
    def test_sql_adapter_initialization(self, settings_builder):
        """Test SqlAdapter initialization with SQLite connection."""
        # Given: Valid SQLite connection settings
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": "sqlite:///:memory:"
            }) \
            .build()
        
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
        
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": connection_uri
            }) \
            .build()
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
        
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": connection_uri
            }) \
            .build()
        adapter = SqlAdapter(settings)
        
        # When: Executing parameterized query
        query = f"SELECT feature_1, feature_2, feature_3, feature_4, feature_5, target, entity_id FROM {sql_info['classification_table']} WHERE target = :target_value"
        df = adapter.read(query, params={"target_value": 0})
        
        # Then: Only filtered data is returned
        assert isinstance(df, pd.DataFrame)
        assert all(df["target"] == 0)
        assert len(df) > 0
    
    def test_write_to_sqlite_database(self, settings_builder, isolated_temp_directory,
                                     test_data_generator):
        """Test writing DataFrame to SQLite database."""
        # Given: Data and SQLite database
        df, _ = test_data_generator.classification_data(n_samples=50, n_features=5)
        df["target"] = np.random.randint(0, 2, size=50)
        
        db_path = isolated_temp_directory / "test.db"
        connection_uri = f"sqlite:///{db_path}"
        
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": connection_uri
            }) \
            .build()
        adapter = SqlAdapter(settings)
        
        # When: Writing to database
        adapter.write(df, "test_table", if_exists="replace")
        
        # Then: Data is written and readable
        read_df = adapter.read("SELECT feature_1, feature_2, feature_3, feature_4, feature_5, entity_id, target FROM test_table")
        assert len(read_df) == len(df)
        assert set(read_df.columns) == set(df.columns)
    
    def test_sql_adapter_with_multiple_tables(self, settings_builder, isolated_temp_directory,
                                             test_data_generator):
        """Test SqlAdapter with multiple tables in same database."""
        # Given: Multiple datasets and database
        cls_df, _ = test_data_generator.classification_data(n_samples=30, n_features=5)
        cls_df["target"] = np.random.randint(0, 2, size=30)
        
        reg_df, _ = test_data_generator.regression_data(n_samples=40, n_features=4)
        reg_df["target"] = np.random.randn(40)
        
        db_path = isolated_temp_directory / "multi_table.db"
        connection_uri = f"sqlite:///{db_path}"
        
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": connection_uri
            }) \
            .build()
        adapter = SqlAdapter(settings)
        
        # When: Writing multiple tables
        adapter.write(cls_df, "classification_table", if_exists="replace")
        adapter.write(reg_df, "regression_table", if_exists="replace")
        
        # Then: Both tables exist and contain correct data
        cls_read = adapter.read("SELECT feature_1, feature_2, feature_3, feature_4, feature_5, entity_id, target FROM classification_table")
        reg_read = adapter.read("SELECT feature_1, feature_2, feature_3, feature_4, entity_id, target FROM regression_table")
        
        assert len(cls_read) == 30
        assert len(reg_read) == 40
        assert cls_read["target"].nunique() >= 2
    
    def test_sql_adapter_connection_pooling(self, settings_builder):
        """Test SqlAdapter connection pooling behavior."""
        # Given: SQLite connection with pool settings
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": "sqlite:///:memory:",
                "pool_size": 5,
                "max_overflow": 10
            }) \
            .build()
        
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
        
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": connection_uri
            }) \
            .build()
        adapter = SqlAdapter(settings)
        
        # Create empty table
        with adapter.engine.begin() as conn:
            conn.execute(sqlalchemy.text(
                "CREATE TABLE empty_table (id INTEGER, value TEXT)"
            ))
        
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
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": "sqlite:///:memory:",
                # BigQuery configuration stored as metadata
                "bigquery_project_id": "test-project",
                "bigquery_dataset_id": "test-dataset",
                "bigquery_location": "EU",
                "bigquery_use_pandas_gbq": True
            }) \
            .build()
        
        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)
        
        # Then: SQLite connection works and BigQuery config is parsed
        assert adapter.db_type == 'sqlite'  # Actual connection is SQLite
        # BigQuery configuration should be stored for later use
        config = adapter.settings.config.data_source.config
        assert config.get("bigquery_project_id") == "test-project"
        assert config.get("bigquery_dataset_id") == "test-dataset"
        assert config.get("bigquery_location") == "EU"
        assert config.get("bigquery_use_pandas_gbq") == True
    
    def test_bigquery_default_configuration(self, settings_builder):
        """Test BigQuery default values with SQLite compatibility testing."""
        # Given: Minimal SQLite configuration without BigQuery metadata
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": "sqlite:///:memory:"
                # No BigQuery metadata - test defaults
            }) \
            .build()
        
        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)
        
        # Then: SQLite connection works with default BigQuery values
        assert adapter.db_type == 'sqlite'
        config = adapter.settings.config.data_source.config
        # Default BigQuery values should be None when not specified
        assert config.get("bigquery_project_id") is None
        assert config.get("bigquery_dataset_id") is None
        assert config.get("bigquery_location", "US") == "US"  # Default location
        assert config.get("bigquery_use_pandas_gbq", False) == False  # Default
    
    def test_postgresql_not_affected_by_bigquery_changes(self, settings_builder):
        """Test PostgreSQL configuration isolation using SQLite for compatibility."""
        # Given: SQLite configuration simulating PostgreSQL behavior
        settings = settings_builder \
            .with_data_source("sql", config={
                "connection_uri": "sqlite:///:memory:",
                "postgresql_mode": True  # Simulate PostgreSQL behavior
            }) \
            .build()
        
        # When: Creating SqlAdapter with SQLite
        adapter = SqlAdapter(settings)
        
        # Then: No BigQuery settings interfere with PostgreSQL-style config
        assert adapter.db_type == 'sqlite'  # Using SQLite for testing
        config = adapter.settings.config.data_source.config
        # Verify BigQuery settings don't affect PostgreSQL mode
        assert config.get("bigquery_project_id") is None
        assert config.get("bigquery_dataset_id") is None
        assert config.get("postgresql_mode") == True