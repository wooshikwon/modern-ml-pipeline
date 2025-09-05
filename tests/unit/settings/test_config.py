"""
Unit tests for the Config module.
Tests the Config schema and validation logic.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from src.settings.config import (
    Config, Environment, MLflow, DataSource, 
    FeatureStore, FeastConfig, Serving, ArtifactStore
)
from tests.helpers.builders import ConfigBuilder
from tests.helpers.assertions import assert_config_valid


class TestEnvironment:
    """Test the Environment configuration class."""
    
    def test_environment_creation(self):
        """Test creating an Environment object."""
        env = Environment(name="test")
        assert env.name == "test"
    
    def test_environment_validation(self):
        """Test Environment validation."""
        with pytest.raises(ValidationError):
            # Name is required
            Environment()


class TestMLflow:
    """Test the MLflow configuration class."""
    
    def test_mlflow_creation(self):
        """Test creating an MLflow object."""
        mlflow = MLflow(
            tracking_uri="./mlruns",
            experiment_name="test_experiment"
        )
        assert mlflow.tracking_uri == "./mlruns"
        assert mlflow.experiment_name == "test_experiment"
        assert mlflow.tracking_username == ""  # Default value
        assert mlflow.tracking_password == ""  # Default value
    
    def test_mlflow_with_auth(self):
        """Test MLflow with authentication."""
        mlflow = MLflow(
            tracking_uri="http://mlflow.example.com",
            experiment_name="test",
            tracking_username="user",
            tracking_password="pass"
        )
        assert mlflow.tracking_username == "user"
        assert mlflow.tracking_password == "pass"
    
    def test_mlflow_validation(self):
        """Test MLflow validation."""
        with pytest.raises(ValidationError):
            # tracking_uri is required
            MLflow(experiment_name="test")


class TestDataSource:
    """Test the DataSource configuration class."""
    
    @pytest.mark.parametrize("adapter_type", ["sql", "bigquery", "storage"])
    def test_datasource_creation(self, adapter_type):
        """Test creating DataSource with different adapter types."""
        ds = DataSource(
            name="test_source",
            adapter_type=adapter_type,
            config={"key": "value"}
        )
        assert ds.name == "test_source"
        assert ds.adapter_type == adapter_type
        assert ds.config == {"key": "value"}
    
    def test_datasource_invalid_adapter(self):
        """Test DataSource with invalid adapter type."""
        with pytest.raises(ValidationError):
            DataSource(
                name="test",
                adapter_type="invalid",  # Not in allowed values
                config={}
            )
    
    def test_datasource_validation(self):
        """Test DataSource validation."""
        with pytest.raises(ValidationError):
            # name is required
            DataSource(adapter_type="sql", config={})


class TestFeatureStore:
    """Test the FeatureStore configuration class."""
    
    def test_featurestore_none(self):
        """Test FeatureStore with provider='none'."""
        fs = FeatureStore(provider="none")
        assert fs.provider == "none"
        assert fs.feast_config is None
    
    def test_featurestore_feast(self):
        """Test FeatureStore with Feast configuration."""
        feast_config = {
            "project": "test_project",
            "registry": "./feast_repo/registry.db",
            "online_store": {
                "type": "redis",
                "connection_string": "redis://localhost:6379"
            },
            "offline_store": {
                "type": "file",
                "path": "./feast_repo/data"
            }
        }
        
        fs = FeatureStore(
            provider="feast",
            feast_config=FeastConfig(**feast_config)
        )
        assert fs.provider == "feast"
        assert fs.feast_config is not None
        assert fs.feast_config.project == "test_project"
    
    def test_featurestore_validation(self):
        """Test FeatureStore validation."""
        with pytest.raises(ValidationError):
            FeatureStore(provider="invalid")  # Invalid provider


class TestConfig:
    """Test the main Config class."""
    
    def test_config_creation_minimal(self):
        """Test creating Config with minimal required fields."""
        config = Config(
            environment=Environment(name="test"),
            data_source=DataSource(
                name="test_source",
                adapter_type="storage",
                config={}
            ),
            feature_store=FeatureStore(provider="none")
        )
        assert_config_valid(config)
        assert config.environment.name == "test"
        assert config.mlflow is None
        assert config.serving is None
    
    def test_config_creation_full(self):
        """Test creating Config with all fields."""
        config = Config(
            environment=Environment(name="prod"),
            mlflow=MLflow(
                tracking_uri="http://mlflow.example.com",
                experiment_name="prod_experiment"
            ),
            data_source=DataSource(
                name="prod_db",
                adapter_type="sql",
                config={
                    "connection_uri": "postgresql://localhost/db"
                }
            ),
            feature_store=FeatureStore(provider="none"),
            serving=Serving(
                enabled=True,
                host="0.0.0.0",
                port=8000
            ),
            artifact_store=ArtifactStore(
                type="s3",
                config={
                    "bucket": "my-bucket",
                    "prefix": "artifacts/"
                }
            )
        )
        assert_config_valid(config)
        assert config.serving.enabled is True
        assert config.artifact_store.type == "s3"
    
    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {
            "environment": {"name": "test"},
            "mlflow": {
                "tracking_uri": "./mlruns",
                "experiment_name": "test"
            },
            "data_source": {
                "name": "test_source",
                "adapter_type": "storage",
                "config": {}
            },
            "feature_store": {"provider": "none"}
        }
        config = Config(**config_dict)
        assert_config_valid(config)
    
    def test_config_validation_missing_required(self):
        """Test Config validation with missing required fields."""
        with pytest.raises(ValidationError):
            Config(
                environment=Environment(name="test")
                # Missing data_source
            )
    
    def test_config_get_adapter_config(self):
        """Test the get_adapter_config helper method."""
        config = ConfigBuilder.build(adapter_type="sql")
        adapter_config = config.get_adapter_config()
        
        assert "type" in adapter_config
        assert adapter_config["type"] == "sql"
        assert "config" in adapter_config
    
    def test_config_has_feast(self):
        """Test the has_feast helper method."""
        # Without Feast
        config1 = ConfigBuilder.build(feature_store_provider="none")
        assert config1.has_feast() is False
        
        # With Feast
        config2 = ConfigBuilder.build(feature_store_provider="feast")
        # Note: This would normally require feast_config setup
        assert config2.has_feast() is True
    
    def test_config_builder_helper(self):
        """Test using the ConfigBuilder helper."""
        config = ConfigBuilder.build(
            env_name="custom",
            mlflow_tracking_uri="s3://bucket/mlflow",
            adapter_type="bigquery"
        )
        assert config.environment.name == "custom"
        assert config.mlflow.tracking_uri == "s3://bucket/mlflow"
        assert config.data_source.adapter_type == "bigquery"


class TestServing:
    """Test the Serving configuration class."""
    
    def test_serving_defaults(self):
        """Test Serving with default values."""
        serving = Serving()
        assert serving.enabled is False
        assert serving.host == "0.0.0.0"
        assert serving.port == 8000
        assert serving.workers == 1
    
    def test_serving_custom(self):
        """Test Serving with custom values."""
        serving = Serving(
            enabled=True,
            host="127.0.0.1",
            port=5000,
            workers=4
        )
        assert serving.enabled is True
        assert serving.host == "127.0.0.1"
        assert serving.port == 5000
        assert serving.workers == 4
    
    def test_serving_port_validation(self):
        """Test Serving port validation."""
        # Valid port range
        serving1 = Serving(port=1024)  # Minimum allowed
        assert serving1.port == 1024
        
        serving2 = Serving(port=65535)  # Maximum allowed
        assert serving2.port == 65535
        
        # Invalid port range
        with pytest.raises(ValidationError):
            Serving(port=1023)  # Too low
        
        with pytest.raises(ValidationError):
            Serving(port=65536)  # Too high


class TestArtifactStore:
    """Test the ArtifactStore configuration class."""
    
    @pytest.mark.parametrize("store_type", ["local", "s3", "gcs"])
    def test_artifactstore_types(self, store_type):
        """Test ArtifactStore with different types."""
        store = ArtifactStore(
            type=store_type,
            config={"path": f"/path/to/{store_type}"}
        )
        assert store.type == store_type
    
    def test_artifactstore_invalid_type(self):
        """Test ArtifactStore with invalid type."""
        with pytest.raises(ValidationError):
            ArtifactStore(type="invalid", config={})
    
    def test_artifactstore_config_structure(self):
        """Test ArtifactStore config structure."""
        # S3 config
        s3_store = ArtifactStore(
            type="s3",
            config={
                "bucket": "my-bucket",
                "prefix": "models/",
                "region": "us-west-2"
            }
        )
        assert s3_store.config["bucket"] == "my-bucket"
        
        # GCS config
        gcs_store = ArtifactStore(
            type="gcs",
            config={
                "bucket": "my-gcs-bucket",
                "project": "my-project"
            }
        )
        assert gcs_store.config["project"] == "my-project"