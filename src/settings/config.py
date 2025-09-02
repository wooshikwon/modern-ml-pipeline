"""
Config Schema - Infrastructure Settings (v2.0)
Simplified from 377 lines to ~150 lines

This module defines the infrastructure configuration schema.
Completely separated from Recipe (workflow) definitions.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class Environment(BaseModel):
    """
    Environment configuration (v2.0).
    
    Note: app_env field has been removed. Use ENV_NAME environment variable instead.
    """
    project_id: str = Field(..., description="Cloud project ID (GCP/AWS/Azure)")
    credential_path: Optional[str] = Field(None, description="Path to service account credentials")
    region: Optional[str] = Field("us-central1", description="Default cloud region")


class MLflow(BaseModel):
    """MLflow experiment tracking configuration."""
    tracking_uri: str = Field(..., description="MLflow tracking server URI or local path")
    experiment_name: str = Field(..., description="Experiment name for grouping runs")
    artifact_location: Optional[str] = Field(None, description="Custom artifact storage location")


class Adapter(BaseModel):
    """
    Unified data adapter configuration.
    Supports both SQL and Storage adapters.
    """
    type: str = Field(..., description="Adapter type: 'sql' or 'storage'")
    config: Dict[str, Any] = Field(default_factory=dict, description="Adapter-specific configuration")
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "type": "sql",
                    "config": {
                        "connection_uri": "postgresql://user:pass@localhost:5432/db",
                        "query_timeout": 30
                    }
                },
                {
                    "type": "storage", 
                    "config": {
                        "base_path": "./data",
                        "storage_options": {}
                    }
                }
            ]
        }


class Serving(BaseModel):
    """API serving configuration."""
    enabled: bool = Field(False, description="Whether serving is enabled")
    host: str = Field("0.0.0.0", description="Host to bind the API server")
    port: int = Field(8000, description="Port to bind the API server")
    model_stage: Optional[str] = Field(None, description="MLflow model stage to serve")
    workers: int = Field(1, description="Number of worker processes")


class ArtifactStore(BaseModel):
    """Artifact storage configuration."""
    enabled: bool = Field(True, description="Whether to use this artifact store")
    base_uri: str = Field(..., description="Base URI for artifact storage")
    storage_type: str = Field("local", description="Storage type: 'local', 'gcs', 's3'")


class Config(BaseModel):
    """
    Root infrastructure configuration (configs/*.yaml).
    
    This is the complete infrastructure settings loaded from configs/{env_name}.yaml.
    Completely separated from Recipe (workflow) settings.
    """
    environment: Environment
    mlflow: MLflow
    adapters: Dict[str, Adapter] = Field(
        default_factory=dict,
        description="Named adapters for data loading"
    )
    serving: Optional[Serving] = None
    artifact_stores: Optional[Dict[str, ArtifactStore]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "environment": {
                    "project_id": "my-gcp-project",
                    "credential_path": "/path/to/credentials.json"
                },
                "mlflow": {
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "my-experiment"
                },
                "adapters": {
                    "sql": {
                        "type": "sql",
                        "config": {
                            "connection_uri": "${DB_CONNECTION_URI}"
                        }
                    },
                    "storage": {
                        "type": "storage",
                        "config": {
                            "base_path": "./data"
                        }
                    }
                },
                "serving": {
                    "enabled": True,
                    "port": 8080
                }
            }
        }
    
    def get_adapter(self, name: str) -> Adapter:
        """Get adapter by name with validation."""
        if name not in self.adapters:
            raise KeyError(
                f"Adapter '{name}' not found in config. "
                f"Available adapters: {list(self.adapters.keys())}"
            )
        return self.adapters[name]