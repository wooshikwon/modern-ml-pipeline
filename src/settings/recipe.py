"""
Recipe Schema - Workflow Definitions (v2.0)
Simplified from 259 lines to ~150 lines

This module defines the ML workflow (Recipe) schema.
Completely separated from Config (infrastructure) settings.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional


class Model(BaseModel):
    """
    Model configuration.
    Defines the ML model and its hyperparameters.
    """
    class_path: str = Field(..., description="Full Python path to model class")
    library: Optional[str] = Field(None, description="Library name (sklearn, xgboost, etc.)")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Model hyperparameters"
    )
    computed: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Runtime computed fields (run_name, etc.)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "class_path": "sklearn.ensemble.RandomForestClassifier",
                "library": "sklearn",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            }
        }


class Loader(BaseModel):
    """Data loader configuration."""
    name: str = Field("data_loader", description="Name of the data loader")
    adapter: Optional[str] = Field(None, description="Adapter name (optional, auto-detected from source_uri if not provided)")
    source_uri: str = Field(..., description="Data source URI or SQL query")
    cache_enabled: bool = Field(False, description="Whether to cache loaded data")
    
    
class DataInterface(BaseModel):
    """Data interface configuration."""
    task_type: str = Field(..., description="ML task type: 'classification' or 'regression'")
    target_column: str = Field(..., description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns (None = all)")
    id_column: Optional[str] = Field(None, description="ID column for tracking")


class Data(BaseModel):
    """
    Data configuration.
    Defines data loading and interface.
    """
    loader: Loader
    data_interface: DataInterface
    entity_schema: Optional[Dict[str, str]] = Field(
        None,
        description="Entity column schema for feature engineering"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "loader": {
                    "name": "training_data",
                    "adapter": "sql",
                    "source_uri": "SELECT * FROM training_table"
                },
                "data_interface": {
                    "task_type": "classification",
                    "target_column": "label",
                    "feature_columns": ["feature1", "feature2"]
                }
            }
        }


class Preprocessor(BaseModel):
    """Data preprocessing configuration."""
    steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of preprocessing steps"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "steps": [
                    {"type": "missing_handler", "config": {"strategy": "drop"}},
                    {"type": "scaler", "config": {"method": "standard"}}
                ]
            }
        }


class Evaluation(BaseModel):
    """
    Evaluation configuration.
    Defines metrics and validation strategy.
    """
    metrics: List[str] = Field(..., description="Evaluation metrics")
    validation: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "split", "test_size": 0.2},
        description="Validation strategy configuration"
    )
    
    @validator('metrics')
    def validate_metrics(cls, v, values):
        """Ensure metrics are valid strings."""
        if not v:
            raise ValueError("At least one metric must be specified")
        return [m.lower() for m in v]
    
    class Config:
        schema_extra = {
            "example": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "validation": {
                    "method": "split",
                    "test_size": 0.2,
                    "random_state": 42
                }
            }
        }


class Recipe(BaseModel):
    """
    Root Recipe configuration (recipes/*.yaml).
    
    This is the complete ML workflow definition.
    Completely separated from Config (infrastructure) settings.
    """
    name: str = Field(..., description="Recipe name")
    model: Model
    data: Data
    preprocessor: Optional[Preprocessor] = None
    evaluation: Evaluation
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (author, version, etc.)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "name": "classification_rf",
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "hyperparameters": {"n_estimators": 100}
                },
                "data": {
                    "loader": {
                        "name": "train_data",
                        "adapter": "sql",
                        "source_uri": "SELECT * FROM features"
                    },
                    "data_interface": {
                        "task_type": "classification",
                        "target_column": "target"
                    }
                },
                "evaluation": {
                    "metrics": ["accuracy", "f1"],
                    "validation": {"method": "split"}
                }
            }
        }
    
    def get_task_type(self) -> str:
        """Get the ML task type."""
        return self.data.data_interface.task_type