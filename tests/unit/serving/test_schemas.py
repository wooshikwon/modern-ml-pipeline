"""
Unit tests for serving schemas.
Tests dynamic schema generation and validation.
"""

import pytest
from typing import Any, List
from pydantic import BaseModel, ValidationError

from src.serving.schemas import (
    get_pk_from_loader_sql,
    create_dynamic_prediction_request,
    HealthCheckResponse,
    BatchPredictionResponse,
    ModelMetadataResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
)


class TestGetPkFromLoaderSql:
    """Test extraction of primary keys from SQL templates."""
    
    def test_get_pk_single_variable(self):
        """Test extraction of single Jinja2 variable."""
        # Arrange
        sql_template = "SELECT * FROM table WHERE id = {{ user_id }}"
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        assert result == ["user_id"]
    
    def test_get_pk_multiple_variables(self):
        """Test extraction of multiple Jinja2 variables."""
        # Arrange
        sql_template = """
        SELECT * FROM table 
        WHERE user_id = {{ user_id }} 
        AND campaign_id = {{ campaign_id }}
        AND date = {{ date_param }}
        """
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        expected = ["campaign_id", "date_param", "user_id"]  # Sorted
        assert result == expected
    
    def test_get_pk_with_gcp_project_id_excluded(self):
        """Test that gcp_project_id is excluded from results."""
        # Arrange
        sql_template = """
        SELECT * FROM `{{ gcp_project_id }}.dataset.table`
        WHERE user_id = {{ user_id }}
        AND session_id = {{ session_id }}
        """
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        expected = ["session_id", "user_id"]  # gcp_project_id excluded
        assert result == expected
    
    def test_get_pk_duplicate_variables(self):
        """Test handling of duplicate variables."""
        # Arrange
        sql_template = """
        SELECT * FROM table1 WHERE id = {{ user_id }}
        UNION
        SELECT * FROM table2 WHERE id = {{ user_id }}
        AND category = {{ category }}
        """
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        expected = ["category", "user_id"]  # Duplicates removed
        assert result == expected
    
    def test_get_pk_no_variables(self):
        """Test SQL template with no Jinja2 variables."""
        # Arrange
        sql_template = "SELECT * FROM static_table WHERE status = 'active'"
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        assert result == []
    
    def test_get_pk_whitespace_handling(self):
        """Test handling of whitespace in Jinja2 variables."""
        # Arrange
        sql_template = """
        SELECT * FROM table 
        WHERE id = {{  user_id  }}
        AND name = {{ user_name   }}
        AND type = {{category}}
        """
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        expected = ["category", "user_id", "user_name"]
        assert result == expected
    
    def test_get_pk_complex_sql(self):
        """Test extraction from complex SQL template."""
        # Arrange
        sql_template = """
        WITH filtered_data AS (
            SELECT * FROM `{{ gcp_project_id }}.analytics.events`
            WHERE date = {{ event_date }}
            AND user_id = {{ user_id }}
        ),
        aggregated AS (
            SELECT 
                user_id,
                campaign_id,
                SUM(value) as total_value
            FROM filtered_data
            WHERE campaign_id = {{ campaign_id }}
            GROUP BY user_id, campaign_id
        )
        SELECT * FROM aggregated
        WHERE user_id = {{ user_id }}
        """
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert
        expected = ["campaign_id", "event_date", "user_id"]
        assert result == expected


class TestCreateDynamicPredictionRequest:
    """Test dynamic prediction request model creation."""
    
    def test_create_dynamic_model_single_field(self):
        """Test creation of dynamic model with single field."""
        # Arrange
        model_name = "TestModel"
        pk_fields = ["user_id"]
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert
        assert DynamicModel.__name__ == "TestModelPredictionRequest"
        assert issubclass(DynamicModel, BaseModel)
        
        # Test model instantiation
        instance = DynamicModel(user_id="123")
        assert instance.user_id == "123"
    
    def test_create_dynamic_model_multiple_fields(self):
        """Test creation of dynamic model with multiple fields."""
        # Arrange
        model_name = "ComplexModel"
        pk_fields = ["user_id", "campaign_id", "session_id"]
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert
        assert DynamicModel.__name__ == "ComplexModelPredictionRequest"
        
        # Test model instantiation
        instance = DynamicModel(
            user_id="user123",
            campaign_id="camp456",
            session_id="sess789"
        )
        assert instance.user_id == "user123"
        assert instance.campaign_id == "camp456"
        assert instance.session_id == "sess789"
    
    def test_create_dynamic_model_empty_fields(self):
        """Test creation of dynamic model with empty fields list."""
        # Arrange
        model_name = "EmptyModel"
        pk_fields = []
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert
        assert DynamicModel.__name__ == "EmptyModelPredictionRequest"
        
        # Test model instantiation (should work with no fields)
        instance = DynamicModel()
        assert instance is not None
    
    def test_create_dynamic_model_validation(self):
        """Test validation of dynamic model."""
        # Arrange
        model_name = "ValidatedModel"
        pk_fields = ["required_field"]
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert - Required field validation
        with pytest.raises(ValidationError):
            DynamicModel()  # Missing required field
        
        # Should work with required field
        instance = DynamicModel(required_field="value")
        assert instance.required_field == "value"
    
    def test_create_dynamic_model_field_types(self):
        """Test that dynamic model accepts various field types."""
        # Arrange
        model_name = "FlexibleModel"
        pk_fields = ["flexible_field"]
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert - Should accept various types
        string_instance = DynamicModel(flexible_field="string_value")
        assert string_instance.flexible_field == "string_value"
        
        int_instance = DynamicModel(flexible_field=123)
        assert int_instance.flexible_field == 123
        
        list_instance = DynamicModel(flexible_field=[1, 2, 3])
        assert list_instance.flexible_field == [1, 2, 3]
        
        dict_instance = DynamicModel(flexible_field={"key": "value"})
        assert dict_instance.flexible_field == {"key": "value"}


class TestHealthCheckResponse:
    """Test HealthCheckResponse schema."""
    
    def test_health_check_response_valid(self):
        """Test valid HealthCheckResponse creation."""
        # Arrange & Act
        response = HealthCheckResponse(
            status="healthy",
            model_uri="runs/123/model",
            model_name="TestModel"
        )
        
        # Assert
        assert response.status == "healthy"
        assert response.model_uri == "runs/123/model"
        assert response.model_name == "TestModel"
    
    def test_health_check_response_validation(self):
        """Test HealthCheckResponse validation."""
        # Assert - Required fields
        with pytest.raises(ValidationError):
            HealthCheckResponse()  # Missing required fields


class TestBatchPredictionResponse:
    """Test BatchPredictionResponse schema."""
    
    def test_batch_prediction_response_valid(self):
        """Test valid BatchPredictionResponse creation."""
        # Arrange & Act
        response = BatchPredictionResponse(
            predictions=[{"prediction": 0.8, "confidence": 0.9}],
            model_uri="runs/123/model",
            sample_count=1
        )
        
        # Assert
        assert response.predictions == [{"prediction": 0.8, "confidence": 0.9}]
        assert response.model_uri == "runs/123/model"
        assert response.sample_count == 1
    
    def test_batch_prediction_response_empty_predictions(self):
        """Test BatchPredictionResponse with empty predictions."""
        # Arrange & Act
        response = BatchPredictionResponse(
            predictions=[],
            model_uri="runs/123/model",
            sample_count=0
        )
        
        # Assert
        assert response.predictions == []
        assert response.sample_count == 0


class TestHyperparameterOptimizationInfo:
    """Test HyperparameterOptimizationInfo schema."""
    
    def test_hyperparameter_optimization_info_enabled(self):
        """Test HyperparameterOptimizationInfo when enabled."""
        # Arrange & Act
        hpo_info = HyperparameterOptimizationInfo(
            enabled=True,
            best_params={"lr": 0.001, "batch_size": 32},
            best_score=0.95,
            total_trials=100
        )
        
        # Assert
        assert hpo_info.enabled is True
        assert hpo_info.best_params == {"lr": 0.001, "batch_size": 32}
        assert hpo_info.best_score == 0.95
        assert hpo_info.total_trials == 100
    
    def test_hyperparameter_optimization_info_disabled(self):
        """Test HyperparameterOptimizationInfo when disabled."""
        # Arrange & Act
        hpo_info = HyperparameterOptimizationInfo(
            enabled=False,
            best_params={},
            best_score=0.0,
            total_trials=0
        )
        
        # Assert
        assert hpo_info.enabled is False
        assert hpo_info.best_params == {}
        assert hpo_info.best_score == 0.0
        assert hpo_info.total_trials == 0


class TestModelMetadataResponse:
    """Test ModelMetadataResponse schema."""
    
    def test_model_metadata_response_complete(self):
        """Test complete ModelMetadataResponse."""
        # Arrange & Act
        hpo_info = HyperparameterOptimizationInfo(
            enabled=True,
            best_params={"lr": 0.001},
            best_score=0.95,
            total_trials=50
        )
        
        training_methodology = TrainingMethodologyInfo(
            train_test_split_method="temporal",
            train_ratio=0.8,
            validation_strategy="cross_validation",
            preprocessing_fit_scope="train_only"
        )
        
        response = ModelMetadataResponse(
            model_uri="runs/123/model",
            model_class_path="src.models.TestModel",
            hyperparameter_optimization=hpo_info,
            training_methodology=training_methodology,
            training_metadata={"accuracy": 0.95, "loss": 0.05}
        )
        
        # Assert
        assert response.model_uri == "runs/123/model"
        assert response.model_class_path == "src.models.TestModel"
        assert response.training_metadata["accuracy"] == 0.95
        assert response.hyperparameter_optimization.enabled is True
        assert response.training_methodology.train_test_split_method == "temporal"
    
    def test_model_metadata_response_minimal(self):
        """Test minimal ModelMetadataResponse."""
        # Arrange & Act
        hpo_info = HyperparameterOptimizationInfo(
            enabled=False,
            best_params={},
            best_score=0.0,
            total_trials=0
        )
        
        training_methodology = TrainingMethodologyInfo(
            train_test_split_method="random",
            train_ratio=0.8,
            validation_strategy="holdout",
            preprocessing_fit_scope="train_only"
        )
        
        response = ModelMetadataResponse(
            model_uri="runs/456/model",
            model_class_path="MinimalModel",
            hyperparameter_optimization=hpo_info,
            training_methodology=training_methodology,
            training_metadata={}
        )
        
        # Assert
        assert response.model_uri == "runs/456/model"
        assert response.model_class_path == "MinimalModel"
        assert response.training_metadata == {}
        assert response.hyperparameter_optimization.enabled is False


class TestSchemaIntegration:
    """Test schema integration scenarios."""
    
    def test_end_to_end_schema_workflow(self):
        """Test complete schema workflow from SQL to response."""
        # Arrange
        sql_template = """
        SELECT features.*, target
        FROM `{{ gcp_project_id }}.ml.features` features
        JOIN `{{ gcp_project_id }}.ml.targets` targets
            ON features.user_id = targets.user_id
        WHERE features.user_id = {{ user_id }}
        AND features.campaign_id = {{ campaign_id }}
        """
        
        # Act - Extract PK fields
        pk_fields = get_pk_from_loader_sql(sql_template)
        
        # Act - Create dynamic model
        DynamicModel = create_dynamic_prediction_request("TestModel", pk_fields)
        
        # Act - Create request instance
        request = DynamicModel(user_id="user123", campaign_id="camp456")
        
        # Act - Create response
        response = BatchPredictionResponse(
            predictions=[{"prediction": 0.85}],
            model_uri="runs/test/model",
            sample_count=1
        )
        
        # Assert
        assert pk_fields == ["campaign_id", "user_id"]
        assert request.user_id == "user123"
        assert request.campaign_id == "camp456"
        assert response.predictions[0]["prediction"] == 0.85
    
    def test_schema_serialization(self):
        """Test schema serialization to JSON."""
        # Arrange
        hpo_info = HyperparameterOptimizationInfo(
            enabled=True,
            best_params={"lr": 0.001},
            best_score=0.9,
            total_trials=25
        )
        
        training_methodology = TrainingMethodologyInfo(
            train_test_split_method="temporal",
            train_ratio=0.8,
            validation_strategy="cross_validation",
            preprocessing_fit_scope="train_only"
        )
        
        response = ModelMetadataResponse(
            model_uri="runs/serialize/model",
            model_class_path="SerializationTest",
            hyperparameter_optimization=hpo_info,
            training_methodology=training_methodology,
            training_metadata={"accuracy": 0.9}
        )
        
        # Act
        json_data = response.model_dump()
        
        # Assert
        assert json_data["model_uri"] == "runs/serialize/model"
        assert json_data["model_class_path"] == "SerializationTest"
        assert json_data["training_metadata"]["accuracy"] == 0.9
        assert json_data["hyperparameter_optimization"]["enabled"] is True
        assert json_data["hyperparameter_optimization"]["best_params"]["lr"] == 0.001


class TestSchemaEdgeCases:
    """Test schema edge cases."""
    
    def test_get_pk_malformed_jinja(self):
        """Test handling of malformed Jinja2 syntax."""
        # Arrange
        sql_template = "SELECT * WHERE id = { user_id } AND name = {{ incomplete"
        
        # Act
        result = get_pk_from_loader_sql(sql_template)
        
        # Assert - Should not extract malformed patterns
        assert result == []
    
    def test_create_dynamic_model_special_characters(self):
        """Test dynamic model creation with special characters in field names."""
        # Arrange - Note: In real scenarios, field names should be valid Python identifiers
        model_name = "SpecialModel"
        pk_fields = ["user_id", "campaign_id"]  # Valid identifiers
        
        # Act
        DynamicModel = create_dynamic_prediction_request(model_name, pk_fields)
        
        # Assert
        instance = DynamicModel(user_id="test", campaign_id="special-123")
        assert instance.user_id == "test"
        assert instance.campaign_id == "special-123"
    
    def test_batch_prediction_response_large_payload(self):
        """Test BatchPredictionResponse with large prediction payload."""
        # Arrange
        large_predictions = [{"prediction": i/1000.0} for i in range(1000)]
        
        # Act
        response = BatchPredictionResponse(
            predictions=large_predictions,
            model_uri="runs/large/model",
            sample_count=1000
        )
        
        # Assert
        assert len(response.predictions) == 1000
        assert response.sample_count == 1000
        assert response.predictions[999]["prediction"] == 0.999