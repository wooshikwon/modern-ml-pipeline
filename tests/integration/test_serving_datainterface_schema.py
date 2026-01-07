"""
Integration tests for DataInterface-based API schema generation.
Tests the automatic schema generation from Recipe YAML's data_interface.
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from src.serving._context import app_context
from src.serving._lifespan import setup_api_context
from src.serving.schemas import (
    create_batch_prediction_request,
    create_datainterface_based_prediction_request_v2,
)
from src.settings import Settings


class TestDataInterfaceSchemaGeneration:
    """Test automatic schema generation from DataInterface"""

    def test_v2_schema_excludes_target_column(self):
        """Test that target_column is automatically excluded from API schema"""
        # Given: DataInterface schema with target_column
        data_interface_schema = {
            "target_column": "price",  # Should be excluded
            "entity_columns": ["product_id", "store_id"],
            "feature_columns": ["category", "brand", "stock_level", "price"],  # price is target
            "task_type": "regression",
            "all_columns": ["product_id", "store_id", "category", "brand", "stock_level", "price"],
        }

        # When: Generate prediction request schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="TestModel", data_interface_schema=data_interface_schema, exclude_target=True
        )

        # Then: target_column should be excluded
        fields = PredictionRequest.model_fields
        assert "price" not in fields, "Target column 'price' should be excluded"
        assert "product_id" in fields
        assert "store_id" in fields
        assert "category" in fields
        assert "brand" in fields
        assert "stock_level" in fields

    def test_v2_schema_handles_timeseries_columns(self):
        """Test schema generation for timeseries tasks"""
        # Given: Timeseries DataInterface schema
        data_interface_schema = {
            "target_column": "sales",
            "entity_columns": ["store_id"],
            "timestamp_column": "date",  # Timeseries specific
            "feature_columns": ["temperature", "holiday", "sales"],
            "task_type": "timeseries",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="TimeseriesModel",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # Then: Include timestamp but exclude target
        fields = PredictionRequest.model_fields
        assert "date" in fields, "Timestamp column should be included"
        assert "sales" not in fields, "Target should be excluded"
        assert "store_id" in fields
        assert "temperature" in fields
        assert "holiday" in fields

    def test_v2_schema_handles_causal_columns(self):
        """Test schema generation for causal inference tasks"""
        # Given: Causal DataInterface schema
        data_interface_schema = {
            "target_column": "conversion",
            "treatment_column": "treatment_group",  # Causal specific
            "entity_columns": ["user_id"],
            "feature_columns": ["age", "gender", "previous_purchases"],
            "task_type": "causal",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="CausalModel",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # Then: Include treatment but exclude target
        fields = PredictionRequest.model_fields
        assert "treatment_group" in fields, "Treatment column should be included"
        assert "conversion" not in fields, "Target should be excluded"
        assert "user_id" in fields
        assert "age" in fields

    def test_v2_schema_with_no_explicit_features(self):
        """Test schema when feature_columns is None (use all except target/entity)"""
        # Given: Schema without explicit feature_columns
        data_interface_schema = {
            "target_column": "label",
            "entity_columns": ["id"],
            "feature_columns": None,  # Not specified
            "all_columns": ["id", "feature1", "feature2", "feature3", "label"],
            "task_type": "classification",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="AutoFeatureModel",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # Then: Include all columns except target
        fields = PredictionRequest.model_fields
        assert "label" not in fields, "Target should be excluded"
        assert "id" in fields
        assert "feature1" in fields
        assert "feature2" in fields
        assert "feature3" in fields

    def test_batch_request_generation(self):
        """Test batch prediction request generation"""
        # Given: Base prediction request
        data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["user_id"],
            "feature_columns": ["feature1", "feature2"],
            "task_type": "classification",
        }

        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="BatchTest", data_interface_schema=data_interface_schema
        )

        # When: Create batch request
        BatchRequest = create_batch_prediction_request(PredictionRequest)

        # Then: Should have samples field
        fields = BatchRequest.model_fields
        assert "samples" in fields

    def test_schema_field_descriptions(self):
        """Test that generated fields have appropriate descriptions"""
        # Given: DataInterface schema
        data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["user_id"],
            "feature_columns": ["age", "income"],
            "timestamp_column": "date",
            "task_type": "timeseries",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="DescTest", data_interface_schema=data_interface_schema
        )

        # Then: Fields should have descriptions
        assert "Entity column" in PredictionRequest.model_fields["user_id"].description
        assert "Feature column" in PredictionRequest.model_fields["age"].description
        assert "Timestamp column" in PredictionRequest.model_fields["date"].description


class TestAPIContextSetup:
    """Test API context initialization with DataInterface schema"""

    @patch("src.serving._lifespan.mlflow")
    @patch("src.serving._lifespan.bootstrap")
    def test_setup_with_datainterface_schema(self, mock_bootstrap, mock_mlflow):
        """Test API context setup prioritizes DataInterface schema"""
        # Given: Mock model with DataInterface schema
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["id"],
            "feature_columns": ["f1", "f2"],
            "task_type": "classification",
        }

        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        settings = Mock(spec=Settings)

        # When: Setup API context
        setup_api_context(run_id="test_run", settings=settings)

        # Then: Should use DataInterface schema
        assert app_context.PredictionRequest is not None
        assert app_context.model_uri == "runs:/test_run/model"

        # Verify correct schema was created (target excluded)
        fields = app_context.PredictionRequest.model_fields
        assert "target" not in fields
        assert "id" in fields
        assert "f1" in fields
        assert "f2" in fields

    @patch("src.serving._lifespan.mlflow")
    @patch("src.serving._lifespan.bootstrap")
    def test_fallback_to_legacy_schema(self, mock_bootstrap, mock_mlflow):
        """Test fallback to legacy schema when DataInterface not available"""
        # Given: Mock model without DataInterface schema
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        mock_wrapper.data_schema = {"entity_columns": ["user_id", "item_id"]}

        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        settings = Mock(spec=Settings)

        # When: Setup API context
        with patch("src.serving._lifespan.create_dynamic_prediction_request") as mock_create:
            # Use real simple call signature to avoid pydantic internals mocking issues
            class _Dummy(BaseModel):
                user_id: int
                item_id: int

            mock_create.return_value = _Dummy
            setup_api_context(run_id="test_run", settings=settings)

            # Then: Should fall back to legacy method
            mock_create.assert_called_once()
            assert mock_create.call_args[1]["pk_fields"] == ["user_id", "item_id"]


class TestEndToEndSchemaGeneration:
    """End-to-end tests for complete schema generation flow"""

    def test_clustering_task_no_target(self):
        """Test schema for clustering tasks (no target column)"""
        # Given: Clustering task with no target
        data_interface_schema = {
            "target_column": None,  # Clustering has no target
            "entity_columns": ["customer_id"],
            "feature_columns": ["age", "income", "spending_score"],
            "task_type": "clustering",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="ClusteringModel",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # Then: All features should be included
        fields = PredictionRequest.model_fields
        assert "customer_id" in fields
        assert "age" in fields
        assert "income" in fields
        assert "spending_score" in fields

    def test_required_columns_handling(self):
        """Test handling of required_columns from training"""
        # Given: Schema with required_columns from training
        data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["id"],
            "feature_columns": [],  # Empty
            "required_columns": ["computed_feature1", "computed_feature2"],  # From training
            "task_type": "regression",
        }

        # When: Generate schema
        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="RequiredColsModel",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # Then: Required columns should be included
        fields = PredictionRequest.model_fields
        assert "computed_feature1" in fields
        assert "computed_feature2" in fields
        assert "target" not in fields

    def test_schema_validation_with_sample_data(self):
        """Test that generated schema validates correct data"""
        # Given: Schema and sample data
        data_interface_schema = {
            "target_column": "price",
            "entity_columns": ["product_id"],
            "feature_columns": ["brand", "category"],
            "task_type": "regression",
        }

        PredictionRequest = create_datainterface_based_prediction_request_v2(
            model_name="ValidationTest",
            data_interface_schema=data_interface_schema,
            exclude_target=True,
        )

        # When: Create instance with valid data
        valid_data = {"product_id": "PROD123", "brand": "BrandA", "category": "Electronics"}

        # Then: Should validate successfully
        request = PredictionRequest(**valid_data)
        assert request.product_id == "PROD123"
        assert request.brand == "BrandA"

        # And: Should fail with invalid data (missing required field)
        invalid_data = {
            "product_id": "PROD123",
            # Missing 'brand' and 'category'
        }

        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)
