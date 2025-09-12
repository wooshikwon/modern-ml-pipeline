"""
Unit tests for DataInterface v2 schema generation.
Tests the automatic target_column exclusion and various task types.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from src.serving.schemas import (
    create_datainterface_based_prediction_request_v2,
    create_dynamic_prediction_request,
    get_pk_from_loader_sql,
    create_batch_prediction_request,
    MinimalPredictionResponse,
    PredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelMetadataResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
    OptimizationHistoryResponse
)


class TestDataInterfaceV2SchemaGeneration:
    """Test v2 schema generation with target exclusion"""
    
    def test_basic_target_exclusion(self):
        """Test that target_column is excluded by default"""
        schema = {
            'target_column': 'price',
            'entity_columns': ['product_id'],
            'feature_columns': ['brand', 'category', 'stock'],
            'task_type': 'regression'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Test",
            data_interface_schema=schema,
            exclude_target=True
        )
        
        # Target should be excluded
        assert 'price' not in RequestModel.__fields__
        assert 'product_id' in RequestModel.__fields__
        assert 'brand' in RequestModel.__fields__
        
    def test_target_inclusion_option(self):
        """Test that target can be included when exclude_target=False"""
        schema = {
            'target_column': 'label',
            'entity_columns': ['id'],
            'feature_columns': ['f1', 'f2', 'label'],
            'task_type': 'classification'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Test",
            data_interface_schema=schema,
            exclude_target=False  # Include target
        )
        
        # Target should be included
        assert 'label' in RequestModel.__fields__
        assert 'id' in RequestModel.__fields__
        
    def test_timeseries_schema(self):
        """Test timeseries-specific columns"""
        schema = {
            'target_column': 'sales',
            'entity_columns': ['store_id'],
            'timestamp_column': 'date',
            'feature_columns': ['temperature', 'holiday'],
            'task_type': 'timeseries'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Timeseries",
            data_interface_schema=schema
        )
        
        fields = RequestModel.__fields__
        assert 'date' in fields  # Timestamp included
        assert 'sales' not in fields  # Target excluded
        assert 'store_id' in fields
        
    def test_causal_schema(self):
        """Test causal inference schema with treatment column"""
        schema = {
            'target_column': 'outcome',
            'treatment_column': 'treatment',
            'entity_columns': ['user_id'],
            'feature_columns': ['age', 'income'],
            'task_type': 'causal'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Causal",
            data_interface_schema=schema
        )
        
        fields = RequestModel.__fields__
        assert 'treatment' in fields  # Treatment included
        assert 'outcome' not in fields  # Target excluded
        assert 'user_id' in fields
        
    def test_clustering_no_target(self):
        """Test clustering task with no target column"""
        schema = {
            'target_column': None,  # No target in clustering
            'entity_columns': ['customer_id'],
            'feature_columns': ['age', 'income', 'spending'],
            'task_type': 'clustering'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Clustering",
            data_interface_schema=schema
        )
        
        fields = RequestModel.__fields__
        assert 'customer_id' in fields
        assert 'age' in fields
        assert 'income' in fields
        assert 'spending' in fields
        
    def test_all_columns_fallback(self):
        """Test fallback to all_columns when feature_columns is empty"""
        schema = {
            'target_column': 'target',
            'entity_columns': ['id'],
            'feature_columns': [],  # Empty
            'all_columns': ['id', 'f1', 'f2', 'f3', 'target'],
            'task_type': 'classification'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="AllColumns",
            data_interface_schema=schema
        )
        
        fields = RequestModel.__fields__
        assert 'target' not in fields
        assert 'id' in fields
        assert 'f1' in fields
        assert 'f2' in fields
        assert 'f3' in fields
        
    def test_required_columns_handling(self):
        """Test required_columns from training"""
        schema = {
            'target_column': 'y',
            'entity_columns': ['id'],
            'required_columns': ['computed_f1', 'computed_f2', 'y'],
            'task_type': 'regression'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Required",
            data_interface_schema=schema
        )
        
        fields = RequestModel.__fields__
        assert 'y' not in fields  # Target excluded
        assert 'computed_f1' in fields
        assert 'computed_f2' in fields
        
    def test_field_descriptions(self):
        """Test that generated fields have proper descriptions"""
        schema = {
            'target_column': 'target',
            'entity_columns': ['user_id'],
            'feature_columns': ['age'],
            'timestamp_column': 'ts',
            'task_type': 'timeseries'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Desc",
            data_interface_schema=schema
        )
        
        # Check descriptions
        assert 'Entity column' in RequestModel.__fields__['user_id'].field_info.description
        assert 'Feature' in RequestModel.__fields__['age'].field_info.description
        assert 'Timestamp' in RequestModel.__fields__['ts'].field_info.description
        
    def test_duplicate_column_handling(self):
        """Test that duplicate columns are not added multiple times"""
        schema = {
            'target_column': 'y',
            'entity_columns': ['id', 'user_id'],
            'feature_columns': ['id', 'f1', 'f2'],  # 'id' duplicated
            'required_columns': ['f1', 'f3'],  # 'f1' duplicated
            'task_type': 'classification'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Duplicate",
            data_interface_schema=schema
        )
        
        fields = list(RequestModel.__fields__.keys())
        # Each column should appear only once
        assert fields.count('id') == 1
        assert fields.count('f1') == 1
        assert 'user_id' in fields
        assert 'f2' in fields
        assert 'f3' in fields


class TestLegacySchemaFunctions:
    """Test legacy schema functions still work"""
    
    def test_get_pk_from_loader_sql(self):
        """Test PK extraction from SQL template"""
        sql = "SELECT * FROM table WHERE user_id = {{ user_id }} AND campaign_id = {{ campaign_id }}"
        pks = get_pk_from_loader_sql(sql)
        assert pks == ['campaign_id', 'user_id']  # Sorted
        
    def test_get_pk_excludes_env_vars(self):
        """Test that environment variables are excluded"""
        sql = "SELECT * FROM {{ gcp_project_id }}.dataset.table WHERE id = {{ entity_id }}"
        pks = get_pk_from_loader_sql(sql)
        assert pks == ['entity_id']  # gcp_project_id excluded
        
    def test_create_dynamic_prediction_request(self):
        """Test legacy dynamic request creation"""
        RequestModel = create_dynamic_prediction_request(
            model_name="Legacy",
            pk_fields=['user_id', 'item_id']
        )
        
        fields = RequestModel.__fields__
        assert 'user_id' in fields
        assert 'item_id' in fields
        assert len(fields) == 2
        
    def test_batch_prediction_request(self):
        """Test batch request wrapper creation"""
        BaseRequest = create_dynamic_prediction_request(
            model_name="Base",
            pk_fields=['id']
        )
        
        BatchRequest = create_batch_prediction_request(BaseRequest)
        
        assert 'samples' in BatchRequest.__fields__
        assert 'BatchBasePredictionRequest' in BatchRequest.__name__


class TestResponseSchemas:
    """Test response schema models"""
    
    def test_minimal_prediction_response(self):
        """Test minimal response schema"""
        response = MinimalPredictionResponse(
            prediction=[0, 1, 0],
            model_uri="runs:/abc123/model"
        )
        assert response.prediction == [0, 1, 0]
        assert response.model_uri == "runs:/abc123/model"
        
    def test_prediction_response_with_uplift(self):
        """Test legacy uplift response"""
        response = PredictionResponse(
            uplift_score=0.123,
            model_uri="models:/uplift/1"
        )
        assert response.uplift_score == 0.123
        assert response.optimization_enabled is False  # Default
        
    def test_batch_prediction_response(self):
        """Test batch response schema"""
        response = BatchPredictionResponse(
            predictions=[
                {'id': 1, 'score': 0.8},
                {'id': 2, 'score': 0.3}
            ],
            model_uri="runs:/xyz/model",
            sample_count=2
        )
        assert len(response.predictions) == 2
        assert response.sample_count == 2
        
    def test_health_check_response(self):
        """Test health check response"""
        response = HealthCheckResponse(
            status="healthy",
            model_uri="runs:/abc/model",
            model_name="xgboost"
        )
        assert response.status == "healthy"
        assert response.model_name == "xgboost"
        
    def test_model_metadata_response(self):
        """Test metadata response with nested objects"""
        opt_info = HyperparameterOptimizationInfo(
            enabled=True,
            engine="optuna",
            best_params={'n_estimators': 100},
            best_score=0.95,
            total_trials=50,
            pruned_trials=10,
            optimization_time="2h 30m"
        )
        
        method_info = TrainingMethodologyInfo(
            train_test_split_method="stratified",
            train_ratio=0.8,
            validation_strategy="cross-validation",
            preprocessing_fit_scope="train_only",
            random_state=42
        )
        
        response = ModelMetadataResponse(
            model_uri="runs:/abc/model",
            model_class_path="sklearn.ensemble.RandomForestClassifier",
            hyperparameter_optimization=opt_info,
            training_methodology=method_info
        )
        
        assert response.hyperparameter_optimization.enabled is True
        assert response.hyperparameter_optimization.best_score == 0.95
        assert response.training_methodology.train_ratio == 0.8
        
    def test_optimization_history_response(self):
        """Test optimization history response"""
        response = OptimizationHistoryResponse(
            enabled=True,
            optimization_history=[
                {'trial': 1, 'score': 0.8},
                {'trial': 2, 'score': 0.85}
            ],
            search_space={'n_estimators': [50, 100, 200]},
            convergence_info={'converged_at_trial': 45}
        )
        
        assert response.enabled is True
        assert len(response.optimization_history) == 2
        assert 'n_estimators' in response.search_space


class TestSchemaValidation:
    """Test schema validation with actual data"""
    
    def test_valid_request_data(self):
        """Test that valid data passes validation"""
        schema = {
            'target_column': 'price',
            'entity_columns': ['product_id'],
            'feature_columns': ['brand', 'category'],
            'task_type': 'regression'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Valid",
            data_interface_schema=schema
        )
        
        # Valid data
        valid_data = {
            'product_id': 'PROD123',
            'brand': 'BrandA',
            'category': 'Electronics'
        }
        
        request = RequestModel(**valid_data)
        assert request.product_id == 'PROD123'
        assert request.brand == 'BrandA'
        
    def test_invalid_request_missing_field(self):
        """Test that missing required fields raise validation error"""
        schema = {
            'target_column': 'y',
            'entity_columns': ['id'],
            'feature_columns': ['f1', 'f2'],
            'task_type': 'classification'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Invalid",
            data_interface_schema=schema
        )
        
        # Missing 'f2'
        invalid_data = {
            'id': '123',
            'f1': 'value1'
            # 'f2' missing
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RequestModel(**invalid_data)
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('f2',) for error in errors)
        
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored"""
        schema = {
            'target_column': 'y',
            'entity_columns': ['id'],
            'feature_columns': ['f1'],
            'task_type': 'classification'
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Extra",
            data_interface_schema=schema
        )
        
        # Extra field 'extra_field'
        data = {
            'id': '123',
            'f1': 'value1',
            'extra_field': 'should_be_ignored'
        }
        
        request = RequestModel(**data)
        assert request.id == '123'
        assert request.f1 == 'value1'
        assert not hasattr(request, 'extra_field')


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_schema(self):
        """Test handling of empty schema"""
        schema = {
            'target_column': None,
            'entity_columns': [],
            'feature_columns': [],
            'task_type': ''
        }
        
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="Empty",
            data_interface_schema=schema
        )
        
        # Should create model with no fields
        assert len(RequestModel.__fields__) == 0
        
    def test_none_values_in_schema(self):
        """Test handling of None values"""
        schema = {
            'target_column': None,
            'entity_columns': None,
            'feature_columns': None,
            'timestamp_column': None,
            'treatment_column': None,
            'task_type': 'unknown'
        }
        
        # Should not crash
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="NoneValues",
            data_interface_schema=schema
        )
        
        assert RequestModel is not None
        
    def test_special_characters_in_column_names(self):
        """Test handling of special characters in column names"""
        schema = {
            'target_column': 'target-column',
            'entity_columns': ['user.id', 'item_id'],
            'feature_columns': ['feature-1', 'feature_2'],
            'task_type': 'classification'
        }
        
        # Should handle special characters
        RequestModel = create_datainterface_based_prediction_request_v2(
            model_name="SpecialChars",
            data_interface_schema=schema
        )
        
        # Python identifiers can't have dots or hyphens, 
        # but Pydantic can handle them as field names
        assert 'user.id' in RequestModel.__fields__
        assert 'feature-1' in RequestModel.__fields__