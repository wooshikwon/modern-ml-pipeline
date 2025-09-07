"""
Unit tests for the schema utilities module.
Tests data validation, schema consistency, and type conversion functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.utils.system.schema_utils import (
    validate_schema,
    convert_schema,
    SchemaConsistencyValidator,
    generate_training_schema_metadata
)


class TestValidateSchema:
    """Test validate_schema function for data validation."""

    def setup_method(self):
        """Setup test fixtures."""
        # Mock Settings objects for different scenarios
        self.mock_settings = MagicMock()
        
        # Basic classification setup
        self.mock_settings.recipe.data.data_interface.entity_columns = ['user_id', 'product_id']
        self.mock_settings.recipe.data.data_interface.target_column = 'target'
        self.mock_settings.recipe.data.data_interface.task_type = 'classification'
        self.mock_settings.recipe.data.data_interface.treatment_column = None
        self.mock_settings.recipe.data.fetcher.timestamp_column = 'timestamp'
        
        # Sample valid dataframe
        self.valid_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'product_id': ['A', 'B', 'C', 'D'],
            'timestamp': pd.date_range('2023-01-01', periods=4),
            'target': [0, 1, 0, 1],
            'feature1': [10.5, 20.3, 15.7, 12.1],
            'feature2': ['high', 'medium', 'low', 'high']
        })

    def test_validate_schema_success_original_data(self):
        """Test successful validation for original data (not training)."""
        validate_schema(self.valid_df, self.mock_settings, for_training=False)
        # Should pass without raising exception

    def test_validate_schema_success_training_data(self):
        """Test successful validation for training data."""
        # Training data excludes entity and timestamp columns
        training_df = self.valid_df.drop(columns=['user_id', 'product_id', 'timestamp'])
        validate_schema(training_df, self.mock_settings, for_training=True)
        # Should pass without raising exception

    def test_validate_schema_missing_entity_column(self):
        """Test validation failure when entity column is missing."""
        df_missing_entity = self.valid_df.drop(columns=['user_id'])
        
        with pytest.raises(TypeError, match="필수 컬럼 누락: 'user_id'"):
            validate_schema(df_missing_entity, self.mock_settings, for_training=False)

    def test_validate_schema_missing_timestamp_column(self):
        """Test validation failure when timestamp column is missing."""
        df_missing_timestamp = self.valid_df.drop(columns=['timestamp'])
        
        with pytest.raises(TypeError, match="필수 컬럼 누락: 'timestamp'"):
            validate_schema(df_missing_timestamp, self.mock_settings, for_training=False)

    def test_validate_schema_missing_target_column(self):
        """Test validation failure when target column is missing."""
        df_missing_target = self.valid_df.drop(columns=['target'])
        
        with pytest.raises(TypeError, match="필수 컬럼 누락: 'target'"):
            validate_schema(df_missing_target, self.mock_settings, for_training=False)

    def test_validate_schema_clustering_task_no_target_required(self):
        """Test that clustering tasks don't require target column."""
        self.mock_settings.recipe.data.data_interface.task_type = 'clustering'
        self.mock_settings.recipe.data.data_interface.target_column = None
        
        df_no_target = self.valid_df.drop(columns=['target'])
        validate_schema(df_no_target, self.mock_settings, for_training=False)
        # Should pass without raising exception

    def test_validate_schema_causal_task_requires_treatment(self):
        """Test that causal tasks require treatment column."""
        self.mock_settings.recipe.data.data_interface.task_type = 'causal'
        self.mock_settings.recipe.data.data_interface.treatment_column = 'treatment'
        
        # Add treatment column to valid data
        valid_causal_df = self.valid_df.copy()
        valid_causal_df['treatment'] = [1, 0, 1, 0]
        
        validate_schema(valid_causal_df, self.mock_settings, for_training=False)
        # Should pass

    def test_validate_schema_causal_task_missing_treatment(self):
        """Test causal task validation failure when treatment column is missing."""
        self.mock_settings.recipe.data.data_interface.task_type = 'causal'
        self.mock_settings.recipe.data.data_interface.treatment_column = 'treatment'
        
        with pytest.raises(TypeError, match="필수 컬럼 누락: 'treatment'"):
            validate_schema(self.valid_df, self.mock_settings, for_training=False)

    def test_validate_schema_invalid_timestamp_type(self):
        """Test validation failure when timestamp column has invalid type."""
        df_invalid_timestamp = self.valid_df.copy()
        df_invalid_timestamp['timestamp'] = ['not-a-date', 'invalid', 'bad-timestamp', 'wrong']
        
        with pytest.raises(TypeError, match="Timestamp 컬럼 'timestamp' 타입 오류"):
            validate_schema(df_invalid_timestamp, self.mock_settings, for_training=False)

    def test_validate_schema_convertible_timestamp_type(self):
        """Test that convertible timestamp types are allowed with warning."""
        df_convertible_timestamp = self.valid_df.copy()
        df_convertible_timestamp['timestamp'] = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        
        with patch('src.utils.system.schema_utils.logger') as mock_logger:
            validate_schema(df_convertible_timestamp, self.mock_settings, for_training=False)
            # Should log conversion message
            mock_logger.info.assert_any_call("Timestamp 컬럼 'timestamp' 자동 변환 가능")

    def test_validate_schema_no_fetcher_config(self):
        """Test validation when fetcher config is None."""
        self.mock_settings.recipe.data.fetcher = None
        
        # Should work without timestamp validation
        df_no_timestamp = self.valid_df.drop(columns=['timestamp'])
        validate_schema(df_no_timestamp, self.mock_settings, for_training=False)

    def test_validate_schema_training_mode_excludes_entity_timestamp(self):
        """Test that training mode doesn't require entity/timestamp columns."""
        # Training data with only features and target
        training_df = pd.DataFrame({
            'feature1': [10.5, 20.3, 15.7, 12.1],
            'feature2': ['high', 'medium', 'low', 'high']
        })
        
        validate_schema(training_df, self.mock_settings, for_training=True)
        # Should pass without entity/timestamp validation

    @patch('src.utils.system.schema_utils.logger')
    def test_validate_schema_logging(self, mock_logger):
        """Test that appropriate logging occurs during validation."""
        validate_schema(self.valid_df, self.mock_settings, for_training=False)
        
        # Check logging calls
        mock_logger.info.assert_any_call("모델 입력 데이터 스키마를 검증합니다... (for_training: False)")
        mock_logger.info.assert_any_call("스키마 검증 성공 (task_type: classification)")

    def test_validate_schema_error_message_details(self):
        """Test that error messages contain helpful details."""
        df_multiple_issues = pd.DataFrame({
            'feature1': [1, 2, 3]
        })
        
        with pytest.raises(TypeError) as exc_info:
            validate_schema(df_multiple_issues, self.mock_settings, for_training=False)
        
        error_message = str(exc_info.value)
        assert "모델 입력 데이터 스키마 검증 실패" in error_message
        assert "필수 컬럼:" in error_message
        assert "실제 컬럼:" in error_message


class TestConvertSchema:
    """Test convert_schema function for type conversion."""

    def test_convert_schema_numeric_conversion(self):
        """Test conversion to numeric types."""
        df = pd.DataFrame({
            'numeric_col': ['10', '20', '30', '40'],
            'other_col': ['a', 'b', 'c', 'd']
        })
        
        expected_schema = {
            'numeric_col': 'numeric',
            'other_col': 'unchanged'
        }
        
        result_df = convert_schema(df, expected_schema)
        
        assert pd.api.types.is_numeric_dtype(result_df['numeric_col'])
        assert result_df['numeric_col'].tolist() == [10.0, 20.0, 30.0, 40.0]
        assert result_df['other_col'].tolist() == ['a', 'b', 'c', 'd']

    def test_convert_schema_category_conversion(self):
        """Test conversion to category types."""
        df = pd.DataFrame({
            'category_col': ['A', 'B', 'A', 'C'],
            'other_col': [1, 2, 3, 4]
        })
        
        expected_schema = {
            'category_col': 'category'
        }
        
        result_df = convert_schema(df, expected_schema)
        
        assert pd.api.types.is_categorical_dtype(result_df['category_col'])
        assert set(result_df['category_col'].cat.categories) == {'A', 'B', 'C'}

    def test_convert_schema_numeric_with_errors(self):
        """Test numeric conversion with invalid values."""
        df = pd.DataFrame({
            'mixed_col': ['10', 'invalid', '30', 'bad_number']
        })
        
        expected_schema = {
            'mixed_col': 'numeric'
        }
        
        result_df = convert_schema(df, expected_schema)
        
        # Invalid values should become NaN
        assert pd.isna(result_df['mixed_col'].iloc[1])
        assert pd.isna(result_df['mixed_col'].iloc[3])
        assert result_df['mixed_col'].iloc[0] == 10.0
        assert result_df['mixed_col'].iloc[2] == 30.0

    def test_convert_schema_missing_columns(self):
        """Test that missing columns in schema are ignored."""
        df = pd.DataFrame({
            'existing_col': [1, 2, 3]
        })
        
        expected_schema = {
            'existing_col': 'numeric',
            'missing_col': 'numeric'
        }
        
        result_df = convert_schema(df, expected_schema)
        
        # Should not fail and only process existing columns
        assert 'existing_col' in result_df.columns
        assert 'missing_col' not in result_df.columns

    def test_convert_schema_preserves_original_dataframe(self):
        """Test that original dataframe is not modified."""
        original_df = pd.DataFrame({
            'col1': ['10', '20', '30']
        })
        
        expected_schema = {
            'col1': 'numeric'
        }
        
        result_df = convert_schema(original_df, expected_schema)
        
        # Original should remain unchanged
        assert original_df['col1'].dtype == 'object'
        assert result_df['col1'].dtype in ['float64', 'int64']

    @patch('src.utils.system.schema_utils.logger')
    def test_convert_schema_logging(self, mock_logger):
        """Test that conversion process is logged."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        schema = {'col1': 'numeric'}
        
        convert_schema(df, schema)
        
        mock_logger.info.assert_any_call("데이터 타입 변환을 시작합니다...")
        mock_logger.info.assert_any_call("데이터 타입 변환 완료.")


class TestSchemaConsistencyValidator:
    """Test SchemaConsistencyValidator class for advanced schema validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.training_schema = {
            'entity_columns': ['user_id', 'product_id'],
            'timestamp_column': 'timestamp',
            'target_column': 'target',
            'task_type': 'classification',
            'training_columns': ['user_id', 'product_id', 'timestamp', 'feature1', 'feature2'],
            'inference_columns': ['user_id', 'product_id'],
            'column_types': {
                'user_id': 'int64',
                'product_id': 'object',
                'timestamp': 'datetime64[ns]',
                'feature1': 'float64',
                'feature2': 'object'
            },
            'schema_version': '2.0',
            'created_at': '2023-01-01T00:00:00',
            'point_in_time_safe': True,
            'sql_injection_safe': True
        }
        
        self.validator = SchemaConsistencyValidator(self.training_schema)
        
        self.valid_inference_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'product_id': ['A', 'B', 'C'],
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'feature1': [10.5, 20.3, 15.7],
            'feature2': ['high', 'medium', 'low']
        })

    def test_schema_consistency_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.training_schema == self.training_schema
        assert isinstance(self.validator, SchemaConsistencyValidator)

    @patch('src.utils.system.schema_utils.logger')
    def test_schema_consistency_validator_initialization_logging(self, mock_logger):
        """Test that initialization is logged."""
        SchemaConsistencyValidator(self.training_schema)
        
        mock_logger.info.assert_called_with(
            "SchemaConsistencyValidator 초기화 완료 - 검증 대상: 2개 컬럼"
        )

    def test_validate_inference_consistency_success(self):
        """Test successful inference consistency validation."""
        result = self.validator.validate_inference_consistency(self.valid_inference_df)
        
        assert result is True

    def test_validate_inference_consistency_missing_required_column(self):
        """Test validation failure when required column is missing."""
        invalid_df = self.valid_inference_df.drop(columns=['user_id'])
        
        with pytest.raises(ValueError, match="Inference 데이터에 Training 시 필수 컬럼이 없습니다"):
            self.validator.validate_inference_consistency(invalid_df)

    def test_validate_inference_consistency_extra_columns_warning(self):
        """Test that extra columns generate warning but don't fail."""
        df_with_extra = self.valid_inference_df.copy()
        df_with_extra['extra_column'] = [1, 2, 3]
        
        with patch('src.utils.system.schema_utils.logger') as mock_logger:
            result = self.validator.validate_inference_consistency(df_with_extra)
            
            assert result is True
            # Should log warning about extra columns
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'Training에 없던 추가 컬럼 발견' in str(call)]
            assert len(warning_calls) > 0

    def test_validate_dtype_compatibility_compatible_types(self):
        """Test that compatible types pass validation."""
        # Create inference data with compatible types
        compatible_df = pd.DataFrame({
            'user_id': [1, 2, 3],  # int64 compatible with int64
            'product_id': ['A', 'B', 'C']  # object compatible with object
        })
        
        # This should pass without error
        result = self.validator.validate_inference_consistency(compatible_df)
        assert result is True

    def test_validate_dtype_compatibility_incompatible_types(self):
        """Test that incompatible types fail validation."""
        # Create inference data with incompatible types
        incompatible_df = pd.DataFrame({
            'user_id': ['str1', 'str2', 'str3'],  # string instead of int64
            'product_id': [1, 2, 3]  # int instead of object
        })
        
        with pytest.raises(ValueError, match="컬럼.*타입 불일치"):
            self.validator.validate_inference_consistency(incompatible_df)

    def test_is_compatible_dtype_same_types(self):
        """Test that identical types are compatible."""
        assert self.validator._is_compatible_dtype('int64', 'int64') is True
        assert self.validator._is_compatible_dtype('object', 'object') is True

    def test_is_compatible_dtype_numeric_compatibility(self):
        """Test numeric type compatibility."""
        assert self.validator._is_compatible_dtype('int64', 'int32') is True
        assert self.validator._is_compatible_dtype('float64', 'float32') is True
        assert self.validator._is_compatible_dtype('int', 'int64') is True

    def test_is_compatible_dtype_string_compatibility(self):
        """Test string type compatibility."""
        assert self.validator._is_compatible_dtype('object', 'string') is True
        assert self.validator._is_compatible_dtype('string', 'object') is True

    def test_is_compatible_dtype_datetime_compatibility(self):
        """Test datetime type compatibility."""
        assert self.validator._is_compatible_dtype('datetime64', 'datetime64[ns]') is True
        assert self.validator._is_compatible_dtype('datetime', 'datetime64') is True

    def test_is_compatible_dtype_bool_compatibility(self):
        """Test boolean type compatibility."""
        assert self.validator._is_compatible_dtype('bool', 'boolean') is True
        assert self.validator._is_compatible_dtype('boolean', 'bool') is True

    def test_is_compatible_dtype_incompatible_types(self):
        """Test that incompatible types are correctly identified."""
        assert self.validator._is_compatible_dtype('int64', 'object') is False
        assert self.validator._is_compatible_dtype('string', 'int') is False
        assert self.validator._is_compatible_dtype('float64', 'bool') is False

    def test_validate_point_in_time_columns_success(self):
        """Test successful point-in-time column validation."""
        # Valid inference data with proper entity and timestamp columns
        result = self.validator.validate_inference_consistency(self.valid_inference_df)
        assert result is True

    def test_validate_point_in_time_columns_missing_entity(self):
        """Test point-in-time validation failure when entity column is missing."""
        df_missing_entity = self.valid_inference_df.drop(columns=['user_id'])
        
        with pytest.raises(ValueError, match="Inference 데이터에 Training 시 필수 컬럼이 없습니다"):
            self.validator.validate_inference_consistency(df_missing_entity)

    def test_validate_point_in_time_columns_invalid_timestamp_type(self):
        """Test point-in-time validation failure with invalid timestamp type."""
        invalid_timestamp_df = self.valid_inference_df.copy()
        invalid_timestamp_df['timestamp'] = ['2023-01-01', '2023-01-02', '2023-01-03']
        
        with pytest.raises(ValueError, match="Timestamp 컬럼.*datetime 타입이 아닙니다"):
            self.validator.validate_inference_consistency(invalid_timestamp_df)

    def test_validate_point_in_time_columns_future_data_warning(self):
        """Test warning for future data in timestamp column."""
        future_df = self.valid_inference_df.copy()
        future_df['timestamp'] = pd.date_range(start=pd.Timestamp.now() + timedelta(days=1), periods=3)
        
        with patch('src.utils.system.schema_utils.logger') as mock_logger:
            result = self.validator.validate_inference_consistency(future_df)
            
            assert result is True
            # Should log warning about future data
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if '미래 데이터 감지' in str(call)]
            assert len(warning_calls) > 0

    @patch('src.utils.system.schema_utils.logger')
    def test_validate_inference_consistency_comprehensive_logging(self, mock_logger):
        """Test comprehensive logging throughout validation process."""
        self.validator.validate_inference_consistency(self.valid_inference_df)
        
        # Check that all phases are logged
        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        
        assert any("Phase 1: 기본 스키마 구조 검증 시작" in msg for msg in log_messages)
        assert any("Phase 2: Training/Inference 컬럼 일관성 검증 시작" in msg for msg in log_messages)
        assert any("Phase 3: 고급 타입 호환성 검증 시작" in msg for msg in log_messages)
        assert any("Phase 4: Point-in-Time 컬럼 특별 검증 시작" in msg for msg in log_messages)
        assert any("모든 스키마 일관성 검증 통과" in msg for msg in log_messages)


class TestGenerateTrainingSchemaMetadata:
    """Test generate_training_schema_metadata function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.training_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'product_id': ['A', 'B', 'C', 'D'],
            'timestamp': pd.date_range('2023-01-01', periods=4),
            'feature1': [10.5, 20.3, 15.7, 12.1],
            'feature2': ['high', 'medium', 'low', 'high'],
            'target': [0, 1, 0, 1]
        })
        
        self.data_interface_config = {
            'entity_columns': ['user_id', 'product_id'],
            'timestamp_column': 'timestamp',
            'target_column': 'target',
            'task_type': 'classification'
        }

    def test_generate_training_schema_metadata_complete(self):
        """Test generation of complete schema metadata."""
        metadata = generate_training_schema_metadata(
            self.training_df, 
            self.data_interface_config
        )
        
        # Check all required fields
        assert metadata['entity_columns'] == ['user_id', 'product_id']
        assert metadata['timestamp_column'] == 'timestamp'
        assert metadata['target_column'] == 'target'
        assert metadata['task_type'] == 'classification'
        
        assert metadata['training_columns'] == list(self.training_df.columns)
        assert metadata['inference_columns'] == ['user_id', 'product_id']
        assert len(metadata['column_types']) == len(self.training_df.columns)
        
        assert metadata['schema_version'] == '2.0'
        assert 'created_at' in metadata
        assert metadata['point_in_time_safe'] is True
        assert metadata['sql_injection_safe'] is True
        assert metadata['total_training_samples'] == 4
        assert metadata['column_count'] == 6

    def test_generate_training_schema_metadata_column_types(self):
        """Test that column types are correctly captured."""
        metadata = generate_training_schema_metadata(
            self.training_df, 
            self.data_interface_config
        )
        
        column_types = metadata['column_types']
        
        assert 'int64' in column_types['user_id']
        assert 'object' in column_types['product_id']
        assert 'datetime64' in column_types['timestamp']
        assert 'float64' in column_types['feature1']
        assert 'object' in column_types['feature2']
        assert 'int64' in column_types['target']

    def test_generate_training_schema_metadata_no_target(self):
        """Test metadata generation without target column (clustering)."""
        clustering_config = self.data_interface_config.copy()
        clustering_config['target_column'] = None
        clustering_config['task_type'] = 'clustering'
        
        metadata = generate_training_schema_metadata(
            self.training_df, 
            clustering_config
        )
        
        assert metadata['target_column'] is None
        assert metadata['task_type'] == 'clustering'

    def test_generate_training_schema_metadata_empty_entity_columns(self):
        """Test metadata generation with empty entity columns."""
        config_no_entities = self.data_interface_config.copy()
        config_no_entities['entity_columns'] = []
        
        metadata = generate_training_schema_metadata(
            self.training_df, 
            config_no_entities
        )
        
        assert metadata['entity_columns'] == []
        assert metadata['inference_columns'] == []

    def test_generate_training_schema_metadata_missing_config_fields(self):
        """Test metadata generation with missing configuration fields."""
        minimal_config = {'task_type': 'regression'}
        
        metadata = generate_training_schema_metadata(
            self.training_df, 
            minimal_config
        )
        
        # Should handle missing fields gracefully
        assert metadata['entity_columns'] == []
        assert metadata['timestamp_column'] == ''
        assert metadata['target_column'] is None
        assert metadata['task_type'] == 'regression'

    @patch('src.utils.system.schema_utils.logger')
    def test_generate_training_schema_metadata_logging(self, mock_logger):
        """Test that metadata generation is logged."""
        metadata = generate_training_schema_metadata(
            self.training_df, 
            self.data_interface_config
        )
        
        # Check logging
        mock_logger.info.assert_called_with(
            "✅ Training 스키마 메타데이터 생성 완료: 2개 inference 컬럼, 6개 전체 컬럼"
        )

    def test_generate_training_schema_metadata_datetime_format(self):
        """Test that created_at timestamp is in ISO format."""
        metadata = generate_training_schema_metadata(
            self.training_df, 
            self.data_interface_config
        )
        
        created_at = metadata['created_at']
        # Should be parseable as datetime
        parsed_datetime = datetime.fromisoformat(created_at.replace('Z', '+00:00') if created_at.endswith('Z') else created_at)
        assert isinstance(parsed_datetime, datetime)


class TestSchemaUtilsIntegration:
    """Integration tests for schema utilities."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.training_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'product_id': ['A', 'B', 'C', 'A', 'B'],
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'feature1': [10.5, 20.3, 15.7, 12.1, 18.9],
            'feature2': ['high', 'medium', 'low', 'high', 'medium'],
            'target': [0, 1, 0, 1, 0]
        })
        
        self.config = {
            'entity_columns': ['user_id', 'product_id'],
            'timestamp_column': 'timestamp',
            'target_column': 'target',
            'task_type': 'classification'
        }

    def test_full_schema_workflow(self):
        """Test complete schema validation workflow from training to inference."""
        # 1. Generate training schema metadata
        training_metadata = generate_training_schema_metadata(self.training_df, self.config)
        
        # 2. Create validator from training metadata
        validator = SchemaConsistencyValidator(training_metadata)
        
        # 3. Create inference data (subset of training data)
        inference_df = pd.DataFrame({
            'user_id': [6, 7, 8],
            'product_id': ['C', 'D', 'A'],
            'timestamp': pd.date_range('2023-01-06', periods=3),
            'feature1': [22.1, 14.5, 16.8],
            'feature2': ['low', 'high', 'medium']
        })
        
        # 4. Validate inference consistency
        result = validator.validate_inference_consistency(inference_df)
        assert result is True

    def test_schema_conversion_integration(self):
        """Test integration of schema validation with type conversion."""
        # Create data that needs type conversion
        df_with_strings = pd.DataFrame({
            'numeric_feature': ['10.5', '20.3', '15.7'],
            'category_feature': ['A', 'B', 'C']
        })
        
        # Define conversion schema
        conversion_schema = {
            'numeric_feature': 'numeric',
            'category_feature': 'category'
        }
        
        # Convert types
        converted_df = convert_schema(df_with_strings, conversion_schema)
        
        # Validate converted types
        assert pd.api.types.is_numeric_dtype(converted_df['numeric_feature'])
        assert pd.api.types.is_categorical_dtype(converted_df['category_feature'])

    def test_error_handling_integration(self):
        """Test error handling across different schema utilities."""
        # Test with completely invalid data
        invalid_df = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        mock_settings = MagicMock()
        mock_settings.recipe.data.data_interface.entity_columns = ['required_column']
        mock_settings.recipe.data.data_interface.target_column = 'target'
        mock_settings.recipe.data.data_interface.task_type = 'classification'
        mock_settings.recipe.data.fetcher.timestamp_column = 'timestamp'
        
        # Should raise detailed error
        with pytest.raises(TypeError, match="모델 입력 데이터 스키마 검증 실패"):
            validate_schema(invalid_df, mock_settings, for_training=False)

    @patch('src.utils.system.schema_utils.logger')
    def test_comprehensive_logging_integration(self, mock_logger):
        """Test comprehensive logging across all schema utilities."""
        # Test logging from all major functions
        
        # 1. Schema validation logging
        mock_settings = MagicMock()
        mock_settings.recipe.data.data_interface.entity_columns = ['user_id']
        mock_settings.recipe.data.data_interface.target_column = 'target'
        mock_settings.recipe.data.data_interface.task_type = 'classification'
        mock_settings.recipe.data.fetcher.timestamp_column = 'timestamp'
        
        simple_df = pd.DataFrame({
            'user_id': [1, 2],
            'timestamp': pd.date_range('2023-01-01', periods=2),
            'target': [0, 1]
        })
        
        validate_schema(simple_df, mock_settings, for_training=False)
        
        # 2. Schema conversion logging
        convert_schema(simple_df, {'user_id': 'numeric'})
        
        # 3. Metadata generation logging
        generate_training_schema_metadata(simple_df, self.config)
        
        # Verify that logging occurred from all components
        log_calls = mock_logger.info.call_args_list
        log_messages = [call.args[0] for call in log_calls]
        
        assert any("스키마를 검증합니다" in msg for msg in log_messages)
        assert any("데이터 타입 변환을 시작합니다" in msg for msg in log_messages)  
        assert any("스키마 메타데이터 생성 완료" in msg for msg in log_messages)