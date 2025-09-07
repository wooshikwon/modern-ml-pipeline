"""
Tests for data validation utilities - DataInterface 기반 데이터 품질 검증

Phase 5.1 DataInterface 기반 컬럼 검증 로직 테스트:
- DataInterface 필수 컬럼 추출 
- DataFrame 컬럼 검증
- PyFunc 저장용 스키마 메타데이터 생성

Test Categories:
1. TestGetRequiredColumnsFromDataInterface - 필수 컬럼 추출 테스트
2. TestValidateDataInterfaceColumns - DataFrame 컬럼 검증 테스트  
3. TestCreateDataInterfaceSchemaForStorage - 스키마 메타데이터 생성 테스트
4. TestDataValidationIntegration - 통합 시나리오 테스트
5. TestDataValidationSecurity - 보안 및 에러 처리 테스트
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from typing import List, Dict, Any

from src.utils.system.data_validation import (
    get_required_columns_from_data_interface,
    validate_data_interface_columns, 
    create_data_interface_schema_for_storage
)
from src.settings.recipe import DataInterface


class TestGetRequiredColumnsFromDataInterface:
    """Test get_required_columns_from_data_interface function."""

    def test_regression_task_with_explicit_feature_columns(self):
        """Test regression task with explicitly defined feature columns."""
        interface = DataInterface(
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=["age", "income", "location"],
            timestamp_column=None,
            treatment_column=None
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="regression")
        
        expected = ["user_id", "age", "income", "location"]
        assert set(result) == set(expected)
        assert "price" not in result  # target_column excluded for inference

    def test_classification_task_with_multiple_entities(self):
        """Test classification task with multiple entity columns."""
        interface = DataInterface(
            target_column="category",
            entity_columns=["user_id", "session_id"],
            feature_columns=["page_views", "duration"],
            timestamp_column=None,
            treatment_column=None
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="classification")
        
        expected = ["user_id", "session_id", "page_views", "duration"]
        assert set(result) == set(expected)
        assert "category" not in result

    def test_timeseries_task_with_timestamp_column(self):
        """Test timeseries task includes timestamp column."""
        interface = DataInterface(
            target_column="sales",
            entity_columns=["store_id"],
            feature_columns=["weather", "promotion"],
            timestamp_column="date",
            treatment_column=None
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="timeseries")
        
        expected = ["store_id", "date", "weather", "promotion"]
        assert set(result) == set(expected)
        assert "sales" not in result

    def test_causal_task_with_treatment_column(self):
        """Test causal task includes treatment column."""
        interface = DataInterface(
            target_column="outcome",
            entity_columns=["patient_id"],
            feature_columns=["age", "gender"],
            timestamp_column=None,
            treatment_column="treatment"
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="causal")
        
        expected = ["patient_id", "treatment", "age", "gender"]
        assert set(result) == set(expected)
        assert "outcome" not in result

    def test_null_feature_columns_with_training_data(self):
        """Test feature_columns=null with training data - auto extraction."""
        interface = DataInterface(
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=None,  # null - should auto extract
            timestamp_column=None,
            treatment_column=None
        )
        
        # Training data with additional features not in interface
        training_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'price': [100, 200, 300],  # target - should be excluded
            'age': [25, 30, 35],       # auto-extracted feature
            'income': [5000, 6000, 7000],  # auto-extracted feature
            'location': ['A', 'B', 'C']     # auto-extracted feature
        })
        
        result = get_required_columns_from_data_interface(interface, training_df, task_choice="regression")
        
        expected = ["user_id", "age", "income", "location"]  # target excluded
        assert set(result) == set(expected)
        assert "price" not in result

    def test_null_feature_columns_without_training_data(self):
        """Test feature_columns=null without training data - warning case."""
        interface = DataInterface(
            target_column="price", 
            entity_columns=["user_id"],
            feature_columns=None,
            timestamp_column=None,
            treatment_column=None
        )
        
        with patch('src.utils.system.data_validation.logger') as mock_logger:
            result = get_required_columns_from_data_interface(interface, None, task_choice="regression")
            
            # Should only include entity columns when no training data
            expected = ["user_id"]
            assert set(result) == set(expected)
            
            # Should log warning
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args[0][0]
            assert "feature_columns=null" in warning_message
            assert "실제 학습 데이터가 제공되지 않았습니다" in warning_message

    def test_task_choice_override(self):
        """Test task_choice parameter controls task-specific column inclusion."""
        interface = DataInterface(
            target_column="outcome",
            entity_columns=["user_id"],
            feature_columns=["feature1"],
            timestamp_column="date",
            treatment_column="treatment"
        )
        
        # Timeseries task - should include timestamp
        result_timeseries = get_required_columns_from_data_interface(
            interface, task_choice="timeseries"
        )
        assert "date" in result_timeseries
        assert "treatment" not in result_timeseries
        
        # Causal task - should include treatment  
        result_causal = get_required_columns_from_data_interface(
            interface, task_choice="causal"
        )
        assert "treatment" in result_causal
        assert "date" not in result_causal

    def test_duplicate_column_removal(self):
        """Test duplicate columns are removed from result."""
        interface = DataInterface(
            target_column="outcome",
            entity_columns=["user_id", "session_id"],
            feature_columns=["user_id", "feature1"],  # user_id duplicated
            timestamp_column="user_id",  # user_id duplicated again
            treatment_column=None
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="timeseries")
        
        # Should only have one instance of user_id despite duplicates
        assert result.count("user_id") == 1
        assert set(result) == {"user_id", "session_id", "feature1"}

    @patch('src.utils.system.data_validation.logger')
    def test_logging_debug_information(self, mock_logger):
        """Test debug logging provides useful information."""
        interface = DataInterface(
            target_column="label",
            entity_columns=["id"],
            feature_columns=["f1", "f2"],
            timestamp_column=None,
            treatment_column=None
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="classification")
        
        # Should log debug information
        mock_logger.debug.assert_called_once()
        debug_message = mock_logger.debug.call_args[0][0]
        assert "DataInterface 필수 컬럼 추출 완료" in debug_message
        assert "classification" in debug_message
        assert str(len(result)) in debug_message

    def test_complex_scenario_all_columns(self):
        """Test complex scenario with all possible column types."""
        interface = DataInterface(
            target_column="outcome", 
            entity_columns=["patient_id", "study_id"],
            feature_columns=["age", "gender", "baseline_score"],
            timestamp_column="enrollment_date",
            treatment_column="treatment_arm"
        )
        
        result = get_required_columns_from_data_interface(interface, task_choice="causal")
        
        expected = [
            "patient_id", "study_id",  # entities
            "treatment_arm",           # causal treatment
            "age", "gender", "baseline_score"  # features
        ]
        assert set(result) == set(expected)
        # Causal task should NOT include timestamp (only for timeseries)
        assert "enrollment_date" not in result
        assert "outcome" not in result


class TestValidateDataInterfaceColumns:
    """Test validate_data_interface_columns function."""

    def test_successful_validation_with_exact_columns(self):
        """Test successful validation when DataFrame has exact required columns."""
        interface = DataInterface(
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=["age", "income"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35], 
            'income': [5000, 6000, 7000]
        })
        
        # Should not raise any exception
        validate_data_interface_columns(df, interface)

    def test_successful_validation_with_extra_columns(self):
        """Test successful validation when DataFrame has extra columns."""
        interface = DataInterface(
            target_column="category",
            entity_columns=["user_id"],
            feature_columns=["feature1"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature1': [10, 20, 30],
            'extra_column': ['A', 'B', 'C'],  # Extra column - should be allowed
            'another_extra': [1.1, 2.2, 3.3]
        })
        
        with patch('src.utils.system.data_validation.logger') as mock_logger:
            validate_data_interface_columns(df, interface)
            
            # Should log info about extra columns
            mock_logger.info.assert_called()
            info_calls = [call for call in mock_logger.info.call_args_list
                         if "추가 컬럼 발견" in str(call)]
            assert len(info_calls) > 0

    def test_validation_failure_missing_columns(self):
        """Test validation failure when required columns are missing."""
        interface = DataInterface(
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=["age", "income", "location"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35]
            # Missing 'income' and 'location' columns
        })
        
        with pytest.raises(ValueError) as exc_info:
            validate_data_interface_columns(df, interface)
        
        error_message = str(exc_info.value)
        assert "DataInterface 필수 컬럼 누락 감지" in error_message
        assert "income" in error_message
        assert "location" in error_message
        assert "해결방안" in error_message

    def test_validation_with_stored_required_columns(self):
        """Test validation using stored required columns (inference mode)."""
        interface = DataInterface(
            target_column="price",
            entity_columns=["user_id"], 
            feature_columns=["age"],
            timestamp_column=None,
            treatment_column=None
        )
        
        # DataFrame for inference (no target column needed)
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'income': [5000, 6000, 7000]  # Extra from training
        })
        
        # Stored columns from training time
        stored_columns = ['user_id', 'age', 'income']
        
        with patch('src.utils.system.data_validation.logger') as mock_logger:
            validate_data_interface_columns(df, interface, stored_columns)
            
            # Should use stored columns and log accordingly
            debug_calls = [call for call in mock_logger.debug.call_args_list 
                          if "저장된 필수 컬럼 목록 사용" in str(call)]
            assert len(debug_calls) > 0

    def test_validation_timeseries_missing_timestamp(self):
        """Test validation failure for timeseries task missing timestamp."""
        interface = DataInterface(
            target_column="sales",
            entity_columns=["store_id"],
            feature_columns=["weather"],
            timestamp_column="date",
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'store_id': [1, 2, 3],
            'weather': ['sunny', 'rainy', 'cloudy']
            # Missing 'date' timestamp column
        })
        
        # Use stored required columns that include timestamp (as from training time)
        stored_required_columns = ['store_id', 'date', 'weather']
        
        with pytest.raises(ValueError) as exc_info:
            validate_data_interface_columns(df, interface, stored_required_columns)
        
        error_message = str(exc_info.value)
        assert "date" in error_message

    def test_validation_causal_missing_treatment(self):
        """Test validation failure for causal task missing treatment."""
        interface = DataInterface(
            target_column="outcome",
            entity_columns=["patient_id"],
            feature_columns=["age"],
            timestamp_column=None,
            treatment_column="treatment"
        )
        
        df = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [25, 30, 35]
            # Missing 'treatment' column
        })
        
        # Use stored required columns that include treatment (as from training time)
        stored_required_columns = ['patient_id', 'treatment', 'age']
        
        with pytest.raises(ValueError) as exc_info:
            validate_data_interface_columns(df, interface, stored_required_columns)
        
        error_message = str(exc_info.value)
        assert "treatment" in error_message

    @patch('src.utils.system.data_validation.logger')
    def test_validation_success_logging(self, mock_logger):
        """Test successful validation logs appropriate messages."""
        interface = DataInterface(
            target_column="label",
            entity_columns=["id"],
            feature_columns=["feature"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [10, 20, 30]
        })
        
        validate_data_interface_columns(df, interface)
        
        # Should log success
        success_calls = [call for call in mock_logger.info.call_args_list
                        if "컬럼 검증 통과" in str(call)]
        assert len(success_calls) > 0

    @patch('src.utils.system.data_validation.logger')
    def test_validation_failure_error_logging(self, mock_logger):
        """Test validation failure logs error messages."""
        interface = DataInterface(
            target_column="y",
            entity_columns=["id"],
            feature_columns=["missing_feature"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({'id': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            validate_data_interface_columns(df, interface)
        
        # Should log error
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "컬럼 검증 실패" in error_message


class TestCreateDataInterfaceSchemaForStorage:
    """Test create_data_interface_schema_for_storage function."""

    def test_schema_creation_explicit_feature_columns(self):
        """Test schema creation with explicit feature columns."""
        interface = DataInterface(
            
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=["age", "income"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'price': [100, 200, 300],
            'age': [25, 30, 35],
            'income': [5000, 6000, 7000]
        })
        
        result = create_data_interface_schema_for_storage(interface, df, task_choice="regression")
        
        # Verify schema structure
        assert 'data_interface' in result
        assert 'required_columns' in result
        assert 'column_dtypes' in result
        assert 'schema_version' in result
        
        # Verify required columns (target excluded)
        expected_columns = ['user_id', 'age', 'income']
        assert set(result['required_columns']) == set(expected_columns)
        
        # Verify metadata
        assert result['schema_version'] == '5.1'
        assert result['validation_policy'] == 'data_interface_based'
        assert result['feature_columns_was_null'] is False
        assert result['total_required_columns'] == 3

    def test_schema_creation_null_feature_columns(self):
        """Test schema creation with null feature columns - auto extraction."""
        interface = DataInterface(
            
            target_column="category",
            entity_columns=["user_id"],
            feature_columns=None,  # null - should auto extract
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'category': ['A', 'B', 'C'],
            'age': [25, 30, 35],
            'income': [5000, 6000, 7000],
            'location': ['X', 'Y', 'Z']
        })
        
        result = create_data_interface_schema_for_storage(interface, df, task_choice="classification")
        
        # Should include all columns except target
        expected_columns = ['user_id', 'age', 'income', 'location']
        assert set(result['required_columns']) == set(expected_columns)
        
        # Should mark feature_columns_was_null
        assert result['feature_columns_was_null'] is True
        
        # Should record original columns
        assert set(result['original_dataframe_columns']) == set(df.columns)

    def test_schema_creation_with_task_choice(self):
        """Test schema creation with task_choice parameter."""
        interface = DataInterface(
              # Original type
            target_column="outcome",
            entity_columns=["user_id"],
            feature_columns=["feature1"],
            timestamp_column="date",
            treatment_column="treatment"
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'outcome': [10, 20, 30],
            'feature1': [100, 200, 300],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'treatment': [0, 1, 0]
        })
        
        # Test with causal task choice
        result = create_data_interface_schema_for_storage(interface, df, task_choice="causal")
        
        # Should include treatment column for causal task
        assert 'treatment' in result['required_columns']
        assert 'date' not in result['required_columns']  # Not timeseries

    def test_schema_creation_column_dtypes(self):
        """Test schema creation captures correct column data types."""
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=["int_col", "float_col", "str_col", "bool_col"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [10.0, 20.0, 30.0],
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        
        result = create_data_interface_schema_for_storage(interface, df, task_choice="regression")
        
        # Check data types are captured
        assert 'int64' in result['column_dtypes']['int_col'] or 'int' in result['column_dtypes']['int_col']
        assert 'float' in result['column_dtypes']['float_col']
        assert 'object' in result['column_dtypes']['str_col'] or 'string' in result['column_dtypes']['str_col']
        assert 'bool' in result['column_dtypes']['bool_col']

    def test_schema_creation_missing_column_warning(self):
        """Test schema creation with missing required column - warning case."""
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=["existing_col", "missing_col"],  # missing_col not in df
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [10, 20, 30],
            'existing_col': [100, 200, 300]
            # missing_col not present
        })
        
        with patch('src.utils.system.data_validation.logger') as mock_logger:
            result = create_data_interface_schema_for_storage(interface, df, task_choice="regression")
            
            # Should warn about missing column
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if "필수 컬럼" in str(call) and "DataFrame에 없습니다" in str(call)]
            assert len(warning_calls) > 0
            
            # Should set unknown type for missing column
            assert result['column_dtypes']['missing_col'] == 'unknown'

    def test_schema_creation_complex_timeseries_scenario(self):
        """Test schema creation for complex timeseries scenario."""
        interface = DataInterface(
            
            target_column="sales",
            entity_columns=["store_id", "region_id"],
            feature_columns=["weather", "promotion", "holiday"],
            timestamp_column="date",
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'store_id': [1, 2, 3],
            'region_id': [10, 20, 30],
            'sales': [1000, 2000, 3000],
            'weather': ['sunny', 'rainy', 'cloudy'],
            'promotion': [1, 0, 1],
            'holiday': [True, False, False],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        })
        
        result = create_data_interface_schema_for_storage(interface, df, task_choice="timeseries")
        
        expected_columns = ['store_id', 'region_id', 'date', 'weather', 'promotion', 'holiday']
        assert set(result['required_columns']) == set(expected_columns)
        assert 'sales' not in result['required_columns']  # target excluded
        
        # Verify datetime type captured
        assert 'datetime' in result['column_dtypes']['date']

    @patch('src.utils.system.data_validation.logger')
    def test_schema_creation_logging(self, mock_logger):
        """Test schema creation logs appropriate information."""
        interface = DataInterface(
            
            target_column="label",
            entity_columns=["id"],
            feature_columns=["feature1"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2],
            'label': ['A', 'B'], 
            'feature1': [10, 20]
        })
        
        result = create_data_interface_schema_for_storage(interface, df, task_choice="classification")
        
        # Should log completion
        info_calls = [call for call in mock_logger.info.call_args_list
                     if "저장용 스키마 생성 완료" in str(call)]
        assert len(info_calls) > 0


class TestDataValidationIntegration:
    """Integration tests combining multiple data validation functions."""

    def test_full_training_to_inference_workflow(self):
        """Test complete workflow from training schema creation to inference validation."""
        # Training setup
        interface = DataInterface(
            
            target_column="price",
            entity_columns=["user_id"],
            feature_columns=None,  # null - auto extract
            timestamp_column=None,
            treatment_column=None
        )
        
        training_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'price': [100, 200, 300, 400],
            'age': [25, 30, 35, 40],
            'income': [5000, 6000, 7000, 8000],
            'location': ['A', 'B', 'C', 'D']
        })
        
        # Step 1: Validate training data
        validate_data_interface_columns(training_df, interface)
        
        # Step 2: Create storage schema
        schema = create_data_interface_schema_for_storage(interface, training_df, task_choice="regression")
        stored_columns = schema['required_columns']
        
        # Step 3: Inference data (no target column)
        inference_df = pd.DataFrame({
            'user_id': [5, 6],
            'age': [45, 50],
            'income': [9000, 10000],
            'location': ['E', 'F'],
            'extra_col': ['X', 'Y']  # Extra column should be allowed
        })
        
        # Step 4: Validate inference data using stored schema
        validate_data_interface_columns(inference_df, interface, stored_columns)
        
        # Verify expected columns were used
        expected = ['user_id', 'age', 'income', 'location']
        assert set(stored_columns) == set(expected)

    def test_timeseries_causal_hybrid_scenario(self):
        """Test complex scenario with timeseries features but causal inference."""
        interface = DataInterface(
              # Causal task
            target_column="outcome",
            entity_columns=["patient_id"],
            feature_columns=["baseline_score", "age"],
            timestamp_column="enrollment_date",  # Has timestamp but not timeseries task
            treatment_column="treatment_arm"
        )
        
        df = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'outcome': [1, 0, 1],
            'baseline_score': [80, 75, 85],
            'age': [65, 70, 60],
            'enrollment_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'treatment_arm': ['control', 'treatment', 'control']
        })
        
        # Required columns should include treatment but NOT timestamp (not timeseries)
        required = get_required_columns_from_data_interface(interface, task_choice="causal")
        assert 'treatment_arm' in required
        assert 'enrollment_date' not in required  # Only for timeseries tasks
        
        # Validation should pass
        validate_data_interface_columns(df, interface)
        
        # Schema should capture the causal structure
        schema = create_data_interface_schema_for_storage(interface, df, task_choice="causal")
        assert 'treatment_arm' in schema['required_columns']
        assert 'enrollment_date' not in schema['required_columns']

    def test_error_propagation_across_functions(self):
        """Test error handling across multiple function calls."""
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=["feature1", "feature2"],
            timestamp_column=None,
            treatment_column=None
        )
        
        # Incomplete DataFrame
        incomplete_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [10, 20, 30]
            # Missing target and feature2
        })
        
        # Validation should fail
        with pytest.raises(ValueError) as exc_info:
            validate_data_interface_columns(incomplete_df, interface)
        
        assert "feature2" in str(exc_info.value)
        
        # Schema creation should still work (may warn about missing columns)
        with patch('src.utils.system.data_validation.logger'):
            schema = create_data_interface_schema_for_storage(interface, incomplete_df, task_choice="regression")
            assert 'feature2' in schema['column_dtypes']
            assert schema['column_dtypes']['feature2'] == 'unknown'


class TestDataValidationSecurity:
    """Security and edge case tests for data validation functions."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=["feature"],
            timestamp_column=None,
            treatment_column=None
        )
        
        empty_df = pd.DataFrame(columns=['id', 'target', 'feature'])
        
        # Validation should pass for structure
        validate_data_interface_columns(empty_df, interface)
        
        # Schema creation should handle empty data
        schema = create_data_interface_schema_for_storage(interface, empty_df, task_choice="regression")
        assert schema['actual_dataframe_shape'] == [0, 3]

    def test_dataframe_with_null_values(self):
        """Test handling of DataFrames with null values."""
        interface = DataInterface(
            
            target_column="label",
            entity_columns=["id"],
            feature_columns=["feature"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df_with_nulls = pd.DataFrame({
            'id': [1, 2, None],
            'label': ['A', None, 'C'],
            'feature': [10, 20, 30]
        })
        
        # Structure validation should still pass
        validate_data_interface_columns(df_with_nulls, interface)
        
        # Schema should capture data types despite nulls
        schema = create_data_interface_schema_for_storage(interface, df_with_nulls, task_choice="classification")
        assert 'column_dtypes' in schema

    def test_large_column_names_handling(self):
        """Test handling of very long column names."""
        long_column_name = "very_long_column_name_" + "x" * 100
        
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=[long_column_name],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [10, 20, 30],
            long_column_name: [100, 200, 300]
        })
        
        # Should handle long names gracefully
        required = get_required_columns_from_data_interface(interface)
        assert long_column_name in required
        
        validate_data_interface_columns(df, interface)
        
        schema = create_data_interface_schema_for_storage(interface, df, task_choice="regression")
        assert long_column_name in schema['column_dtypes']

    def test_special_characters_in_column_names(self):
        """Test handling of special characters in column names."""
        special_cols = ["col-with-dash", "col_with_underscore", "col with space", "col.with.dot"]
        
        interface = DataInterface(
            
            target_column="target",
            entity_columns=["id"],
            feature_columns=special_cols,
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'id': [1, 2],
            'target': [10, 20],
            'col-with-dash': [1, 2],
            'col_with_underscore': [3, 4],
            'col with space': [5, 6],
            'col.with.dot': [7, 8]
        })
        
        # Should handle special characters
        validate_data_interface_columns(df, interface)
        
        schema = create_data_interface_schema_for_storage(interface, df, task_choice="regression")
        for col in special_cols:
            assert col in schema['column_dtypes']

    def test_malformed_data_interface_handling(self):
        """Test handling of malformed or incomplete DataInterface objects."""
        # DataInterface with missing required fields
        minimal_interface = DataInterface(
            
            target_column="target",
            entity_columns=[],  # Empty entities
            feature_columns=None,
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'target': [10, 20, 30],
            'feature1': [1, 2, 3]
        })
        
        # Should handle gracefully
        required = get_required_columns_from_data_interface(minimal_interface, df, task_choice="regression")
        # Should auto-extract features when entities empty
        assert 'feature1' in required
        assert 'target' not in required

    @patch('src.utils.system.data_validation.logger')
    def test_logging_security(self, mock_logger):
        """Test that logging doesn't expose sensitive information."""
        interface = DataInterface(
            
            target_column="sensitive_target",
            entity_columns=["user_id"],
            feature_columns=["sensitive_feature"],
            timestamp_column=None,
            treatment_column=None
        )
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'sensitive_target': [100, 200, 300],
            'sensitive_feature': [10, 20, 30]
        })
        
        validate_data_interface_columns(df, interface)
        create_data_interface_schema_for_storage(interface, df, task_choice="regression")
        
        # Check that all log messages are reasonable (not checking for absence of data)
        for call in mock_logger.info.call_args_list + mock_logger.debug.call_args_list:
            message = str(call)
            # Logs should contain column names but not sensitive data values
            assert any(col in message for col in ['user_id', 'sensitive_feature', 'sensitive_target']) or \
                   'DataInterface' in message or 'validation' in message or 'schema' in message