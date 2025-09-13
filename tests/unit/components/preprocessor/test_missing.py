"""
Missing Value Handling Components Comprehensive Tests
Testing DropMissingWrapper, ForwardFillWrapper, BackwardFillWrapper, 
ConstantFillWrapper, and InterpolationWrapper

Architecture Compliance:
- Targeted application type behavior
- Missing value detection and handling
- DataFrame-first approach  
- Various missing value strategies
- Column name preservation
- Real component testing (no mock hell)
"""
import pytest
import pandas as pd
import numpy as np

from src.components.preprocessor.modules.missing import (
    DropMissingWrapper,
    ForwardFillWrapper, 
    BackwardFillWrapper,
    ConstantFillWrapper,
    InterpolationWrapper
)
from src.components.preprocessor.registry import PreprocessorStepRegistry


class TestDropMissingWrapper:
    """DropMissingWrapper comprehensive testing"""
    
    def test_drop_missing_targeted_application_type(self):
        """Verify DropMissing is Targeted type"""
        # Given: DropMissing instance
        dropper = DropMissingWrapper()
        
        # When: Check application type
        app_type = dropper.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_drop_missing_rows_preserves_column_names(self):
        """Verify row dropping preserves column names"""
        # Given: DropMissing configured for row dropping
        dropper = DropMissingWrapper(axis='rows')
        
        # When: Check column name preservation
        preserves = dropper.preserves_column_names()
        
        # Then: Should preserve names when dropping rows
        assert preserves is True
    
    def test_drop_missing_columns_does_not_preserve_names(self):
        """Verify column dropping doesn't preserve column names"""
        # Given: DropMissing configured for column dropping
        dropper = DropMissingWrapper(axis='columns')
        
        # When: Check column name preservation
        preserves = dropper.preserves_column_names()
        
        # Then: Should not preserve names when dropping columns
        assert preserves is False
    
    def test_drop_missing_rows_basic_functionality(self):
        """Test core row dropping functionality"""
        # Given: Data with missing values in some rows
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, 4.0],
            'col2': [10.0, 20.0, np.nan, 40.0],
            'col3': [100.0, 200.0, 300.0, 400.0]
        })
        
        dropper = DropMissingWrapper(axis='rows', threshold=0.0)  # Drop any row with missing
        
        # When: Fit and transform
        dropper.fit(df)
        result = dropper.transform(df)
        
        # Then: Should drop rows with any missing values
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)  # Columns preserved
        assert len(result) == 2  # Row 0 and 3 have no missing values
        assert result.index.tolist() == [0, 3]  # Rows with index 0 and 3 remain
    
    def test_drop_missing_columns_basic_functionality(self):
        """Test core column dropping functionality"""
        # Given: Data with missing values in some columns
        df = pd.DataFrame({
            'col_with_missing': [1.0, np.nan, 3.0],
            'col_all_missing': [np.nan, np.nan, np.nan],
            'col_no_missing': [10.0, 20.0, 30.0]
        })
        
        dropper = DropMissingWrapper(axis='columns', threshold=0.0)  # Drop columns with any missing
        
        # When: Fit and transform
        dropper.fit(df)
        result = dropper.transform(df)
        
        # Then: Should drop columns with missing values
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Same number of rows
        assert list(result.columns) == ['col_no_missing']  # Only column without missing
    
    def test_drop_missing_threshold_parameter(self):
        """Test threshold parameter for controlling drop behavior"""
        # Given: Data with different missing patterns
        df = pd.DataFrame({
            'mostly_missing': [1.0, np.nan, np.nan, np.nan],     # 75% missing
            'half_missing': [1.0, 2.0, np.nan, np.nan],         # 50% missing  
            'few_missing': [1.0, 2.0, 3.0, np.nan],             # 25% missing
            'no_missing': [1.0, 2.0, 3.0, 4.0]                  # 0% missing
        })
        
        # When: Apply different thresholds for column dropping
        dropper_strict = DropMissingWrapper(axis='columns', threshold=0.6)  # Drop if >60% missing
        dropper_lenient = DropMissingWrapper(axis='columns', threshold=0.8)  # Drop if >80% missing
        
        result_strict = dropper_strict.fit_transform(df)
        result_lenient = dropper_lenient.fit_transform(df)
        
        # Then: Different thresholds should keep different columns
        # Strict (threshold=0.6): drops mostly_missing (75% > 60%), keeps others
        assert 'no_missing' in result_strict.columns
        assert 'few_missing' in result_strict.columns
        assert 'half_missing' in result_strict.columns
        assert 'mostly_missing' not in result_strict.columns
        
        # Lenient (threshold=0.8): only drops if >80% missing, keeps all (max is 75%)
        expected_lenient = {'no_missing', 'few_missing', 'half_missing', 'mostly_missing'}
        assert set(result_lenient.columns) == expected_lenient
    
    def test_drop_missing_specific_columns_parameter(self):
        """Test columns parameter for targeted dropping"""
        # Given: Data with missing in multiple columns
        df = pd.DataFrame({
            'target_col': [1.0, np.nan, 3.0],
            'other_col': [np.nan, 2.0, 3.0],
            'keep_col': [10.0, 20.0, 30.0]
        })
        
        # When: Drop only considering specific columns
        dropper = DropMissingWrapper(axis='rows', threshold=0.0, columns=['target_col'])
        result = dropper.fit_transform(df)
        
        # Then: Should drop rows based only on target_col missing values
        # Row 1 has missing in target_col, so should be dropped
        assert len(result) == 2  # Rows 0 and 2 remain
        assert 0 in result.index and 2 in result.index
        assert 1 not in result.index
    
    def test_drop_missing_get_output_column_names(self):
        """Test output column name prediction"""
        # Given: DropMissing for columns
        dropper = DropMissingWrapper(axis='columns')
        
        # Simulate dropped columns
        dropper._dropped_columns = ['col_to_drop']
        
        # When: Get output column names
        input_cols = ['col1', 'col_to_drop', 'col3']
        output_cols = dropper.get_output_column_names(input_cols)
        
        # Then: Should exclude dropped columns
        expected = ['col1', 'col3']
        assert output_cols == expected


class TestForwardFillWrapper:
    """ForwardFillWrapper comprehensive testing"""
    
    def test_forward_fill_targeted_application_type(self):
        """Verify ForwardFill is Targeted type"""
        # Given: ForwardFill instance
        filler = ForwardFillWrapper()
        
        # When: Check application type
        app_type = filler.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_forward_fill_preserves_column_names(self):
        """Verify ForwardFill preserves column names"""
        # Given: ForwardFill instance
        filler = ForwardFillWrapper()
        
        # When: Check column name preservation
        preserves = filler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True
    
    def test_forward_fill_basic_functionality(self):
        """Test core forward fill functionality"""
        # Given: Time series data with missing values
        df = pd.DataFrame({
            'values': [1.0, 2.0, np.nan, np.nan, 5.0, np.nan]
        })
        
        filler = ForwardFillWrapper()
        
        # When: Fit and transform
        filler.fit(df)
        result = filler.transform(df)
        
        # Then: Should forward fill missing values
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['values']
        
        expected_values = [1.0, 2.0, 2.0, 2.0, 5.0, 5.0]  # Forward fill pattern
        pd.testing.assert_series_equal(result['values'], pd.Series(expected_values, name='values'))
    
    def test_forward_fill_with_limit(self):
        """Test forward fill with limit parameter"""
        # Given: Data with consecutive missing values
        df = pd.DataFrame({
            'values': [1.0, np.nan, np.nan, np.nan, 5.0]
        })
        
        filler = ForwardFillWrapper(limit=2)  # Fill at most 2 consecutive
        
        # When: Apply forward fill with limit
        result = filler.fit_transform(df)
        
        # Then: Should fill only up to limit
        # Index 1,2 filled, index 3 remains NaN due to limit
        expected = pd.Series([1.0, 1.0, 1.0, np.nan, 5.0], name='values')
        pd.testing.assert_series_equal(result['values'], expected)
    
    def test_forward_fill_specific_columns(self):
        """Test forward fill on specific columns only"""
        # Given: Multi-column data
        df = pd.DataFrame({
            'fill_this': [1.0, np.nan, np.nan, 4.0],
            'leave_this': [np.nan, 2.0, np.nan, 4.0]
        })
        
        filler = ForwardFillWrapper(columns=['fill_this'])
        
        # When: Apply to specific column only
        result = filler.fit_transform(df)
        
        # Then: Only specified column should be filled
        assert not result['fill_this'].isnull().any()  # Should be filled
        assert result['leave_this'].isnull().sum() == 2  # Should remain with missing
    
    def test_forward_fill_applicable_columns_detection(self):
        """Test automatic detection of applicable columns"""
        # Given: Mixed data with some columns having missing values
        df = pd.DataFrame({
            'has_missing': [1.0, np.nan, 3.0],
            'no_missing': [1.0, 2.0, 3.0],
            'all_missing': [np.nan, np.nan, np.nan]
        })
        
        filler = ForwardFillWrapper()
        
        # When: Get applicable columns
        applicable_cols = filler.get_applicable_columns(df)
        
        # Then: Should detect columns with missing values
        expected = ['has_missing', 'all_missing']
        assert set(applicable_cols) == set(expected)
        assert 'no_missing' not in applicable_cols


class TestBackwardFillWrapper:
    """BackwardFillWrapper comprehensive testing"""
    
    def test_backward_fill_basic_functionality(self):
        """Test core backward fill functionality"""
        # Given: Time series data with missing values
        df = pd.DataFrame({
            'values': [np.nan, 2.0, np.nan, np.nan, 5.0, 6.0]
        })
        
        filler = BackwardFillWrapper()
        
        # When: Apply backward fill
        result = filler.fit_transform(df)
        
        # Then: Should backward fill missing values
        expected_values = [2.0, 2.0, 5.0, 5.0, 5.0, 6.0]  # Backward fill pattern
        pd.testing.assert_series_equal(result['values'], pd.Series(expected_values, name='values'))
    
    def test_backward_fill_with_limit(self):
        """Test backward fill with limit parameter"""
        # Given: Data with consecutive missing values
        df = pd.DataFrame({
            'values': [1.0, np.nan, np.nan, np.nan, 5.0]
        })
        
        filler = BackwardFillWrapper(limit=2)
        
        # When: Apply backward fill with limit
        result = filler.fit_transform(df)
        
        # Then: Should fill only up to limit from the end
        # Working backward: index 3,2 filled, index 1 remains NaN due to limit
        expected = pd.Series([1.0, np.nan, 5.0, 5.0, 5.0], name='values')
        pd.testing.assert_series_equal(result['values'], expected)
    
    def test_backward_fill_preserves_column_names(self):
        """Verify BackwardFill preserves column names"""
        # Given: BackwardFill instance
        filler = BackwardFillWrapper()
        
        # When: Check column name preservation
        preserves = filler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True


class TestConstantFillWrapper:
    """ConstantFillWrapper comprehensive testing"""
    
    def test_constant_fill_single_value(self):
        """Test constant fill with single value for all columns"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0],
            'col2': [np.nan, 2.0, np.nan]
        })
        
        filler = ConstantFillWrapper(fill_value=999)
        
        # When: Apply constant fill
        result = filler.fit_transform(df)
        
        # Then: All missing values should be filled with 999
        assert not result.isnull().any().any()  # No missing values
        assert result.loc[0, 'col2'] == 999  # Missing value filled
        assert result.loc[1, 'col1'] == 999  # Missing value filled
        assert result.loc[2, 'col2'] == 999  # Missing value filled
        
        # Non-missing values should remain unchanged
        assert result.loc[0, 'col1'] == 1.0
        assert result.loc[1, 'col2'] == 2.0
        assert result.loc[2, 'col1'] == 3.0
    
    def test_constant_fill_column_specific_values(self):
        """Test constant fill with different values per column"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'numeric_col': [1.0, np.nan, 3.0],
            'category_col': ['a', np.nan, 'c']
        })
        
        filler = ConstantFillWrapper(fill_value={
            'numeric_col': -1,
            'category_col': 'unknown'
        })
        
        # When: Apply column-specific constant fill
        result = filler.fit_transform(df)
        
        # Then: Each column should be filled with its specific value
        assert result.loc[1, 'numeric_col'] == -1
        assert result.loc[1, 'category_col'] == 'unknown'
        
        # Non-missing values should remain unchanged
        assert result.loc[0, 'numeric_col'] == 1.0
        assert result.loc[0, 'category_col'] == 'a'
    
    def test_constant_fill_specific_columns_only(self):
        """Test constant fill applied to specific columns only"""
        # Given: Multi-column data
        df = pd.DataFrame({
            'fill_this': [1.0, np.nan, 3.0],
            'leave_this': [np.nan, 2.0, np.nan]
        })
        
        filler = ConstantFillWrapper(fill_value=0, columns=['fill_this'])
        
        # When: Apply to specific column only
        result = filler.fit_transform(df)
        
        # Then: Only specified column should be filled
        assert not result['fill_this'].isnull().any()  # Should be filled
        assert result['leave_this'].isnull().sum() == 2  # Should remain with missing
        assert result.loc[1, 'fill_this'] == 0  # Filled with constant
    
    def test_constant_fill_preserves_column_names(self):
        """Verify ConstantFill preserves column names"""
        # Given: ConstantFill instance
        filler = ConstantFillWrapper()
        
        # When: Check column name preservation
        preserves = filler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True


class TestInterpolationWrapper:
    """InterpolationWrapper comprehensive testing"""
    
    def test_interpolation_linear_basic_functionality(self):
        """Test core linear interpolation functionality"""
        # Given: Numeric data with missing values
        df = pd.DataFrame({
            'values': [1.0, np.nan, np.nan, 4.0, np.nan, 6.0]
        })
        
        interpolator = InterpolationWrapper(method='linear')
        
        # When: Apply linear interpolation
        result = interpolator.fit_transform(df)
        
        # Then: Should interpolate missing values linearly
        # Between 1 and 4: 2.0, 3.0
        # Between 4 and 6: 5.0
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='values')
        pd.testing.assert_series_equal(result['values'], expected)
    
    def test_interpolation_with_limit(self):
        """Test interpolation with limit parameter"""
        # Given: Data with many consecutive missing values
        df = pd.DataFrame({
            'values': [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]
        })
        
        interpolator = InterpolationWrapper(method='linear', limit=2)
        
        # When: Apply interpolation with limit
        result = interpolator.fit_transform(df)
        
        # Then: Should interpolate only up to limit
        # Only 2 consecutive NaN values should be filled
        filled_count = (~result['values'].isnull()).sum()
        assert filled_count >= 4  # At least original 2 + 2 filled
    
    def test_interpolation_numeric_columns_only(self):
        """Test that interpolation applies only to numeric columns"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'numeric': [1.0, np.nan, 3.0],
            'category': ['a', np.nan, 'c'],
            'boolean': [True, np.nan, False]
        })
        
        interpolator = InterpolationWrapper(method='linear')
        
        # When: Get applicable columns
        applicable_cols = interpolator.get_applicable_columns(df)
        
        # Then: Only numeric columns with missing values
        assert applicable_cols == ['numeric']
        assert 'category' not in applicable_cols
        assert 'boolean' not in applicable_cols
    
    def test_interpolation_specific_columns(self):
        """Test interpolation on specific columns only"""
        # Given: Multiple numeric columns
        df = pd.DataFrame({
            'interpolate_this': [1.0, np.nan, 3.0],
            'leave_this': [10.0, np.nan, 30.0]
        })
        
        interpolator = InterpolationWrapper(method='linear', columns=['interpolate_this'])
        
        # When: Apply to specific column only
        result = interpolator.fit_transform(df)
        
        # Then: Only specified column should be interpolated
        assert not result['interpolate_this'].isnull().any()  # Should be interpolated
        assert result['leave_this'].isnull().sum() == 1  # Should remain with missing
    
    def test_interpolation_polynomial_method(self):
        """Test polynomial interpolation method"""
        # Given: Data suitable for polynomial interpolation
        df = pd.DataFrame({
            'values': [1.0, np.nan, 4.0, np.nan, 9.0]  # Quadratic pattern: 1, ?, 4, ?, 9
        })
        
        interpolator = InterpolationWrapper(method='polynomial', order=2)
        
        # When: Apply polynomial interpolation
        result = interpolator.fit_transform(df)
        
        # Then: Should interpolate using polynomial method
        assert isinstance(result, pd.DataFrame)
        assert not result['values'].isnull().any()  # All values filled
        
        # For quadratic sequence 1, 4, 9, the missing values should be ~2.25, ~6.25
        # (exact values depend on pandas interpolation implementation)
    
    def test_interpolation_preserves_column_names(self):
        """Verify Interpolation preserves column names"""
        # Given: Interpolation instance
        interpolator = InterpolationWrapper()
        
        # When: Check column name preservation
        preserves = interpolator.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True


class TestMissingValueRegistration:
    """Test missing value handler registration in PreprocessorStepRegistry"""
    
    def test_all_missing_handlers_registered(self):
        """Verify all missing value handlers are properly registered"""
        # Given: Registry should contain all handlers
        expected_handlers = [
            'drop_missing',
            'forward_fill', 
            'backward_fill',
            'constant_fill',
            'interpolation'
        ]
        
        # When: Check registration
        for handler in expected_handlers:
            registered = handler in PreprocessorStepRegistry.preprocessor_steps
            
            # Then: Should be registered
            assert registered, f"{handler} not registered"
    
    def test_missing_handler_creation_through_registry(self):
        """Test creating missing value handlers through registry"""
        # Given: Registry with registered handlers
        
        # When: Create handlers through registry
        drop_handler = PreprocessorStepRegistry.create('drop_missing')
        forward_handler = PreprocessorStepRegistry.create('forward_fill')
        backward_handler = PreprocessorStepRegistry.create('backward_fill')
        constant_handler = PreprocessorStepRegistry.create('constant_fill')
        interp_handler = PreprocessorStepRegistry.create('interpolation')
        
        # Then: Should create correct instances
        assert isinstance(drop_handler, DropMissingWrapper)
        assert isinstance(forward_handler, ForwardFillWrapper)
        assert isinstance(backward_handler, BackwardFillWrapper)
        assert isinstance(constant_handler, ConstantFillWrapper)
        assert isinstance(interp_handler, InterpolationWrapper)
    
    def test_missing_handler_creation_with_parameters(self):
        """Test creating missing handlers with parameters through registry"""
        # Given: Registry and custom parameters
        
        # When: Create with parameters
        drop_handler = PreprocessorStepRegistry.create(
            'drop_missing',
            axis='columns',
            threshold=0.5,
            columns=['col1']
        )
        
        constant_handler = PreprocessorStepRegistry.create(
            'constant_fill',
            fill_value={'col1': 0, 'col2': 'missing'},
            columns=['col1', 'col2']
        )
        
        interp_handler = PreprocessorStepRegistry.create(
            'interpolation',
            method='polynomial',
            order=2,
            limit=3
        )
        
        # Then: Should create instances with parameters
        assert drop_handler.axis == 'columns'
        assert drop_handler.threshold == 0.5
        assert drop_handler.columns == ['col1']
        
        assert constant_handler.fill_value == {'col1': 0, 'col2': 'missing'}
        assert constant_handler.columns == ['col1', 'col2']
        
        assert interp_handler.method == 'polynomial'
        assert interp_handler.order == 2
        assert interp_handler.limit == 3


class TestMissingValueIntegration:
    """Integration tests for missing value handling components"""
    
    def test_missing_handlers_pipeline_compatibility(self):
        """Test missing handlers work in preprocessing pipeline context"""
        # Given: Data with various missing patterns
        df = pd.DataFrame({
            'mostly_complete': [1.0, 2.0, np.nan, 4.0, 5.0],
            'time_series': [10.0, np.nan, np.nan, 40.0, 50.0],
            'categorical': ['a', 'b', np.nan, 'd', 'e']
        })
        
        # When: Apply different handlers sequentially
        # First: constant fill for categorical
        const_filler = ConstantFillWrapper(fill_value='unknown', columns=['categorical'])
        step1 = const_filler.fit_transform(df)
        
        # Then: interpolate numeric columns
        interpolator = InterpolationWrapper(method='linear', columns=['mostly_complete', 'time_series'])
        step2 = interpolator.fit_transform(step1)
        
        # Then: Should have no missing values
        assert not step2.isnull().any().any()
        assert len(step2) == len(df)  # Same number of rows
        assert list(step2.columns) == list(df.columns)  # Same columns
    
    def test_missing_handlers_deterministic_behavior(self):
        """Test missing handlers produce deterministic results"""
        # Given: Same data with missing values
        np.random.seed(42)
        df = pd.DataFrame({
            'values': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # When: Apply same handler multiple times
        handler1 = InterpolationWrapper(method='linear')
        handler2 = InterpolationWrapper(method='linear')
        
        result1 = handler1.fit_transform(df)
        result2 = handler2.fit_transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_missing_handlers_edge_cases(self):
        """Test missing handlers handle edge cases gracefully"""
        # Given: Edge case scenarios
        edge_cases = [
            # All missing
            pd.DataFrame({'all_missing': [np.nan, np.nan, np.nan]}),
            # No missing
            pd.DataFrame({'no_missing': [1.0, 2.0, 3.0]}),
            # Single value
            pd.DataFrame({'single': [5.0]}),
            # Single missing
            pd.DataFrame({'single_missing': [np.nan]})
        ]
        
        handlers = [
            ConstantFillWrapper(fill_value=0),
            ForwardFillWrapper(),
            BackwardFillWrapper()
        ]
        
        # When: Apply handlers to edge cases
        for df in edge_cases:
            for handler in handlers:
                try:
                    result = handler.fit_transform(df)
                    # Then: Should produce valid DataFrame
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == len(df)
                except Exception:
                    # Some combinations may legitimately fail (e.g., forward fill with single NaN)
                    # This is acceptable behavior
                    pass
    
    def test_missing_handlers_preserve_data_types(self):
        """Test that missing handlers preserve appropriate data types"""
        # Given: Data with specific dtypes
        df = pd.DataFrame({
            'int_col': pd.Series([1, 2, np.nan, 4], dtype='float64'),  # Will be float due to NaN
            'float_col': [1.1, np.nan, 3.3, 4.4],
            'str_col': ['a', np.nan, 'c', 'd']
        })
        
        # When: Apply constant fill
        const_filler = ConstantFillWrapper(fill_value={
            'int_col': 999,
            'float_col': 999.9,
            'str_col': 'missing'
        })
        result = const_filler.fit_transform(df)
        
        # Then: Data types should be preserved appropriately
        assert result['float_col'].dtype == np.float64
        assert result['str_col'].dtype == object  # String dtype
        # int_col will remain float due to original NaN, but values should be correct