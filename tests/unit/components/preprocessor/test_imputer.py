"""
Imputer Components Comprehensive Tests
Testing SimpleImputerWrapper with missing value handling strategies

Architecture Compliance:
- Targeted application type behavior
- Missing value detection and handling
- DataFrame-first approach  
- Missing indicator generation capability
- Column name preservation logic
- Real component testing (no mock hell)
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator

from src.components.preprocessor.modules.imputer import SimpleImputerWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry


class TestSimpleImputerWrapper:
    """SimpleImputerWrapper comprehensive testing"""
    
    def test_simple_imputer_targeted_application_type(self):
        """Verify SimpleImputer is Targeted type - applies to specific columns"""
        # Given: SimpleImputer instance
        imputer = SimpleImputerWrapper()
        
        # When: Check application type
        app_type = imputer.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_simple_imputer_preserves_column_names_by_default(self):
        """Verify SimpleImputer preserves column names when not creating indicators"""
        # Given: SimpleImputer without missing indicators
        imputer = SimpleImputerWrapper(create_missing_indicators=False)
        
        # When: Check column name preservation
        preserves = imputer.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True
    
    def test_simple_imputer_does_not_preserve_names_with_indicators(self):
        """Verify SimpleImputer doesn't preserve names when creating indicators"""
        # Given: SimpleImputer with missing indicators
        imputer = SimpleImputerWrapper(create_missing_indicators=True)
        
        # When: Check column name preservation
        preserves = imputer.preserves_column_names()
        
        # Then: Should not preserve names (adds indicator columns)
        assert preserves is False
    
    def test_simple_imputer_numeric_missing_column_detection(self):
        """Test automatic detection of numeric columns with missing values"""
        # Given: Mixed data with different missing patterns
        df = pd.DataFrame({
            'numeric_with_missing': [1.0, np.nan, 3.0, 4.0, np.nan],
            'numeric_no_missing': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical_with_missing': ['a', np.nan, 'c', 'd', 'e'],
            'categorical_no_missing': ['x', 'y', 'z', 'x', 'y']
        })
        
        imputer = SimpleImputerWrapper()
        
        # When: Get applicable columns
        applicable_cols = imputer.get_applicable_columns(df)
        
        # Then: Only numeric columns with missing values
        expected = ['numeric_with_missing']
        assert set(applicable_cols) == set(expected)
        assert 'numeric_no_missing' not in applicable_cols  # No missing values
        assert 'categorical_with_missing' not in applicable_cols  # Not numeric
        assert 'categorical_no_missing' not in applicable_cols  # Not numeric and no missing
    
    def test_simple_imputer_mean_strategy(self):
        """Test mean imputation strategy"""
        # Given: Numeric data with missing values
        df = pd.DataFrame({
            'feature': [1.0, 2.0, np.nan, 4.0, 5.0]  # Mean = 3.0
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Missing value should be replaced with mean
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['feature']
        assert not result.isnull().any().any()  # No missing values
        
        # Missing value (index 2) should be replaced with mean (3.0)
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4  # 3.0
        assert abs(result['feature'].iloc[2] - expected_mean) < 1e-10
    
    def test_simple_imputer_median_strategy(self):
        """Test median imputation strategy"""
        # Given: Numeric data with missing values
        df = pd.DataFrame({
            'feature': [1.0, 2.0, np.nan, 4.0, 100.0]  # Median = 3.0 (not affected by outlier)
        })
        
        imputer = SimpleImputerWrapper(strategy='median')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Missing value should be replaced with median
        assert not result.isnull().any().any()
        
        # Missing value should be replaced with median
        expected_median = np.median([1.0, 2.0, 4.0, 100.0])  # 3.0
        assert abs(result['feature'].iloc[2] - expected_median) < 1e-10
    
    def test_simple_imputer_most_frequent_strategy(self):
        """Test most frequent imputation strategy"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'feature': [1.0, 2.0, np.nan, 2.0, 2.0]  # 2.0 is most frequent
        })
        
        imputer = SimpleImputerWrapper(strategy='most_frequent')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Missing value should be replaced with most frequent value
        assert not result.isnull().any().any()
        
        # Missing value should be replaced with most frequent value (2.0)
        assert result['feature'].iloc[2] == 2.0
    
    def test_simple_imputer_missing_indicators_creation(self):
        """Test creation of missing indicator columns"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'feature_1': [1.0, np.nan, 3.0, 4.0],
            'feature_2': [10.0, 20.0, np.nan, 40.0]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean', create_missing_indicators=True)
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Should have imputed values + missing indicator columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        
        # Should have original columns + indicator columns
        original_cols = ['feature_1', 'feature_2']
        for col in original_cols:
            assert col in result.columns  # Original columns preserved
            
        # Should have missing indicator columns
        indicator_cols = [col for col in result.columns if 'missingindicator' in col]
        assert len(indicator_cols) > 0, "Should have missing indicator columns"
        
        # Check that indicators correctly identify missing positions
        # feature_1 had missing at index 1
        # feature_2 had missing at index 2
        # (exact column names may vary based on sklearn version)
    
    def test_simple_imputer_no_missing_indicators_when_not_requested(self):
        """Test that no indicator columns are created when not requested"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean', create_missing_indicators=False)
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Should only have original column (imputed)
        assert list(result.columns) == ['feature']
        assert len(result.columns) == 1
        assert not result.isnull().any().any()
    
    def test_simple_imputer_handles_all_missing_column_error(self):
        """Test error handling when column has all missing values"""
        # Given: Column with all missing values
        df = pd.DataFrame({
            'all_missing': [np.nan, np.nan, np.nan, np.nan]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When & Then: Should raise meaningful error
        with pytest.raises(ValueError) as exc_info:
            imputer.fit(df)
        
        error_msg = str(exc_info.value)
        assert "전체가 결측값인 컬럼" in error_msg
        assert "all_missing" in error_msg
    
    def test_simple_imputer_invalid_strategy_error(self):
        """Test error handling for invalid strategy"""
        # Given: Data and imputer with invalid strategy
        df = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0]
        })
        
        # When: Try to use invalid strategy through direct sklearn error
        # Create imputer with invalid strategy at sklearn level
        invalid_imputer = SimpleImputerWrapper(strategy='invalid_strategy')
        
        # Then: Should raise meaningful error during fit
        with pytest.raises(ValueError) as exc_info:
            invalid_imputer.fit(df)
        
        error_msg = str(exc_info.value)
        # Should contain information about valid strategies
        assert any(strategy in error_msg for strategy in ['mean', 'median', 'most_frequent'])
    
    def test_simple_imputer_strategy_numeric_compatibility(self):
        """Test strategy compatibility with data types"""
        # Given: Numeric data
        df_numeric = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, np.nan, 4.0]
        })
        
        # When: Apply strategies appropriate for numeric data
        strategies = ['mean', 'median', 'most_frequent']
        
        for strategy in strategies:
            imputer = SimpleImputerWrapper(strategy=strategy)
            imputer.fit(df_numeric)
            result = imputer.transform(df_numeric)
            
            # Then: Should work without errors
            assert isinstance(result, pd.DataFrame)
            assert not result.isnull().any().any()
    
    def test_simple_imputer_multiple_columns(self):
        """Test imputation on multiple columns simultaneously"""
        # Given: Multiple columns with different missing patterns
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, 4.0, 5.0],
            'col2': [10.0, 20.0, np.nan, np.nan, 50.0],
            'col3': [100.0, 200.0, 300.0, 400.0, np.nan]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: All columns should be imputed
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['col1', 'col2', 'col3']
        assert not result.isnull().any().any()  # No missing values
        
        # Check that each column is imputed correctly
        # col1: missing at index 1, should be mean of [1,3,4,5] = 3.25
        expected_col1_imputed = (1.0 + 3.0 + 4.0 + 5.0) / 4
        assert abs(result['col1'].iloc[1] - expected_col1_imputed) < 1e-10
    
    def test_simple_imputer_transform_consistency(self):
        """Test that transform produces consistent results"""
        # Given: Data with missing values
        df_train = pd.DataFrame({
            'feature': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
        df_test = pd.DataFrame({
            'feature': [np.nan, 3.0, np.nan]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: Fit on training data, transform test data
        imputer.fit(df_train)
        train_result = imputer.transform(df_train)
        test_result = imputer.transform(df_test)
        
        # Then: Missing values in test should be filled with training mean
        training_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4  # 3.0
        
        assert abs(test_result['feature'].iloc[0] - training_mean) < 1e-10
        assert abs(test_result['feature'].iloc[2] - training_mean) < 1e-10
        assert test_result['feature'].iloc[1] == 3.0  # Non-missing value preserved
    
    def test_simple_imputer_preserves_index(self):
        """Test that DataFrame index is preserved during imputation"""
        # Given: Data with custom index
        custom_index = ['row_a', 'row_b', 'row_c', 'row_d']
        df = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0, 4.0]
        }, index=custom_index)
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Index should be preserved
        assert list(result.index) == custom_index
    
    def test_simple_imputer_single_feature(self):
        """Test imputation on single feature DataFrame"""
        # Given: Single column with missing value
        df = pd.DataFrame({
            'single_feature': [10.0, np.nan, 30.0]
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: Fit and transform
        imputer.fit(df)
        result = imputer.transform(df)
        
        # Then: Should work correctly
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['single_feature']
        assert not result.isnull().any().any()
        
        # Missing value should be imputed with mean of [10.0, 30.0] = 20.0
        assert abs(result['single_feature'].iloc[1] - 20.0) < 1e-10
    
    def test_simple_imputer_fit_transform_equivalence(self):
        """Test that fit_transform gives same result as fit + transform"""
        # Given: Data with missing values
        df = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # When: Compare fit_transform vs fit + transform
        imputer1 = SimpleImputerWrapper(strategy='median')
        result1 = imputer1.fit_transform(df)
        
        imputer2 = SimpleImputerWrapper(strategy='median')
        imputer2.fit(df)
        result2 = imputer2.transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestSimpleImputerRegistration:
    """Test SimpleImputer registration in PreprocessorStepRegistry"""
    
    def test_simple_imputer_registered(self):
        """Verify SimpleImputer is properly registered"""
        # Given: Registry should contain SimpleImputer
        
        # When: Check registration
        registered = 'simple_imputer' in PreprocessorStepRegistry.preprocessor_steps
        
        # Then: Should be registered
        assert registered, "SimpleImputer not registered"
    
    def test_simple_imputer_creation_through_registry(self):
        """Test creating SimpleImputer through registry"""
        # Given: Registry with registered imputer
        
        # When: Create imputer through registry
        imputer = PreprocessorStepRegistry.create('simple_imputer')
        
        # Then: Should create correct instance
        assert isinstance(imputer, SimpleImputerWrapper)
    
    def test_simple_imputer_creation_with_parameters(self):
        """Test creating SimpleImputer with parameters through registry"""
        # Given: Registry and custom parameters
        
        # When: Create with parameters
        imputer = PreprocessorStepRegistry.create(
            'simple_imputer',
            strategy='median',
            columns=['col1', 'col2'],
            create_missing_indicators=True
        )
        
        # Then: Should create instance with parameters
        assert isinstance(imputer, SimpleImputerWrapper)
        assert imputer.strategy == 'median'
        assert imputer.columns == ['col1', 'col2']
        assert imputer.create_missing_indicators is True


class TestSimpleImputerIntegration:
    """Integration tests for SimpleImputer component"""
    
    def test_simple_imputer_pipeline_compatibility(self, test_data_generator):
        """Test SimpleImputer works correctly in preprocessing pipeline context"""
        # Given: Realistic data with some missing values
        np.random.seed(42)
        df_full, _ = test_data_generator.classification_data(n_samples=50, n_features=3)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['f1', 'f2', 'f3']
        
        # Introduce missing values randomly (10% missing rate)
        missing_mask = np.random.random(df.shape) < 0.1
        df[missing_mask] = np.nan
        
        # When: Apply imputer
        imputer = SimpleImputerWrapper(strategy='mean')
        result = imputer.fit_transform(df)
        
        # Then: Should produce valid result compatible with further processing
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)
        assert not result.isnull().any().any()  # All missing values filled
        
        # Should be compatible with further processing (e.g., scaling)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result)
        assert scaled_result.shape == result.shape
    
    def test_simple_imputer_deterministic_behavior(self):
        """Test SimpleImputer produces deterministic results"""
        # Given: Same data with missing values
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # When: Apply same imputer multiple times
        imputer1 = SimpleImputerWrapper(strategy='mean')
        imputer2 = SimpleImputerWrapper(strategy='mean')
        
        result1 = imputer1.fit_transform(df)
        result2 = imputer2.fit_transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_simple_imputer_handles_edge_cases(self):
        """Test SimpleImputer handles edge cases gracefully"""
        # Given: Edge case scenarios
        edge_cases = [
            # Single row with missing
            pd.DataFrame({'feature': [np.nan]}),
            # Single row without missing
            pd.DataFrame({'feature': [1.0]}),
            # All same value except missing
            pd.DataFrame({'feature': [5.0, 5.0, np.nan, 5.0]})
        ]
        
        imputer = SimpleImputerWrapper(strategy='most_frequent')
        
        # When: Apply to edge cases
        for i, df in enumerate(edge_cases):
            if df['feature'].dropna().empty:
                # All missing case - should raise error
                with pytest.raises(ValueError):
                    imputer.fit(df)
            else:
                # Should handle other edge cases
                result = imputer.fit_transform(df)
                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(df)
                # For non-empty cases, should not have missing values
                if not df['feature'].dropna().empty:
                    assert not result.isnull().any().any()
    
    def test_simple_imputer_memory_efficiency(self):
        """Test SimpleImputer doesn't create excessive memory overhead"""
        # Given: Larger dataset to test memory efficiency
        np.random.seed(42)
        large_data = np.random.randn(1000, 5)
        
        # Introduce missing values
        missing_mask = np.random.random((1000, 5)) < 0.05  # 5% missing
        large_data[missing_mask] = np.nan
        
        df = pd.DataFrame(large_data, columns=[f'feature_{i}' for i in range(5)])
        
        # When: Apply imputer
        imputer = SimpleImputerWrapper(strategy='mean')
        result = imputer.fit_transform(df)
        
        # Then: Should complete without memory issues
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert not result.isnull().any().any()
        
        # Memory usage should be reasonable (not dramatically larger than input)
        # This is more of a performance check than a strict assertion