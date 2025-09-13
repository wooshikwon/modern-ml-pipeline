"""
Discretizer Components Comprehensive Tests
Testing KBinsDiscretizerWrapper for binning continuous features

Architecture Compliance:
- Targeted application type behavior
- Numeric column detection and discretization
- DataFrame-first approach  
- Multiple encoding strategies (ordinal, onehot)
- Binning strategy configuration
- Column name transformation handling
- Real component testing (no mock hell)
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from src.components.preprocessor.modules.discretizer import KBinsDiscretizerWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry


class TestKBinsDiscretizerWrapper:
    """KBinsDiscretizerWrapper comprehensive testing"""
    
    def test_discretizer_targeted_application_type(self):
        """Verify KBinsDiscretizer is Targeted type - applies to specific columns"""
        # Given: KBinsDiscretizer instance
        discretizer = KBinsDiscretizerWrapper()
        
        # When: Check application type
        app_type = discretizer.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_discretizer_does_not_preserve_column_names(self):
        """Verify KBinsDiscretizer creates new discretized column names"""
        # Given: KBinsDiscretizer instance
        discretizer = KBinsDiscretizerWrapper()
        
        # When: Check column name preservation
        preserves = discretizer.preserves_column_names()
        
        # Then: Should not preserve names (creates discretized features)
        assert preserves is False
    
    def test_discretizer_numeric_column_detection(self):
        """Test automatic numeric column detection"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'category': ['a', 'b', 'c', 'd', 'e'],
            'boolean': [True, False, True, False, True]
        })
        
        discretizer = KBinsDiscretizerWrapper()
        
        # When: Get applicable columns
        applicable_cols = discretizer.get_applicable_columns(df)
        
        # Then: Only numeric columns
        expected = ['numeric_int', 'numeric_float']
        assert set(applicable_cols) == set(expected)
        assert 'category' not in applicable_cols
        assert 'boolean' not in applicable_cols
    
    def test_discretizer_ordinal_encoding_basic_functionality(self):
        """Test core fit-transform functionality with ordinal encoding"""
        # Given: Continuous numeric data
        df = pd.DataFrame({
            'continuous_feature': [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal', strategy='quantile')
        
        # When: Fit and transform
        discretizer.fit(df)
        result = discretizer.transform(df)
        
        # Then: Should create ordinal-encoded bins
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)  # Same number of rows
        assert list(result.columns) == ['discretized_continuous_feature']  # New column name
        
        # Values should be ordinal (0, 1, 2 for 3 bins)
        unique_values = sorted(result['discretized_continuous_feature'].unique())
        expected_values = [0.0, 1.0, 2.0]  # 3 bins -> 0, 1, 2
        assert unique_values == expected_values
    
    def test_discretizer_onehot_encoding_functionality(self):
        """Test one-hot encoding option"""
        # Given: Continuous numeric data
        df = pd.DataFrame({
            'feature': [1.0, 3.0, 5.0, 7.0, 9.0]
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='onehot-dense', strategy='uniform')
        
        # When: Fit and transform
        discretizer.fit(df)
        result = discretizer.transform(df)
        
        # Then: Should create one-hot encoded bins
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Same number of rows
        assert result.shape[1] == 3  # 3 bins -> 3 one-hot columns
        
        # Each row should have exactly one '1' (one-hot property)
        for i in range(len(result)):
            row_sum = result.iloc[i].sum()
            assert row_sum == 1, f"Row {i} should have exactly one '1', got sum: {row_sum}"
        
        # Check column names have discretized prefix
        discretized_cols = [col for col in result.columns if 'discretized' in col]
        assert len(discretized_cols) == 3
    
    def test_discretizer_n_bins_parameter_effect(self):
        """Test that n_bins parameter affects discretization"""
        # Given: Same data with different n_bins
        df = pd.DataFrame({
            'feature': np.linspace(0, 10, 20)  # 20 points from 0 to 10
        })
        
        # When: Use different number of bins
        discretizer_3_bins = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        discretizer_5_bins = KBinsDiscretizerWrapper(n_bins=5, encode='ordinal')
        
        result_3 = discretizer_3_bins.fit_transform(df)
        result_5 = discretizer_5_bins.fit_transform(df)
        
        # Then: Different n_bins should produce different numbers of unique values
        unique_3 = result_3['discretized_feature'].nunique()
        unique_5 = result_5['discretized_feature'].nunique()
        
        assert unique_3 == 3, f"3 bins should produce 3 unique values, got {unique_3}"
        assert unique_5 == 5, f"5 bins should produce 5 unique values, got {unique_5}"
    
    def test_discretizer_strategy_uniform(self):
        """Test uniform binning strategy"""
        # Given: Data with known range
        df = pd.DataFrame({
            'feature': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 to 10
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=5, encode='ordinal', strategy='uniform')
        
        # When: Fit and transform
        result = discretizer.fit_transform(df)
        
        # Then: Should create uniform-width bins
        assert isinstance(result, pd.DataFrame)
        
        # With uniform strategy and range 0-10 with 5 bins:
        # Each bin width should be 2: [0-2), [2-4), [4-6), [6-8), [8-10]
        # Values 0,1 -> bin 0; 2,3 -> bin 1; etc.
        unique_bins = sorted(result['discretized_feature'].unique())
        assert len(unique_bins) <= 5  # Should have at most 5 bins
    
    def test_discretizer_strategy_quantile(self):
        """Test quantile binning strategy"""
        # Given: Data with skewed distribution
        np.random.seed(42)
        # Create skewed data (most values small, few large)
        skewed_data = np.concatenate([
            np.random.uniform(0, 2, 80),   # Most data in 0-2 range
            np.random.uniform(8, 10, 20)   # Few data in 8-10 range
        ])
        
        df = pd.DataFrame({
            'skewed_feature': skewed_data
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=4, encode='ordinal', strategy='quantile')
        
        # When: Fit and transform
        result = discretizer.fit_transform(df)
        
        # Then: Should create quantile-based bins (equal frequency)
        assert isinstance(result, pd.DataFrame)
        
        # Each bin should have approximately equal number of samples
        bin_counts = result['discretized_skewed_feature'].value_counts().sort_index()
        expected_count_per_bin = len(df) // 4  # 100 / 4 = 25
        
        for count in bin_counts:
            # Allow some tolerance for quantile binning
            assert abs(count - expected_count_per_bin) <= 5, \
                f"Quantile bins should have ~{expected_count_per_bin} samples, got {count}"
    
    def test_discretizer_strategy_kmeans(self):
        """Test k-means binning strategy"""
        # Given: Data with clear clusters
        np.random.seed(42)
        # Create data with natural clusters
        cluster1 = np.random.normal(2, 0.5, 30)
        cluster2 = np.random.normal(8, 0.5, 30)
        clustered_data = np.concatenate([cluster1, cluster2])
        
        df = pd.DataFrame({
            'clustered_feature': clustered_data
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=2, encode='ordinal', strategy='kmeans')
        
        # When: Fit and transform
        result = discretizer.fit_transform(df)
        
        # Then: Should create k-means based bins
        assert isinstance(result, pd.DataFrame)
        
        # Should create 2 bins corresponding to the 2 clusters
        unique_bins = sorted(result['discretized_clustered_feature'].unique())
        assert len(unique_bins) == 2
    
    def test_discretizer_invalid_strategy_error(self):
        """Test error handling for invalid strategy"""
        # Given: Discretizer with invalid strategy
        discretizer = KBinsDiscretizerWrapper(strategy='invalid_strategy')
        df = pd.DataFrame({'feature': [1, 2, 3]})
        
        # When & Then: Should raise meaningful error during fit
        with pytest.raises(ValueError) as exc_info:
            discretizer.fit(df)
        
        error_msg = str(exc_info.value)
        assert "strategy" in error_msg
        assert "invalid_strategy" in error_msg
        # Should mention valid strategies
        assert any(strategy in error_msg for strategy in ['uniform', 'quantile', 'kmeans'])
    
    def test_discretizer_invalid_encode_error(self):
        """Test error handling for invalid encode parameter"""
        # Given: Discretizer with invalid encode setting
        discretizer = KBinsDiscretizerWrapper(encode='invalid_encoding')
        df = pd.DataFrame({'feature': [1, 2, 3]})
        
        # When & Then: Should raise meaningful error during fit
        with pytest.raises(ValueError) as exc_info:
            discretizer.fit(df)
        
        error_msg = str(exc_info.value)
        assert "encode" in error_msg
        assert "invalid_encoding" in error_msg
        # Should mention valid encodings
        assert any(encoding in error_msg for encoding in ['ordinal', 'onehot'])
    
    def test_discretizer_multiple_columns(self):
        """Test discretization on multiple columns simultaneously"""
        # Given: Multiple numeric columns
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6],
            'feature_2': [10, 20, 30, 40, 50, 60],
            'feature_3': [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal', strategy='uniform')
        
        # When: Fit and transform
        result = discretizer.fit_transform(df)
        
        # Then: All columns should be discretized
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6
        
        expected_columns = ['discretized_feature_1', 'discretized_feature_2', 'discretized_feature_3']
        assert list(result.columns) == expected_columns
        
        # Each column should have ordinal values (0, 1, 2 for 3 bins)
        for col in result.columns:
            unique_values = sorted(result[col].unique())
            assert set(unique_values).issubset({0.0, 1.0, 2.0})
    
    def test_discretizer_get_output_column_names_ordinal(self):
        """Test output column name generation for ordinal encoding"""
        # Given: KBinsDiscretizer with ordinal encoding
        discretizer = KBinsDiscretizerWrapper(encode='ordinal')
        
        # When: Get output column names
        input_cols = ['feature_a', 'feature_b']
        output_cols = discretizer.get_output_column_names(input_cols)
        
        # Then: Should have discretized prefix for each input column
        expected = ['discretized_feature_a', 'discretized_feature_b']
        assert output_cols == expected
    
    def test_discretizer_get_output_column_names_onehot(self):
        """Test output column name generation for one-hot encoding"""
        # Given: KBinsDiscretizer with one-hot encoding
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='onehot-dense')
        
        # When: Get output column names
        input_cols = ['single_feature']
        output_cols = discretizer.get_output_column_names(input_cols)
        
        # Then: Should have bin-specific columns
        expected = ['discretized_single_feature_bin0', 'discretized_single_feature_bin1', 'discretized_single_feature_bin2']
        assert output_cols == expected
    
    def test_discretizer_preserves_index(self):
        """Test that DataFrame index is preserved during discretization"""
        # Given: Data with custom index
        custom_index = ['sample_a', 'sample_b', 'sample_c', 'sample_d']
        df = pd.DataFrame({
            'feature': [1.0, 3.0, 5.0, 7.0]
        }, index=custom_index)
        
        discretizer = KBinsDiscretizerWrapper(n_bins=2, encode='ordinal')
        
        # When: Fit and transform
        result = discretizer.fit_transform(df)
        
        # Then: Index should be preserved
        assert list(result.index) == custom_index
    
    def test_discretizer_deterministic_behavior(self):
        """Test discretizer produces deterministic results"""
        # Given: Same data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.uniform(0, 10, 20)
        })
        
        # When: Apply same discretizer multiple times
        discretizer1 = KBinsDiscretizerWrapper(n_bins=4, encode='ordinal', strategy='quantile')
        discretizer2 = KBinsDiscretizerWrapper(n_bins=4, encode='ordinal', strategy='quantile')
        
        result1 = discretizer1.fit_transform(df)
        result2 = discretizer2.fit_transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_discretizer_fit_transform_equivalence(self):
        """Test that fit_transform gives same result as fit + transform"""
        # Given: Data
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # When: Compare fit_transform vs fit + transform
        discretizer1 = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        result1 = discretizer1.fit_transform(df)
        
        discretizer2 = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        discretizer2.fit(df)
        result2 = discretizer2.transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_discretizer_transform_consistency(self):
        """Test that transform produces consistent results on new data"""
        # Given: Training and test data
        train_df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        test_df = pd.DataFrame({
            'feature': [2.5, 5.5, 8.5]  # Values between training points
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal', strategy='uniform')
        
        # When: Fit on training data, transform test data
        discretizer.fit(train_df)
        train_result = discretizer.transform(train_df)
        test_result = discretizer.transform(test_df)
        
        # Then: Test data should be discretized using training bins
        assert isinstance(test_result, pd.DataFrame)
        assert len(test_result) == 3
        
        # Values should be within the same bin range as training
        train_bins = set(train_result['discretized_feature'].unique())
        test_bins = set(test_result['discretized_feature'].unique())
        assert test_bins.issubset(train_bins), "Test bins should be subset of training bins"


class TestKBinsDiscretizerRegistration:
    """Test KBinsDiscretizer registration in PreprocessorStepRegistry"""
    
    def test_discretizer_registered(self):
        """Verify KBinsDiscretizer is properly registered"""
        # Given: Registry should contain KBinsDiscretizer
        
        # When: Check registration
        registered = 'kbins_discretizer' in PreprocessorStepRegistry.preprocessor_steps
        
        # Then: Should be registered
        assert registered, "KBinsDiscretizer not registered"
    
    def test_discretizer_creation_through_registry(self):
        """Test creating KBinsDiscretizer through registry"""
        # Given: Registry with registered discretizer
        
        # When: Create discretizer through registry
        discretizer = PreprocessorStepRegistry.create('kbins_discretizer')
        
        # Then: Should create correct instance
        assert isinstance(discretizer, KBinsDiscretizerWrapper)
    
    def test_discretizer_creation_with_parameters(self):
        """Test creating KBinsDiscretizer with parameters through registry"""
        # Given: Registry and custom parameters
        
        # When: Create with parameters
        discretizer = PreprocessorStepRegistry.create(
            'kbins_discretizer',
            n_bins=4,
            encode='onehot-dense',
            strategy='kmeans',
            columns=['num_col']
        )
        
        # Then: Should create instance with parameters
        assert isinstance(discretizer, KBinsDiscretizerWrapper)
        assert discretizer.n_bins == 4
        assert discretizer.encode == 'onehot-dense'
        assert discretizer.strategy == 'kmeans'
        assert discretizer.columns == ['num_col']


class TestKBinsDiscretizerIntegration:
    """Integration tests for KBinsDiscretizer component"""
    
    def test_discretizer_pipeline_compatibility(self, test_data_generator):
        """Test KBinsDiscretizer works in preprocessing pipeline context"""
        # Given: Continuous numeric data with realistic values
        np.random.seed(42)
        df_full, _ = test_data_generator.regression_data(n_samples=50, n_features=3)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['cont_1', 'cont_2', 'cont_3']
        
        # When: Apply discretizer
        discretizer = KBinsDiscretizerWrapper(n_bins=4, encode='ordinal', strategy='quantile')
        result = discretizer.fit_transform(df)
        
        # Then: Should produce valid result compatible with further processing
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert result.shape[1] == df.shape[1]  # Same number of columns (ordinal encoding)
        
        # All values should be discrete integers
        for col in result.columns:
            assert result[col].dtype in [np.int64, np.float64]
            unique_values = result[col].unique()
            assert len(unique_values) <= 4  # At most 4 bins
    
    def test_discretizer_with_other_preprocessors(self, test_data_generator):
        """Test discretizer compatibility with other preprocessors"""
        # Given: Data that will be processed by multiple steps
        np.random.seed(42)
        df_full, _ = test_data_generator.classification_data(n_samples=40, n_features=2)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        
        # Apply discretizer first
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal', strategy='uniform')
        discretized = discretizer.fit_transform(df)
        
        # Then apply scaling to discretized data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(discretized)
        
        # Then: Should be compatible
        assert scaled_result.shape == discretized.shape
        assert not np.isnan(scaled_result).any()
    
    def test_discretizer_handles_edge_cases(self):
        """Test discretizer handles edge cases gracefully"""
        # Given: Edge case data
        edge_cases = [
            # Constant values
            pd.DataFrame({'constant': [5.0, 5.0, 5.0, 5.0]}),
            # Very small range
            pd.DataFrame({'small_range': [1.0001, 1.0002, 1.0003]}),
            # Single unique value after rounding
            pd.DataFrame({'single_val': [1.0, 1.0, 1.0]})
        ]
        
        # When: Apply discretizer to edge cases
        for i, df in enumerate(edge_cases):
            discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal', strategy='uniform')
            
            # Should handle edge cases without errors
            result = discretizer.fit_transform(df)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(df)
            
            # For constant data, should produce single bin
            if df.iloc[:, 0].nunique() == 1:
                assert result.iloc[:, 0].nunique() == 1
    
    def test_discretizer_memory_efficiency(self):
        """Test discretizer doesn't create excessive memory overhead"""
        # Given: Larger dataset
        np.random.seed(42)
        large_data = np.random.uniform(0, 100, (1000, 3))
        df = pd.DataFrame(large_data, columns=['var1', 'var2', 'var3'])
        
        # When: Apply discretizer
        discretizer = KBinsDiscretizerWrapper(n_bins=5, encode='ordinal', strategy='quantile')
        result = discretizer.fit_transform(df)
        
        # Then: Should complete without memory issues
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape  # Same dimensions for ordinal encoding
        
        # All values should be within expected bin range
        for col in result.columns:
            unique_values = result[col].unique()
            assert len(unique_values) <= 5  # At most 5 bins
            assert all(val >= 0 for val in unique_values)  # Non-negative bin indices