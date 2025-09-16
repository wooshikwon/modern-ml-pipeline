"""
Scaler Components Comprehensive Tests
Testing StandardScalerWrapper, MinMaxScalerWrapper, RobustScalerWrapper

Architecture Compliance:
- Global application type behavior
- Column name preservation
- DataFrame-first approach
- Numeric column auto-detection
- Real component testing (no mock hell)
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.components.preprocessor.modules.scaler import (
    StandardScalerWrapper,
    MinMaxScalerWrapper,
    RobustScalerWrapper
)
from src.components.preprocessor.registry import PreprocessorStepRegistry
# Import the entire scaler module to trigger registration
import src.components.preprocessor.modules.scaler  # noqa: F401


class TestStandardScalerWrapper:
    """StandardScalerWrapper comprehensive testing"""
    
    def test_standard_scaler_global_application_type(self):
        """Verify StandardScaler is Global type - auto-applies to all numeric columns"""
        # Given: StandardScaler instance
        scaler = StandardScalerWrapper()
        
        # When: Check application type
        app_type = scaler.get_application_type()
        
        # Then: Should be global
        assert app_type == 'global'
    
    def test_standard_scaler_preserves_column_names(self):
        """Verify StandardScaler preserves original column names"""
        # Given: StandardScaler instance
        scaler = StandardScalerWrapper()
        
        # When: Check column name preservation
        preserves = scaler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True
    
    def test_standard_scaler_numeric_column_detection(self, component_test_context):
        """Test automatic numeric column detection for global application"""
        # Given: ComponentTestContext로 설정 및 데이터 준비
        with component_test_context.classification_stack() as ctx:
            # Context 데이터를 확장하여 다양한 타입 컬럼 추가
            raw_df = ctx.adapter.read(ctx.data_path)
            df = raw_df.copy()
            df['category'] = ['a', 'b', 'c', 'd', 'e'] * (len(df) // 5) + ['a'] * (len(df) % 5)
            df['boolean'] = [True, False] * (len(df) // 2) + [True] * (len(df) % 2)

            scaler = StandardScalerWrapper()

            # When: Get applicable columns
            applicable_cols = scaler.get_applicable_columns(df)

            # Then: Only numeric columns should be selected
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            assert set(applicable_cols) == set(numeric_cols)
            assert 'category' not in applicable_cols
            assert 'boolean' not in applicable_cols
    
    def test_standard_scaler_fit_transform_functionality(self, component_test_context):
        """Test core fit-transform functionality with real data"""
        # Given: ComponentTestContext로 설정 및 데이터 준비
        with component_test_context.classification_stack() as ctx:
            # Context에서 제공하는 결정론적 데이터 사용
            raw_df = ctx.adapter.read(ctx.data_path)
            df = raw_df.copy()

            # Make features have different scales to test normalization
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            if len(feature_cols) >= 3:
                df[feature_cols[0]] = df[feature_cols[0]] * 100  # Large scale
                df[feature_cols[1]] = df[feature_cols[1]] * 0.01  # Small scale
                df[feature_cols[2]] = df[feature_cols[2]] + 1000  # Large offset

            scaler = StandardScalerWrapper()

            # When: Fit and transform
            scaler.fit(df)
            result = scaler.transform(df)

            # Then: Result should be standardized (mean ≈ 0, std ≈ 1)
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == list(df.columns)  # Column names preserved
            assert len(result) == len(df)  # Row count preserved
            assert result.index.equals(df.index)  # Index preserved

            # Check standardization properties for numeric columns
            for col in feature_cols:
                col_mean = result[col].mean()
                col_std = result[col].std()
                if not pd.isna(col_mean):
                    assert abs(col_mean) < 0.1  # Mean ≈ 0 (allowing for numerical precision)
                if not pd.isna(col_std) and col_std > 0:
                    assert abs(col_std - 1.0) < 0.3  # Std ≈ 1 (relaxed tolerance)

            # Context 헬퍼로 데이터 흐름 검증
            assert ctx.validate_data_flow(df, result)
    
    def test_standard_scaler_fit_transform_single_call(self, component_test_context):
        """Test fit_transform convenience method"""
        # Given: ComponentTestContext로 설정 및 데이터 준비
        with component_test_context.classification_stack() as ctx:
            # Context에서 제공하는 결정론적 데이터 사용
            raw_df = ctx.adapter.read(ctx.data_path)
            # 테스트용으로 처음 2개 feature 컬럼만 사용
            feature_cols = [col for col in raw_df.columns if col.startswith('feature_')][:2]
            df = raw_df[feature_cols].copy()

            scaler = StandardScalerWrapper()

            # When: Use fit_transform
            result = scaler.fit_transform(df)

            # Then: Should match separate fit+transform
            scaler2 = StandardScalerWrapper()
            scaler2.fit(df)
            expected = scaler2.transform(df)

            pd.testing.assert_frame_equal(result, expected)

            # Context 헬퍼로 데이터 흐름 검증
            assert ctx.validate_data_flow(df, result)
    
    def test_standard_scaler_handles_single_feature(self):
        """Test handling of single feature DataFrame"""
        # Given: Single column data
        df = pd.DataFrame({
            'single_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        scaler = StandardScalerWrapper()
        
        # When: Fit and transform
        scaler.fit(df)
        result = scaler.transform(df)
        
        # Then: Should work correctly
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['single_feature']
        assert abs(result['single_feature'].mean()) < 1e-10
        assert abs(result['single_feature'].std() - 1.0) < 0.2
    
    def test_standard_scaler_handles_constant_feature(self):
        """Test handling of constant feature (zero variance)"""
        # Given: Constant feature (zero variance)
        df = pd.DataFrame({
            'constant_feature': [5.0, 5.0, 5.0, 5.0, 5.0],
            'normal_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        scaler = StandardScalerWrapper()
        
        # When: Fit and transform
        scaler.fit(df)
        result = scaler.transform(df)
        
        # Then: Constant feature remains unchanged (filtered out from scaling)
        assert isinstance(result, pd.DataFrame)
        assert result['constant_feature'].nunique() == 1  # Remains constant
        assert result['constant_feature'].iloc[0] == 5.0  # Original value preserved
        assert abs(result['normal_feature'].mean()) < 1e-10
        assert abs(result['normal_feature'].std() - 1.0) < 0.2


class TestMinMaxScalerWrapper:
    """MinMaxScalerWrapper comprehensive testing"""
    
    def test_minmax_scaler_global_application_type(self):
        """Verify MinMaxScaler is Global type"""
        # Given: MinMaxScaler instance
        scaler = MinMaxScalerWrapper()
        
        # When: Check application type
        app_type = scaler.get_application_type()
        
        # Then: Should be global
        assert app_type == 'global'
    
    def test_minmax_scaler_preserves_column_names(self):
        """Verify MinMaxScaler preserves original column names"""
        # Given: MinMaxScaler instance
        scaler = MinMaxScalerWrapper()
        
        # When: Check column name preservation
        preserves = scaler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True
    
    def test_minmax_scaler_fit_transform_functionality(self, component_test_context):
        """Test core fit-transform functionality with scaling to [0,1]"""
        # Given: ComponentTestContext로 설정 및 데이터 준비
        with component_test_context.classification_stack() as ctx:
            # Context에서 제공하는 결정론적 데이터 사용하고 범위 조정
            raw_df = ctx.adapter.read(ctx.data_path)
            feature_cols = [col for col in raw_df.columns if col.startswith('feature_')][:2]
            df = raw_df[feature_cols].copy()

            # Create known ranges for testing MinMax scaling
            if len(feature_cols) >= 2:
                df[feature_cols[0]] = [0, 10, 20, 30, 40] * (len(df) // 5) + [0] * (len(df) % 5)  # Range 0-40
                df[feature_cols[1]] = [-5, 0, 5, 10, 15] * (len(df) // 5) + [-5] * (len(df) % 5)   # Range -5 to 15
        
            scaler = MinMaxScalerWrapper()

            # When: Fit and transform
            scaler.fit(df)
            result = scaler.transform(df)

            # Then: Result should be scaled to [0,1]
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == list(df.columns)

            # Check Min-Max scaling properties
            for col in df.columns:
                if df[col].nunique() > 1:  # Only check non-constant columns
                    assert abs(result[col].min() - 0.0) < 1e-10  # Min should be 0
                    assert abs(result[col].max() - 1.0) < 1e-10  # Max should be 1
                    assert result[col].min() <= result[col].max()  # Sanity check

            # Context 헬퍼로 데이터 흐름 검증
            assert ctx.validate_data_flow(df, result)
    
    def test_minmax_scaler_numeric_column_detection(self):
        """Test automatic numeric column detection for MinMax scaler"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['x', 'y', 'z'],
            'bool_col': [True, False, True]
        })
        
        scaler = MinMaxScalerWrapper()
        
        # When: Get applicable columns
        applicable_cols = scaler.get_applicable_columns(df)
        
        # Then: Only numeric columns
        expected = ['int_col', 'float_col']
        assert set(applicable_cols) == set(expected)
    
    def test_minmax_scaler_handles_negative_values(self):
        """Test MinMax scaling with negative values"""
        # Given: Data with negative values
        df = pd.DataFrame({
            'negative_feature': [-10, -5, 0, 5, 10]
        })
        
        scaler = MinMaxScalerWrapper()
        
        # When: Fit and transform
        scaler.fit(df)
        result = scaler.transform(df)
        
        # Then: Should scale to [0,1] regardless of negative values
        assert result['negative_feature'].min() == 0.0
        assert result['negative_feature'].max() == 1.0
        # Original -10 should map to 0, original 10 should map to 1
        assert result['negative_feature'].iloc[0] == 0.0  # -10 → 0
        assert result['negative_feature'].iloc[-1] == 1.0  # 10 → 1


class TestRobustScalerWrapper:
    """RobustScalerWrapper comprehensive testing"""
    
    def test_robust_scaler_global_application_type(self):
        """Verify RobustScaler is Global type"""
        # Given: RobustScaler instance
        scaler = RobustScalerWrapper()
        
        # When: Check application type
        app_type = scaler.get_application_type()
        
        # Then: Should be global
        assert app_type == 'global'
    
    def test_robust_scaler_preserves_column_names(self):
        """Verify RobustScaler preserves original column names"""
        # Given: RobustScaler instance
        scaler = RobustScalerWrapper()
        
        # When: Check column name preservation
        preserves = scaler.preserves_column_names()
        
        # Then: Should preserve names
        assert preserves is True
    
    def test_robust_scaler_outlier_resistance(self):
        """Test RobustScaler's resistance to outliers compared to StandardScaler"""
        # Given: Data with outliers
        normal_data = [1, 2, 3, 4, 5]
        outlier_data = [1, 2, 3, 4, 100]  # 100 is outlier
        
        df_normal = pd.DataFrame({'feature': normal_data})
        df_outlier = pd.DataFrame({'feature': outlier_data})
        
        # When: Apply both scalers
        robust_scaler = RobustScalerWrapper()
        standard_scaler = StandardScalerWrapper()
        
        # Fit on normal data
        robust_scaler.fit(df_normal)
        standard_scaler.fit(df_normal)
        
        # Transform outlier data
        robust_result = robust_scaler.transform(df_outlier)
        standard_result = standard_scaler.transform(df_outlier)
        
        # Then: RobustScaler should be less affected by outlier
        robust_outlier_value = robust_result['feature'].iloc[-1]
        standard_outlier_value = standard_result['feature'].iloc[-1]
        
        # The outlier should have less extreme value with RobustScaler
        assert abs(robust_outlier_value) < abs(standard_outlier_value)
    
    def test_robust_scaler_fit_transform_functionality(self, test_data_generator):
        """Test core fit-transform functionality using median and IQR"""
        # Given: Real numeric data
        np.random.seed(42)
        X, _ = test_data_generator.classification_data(n_samples=100, n_features=2)
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        
        scaler = RobustScalerWrapper()
        
        # When: Fit and transform
        scaler.fit(df)
        result = scaler.transform(df)
        
        # Then: Result should use median centering and IQR scaling
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)
        
        # RobustScaler centers on median (should be close to 0)
        for col in result.columns:
            # Median should be close to 0 (within reasonable tolerance)
            assert abs(result[col].median()) < 0.1
    
    def test_robust_scaler_numeric_column_detection(self):
        """Test automatic numeric column detection for Robust scaler"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'int64_col': [1, 2, 3, 4, 5],
            'float64_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'object_col': ['a', 'b', 'c', 'd', 'e'],
            'category_col': pd.Categorical(['x', 'y', 'z', 'x', 'y'])
        })
        
        scaler = RobustScalerWrapper()
        
        # When: Get applicable columns
        applicable_cols = scaler.get_applicable_columns(df)
        
        # Then: Only int64 and float64 columns
        expected = ['int64_col', 'float64_col']
        assert set(applicable_cols) == set(expected)
        assert 'object_col' not in applicable_cols
        assert 'category_col' not in applicable_cols


class TestScalerRegistration:
    """Test scaler registration in PreprocessorStepRegistry"""
    
    def test_all_scalers_registered(self):
        """Verify all scaler types are properly registered"""
        # Given: Registry should contain all scalers
        
        # When: Check registration
        standard_registered = 'standard_scaler' in PreprocessorStepRegistry.preprocessor_steps
        minmax_registered = 'min_max_scaler' in PreprocessorStepRegistry.preprocessor_steps
        robust_registered = 'robust_scaler' in PreprocessorStepRegistry.preprocessor_steps
        
        # Then: All should be registered
        assert standard_registered, "StandardScaler not registered"
        assert minmax_registered, "MinMaxScaler not registered"
        assert robust_registered, "RobustScaler not registered"
    
    def test_scaler_creation_through_registry(self):
        """Test creating scalers through registry"""
        # Given: Registry with registered scalers
        
        # When: Create scalers through registry
        standard_scaler = PreprocessorStepRegistry.create('standard_scaler')
        minmax_scaler = PreprocessorStepRegistry.create('min_max_scaler')
        robust_scaler = PreprocessorStepRegistry.create('robust_scaler')
        
        # Then: Should create correct instances
        assert isinstance(standard_scaler, StandardScalerWrapper)
        assert isinstance(minmax_scaler, MinMaxScalerWrapper)
        assert isinstance(robust_scaler, RobustScalerWrapper)
    
    def test_scaler_creation_with_parameters(self):
        """Test creating scalers with parameters through registry"""
        # Given: Registry and parameters
        columns_param = ['feature_1', 'feature_2']
        
        # When: Create with parameters (though columns are ignored for global scalers)
        standard_scaler = PreprocessorStepRegistry.create('standard_scaler', columns=columns_param)
        
        # Then: Should create instance with parameters
        assert isinstance(standard_scaler, StandardScalerWrapper)
        assert standard_scaler.columns == columns_param  # Stored but ignored


class TestScalerIntegration:
    """Integration tests for scaler components"""
    
    def test_scaler_pipeline_compatibility(self, test_data_generator):
        """Test scalers work correctly in preprocessing pipeline context"""
        # Given: Realistic data for scaling
        np.random.seed(42)
        df_full, _ = test_data_generator.classification_data(n_samples=50, n_features=3)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['f1', 'f2', 'f3']
        
        # Make features have different characteristics
        df['f1'] = df['f1'] * 1000  # Large scale
        df['f2'] = df['f2'] * 0.001  # Small scale
        df['f3'] = df['f3'] + 100  # Offset
        
        # When: Apply different scalers
        standard_scaler = StandardScalerWrapper()
        minmax_scaler = MinMaxScalerWrapper()
        robust_scaler = RobustScalerWrapper()
        
        # Each scaler should handle the data independently
        standard_result = standard_scaler.fit_transform(df)
        minmax_result = minmax_scaler.fit_transform(df)
        robust_result = robust_scaler.fit_transform(df)
        
        # Then: All should produce valid results with preserved structure
        for result in [standard_result, minmax_result, robust_result]:
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == list(df.columns)
            assert len(result) == len(df)
            assert not result.isnull().all().any()  # No completely null columns
    
    def test_scaler_deterministic_behavior(self, test_data_generator):
        """Test scalers produce deterministic results"""
        # Given: Same input data
        np.random.seed(42)
        X, _ = test_data_generator.regression_data(n_samples=30, n_features=2)
        df = pd.DataFrame(X, columns=['var1', 'var2'])
        
        # When: Apply same scaler multiple times
        scaler1 = StandardScalerWrapper()
        scaler2 = StandardScalerWrapper()
        
        result1 = scaler1.fit_transform(df)
        result2 = scaler2.fit_transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_scalers_handle_edge_case_data(self):
        """Test scalers handle edge cases gracefully"""
        # Given: Edge case data
        edge_cases = [
            # All zeros
            pd.DataFrame({'feature': [0.0, 0.0, 0.0]}),
            # All same value (constant)
            pd.DataFrame({'feature': [5.0, 5.0, 5.0]}),
            # Very small values
            pd.DataFrame({'feature': [1e-10, 2e-10, 3e-10]}),
            # Very large values
            pd.DataFrame({'feature': [1e10, 2e10, 3e10]})
        ]
        
        scalers = [StandardScalerWrapper(), MinMaxScalerWrapper(), RobustScalerWrapper()]
        
        # When: Apply scalers to edge cases
        for df in edge_cases:
            for scaler in scalers:
                # Should not raise exceptions
                scaler.fit(df)
                result = scaler.transform(df)
                
                # Then: Should produce DataFrame output
                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == list(df.columns)