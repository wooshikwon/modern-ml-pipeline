"""
Feature Generator Components Comprehensive Tests
Testing TreeBasedFeatureGenerator and PolynomialFeaturesWrapper

Architecture Compliance:
- Targeted application type behavior
- Supervised learning requirements (target variable needed)
- DataFrame-first approach  
- New feature column generation
- Column name transformation handling
- Real component testing (no mock hell)
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from src.components.preprocessor.modules.feature_generator import (
    TreeBasedFeatureGenerator,
    PolynomialFeaturesWrapper
)
from src.components.preprocessor.registry import PreprocessorStepRegistry


class TestTreeBasedFeatureGenerator:
    """TreeBasedFeatureGenerator comprehensive testing"""
    
    def test_tree_feature_generator_targeted_application_type(self):
        """Verify TreeBasedFeatureGenerator is Targeted type"""
        # Given: TreeBasedFeatureGenerator instance
        generator = TreeBasedFeatureGenerator()
        
        # When: Check application type
        app_type = generator.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_tree_feature_generator_does_not_preserve_column_names(self):
        """Verify TreeBasedFeatureGenerator creates new feature columns"""
        # Given: TreeBasedFeatureGenerator instance
        generator = TreeBasedFeatureGenerator()
        
        # When: Check column name preservation
        preserves = generator.preserves_column_names()
        
        # Then: Should not preserve names (creates new tree features)
        assert preserves is False
    
    def test_tree_feature_generator_numeric_column_detection(self):
        """Test automatic numeric column detection"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'category': ['a', 'b', 'c', 'd', 'e'],
            'boolean': [True, False, True, False, True]
        })
        
        generator = TreeBasedFeatureGenerator()
        
        # When: Get applicable columns
        applicable_cols = generator.get_applicable_columns(df)
        
        # Then: Only numeric columns
        expected = ['numeric_int', 'numeric_float']
        assert set(applicable_cols) == set(expected)
        assert 'category' not in applicable_cols
        assert 'boolean' not in applicable_cols
    
    def test_tree_feature_generator_requires_target_variable(self):
        """Test that TreeBasedFeatureGenerator requires target variable for fitting"""
        # Given: Numeric data without target
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5]
        })
        
        generator = TreeBasedFeatureGenerator(n_estimators=5, max_depth=2)
        
        # When & Then: Should raise error when y is None
        with pytest.raises(ValueError) as exc_info:
            generator.fit(df, y=None)
        
        error_msg = str(exc_info.value)
        assert "target variable" in error_msg
        assert "y" in error_msg
    
    def test_tree_feature_generator_basic_functionality(self, test_data_generator):
        """Test core fit-transform functionality with supervised learning"""
        # Given: Numeric features and target for classification
        np.random.seed(42)
        df_full, y = test_data_generator.classification_data(n_samples=100, n_features=3)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        y_series = pd.Series(y)
        
        generator = TreeBasedFeatureGenerator(n_estimators=5, max_depth=3, random_state=42)
        
        # When: Fit with target and transform
        generator.fit(df, y_series)
        result = generator.transform(df)
        
        # Then: Should create tree-based features
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)  # Same number of rows
        assert result.shape[1] > df.shape[1]  # More columns (new features)
        
        # Check that new column names have 'treefeature' prefix
        tree_feature_cols = [col for col in result.columns if 'treefeature' in col]
        assert len(tree_feature_cols) > 0, "Should have tree feature columns"
        
        # Values should be binary (one-hot encoded leaf indices)
        for col in tree_feature_cols:
            unique_values = result[col].unique()
            assert set(unique_values).issubset({0, 1}), f"Tree features should be binary, got: {unique_values}"
    
    def test_tree_feature_generator_parameters_effect(self, test_data_generator):
        """Test that n_estimators and max_depth parameters affect output"""
        # Given: Same data with different generator configurations
        np.random.seed(42)
        df_full, y = test_data_generator.classification_data(n_samples=50, n_features=2)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['f1', 'f2']
        y_series = pd.Series(y)
        
        # When: Use different parameters
        generator_small = TreeBasedFeatureGenerator(n_estimators=3, max_depth=2, random_state=42)
        generator_large = TreeBasedFeatureGenerator(n_estimators=10, max_depth=4, random_state=42)
        
        result_small = generator_small.fit_transform(df, y_series)
        result_large = generator_large.fit_transform(df, y_series)
        
        # Then: Different parameters should produce different number of features
        # More estimators and depth typically mean more features
        assert result_large.shape[1] > result_small.shape[1], \
            f"Larger config should have more features: {result_large.shape[1]} vs {result_small.shape[1]}"
    
    def test_tree_feature_generator_deterministic_with_random_state(self, test_data_generator):
        """Test that results are deterministic when random_state is set"""
        # Given: Same data and configuration
        np.random.seed(42)
        df_full, y = test_data_generator.classification_data(n_samples=30, n_features=2)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['var1', 'var2']
        y_series = pd.Series(y)
        
        # When: Apply same generator multiple times with same random_state
        generator1 = TreeBasedFeatureGenerator(n_estimators=5, max_depth=2, random_state=123)
        generator2 = TreeBasedFeatureGenerator(n_estimators=5, max_depth=2, random_state=123)
        
        result1 = generator1.fit_transform(df, y_series)
        result2 = generator2.fit_transform(df, y_series)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_tree_feature_generator_handles_small_dataset(self):
        """Test handling of small dataset"""
        # Given: Very small dataset
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0]
        })
        y = pd.Series([0, 1, 0])
        
        generator = TreeBasedFeatureGenerator(n_estimators=2, max_depth=1, random_state=42)
        
        # When: Fit and transform
        generator.fit(df, y)
        result = generator.transform(df)
        
        # Then: Should work without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result.shape[1] > 0  # Should have some features
    
    def test_tree_feature_generator_multiclass_target(self, test_data_generator):
        """Test TreeBasedFeatureGenerator with multiclass target"""
        # Given: Numeric features with multiclass target
        np.random.seed(42)
        # Create synthetic multiclass data
        X = np.random.randn(60, 3)
        y = np.array([0, 1, 2] * 20)  # 3 classes
        
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
        y_series = pd.Series(y)
        
        generator = TreeBasedFeatureGenerator(n_estimators=4, max_depth=2, random_state=42)
        
        # When: Fit and transform
        result = generator.fit_transform(df, y_series)
        
        # Then: Should handle multiclass target
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 60
        assert result.shape[1] > df.shape[1]
    
    def test_tree_feature_generator_get_output_column_names(self):
        """Test output column name generation"""
        # Given: TreeBasedFeatureGenerator
        generator = TreeBasedFeatureGenerator(n_estimators=3, max_depth=2)
        
        # When: Get output column names (before fitting - estimation)
        output_cols = generator.get_output_column_names(['input_col'])
        
        # Then: Should provide estimated column names
        assert isinstance(output_cols, list)
        assert len(output_cols) > 0
        # Should contain 'treefeature' prefix
        assert all('treefeature' in col for col in output_cols)


class TestPolynomialFeaturesWrapper:
    """PolynomialFeaturesWrapper comprehensive testing"""
    
    def test_polynomial_features_targeted_application_type(self):
        """Verify PolynomialFeatures is Targeted type"""
        # Given: PolynomialFeaturesWrapper instance
        poly = PolynomialFeaturesWrapper()
        
        # When: Check application type
        app_type = poly.get_application_type()
        
        # Then: Should be targeted
        assert app_type == 'targeted'
    
    def test_polynomial_features_does_not_preserve_column_names(self):
        """Verify PolynomialFeatures creates new polynomial feature columns"""
        # Given: PolynomialFeaturesWrapper instance
        poly = PolynomialFeaturesWrapper()
        
        # When: Check column name preservation
        preserves = poly.preserves_column_names()
        
        # Then: Should not preserve names (creates polynomial combinations)
        assert preserves is False
    
    def test_polynomial_features_numeric_column_detection(self):
        """Test automatic numeric column detection"""
        # Given: Mixed data types
        df = pd.DataFrame({
            'int_feature': [1, 2, 3],
            'float_feature': [1.1, 2.2, 3.3],
            'string_feature': ['a', 'b', 'c'],
            'bool_feature': [True, False, True]
        })
        
        poly = PolynomialFeaturesWrapper()
        
        # When: Get applicable columns
        applicable_cols = poly.get_applicable_columns(df)
        
        # Then: Only numeric columns
        expected = ['int_feature', 'float_feature']
        assert set(applicable_cols) == set(expected)
        assert 'string_feature' not in applicable_cols
        assert 'bool_feature' not in applicable_cols
    
    def test_polynomial_features_degree_2_basic_functionality(self):
        """Test core fit-transform functionality with degree 2"""
        # Given: Simple numeric data
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [0.5, 1.0, 1.5]
        })
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        # When: Fit and transform
        poly.fit(df)
        result = poly.transform(df)
        
        # Then: Should create polynomial features
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Same number of rows
        assert result.shape[1] > df.shape[1]  # More columns
        
        # Should include original features, squared terms, and interaction terms
        # For 2 features with degree=2: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        expected_min_features = 5
        assert result.shape[1] >= expected_min_features
        
        # Check column names have 'poly' prefix
        poly_cols = [col for col in result.columns if 'poly' in col]
        assert len(poly_cols) == result.shape[1], "All columns should have 'poly' prefix"
    
    def test_polynomial_features_degree_parameter_effect(self):
        """Test that degree parameter affects number of generated features"""
        # Given: Same data with different degrees
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4]
        })
        
        # When: Use different degrees
        poly_degree_1 = PolynomialFeaturesWrapper(degree=1, include_bias=False)
        poly_degree_2 = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        poly_degree_3 = PolynomialFeaturesWrapper(degree=3, include_bias=False)
        
        result_1 = poly_degree_1.fit_transform(df)
        result_2 = poly_degree_2.fit_transform(df)
        result_3 = poly_degree_3.fit_transform(df)
        
        # Then: Higher degree should produce more features
        assert result_1.shape[1] < result_2.shape[1] < result_3.shape[1]
        
        # Degree 1 with 1 feature should have 1 column
        assert result_1.shape[1] == 1
        
        # Degree 2 with 1 feature should have 2 columns (x, x^2)
        assert result_2.shape[1] == 2
        
        # Degree 3 with 1 feature should have 3 columns (x, x^2, x^3)
        assert result_3.shape[1] == 3
    
    def test_polynomial_features_include_bias_parameter(self):
        """Test include_bias parameter effect"""
        # Given: Same data with different bias settings
        df = pd.DataFrame({
            'x': [1, 2, 3]
        })
        
        # When: Use different bias settings
        poly_no_bias = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        poly_with_bias = PolynomialFeaturesWrapper(degree=2, include_bias=True)
        
        result_no_bias = poly_no_bias.fit_transform(df)
        result_with_bias = poly_with_bias.fit_transform(df)
        
        # Then: With bias should have one more column
        assert result_with_bias.shape[1] == result_no_bias.shape[1] + 1
        
        # Bias column should be all ones
        if result_with_bias.shape[1] > result_no_bias.shape[1]:
            # Find bias column (should have all same values, typically 1.0)
            bias_candidates = []
            for col in result_with_bias.columns:
                if result_with_bias[col].nunique() == 1 and result_with_bias[col].iloc[0] == 1.0:
                    bias_candidates.append(col)
            assert len(bias_candidates) >= 1, "Should have bias column with all 1s"
    
    def test_polynomial_features_interaction_only_parameter(self):
        """Test interaction_only parameter"""
        # Given: Two features
        df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [0.5, 1.0, 1.5]
        })
        
        # When: Compare full polynomial vs interaction only
        poly_full = PolynomialFeaturesWrapper(degree=2, include_bias=False, interaction_only=False)
        poly_interaction = PolynomialFeaturesWrapper(degree=2, include_bias=False, interaction_only=True)
        
        result_full = poly_full.fit_transform(df)
        result_interaction = poly_interaction.fit_transform(df)
        
        # Then: Interaction-only should have fewer features (no x1^2, x2^2 terms)
        # Full: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        # Interaction only: x1, x2, x1*x2 = 3 features
        assert result_interaction.shape[1] < result_full.shape[1]
        assert result_interaction.shape[1] == 3  # x1, x2, x1*x2
        assert result_full.shape[1] == 5  # x1, x2, x1^2, x1*x2, x2^2
    
    def test_polynomial_features_multiple_features(self):
        """Test polynomial features with multiple input features"""
        # Given: Multiple numeric features
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [0.5, 1.0, 1.5],
            'feature_3': [2.0, 3.0, 4.0]
        })
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        # When: Fit and transform
        result = poly.fit_transform(df)
        
        # Then: Should create all polynomial combinations
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # For 3 features with degree 2: 3 linear + 3 quadratic + 3 interaction = 9 features
        expected_features = 9
        assert result.shape[1] == expected_features
    
    def test_polynomial_features_deterministic_behavior(self):
        """Test polynomial features produce deterministic results"""
        # Given: Same data
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [0.5, 1.0, 1.5, 2.0]
        })
        
        # When: Apply same transformation multiple times
        poly1 = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        poly2 = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        result1 = poly1.fit_transform(df)
        result2 = poly2.fit_transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_polynomial_features_get_output_column_names(self):
        """Test output column name generation"""
        # Given: PolynomialFeaturesWrapper with known input
        df = pd.DataFrame({
            'x1': [1, 2],
            'x2': [3, 4]
        })
        
        poly = PolynomialFeaturesWrapper(degree=2)
        poly.fit(df)
        
        # When: Get output column names
        output_cols = poly.get_output_column_names(['x1', 'x2'])
        
        # Then: Should provide meaningful column names
        assert isinstance(output_cols, list)
        assert len(output_cols) > 0
        # Should contain 'poly' prefix
        assert all('poly' in col for col in output_cols)
    
    def test_polynomial_features_fit_transform_equivalence(self):
        """Test that fit_transform gives same result as fit + transform"""
        # Given: Data
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        # When: Compare fit_transform vs fit + transform
        poly1 = PolynomialFeaturesWrapper(degree=2)
        result1 = poly1.fit_transform(df)
        
        poly2 = PolynomialFeaturesWrapper(degree=2)
        poly2.fit(df)
        result2 = poly2.transform(df)
        
        # Then: Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestFeatureGeneratorRegistration:
    """Test feature generator registration in PreprocessorStepRegistry"""
    
    def test_all_feature_generators_registered(self):
        """Verify all feature generator types are properly registered"""
        # Given: Registry should contain all feature generators
        
        # When: Check registration
        tree_registered = 'tree_based_feature_generator' in PreprocessorStepRegistry.preprocessor_steps
        poly_registered = 'polynomial_features' in PreprocessorStepRegistry.preprocessor_steps
        
        # Then: All should be registered
        assert tree_registered, "TreeBasedFeatureGenerator not registered"
        assert poly_registered, "PolynomialFeaturesWrapper not registered"
    
    def test_feature_generator_creation_through_registry(self):
        """Test creating feature generators through registry"""
        # Given: Registry with registered generators
        
        # When: Create generators through registry
        tree_gen = PreprocessorStepRegistry.create('tree_based_feature_generator')
        poly_gen = PreprocessorStepRegistry.create('polynomial_features')
        
        # Then: Should create correct instances
        assert isinstance(tree_gen, TreeBasedFeatureGenerator)
        assert isinstance(poly_gen, PolynomialFeaturesWrapper)
    
    def test_feature_generator_creation_with_parameters(self):
        """Test creating feature generators with parameters through registry"""
        # Given: Registry and custom parameters
        
        # When: Create with parameters
        tree_gen = PreprocessorStepRegistry.create(
            'tree_based_feature_generator',
            n_estimators=5,
            max_depth=2,
            random_state=123,
            columns=['num_col']
        )
        poly_gen = PreprocessorStepRegistry.create(
            'polynomial_features',
            degree=3,
            include_bias=True,
            interaction_only=True,
            columns=['poly_col']
        )
        
        # Then: Should create instances with parameters
        assert tree_gen.n_estimators == 5
        assert tree_gen.max_depth == 2
        assert tree_gen.random_state == 123
        assert tree_gen.columns == ['num_col']
        
        assert poly_gen.degree == 3
        assert poly_gen.include_bias is True
        assert poly_gen.interaction_only is True
        assert poly_gen.columns == ['poly_col']


class TestFeatureGeneratorIntegration:
    """Integration tests for feature generator components"""
    
    def test_feature_generators_pipeline_compatibility(self, test_data_generator):
        """Test feature generators work in preprocessing pipeline context"""
        # Given: Numeric data suitable for both generators
        np.random.seed(42)
        df_full, y = test_data_generator.classification_data(n_samples=50, n_features=2)
        # Use only the numeric feature columns, excluding entity_id
        numeric_cols = [col for col in df_full.columns if col.startswith('feature_')]
        df = df_full[numeric_cols].copy()
        df.columns = ['f1', 'f2']
        y_series = pd.Series(y)
        
        # When: Apply both feature generators
        tree_gen = TreeBasedFeatureGenerator(n_estimators=3, max_depth=2, random_state=42)
        poly_gen = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        tree_result = tree_gen.fit_transform(df, y_series)
        poly_result = poly_gen.fit_transform(df)
        
        # Then: Both should produce valid results compatible for further processing
        assert isinstance(tree_result, pd.DataFrame)
        assert isinstance(poly_result, pd.DataFrame)
        assert len(tree_result) == len(poly_result) == len(df)
        
        # Results should be combinable
        combined = pd.concat([tree_result, poly_result], axis=1)
        assert len(combined) == len(df)
        assert combined.shape[1] == tree_result.shape[1] + poly_result.shape[1]
    
    def test_feature_generators_handle_edge_cases(self):
        """Test feature generators handle edge cases gracefully"""
        # Given: Edge case data
        edge_cases = [
            # Very small dataset
            (pd.DataFrame({'x': [1, 2]}), pd.Series([0, 1])),
            # Single feature
            (pd.DataFrame({'single': [1.0, 2.0, 3.0]}), pd.Series([0, 1, 0])),
            # Constant feature
            (pd.DataFrame({'constant': [5.0, 5.0, 5.0]}), pd.Series([0, 1, 0]))
        ]
        
        # When: Apply generators to edge cases
        for df, y in edge_cases:
            # TreeBasedFeatureGenerator
            tree_gen = TreeBasedFeatureGenerator(n_estimators=2, max_depth=1, random_state=42)
            tree_result = tree_gen.fit_transform(df, y)
            assert isinstance(tree_result, pd.DataFrame)
            assert len(tree_result) == len(df)
            
            # PolynomialFeaturesWrapper
            poly_gen = PolynomialFeaturesWrapper(degree=2, include_bias=False)
            poly_result = poly_gen.fit_transform(df)
            assert isinstance(poly_result, pd.DataFrame)
            assert len(poly_result) == len(df)
    
    def test_feature_generators_memory_efficiency(self):
        """Test feature generators don't create excessive memory overhead"""
        # Given: Moderately sized dataset
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.random.randint(0, 2, 200)
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
        y_series = pd.Series(y)
        
        # When: Apply generators
        tree_gen = TreeBasedFeatureGenerator(n_estimators=5, max_depth=2, random_state=42)
        poly_gen = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        tree_result = tree_gen.fit_transform(df, y_series)
        poly_result = poly_gen.fit_transform(df)
        
        # Then: Should complete without memory issues
        assert isinstance(tree_result, pd.DataFrame)
        assert isinstance(poly_result, pd.DataFrame)
        assert len(tree_result) == len(poly_result) == 200
        
        # Feature counts should be reasonable
        assert tree_result.shape[1] < 100  # Not excessive tree features
        assert poly_result.shape[1] < 20   # Not excessive polynomial features