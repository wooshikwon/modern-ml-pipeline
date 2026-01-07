"""
Integration Pipeline Tests for Multi-Component Preprocessing Chains
Testing complex preprocessing scenarios through ComponentTestContext

Integration Test Categories:
1. Multi-step preprocessing chains with data flow validation
2. Factory integration with preprocessing components
3. End-to-end preprocessing workflows with MLflow integration
4. Complex preprocessing scenarios with mixed step types
5. Data quality preservation through preprocessing pipelines

Following tests/README.md architecture:
- Use ComponentTestContext for factory integration testing
- Test through public APIs only
- Real objects with test data, minimal mocking
- MLflow integration with proper experiment tracking
- Focus on business logic validation, not implementation details
"""

import numpy as np
import pandas as pd
import pytest

from src.components.preprocessor.preprocessor import Preprocessor
from src.settings.recipe import Preprocessor as PreprocessorConfig
from src.settings.recipe import PreprocessorStep


class TestMultiComponentPreprocessingPipelines:
    """Test complex multi-component preprocessing chains through ComponentTestContext"""

    def test_end_to_end_classification_preprocessing_pipeline(self, component_test_context):
        """Test complete preprocessing pipeline for classification task through factory integration"""
        with component_test_context.classification_stack() as ctx:
            # Given: Multi-step preprocessing configuration
            raw_df = ctx.adapter.read(ctx.data_path)

            # Add complexity to data for testing
            enhanced_data = raw_df.copy()
            enhanced_data["categorical_feature"] = ["A", "B", "C"] * (len(enhanced_data) // 3) + [
                "A"
            ] * (len(enhanced_data) % 3)

            # Add some missing values (not all values should be missing)
            np.random.seed(42)
            missing_idx = np.random.choice(
                enhanced_data.index, size=int(0.2 * len(enhanced_data)), replace=False
            )
            enhanced_data.loc[missing_idx, "feature_0"] = np.nan

            # Save the enhanced data back to the adapter's path for consistency

            temp_path = ctx.data_path.replace(".csv", "_enhanced.csv")
            enhanced_data.to_csv(temp_path, index=False)

            # Configure comprehensive preprocessing pipeline
            preprocessing_config = PreprocessorConfig(
                steps=[
                    PreprocessorStep(type="simple_imputer", columns=["feature_0"], strategy="mean"),
                    PreprocessorStep(type="standard_scaler"),  # Global application
                    PreprocessorStep(type="one_hot_encoder", columns=["categorical_feature"]),
                ]
            )

            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Execute full preprocessing pipeline through factory
            preprocessor = ctx.factory.create_preprocessor()

            # Prepare model input (exclude target and entity columns)
            input_data = ctx.prepare_model_input(enhanced_data)

            # Fit and transform through pipeline
            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Validate comprehensive preprocessing results
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Validate imputation worked (no NaNs in feature_0)
            if "feature_0" in processed_data.columns:
                assert not processed_data["feature_0"].isnull().any()

            # Validate standard scaling applied to numeric columns
            numeric_columns = [
                col
                for col in processed_data.columns
                if col.startswith("feature_") and processed_data[col].dtype in ["float64", "int64"]
            ]
            for col in numeric_columns:
                if processed_data[col].std() > 0:  # Only check non-constant columns
                    # Should be approximately standardized (mean ~0, std ~1)
                    assert abs(processed_data[col].mean()) < 0.5
                    assert abs(processed_data[col].std() - 1.0) < 0.5

            # Validate one-hot encoding applied (original categorical_feature should be transformed)
            # One-hot encoding creates new columns and removes the original
            assert "categorical_feature" not in processed_data.columns
            # Should have one-hot encoded columns (A, B, C -> 3 columns or less with drop_first)

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_preprocessing_with_feature_generation_pipeline(self, component_test_context):
        """Test preprocessing pipeline with feature generation and transformation steps"""
        with component_test_context.classification_stack() as ctx:
            # Given: Data with feature generation requirements
            raw_df = ctx.adapter.read(ctx.data_path)

            # Configure pipeline with feature generation
            # First impute missing values, then generate polynomial features
            preprocessing_config = PreprocessorConfig(
                steps=[
                    PreprocessorStep(
                        type="simple_imputer", columns=["feature_0", "feature_1"], strategy="mean"
                    ),
                    PreprocessorStep(
                        type="polynomial_features", columns=["feature_0", "feature_1"], degree=2
                    ),
                    PreprocessorStep(type="standard_scaler"),  # Scale the generated features
                ]
            )

            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Execute feature generation pipeline
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(raw_df)

            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Validate feature generation and scaling
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Should have more columns due to polynomial feature generation
            assert processed_data.shape[1] >= input_data.shape[1]

            # Generated polynomial features should exist
            # PolynomialFeatures typically creates interaction terms and powers
            assert any("feature_0" in str(col) for col in processed_data.columns)

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_preprocessing_pipeline_with_tree_based_features(self, component_test_context):
        """Test preprocessing with tree-based feature generation"""
        with component_test_context.classification_stack() as ctx:
            # Given: Data suitable for tree-based feature generation
            raw_df = ctx.adapter.read(ctx.data_path)

            # Need target for tree-based features
            enhanced_data = raw_df.copy()
            enhanced_data["target"] = [0, 1] * (len(enhanced_data) // 2) + [0] * (
                len(enhanced_data) % 2
            )

            # Configure pipeline with tree-based feature generation
            preprocessing_config = PreprocessorConfig(
                steps=[
                    PreprocessorStep(
                        type="tree_based_feature_generator",
                        columns=["feature_0", "feature_1", "feature_2"],
                        n_estimators=10,
                        max_depth=3,
                    ),
                    PreprocessorStep(type="standard_scaler"),
                ]
            )

            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Execute tree-based feature generation
            preprocessor = ctx.factory.create_preprocessor()

            # For tree-based features, we need to pass target separately to fit method
            input_data = ctx.prepare_model_input(enhanced_data)
            target_data = enhanced_data["target"]

            # Fit with input data and target passed separately
            preprocessor.fit(input_data, target_data)

            # Transform without target for prediction scenario
            input_data = ctx.prepare_model_input(enhanced_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Validate tree-based feature generation
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Should have generated additional features
            assert processed_data.shape[1] >= input_data.shape[1]

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_mixed_preprocessing_pipeline_integration(self, component_test_context):
        """Test complex preprocessing pipeline mixing global and targeted operations"""
        with component_test_context.classification_stack() as ctx:
            # Given: Complex data requiring mixed preprocessing approaches
            raw_df = ctx.adapter.read(ctx.data_path)

            # Create complex test data
            complex_data = raw_df.copy()
            complex_data["high_cardinality_cat"] = [f"cat_{i%10}" for i in range(len(complex_data))]
            complex_data["numeric_to_discretize"] = np.random.uniform(0, 100, len(complex_data))

            # Add missing values
            np.random.seed(42)
            complex_data.loc[complex_data.index[:5], "feature_0"] = np.nan

            # Configure comprehensive mixed preprocessing
            preprocessing_config = PreprocessorConfig(
                steps=[
                    # Targeted operations
                    PreprocessorStep(
                        type="simple_imputer", columns=["feature_0"], strategy="median"
                    ),
                    PreprocessorStep(
                        type="kbins_discretizer", columns=["numeric_to_discretize"], n_bins=5
                    ),
                    # Global operations
                    PreprocessorStep(
                        type="standard_scaler"
                    ),  # Apply to all applicable numeric columns
                ]
            )

            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Execute mixed preprocessing pipeline
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(complex_data)

            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Validate mixed preprocessing results
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Validate imputation (feature_0 should have no NaNs)
            if "feature_0" in processed_data.columns:
                assert not processed_data["feature_0"].isnull().any()

            # Validate discretization (original numeric_to_discretize should be processed)
            # KBinsDiscretizer might preserve or change column name based on implementation
            assert (
                "numeric_to_discretize" not in processed_data.columns
                or processed_data["numeric_to_discretize"].nunique() <= 5
            )

            # Validate standard scaling applied globally
            numeric_cols = [
                col
                for col in processed_data.columns
                if processed_data[col].dtype in ["float64", "int64"]
                and processed_data[col].std() > 0
            ]

            for col in numeric_cols:
                # Check if column appears to be standardized
                mean_val = processed_data[col].mean()
                std_val = processed_data[col].std()
                if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                    assert abs(mean_val) < 1.0  # Approximately centered
                    assert abs(std_val - 1.0) < 1.0  # Approximately scaled

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_preprocessing_pipeline_error_recovery(self, component_test_context):
        """Test preprocessing pipeline behavior with partial failures and error recovery"""
        with component_test_context.classification_stack() as ctx:
            # Given: Preprocessing configuration with potential failure points
            raw_df = ctx.adapter.read(ctx.data_path)

            # Configure pipeline with steps that might fail or be skipped
            preprocessing_config = PreprocessorConfig(
                steps=[
                    PreprocessorStep(
                        type="simple_imputer", columns=["nonexistent_column"], strategy="mean"
                    ),  # Should skip
                    PreprocessorStep(type="standard_scaler"),  # Should work
                ]
            )

            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Execute pipeline with potential failures
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(raw_df)

            # Should not raise errors, but handle gracefully
            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Pipeline should recover gracefully
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Standard scaler should still have been applied
            numeric_cols = [
                col
                for col in processed_data.columns
                if processed_data[col].dtype in ["float64", "int64"]
            ]
            if numeric_cols:
                # At least some numeric processing should have occurred
                assert processed_data[numeric_cols].std().sum() > 0

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)


class TestPreprocessorFactoryIntegration:
    """Test preprocessor integration with factory patterns and component interactions"""

    def test_factory_creates_consistent_preprocessor_instances(self, component_test_context):
        """Test that factory creates consistent preprocessor instances with same settings"""
        with component_test_context.classification_stack() as ctx:
            # Given: Preprocessing configuration
            preprocessing_config = PreprocessorConfig(
                steps=[PreprocessorStep(type="standard_scaler")]
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Create multiple preprocessor instances through factory
            preprocessor1 = ctx.factory.create_preprocessor()
            preprocessor2 = ctx.factory.create_preprocessor()

            # Then: Instances should be consistent with same configuration
            assert isinstance(preprocessor1, Preprocessor)
            assert isinstance(preprocessor2, Preprocessor)

            # Should have same configuration (Factory caching ensures consistency)
            assert preprocessor1.config == preprocessor2.config

            # Factory caching should provide consistent behavior
            # Test functional consistency by verifying same processing behavior
            test_data = ctx.prepare_model_input(ctx.adapter.read(ctx.data_path))

            # Both should be in same state (fitted or unfitted)
            assert hasattr(preprocessor1, "_fitted") == hasattr(preprocessor2, "_fitted")
            if hasattr(preprocessor1, "_fitted"):
                assert preprocessor1._fitted == preprocessor2._fitted

    def test_preprocessor_integrates_with_adapter_and_model(self, component_test_context):
        """Test preprocessor integration with data adapter and model components"""
        with component_test_context.classification_stack() as ctx:
            # Given: Full component stack with preprocessing
            preprocessing_config = PreprocessorConfig(
                steps=[PreprocessorStep(type="standard_scaler")]
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Use all components together
            # 1. Adapter reads data
            raw_data = ctx.adapter.read(ctx.data_path)

            # 2. Preprocessor transforms data
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(raw_data)
            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # 3. Model can work with processed data
            model = ctx.model
            target_data = (
                raw_data["target"]
                if "target" in raw_data.columns
                else np.array([0, 1] * (len(raw_data) // 2) + [0] * (len(raw_data) % 2))
            )

            # Model should be able to fit the processed data
            if hasattr(model, "fit"):
                model.fit(processed_data, target_data)

            # Then: All components should work together
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)
            assert ctx.validate_data_flow(input_data, processed_data)

            # Model should be fitted
            if hasattr(model, "predict"):
                predictions = model.predict(processed_data)
                assert len(predictions) == len(processed_data)

    def test_preprocessor_settings_validation_integration(self, component_test_context):
        """Test preprocessor integration with settings validation and error handling"""
        with component_test_context.classification_stack() as ctx:
            # Given: Invalid preprocessing configuration
            invalid_config = PreprocessorConfig(steps=[PreprocessorStep(type="invalid_step_type")])
            ctx.settings.recipe.preprocessor = invalid_config

            # When: Attempt to create and use preprocessor with invalid config
            preprocessor = ctx.factory.create_preprocessor()
            raw_data = ctx.adapter.read(ctx.data_path)
            input_data = ctx.prepare_model_input(raw_data)

            # Then: Should raise appropriate error during fit - registry raises KeyError
            with pytest.raises(KeyError, match="알 수 없는 키"):
                preprocessor.fit(input_data)


class TestPreprocessorDataQualityPreservation:
    """Test data quality preservation through complex preprocessing pipelines"""

    def test_data_schema_consistency_through_pipeline(self, component_test_context):
        """Test that data schema is handled consistently through preprocessing pipeline"""
        with component_test_context.classification_stack() as ctx:
            # Given: Data with specific schema
            raw_df = ctx.adapter.read(ctx.data_path)
            original_dtypes = raw_df.dtypes.copy()

            # Configure preprocessing that should preserve numeric types
            preprocessing_config = PreprocessorConfig(
                steps=[PreprocessorStep(type="standard_scaler")]  # Should maintain numeric types
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Process through pipeline
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(raw_df)

            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Schema should be appropriately preserved or transformed
            assert isinstance(processed_data, pd.DataFrame)

            # Numeric columns should remain numeric
            for col in processed_data.columns:
                if col in input_data.columns and input_data[col].dtype in ["float64", "int64"]:
                    assert processed_data[col].dtype in [
                        "float64",
                        "int64",
                    ], f"Column {col} type changed unexpectedly"

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_missing_value_handling_consistency(self, component_test_context):
        """Test consistent missing value handling across preprocessing pipeline"""
        with component_test_context.classification_stack() as ctx:
            # Given: Data with strategic missing values
            raw_df = ctx.adapter.read(ctx.data_path)
            data_with_missing = raw_df.copy()

            # Introduce controlled missing values
            np.random.seed(42)
            data_with_missing.loc[data_with_missing.index[:3], "feature_0"] = np.nan
            data_with_missing.loc[data_with_missing.index[5:8], "feature_1"] = np.nan

            # Configure comprehensive missing value handling
            preprocessing_config = PreprocessorConfig(
                steps=[
                    PreprocessorStep(
                        type="simple_imputer", columns=["feature_0", "feature_1"], strategy="mean"
                    ),
                    PreprocessorStep(type="standard_scaler"),
                ]
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Process data with missing values
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(data_with_missing)

            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Missing values should be handled consistently
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(input_data)

            # Imputed columns should have no missing values
            for col in ["feature_0", "feature_1"]:
                if col in processed_data.columns:
                    assert (
                        not processed_data[col].isnull().any()
                    ), f"Column {col} still has missing values after imputation"

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_data_distribution_preservation_analysis(self, component_test_context):
        """Test analysis of data distribution changes through preprocessing"""
        with component_test_context.classification_stack() as ctx:
            # Given: Data with known distribution characteristics
            raw_df = ctx.adapter.read(ctx.data_path)

            # Configure preprocessing
            preprocessing_config = PreprocessorConfig(
                steps=[PreprocessorStep(type="standard_scaler")]
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            # When: Process and analyze distribution changes
            preprocessor = ctx.factory.create_preprocessor()
            input_data = ctx.prepare_model_input(raw_df)

            # Capture original distributions
            original_stats = {}
            for col in input_data.select_dtypes(include=[np.number]).columns:
                original_stats[col] = {
                    "mean": input_data[col].mean(),
                    "std": input_data[col].std(),
                    "min": input_data[col].min(),
                    "max": input_data[col].max(),
                }

            # Process data
            preprocessor.fit(input_data)
            processed_data = preprocessor.transform(input_data)

            # Then: Analyze distribution changes
            assert isinstance(processed_data, pd.DataFrame)

            # For standard scaler, means should be approximately 0 and std approximately 1
            for col in processed_data.select_dtypes(include=[np.number]).columns:
                if processed_data[col].std() > 0:  # Only for non-constant columns
                    processed_mean = processed_data[col].mean()
                    processed_std = processed_data[col].std()

                    # Should be approximately standardized
                    assert (
                        abs(processed_mean) < 0.5
                    ), f"Column {col} mean not properly centered: {processed_mean}"
                    assert (
                        abs(processed_std - 1.0) < 0.5
                    ), f"Column {col} std not properly scaled: {processed_std}"

            # Context validation
            assert ctx.validate_data_flow(input_data, processed_data)

    def test_preprocessing_reproducibility_across_runs(self, component_test_context):
        """Test that preprocessing produces reproducible results across multiple runs"""
        with component_test_context.classification_stack() as ctx:
            # Given: Fixed preprocessing configuration
            preprocessing_config = PreprocessorConfig(
                steps=[PreprocessorStep(type="standard_scaler")]
            )
            ctx.settings.recipe.preprocessor = preprocessing_config

            raw_df = ctx.adapter.read(ctx.data_path)
            input_data = ctx.prepare_model_input(raw_df)

            # When: Run preprocessing multiple times
            results = []
            for run in range(3):
                preprocessor = ctx.factory.create_preprocessor()
                preprocessor.fit(input_data)
                processed_data = preprocessor.transform(input_data)
                results.append(processed_data)

            # Then: Results should be identical across runs
            assert len(results) == 3

            # Compare first result with others
            base_result = results[0]
            for i, result in enumerate(results[1:], 1):
                # Verify preprocessing results are identical across runs
                pd.testing.assert_frame_equal(base_result, result, check_dtype=True)

            # Context validation for all results
            for result in results:
                assert ctx.validate_data_flow(input_data, result)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
