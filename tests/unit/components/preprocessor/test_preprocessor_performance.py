"""
Preprocessor Performance Benchmark Tests
Testing performance characteristics of preprocessing operations with realistic MLOps thresholds

Performance Categories:
1. Core Preprocessing Steps: scaling, encoding, imputation, discretization
2. Integration Pipeline Performance: end-to-end preprocessing workflows
3. Memory Efficiency: resource usage patterns across different data scales
4. Factory Pattern Performance: ComponentTestContext stack creation and processing

Data Scales:
- Small: 1K rows (fast feedback)
- Medium: 10K rows (realistic workload)
- Large: 100K rows (stress testing)

Following tests/README.md architecture:
- Use ComponentTestContext for factory integration
- Real objects with test data, minimal mocking
- Performance thresholds based on production MLOps requirements
- Memory tracking and efficiency validation
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from typing import List, Dict, Any
from contextlib import contextmanager

from src.components.preprocessor.registry import PreprocessorStepRegistry
from src.components.preprocessor.preprocessor import Preprocessor
from src.interface import BasePreprocessor


class PerformanceBenchmark:
    """Enhanced performance benchmarking with memory tracking and throughput calculation"""

    def __init__(self):
        self.measurements = {}
        self.memory_measurements = {}
        self.throughput_measurements = {}

    @contextmanager
    def measure_operation(
        self,
        operation_name: str,
        expected_max_time: float = None,
        memory_limit_mb: int = None,
        data_size: int = None
    ):
        """Context manager for comprehensive performance measurement"""
        start_time = time.perf_counter()

        # Memory measurement
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = start_memory

        try:
            yield self
            # Track peak memory during operation
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = peak_memory - start_memory

            # Store measurements
            self.measurements[operation_name] = execution_time
            self.memory_measurements[operation_name] = {
                'start_mb': start_memory,
                'peak_mb': peak_memory,
                'end_mb': end_memory,
                'usage_mb': memory_usage
            }

            # Calculate throughput if data size provided
            if data_size and execution_time > 0:
                throughput = data_size / execution_time
                self.throughput_measurements[operation_name] = {
                    'rows_per_second': throughput,
                    'data_size': data_size,
                    'execution_time': execution_time
                }

            # Assert thresholds if provided
            if expected_max_time and execution_time > expected_max_time:
                raise AssertionError(
                    f"Performance test failed: {operation_name} took {execution_time:.3f}s "
                    f"but should be under {expected_max_time}s"
                )

            if memory_limit_mb and memory_usage > memory_limit_mb:
                raise AssertionError(
                    f"Memory test failed: {operation_name} used {memory_usage:.1f}MB "
                    f"but should be under {memory_limit_mb}MB"
                )

    def record_throughput(self, data_size: int, operation_name: str = None):
        """Record throughput for the last operation or specified operation"""
        if operation_name and operation_name in self.measurements:
            execution_time = self.measurements[operation_name]
            if execution_time > 0:
                throughput = data_size / execution_time
                self.throughput_measurements[operation_name] = {
                    'rows_per_second': throughput,
                    'data_size': data_size,
                    'execution_time': execution_time
                }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'execution_times': self.measurements,
            'memory_usage': self.memory_measurements,
            'throughput': self.throughput_measurements
        }


@pytest.fixture
def performance_benchmark():
    """Provide enhanced performance benchmark utilities"""
    return PerformanceBenchmark()


class TestCorePreprocessingStepPerformance:
    """Test performance of individual preprocessing steps across different data scales"""

    @pytest.fixture
    def small_dataset(self, test_data_generator):
        """Small dataset for fast performance testing (1K rows)"""
        X, y = test_data_generator.classification_data(n_samples=1000, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df['categorical_col'] = ['A', 'B', 'C'] * (len(df) // 3) + ['A'] * (len(df) % 3)
        return df, y

    @pytest.fixture
    def medium_dataset(self, test_data_generator):
        """Medium dataset for realistic workload testing (10K rows)"""
        X, y = test_data_generator.classification_data(n_samples=10000, n_features=8, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        df['categorical_col'] = ['A', 'B', 'C', 'D'] * (len(df) // 4) + ['A'] * (len(df) % 4)
        # Add some missing values for imputation testing
        np.random.seed(42)
        for col in df.columns[:3]:
            missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
        return df, y

    @pytest.fixture
    def large_dataset(self, test_data_generator):
        """Large dataset for stress testing (100K rows)"""
        X, y = test_data_generator.classification_data(n_samples=100000, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df['categorical_col'] = ['A', 'B', 'C', 'D', 'E'] * (len(df) // 5) + ['A'] * (len(df) % 5)
        return df, y

    @pytest.mark.performance
    def test_standard_scaler_performance_small_data(self, performance_benchmark, small_dataset):
        """Test StandardScaler performance on small dataset"""
        df, _ = small_dataset

        # Force reload to ensure scaler is registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        scaler = PreprocessorStepRegistry.create("standard_scaler")

        with performance_benchmark.measure_operation(
            "standard_scaler_fit_small",
            expected_max_time=0.1,  # 100ms for 1K rows
            memory_limit_mb=50,
            data_size=len(df)
        ):
            scaler.fit(df)

        with performance_benchmark.measure_operation(
            "standard_scaler_transform_small",
            expected_max_time=0.05,  # 50ms for transform
            memory_limit_mb=30,
            data_size=len(df)
        ):
            result = scaler.transform(df)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert len(result.columns) == len(df.columns)

        # Check throughput
        performance_benchmark.record_throughput(len(df), "standard_scaler_fit_small")
        report = performance_benchmark.get_performance_report()
        assert report['throughput']['standard_scaler_fit_small']['rows_per_second'] > 10000  # >10K rows/sec

    @pytest.mark.performance
    def test_standard_scaler_performance_medium_data(self, performance_benchmark, medium_dataset):
        """Test StandardScaler performance on medium dataset"""
        df, _ = medium_dataset

        # Force reload to ensure scaler is registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        scaler = PreprocessorStepRegistry.create("standard_scaler")

        with performance_benchmark.measure_operation(
            "standard_scaler_fit_medium",
            expected_max_time=0.5,  # 500ms for 10K rows
            memory_limit_mb=100,
            data_size=len(df)
        ):
            scaler.fit(df)

        with performance_benchmark.measure_operation(
            "standard_scaler_transform_medium",
            expected_max_time=0.2,  # 200ms for transform
            memory_limit_mb=80,
            data_size=len(df)
        ):
            result = scaler.transform(df)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # Check throughput remains good at medium scale
        performance_benchmark.record_throughput(len(df), "standard_scaler_fit_medium")
        report = performance_benchmark.get_performance_report()
        assert report['throughput']['standard_scaler_fit_medium']['rows_per_second'] > 5000  # >5K rows/sec

    @pytest.mark.performance
    @pytest.mark.slow
    def test_standard_scaler_performance_large_data(self, performance_benchmark, large_dataset):
        """Test StandardScaler performance on large dataset (stress test)"""
        df, _ = large_dataset

        # Force reload to ensure scaler is registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        scaler = PreprocessorStepRegistry.create("standard_scaler")

        with performance_benchmark.measure_operation(
            "standard_scaler_fit_large",
            expected_max_time=2.0,  # 2 seconds for 100K rows
            memory_limit_mb=500,
            data_size=len(df)
        ):
            scaler.fit(df)

        with performance_benchmark.measure_operation(
            "standard_scaler_transform_large",
            expected_max_time=1.0,  # 1 second for transform
            memory_limit_mb=400,
            data_size=len(df)
        ):
            result = scaler.transform(df)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # Check throughput scales reasonably
        performance_benchmark.record_throughput(len(df), "standard_scaler_fit_large")
        report = performance_benchmark.get_performance_report()
        assert report['throughput']['standard_scaler_fit_large']['rows_per_second'] > 10000  # >10K rows/sec

    @pytest.mark.performance
    def test_simple_imputer_performance(self, performance_benchmark, medium_dataset):
        """Test SimpleImputer performance with missing data"""
        df, _ = medium_dataset

        # Force reload to ensure imputer is registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.imputer' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.imputer'])
        else:
            __import__('src.components.preprocessor.modules.imputer')

        imputer = PreprocessorStepRegistry.create("simple_imputer", strategy="mean")

        with performance_benchmark.measure_operation(
            "simple_imputer_fit",
            expected_max_time=0.3,  # 300ms for 10K rows with missing data
            memory_limit_mb=100,
            data_size=len(df)
        ):
            imputer.fit(df)

        with performance_benchmark.measure_operation(
            "simple_imputer_transform",
            expected_max_time=0.2,  # 200ms for transform
            memory_limit_mb=80,
            data_size=len(df)
        ):
            result = imputer.transform(df)

        # Validate result - imputer should handle missing values
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Check that missing values were actually imputed (at least some)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            assert result[numeric_cols].isnull().sum().sum() <= df[numeric_cols].isnull().sum().sum()


class TestPreprocessorPipelinePerformance:
    """Test performance of end-to-end preprocessing pipelines and integration patterns"""

    @pytest.mark.performance
    def test_comprehensive_preprocessing_pipeline_performance(
        self, performance_benchmark, component_test_context
    ):
        """Test performance of complete preprocessing pipeline through ComponentTestContext"""

        with component_test_context.classification_stack() as ctx:
            # Read initial data
            with performance_benchmark.measure_operation(
                "data_loading",
                expected_max_time=0.1,  # 100ms for data loading
                memory_limit_mb=50
            ):
                raw_df = ctx.adapter.read(ctx.data_path)

            # Test preprocessor creation performance
            with performance_benchmark.measure_operation(
                "preprocessor_creation",
                expected_max_time=0.2,  # 200ms for factory creation
                memory_limit_mb=100
            ):
                preprocessor = ctx.preprocessor

            # Test end-to-end preprocessing performance
            input_data = ctx.prepare_model_input(raw_df)

            with performance_benchmark.measure_operation(
                "preprocessing_pipeline_fit",
                expected_max_time=0.5,  # 500ms for pipeline fit
                memory_limit_mb=150,
                data_size=len(input_data)
            ):
                preprocessor.fit(input_data)

            with performance_benchmark.measure_operation(
                "preprocessing_pipeline_transform",
                expected_max_time=0.3,  # 300ms for pipeline transform
                memory_limit_mb=120,
                data_size=len(input_data)
            ):
                processed_df = preprocessor.transform(input_data)

            # Validate data flow
            assert ctx.validate_data_flow(input_data, processed_df)
            assert isinstance(processed_df, pd.DataFrame)
            assert len(processed_df) == len(input_data)

            # Check overall pipeline efficiency
            performance_benchmark.record_throughput(len(input_data), "preprocessing_pipeline_fit")
            report = performance_benchmark.get_performance_report()

            # Pipeline should be efficient end-to-end
            total_time = (report['execution_times']['preprocessing_pipeline_fit'] +
                         report['execution_times']['preprocessing_pipeline_transform'])
            assert total_time < 1.0  # Total preprocessing should be under 1 second

            # Memory usage should be reasonable
            max_memory = max(
                report['memory_usage']['preprocessing_pipeline_fit']['usage_mb'],
                report['memory_usage']['preprocessing_pipeline_transform']['usage_mb']
            )
            assert max_memory < 200  # Should use less than 200MB

    @pytest.mark.performance
    def test_multiple_preprocessing_steps_performance(
        self, performance_benchmark, test_data_generator, settings_builder, isolated_temp_directory
    ):
        """Test performance of multi-step preprocessing chains"""

        # Create larger dataset for multi-step testing
        X, y = test_data_generator.classification_data(n_samples=5000, n_features=6, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
        df['categorical_col'] = ['A', 'B', 'C'] * (len(df) // 3) + ['A'] * (len(df) % 3)

        # Add missing values
        np.random.seed(42)
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, 'feature_0'] = np.nan

        data_path = isolated_temp_directory / "multi_step_data.csv"
        df.to_csv(data_path, index=False)

        # Create settings with multiple preprocessing steps
        from src.settings.recipe import PreprocessorStep, PreprocessorConfig
        preprocessing_config = PreprocessorConfig(steps=[
            PreprocessorStep(type="simple_imputer", columns=["feature_0"], strategy="mean"),
            PreprocessorStep(type="standard_scaler", columns=["feature_0", "feature_1", "feature_2"])
        ])

        settings = settings_builder \
            .with_task("classification") \
            .with_model("sklearn.ensemble.RandomForestClassifier") \
            .with_data_path(str(data_path)) \
            .build()

        # Manually set preprocessor config
        settings.recipe.preprocessor = preprocessing_config

        preprocessor = Preprocessor(settings)

        with performance_benchmark.measure_operation(
            "multi_step_preprocessing_fit",
            expected_max_time=1.0,  # 1 second for multi-step fit
            memory_limit_mb=200,
            data_size=len(df)
        ):
            preprocessor.fit(df)

        with performance_benchmark.measure_operation(
            "multi_step_preprocessing_transform",
            expected_max_time=0.5,  # 500ms for multi-step transform
            memory_limit_mb=150,
            data_size=len(df)
        ):
            result = preprocessor.transform(df)

        # Validate multi-step result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # Check that steps were applied (imputation + scaling)
        assert result['feature_0'].isnull().sum() == 0  # Imputation should remove NaNs

        # Check throughput for multi-step processing
        performance_benchmark.record_throughput(len(df), "multi_step_preprocessing_fit")
        report = performance_benchmark.get_performance_report()
        assert report['throughput']['multi_step_preprocessing_fit']['rows_per_second'] > 2000  # >2K rows/sec


class TestPreprocessorMemoryEfficiency:
    """Test memory efficiency and resource usage patterns"""

    @pytest.mark.performance
    def test_memory_efficiency_scaling(self, performance_benchmark):
        """Test that memory usage scales reasonably with data size"""

        # Force reload to ensure modules are registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        memory_usage_by_size = {}

        for data_size in [1000, 5000, 10000]:
            # Generate data
            np.random.seed(42)
            df = pd.DataFrame(np.random.randn(data_size, 5),
                            columns=[f"feature_{i}" for i in range(5)])

            scaler = PreprocessorStepRegistry.create("standard_scaler")

            with performance_benchmark.measure_operation(
                f"memory_test_{data_size}",
                expected_max_time=2.0,
                data_size=data_size
            ):
                scaler.fit(df)
                result = scaler.transform(df)

            memory_usage_by_size[data_size] = performance_benchmark.memory_measurements[f"memory_test_{data_size}"]["usage_mb"]

        # Memory usage should scale sub-linearly (not worse than linear)
        # Check that 10x data doesn't use >20x memory
        ratio_10x = memory_usage_by_size[10000] / max(memory_usage_by_size[1000], 1.0)  # Avoid division by zero
        assert ratio_10x < 20, f"Memory usage scales poorly: 10x data uses {ratio_10x}x memory"

        # All memory usage should be reasonable
        for size, memory in memory_usage_by_size.items():
            assert memory < 100, f"Memory usage too high for {size} rows: {memory}MB"

    @pytest.mark.performance
    def test_memory_cleanup_after_operations(self, performance_benchmark):
        """Test that preprocessing operations don't leak memory"""

        # Force reload to ensure modules are registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        # Measure baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Perform multiple operations
        for i in range(5):
            np.random.seed(42 + i)
            df = pd.DataFrame(np.random.randn(2000, 4), columns=[f"col_{j}" for j in range(4)])

            scaler = PreprocessorStepRegistry.create("standard_scaler")
            scaler.fit(df)
            result = scaler.transform(df)

            # Explicitly delete references
            del scaler, result, df

        # Force garbage collection
        import gc
        gc.collect()

        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory

        # Memory increase should be minimal (< 50MB for 5 operations)
        assert memory_increase < 50, f"Potential memory leak: {memory_increase}MB increase after operations"


class TestPreprocessorFactoryPerformance:
    """Test performance of preprocessor factory patterns and context creation"""

    @pytest.mark.performance
    def test_component_test_context_creation_performance(
        self, performance_benchmark, isolated_temp_directory, settings_builder, test_data_generator
    ):
        """Test performance of ComponentTestContext factory creation"""

        from tests.fixtures.contexts.component_context import ComponentTestContext

        context = ComponentTestContext(isolated_temp_directory, settings_builder, test_data_generator)

        # Test context stack creation performance
        with performance_benchmark.measure_operation(
            "context_stack_creation",
            expected_max_time=0.5,  # 500ms for complete stack creation
            memory_limit_mb=200
        ):
            with context.classification_stack() as ctx:
                # Access all components to ensure they're fully created
                adapter = ctx.adapter
                model = ctx.model
                evaluator = ctx.evaluator
                preprocessor = ctx.preprocessor

                # Validate all components are created
                assert adapter is not None
                assert model is not None
                assert evaluator is not None
                assert preprocessor is not None

        report = performance_benchmark.get_performance_report()

        # Factory creation should be efficient
        creation_time = report['execution_times']['context_stack_creation']
        assert creation_time < 0.5, f"Context creation too slow: {creation_time}s"

        # Memory usage should be reasonable for factory pattern
        memory_usage = report['memory_usage']['context_stack_creation']['usage_mb']
        assert memory_usage < 200, f"Context creation uses too much memory: {memory_usage}MB"

    @pytest.mark.performance
    def test_repeated_factory_creation_performance(
        self, performance_benchmark, isolated_temp_directory, settings_builder, test_data_generator
    ):
        """Test performance of repeated factory creations (testing for caching issues)"""

        from tests.fixtures.contexts.component_context import ComponentTestContext

        context = ComponentTestContext(isolated_temp_directory, settings_builder, test_data_generator)

        creation_times = []

        # Create multiple factory stacks and measure each
        for i in range(3):
            with performance_benchmark.measure_operation(
                f"repeated_context_creation_{i}",
                expected_max_time=0.8,  # Allow slightly more time for repeated creation
                memory_limit_mb=250
            ):
                with context.classification_stack() as ctx:
                    # Use the components to ensure full initialization
                    raw_df = ctx.adapter.read(ctx.data_path)
                    processed_df = ctx.prepare_model_input(raw_df)
                    assert ctx.validate_data_flow(raw_df, processed_df)

            creation_times.append(performance_benchmark.measurements[f"repeated_context_creation_{i}"])

        # Creation times should be consistent (no significant degradation)
        avg_time = sum(creation_times) / len(creation_times)
        assert avg_time < 1.0, f"Average factory creation too slow: {avg_time}s"

        # Variance should be low (consistent performance)
        max_time = max(creation_times)
        min_time = min(creation_times)
        time_variance = max_time - min_time
        assert time_variance < 0.5, f"Factory creation time too variable: {time_variance}s variance"


class TestPreprocessorStepRegistryPerformance:
    """Test performance characteristics of the preprocessor step registry"""

    @pytest.mark.performance
    def test_step_registration_and_lookup_performance(self, performance_benchmark):
        """Test performance of step registration and lookup operations"""

        # Clear registry for clean test
        PreprocessorStepRegistry.preprocessor_steps.clear()

        # Mock preprocessor step for testing
        class MockStep(BasePreprocessor):
            def fit(self, X, y=None): return self
            def transform(self, X): return X
            def get_output_column_names(self, input_columns): return input_columns
            def preserves_column_names(self): return True
            def get_application_type(self): return 'targeted'

        # Test registration performance
        with performance_benchmark.measure_operation(
            "step_registration_batch",
            expected_max_time=0.01,  # 10ms for batch registration
            memory_limit_mb=10
        ):
            for i in range(100):
                PreprocessorStepRegistry.register(f"mock_step_{i}", MockStep)

        # Test lookup performance
        with performance_benchmark.measure_operation(
            "step_lookup_batch",
            expected_max_time=0.01,  # 10ms for batch lookup
            memory_limit_mb=5
        ):
            for i in range(100):
                step_class = PreprocessorStepRegistry.preprocessor_steps.get(f"mock_step_{i}")
                assert step_class is not None

        # Test creation performance
        with performance_benchmark.measure_operation(
            "step_creation_batch",
            expected_max_time=0.1,  # 100ms for batch creation
            memory_limit_mb=50
        ):
            steps = []
            for i in range(20):  # Fewer creations since they're more expensive
                step = PreprocessorStepRegistry.create(f"mock_step_{i}")
                steps.append(step)

        report = performance_benchmark.get_performance_report()

        # Registry operations should be very fast
        assert report['execution_times']['step_registration_batch'] < 0.01
        assert report['execution_times']['step_lookup_batch'] < 0.01
        assert report['execution_times']['step_creation_batch'] < 0.1

        # Memory usage should be minimal for registry operations
        assert report['memory_usage']['step_registration_batch']['usage_mb'] < 10
        assert report['memory_usage']['step_lookup_batch']['usage_mb'] < 5
        assert report['memory_usage']['step_creation_batch']['usage_mb'] < 50


@pytest.mark.performance
class TestPreprocessorStressScenarios:
    """Stress tests for preprocessor under challenging conditions"""

    def test_high_dimensionality_performance(self, performance_benchmark, test_data_generator):
        """Test preprocessor performance with high-dimensional data"""

        # Force reload to ensure modules are registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        # Create high-dimensional dataset (many features, moderate rows)
        X, y = test_data_generator.classification_data(n_samples=1000, n_features=100, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(100)])

        scaler = PreprocessorStepRegistry.create("standard_scaler")

        with performance_benchmark.measure_operation(
            "high_dimensionality_fit",
            expected_max_time=1.0,  # 1 second for 100 features
            memory_limit_mb=200,
            data_size=len(df)
        ):
            scaler.fit(df)

        with performance_benchmark.measure_operation(
            "high_dimensionality_transform",
            expected_max_time=0.5,  # 500ms for transform
            memory_limit_mb=150,
            data_size=len(df)
        ):
            result = scaler.transform(df)

        # Validate high-dimensional result
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert len(result.columns) == 100

        # Performance should still be reasonable with many features
        performance_benchmark.record_throughput(len(df), "high_dimensionality_fit")
        report = performance_benchmark.get_performance_report()
        assert report['throughput']['high_dimensionality_fit']['rows_per_second'] > 500  # >500 rows/sec

    @pytest.mark.slow
    def test_edge_case_data_performance(self, performance_benchmark):
        """Test preprocessor performance with edge case data scenarios"""

        # Force reload to ensure modules are registered
        import sys
        import importlib
        if 'src.components.preprocessor.modules.scaler' in sys.modules:
            importlib.reload(sys.modules['src.components.preprocessor.modules.scaler'])
        else:
            __import__('src.components.preprocessor.modules.scaler')

        # Create challenging dataset with edge cases
        np.random.seed(42)
        data = {
            'constant_col': [1.0] * 1000,  # Constant column
            'high_variance': np.random.randn(1000) * 1000,  # High variance
            'low_variance': np.random.randn(1000) * 0.001,  # Low variance
            'outliers': np.random.randn(1000),  # Will add outliers
            'normal_col': np.random.randn(1000)  # Normal column
        }

        # Add extreme outliers
        data['outliers'][0] = 10000
        data['outliers'][1] = -10000

        df = pd.DataFrame(data)

        scaler = PreprocessorStepRegistry.create("standard_scaler")

        with performance_benchmark.measure_operation(
            "edge_case_data_processing",
            expected_max_time=0.5,  # 500ms even for challenging data
            memory_limit_mb=100,
            data_size=len(df)
        ):
            scaler.fit(df)
            result = scaler.transform(df)

        # Should handle edge cases gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert not result.isnull().any().any(), "Edge case processing should not introduce NaNs"


if __name__ == "__main__":
    # Run performance tests specifically
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])