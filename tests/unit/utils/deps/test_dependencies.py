"""
Dependencies validation comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/deps/dependencies.py

Author: Phase 2A Development
Date: 2025-09-13
"""

from unittest.mock import Mock, patch

import pytest

from src.utils.deps.dependencies import _requires_pyarrow, validate_dependencies


class TestDependencyValidation:
    """의존성 검증 핵심 기능 테스트 - Context 클래스 기반"""

    def test_requires_pyarrow_with_parquet_file(self, component_test_context):
        """Parquet 파일 사용 시 pyarrow 요구 사항 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Modify settings to use parquet file (schema: recipe.data.loader)
            ctx.settings.recipe.data.loader.source_uri = "/data/test.parquet"

            # Test pyarrow requirement detection
            requires_pyarrow = _requires_pyarrow(ctx.settings)
            assert requires_pyarrow is True

    def test_requires_pyarrow_with_csv_file(self, component_test_context):
        """CSV 파일 사용 시 pyarrow 불필요 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Settings already use CSV by default in context
            requires_pyarrow = _requires_pyarrow(ctx.settings)
            assert requires_pyarrow is False

    def test_requires_pyarrow_with_exception_handling(self, component_test_context):
        """설정 접근 오류 시 안전한 fallback 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create broken settings without loader configuration
            broken_settings = Mock()
            broken_settings.recipe = Mock()
            broken_settings.recipe.data = Mock()
            broken_settings.recipe.data.loader = Mock()
            broken_settings.recipe.data.loader.source_uri = Mock(side_effect=AttributeError)

            # Should return False on exception
            requires_pyarrow = _requires_pyarrow(broken_settings)
            assert requires_pyarrow is False

    def test_validate_dependencies_storage_adapter_csv(self, component_test_context):
        """Storage adapter + CSV 파일 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Context uses storage adapter with CSV by default
            # Should not require additional dependencies

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                # Test validation - should not fail
                validate_dependencies(ctx.settings)
                # __import__ should not be called for CSV files
                mock_import.assert_not_called()

    def test_validate_dependencies_storage_adapter_parquet(self, component_test_context):
        """Storage adapter + Parquet 파일 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Modify settings to use parquet
            ctx.settings.recipe.data.loader.source_uri = (
                "/data/test.PARQUET"  # Test case insensitive
            )

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()  # Simulate successful import

                validate_dependencies(ctx.settings)
                mock_import.assert_called_with("pyarrow")

    def test_validate_dependencies_sql_adapter(self, component_test_context):
        """SQL adapter 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Modify settings to use SQL adapter via config
            ctx.settings.config.data_source.adapter_type = "sql"

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()

                validate_dependencies(ctx.settings)
                mock_import.assert_called_with("sqlalchemy")

    def test_validate_dependencies_feast_feature_store(self, component_test_context):
        """Feast feature store 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Add feast feature store configuration
            mock_feature_store = Mock()
            mock_feature_store.provider = "feast"
            ctx.settings.feature_store = mock_feature_store

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()

                validate_dependencies(ctx.settings)
                mock_import.assert_called_with("feast")

    def test_validate_dependencies_hyperparameter_tuning_enabled(self, component_test_context):
        """하이퍼파라미터 튜닝 활성화 시 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Enable both global and recipe-level tuning
            mock_global_tuning = Mock()
            mock_global_tuning.enabled = True
            ctx.settings.hyperparameter_tuning = mock_global_tuning

            # New schema toggles live under model.hyperparameters.tuning_enabled
            ctx.settings.recipe.model.hyperparameters.tuning_enabled = True

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()

                validate_dependencies(ctx.settings)
                mock_import.assert_called_with("optuna")

    def test_validate_dependencies_hyperparameter_tuning_partial(self, component_test_context):
        """하이퍼파라미터 튜닝 부분 활성화 시 optuna 불필요 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Enable only global tuning (recipe tuning disabled)
            mock_global_tuning = Mock()
            mock_global_tuning.enabled = True
            ctx.settings.hyperparameter_tuning = mock_global_tuning

            # Recipe-level toggle disabled in new schema
            ctx.settings.recipe.model.hyperparameters.tuning_enabled = False

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                validate_dependencies(ctx.settings)

                # optuna should not be required
                calls = [str(call) for call in mock_import.call_args_list]
                assert not any("optuna" in call for call in calls)

    def test_validate_dependencies_serving_enabled(self, component_test_context):
        """서빙 기능 활성화 시 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Enable serving
            mock_serving = Mock()
            mock_serving.enabled = True
            ctx.settings.serving = mock_serving

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()

                validate_dependencies(ctx.settings)

                # Check both fastapi and uvicorn are required
                call_args = [call[0][0] for call in mock_import.call_args_list]
                assert "fastapi" in call_args
                assert "uvicorn" in call_args


class TestDependencyValidationErrorHandling:
    """의존성 검증 에러 처리 테스트"""

    def test_validate_dependencies_missing_package_raises_error(self, component_test_context):
        """필수 패키지 누락 시 ImportError 발생 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Configure to require pyarrow
            ctx.settings.recipe.data.loader.source_uri = "/data/test.parquet"

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.side_effect = ImportError("No module named 'pyarrow'")

                # Should raise ImportError with Korean message
                with pytest.raises(ImportError) as exc_info:
                    validate_dependencies(ctx.settings)

                assert "필수 패키지가 설치되지 않았습니다" in str(exc_info.value)
                assert "pyarrow" in str(exc_info.value)

    def test_validate_dependencies_multiple_missing_packages(self, component_test_context):
        """여러 패키지 누락 시 정렬된 리스트 반환 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Configure to require multiple packages
            ctx.settings.config.data_source.adapter_type = "sql"
            ctx.settings.recipe.data.loader.source_uri = "/data/test.parquet"

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.side_effect = ImportError("Missing package")

                with pytest.raises(ImportError) as exc_info:
                    validate_dependencies(ctx.settings)

                error_msg = str(exc_info.value)
                # Packages should be sorted alphabetically
                assert "pyarrow" in error_msg
                assert "sqlalchemy" in error_msg

    def test_validate_dependencies_with_broken_settings_attributes(self, component_test_context):
        """설정 속성 접근 오류 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Break various settings attributes (feature_store may not exist on Settings)
            if hasattr(ctx.settings, "feature_store"):
                delattr(ctx.settings, "feature_store")

            # Should not raise exception
            with patch("src.utils.deps.dependencies.__import__"):
                validate_dependencies(ctx.settings)


class TestDependencyValidationIntegration:
    """통합 시나리오 의존성 검증 테스트"""

    def test_validate_dependencies_complex_configuration(self, component_test_context):
        """복합 설정에서의 종합 의존성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Configure multiple features requiring different dependencies
            ctx.settings.config.data_source.adapter_type = "sql"
            ctx.settings.recipe.data.loader.source_uri = "/data/test.parquet"

            # Feature store
            mock_feature_store = Mock()
            mock_feature_store.provider = "feast"
            ctx.settings.feature_store = mock_feature_store

            # Hyperparameter tuning
            mock_global_tuning = Mock()
            mock_global_tuning.enabled = True
            ctx.settings.hyperparameter_tuning = mock_global_tuning

            # New schema toggles live under model.hyperparameters.tuning_enabled
            ctx.settings.recipe.model.hyperparameters.tuning_enabled = True

            # Serving
            mock_serving = Mock()
            mock_serving.enabled = True
            ctx.settings.serving = mock_serving

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                mock_import.return_value = Mock()

                validate_dependencies(ctx.settings)

                # Verify all expected packages are checked
                call_args = [call[0][0] for call in mock_import.call_args_list]
                expected_packages = {
                    "pyarrow",
                    "sqlalchemy",
                    "feast",
                    "optuna",
                    "fastapi",
                    "uvicorn",
                }

                for pkg in expected_packages:
                    assert pkg in call_args

    def test_validate_dependencies_no_requirements(self, component_test_context):
        """의존성 요구사항이 없는 기본 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Use default CSV-based configuration
            # Remove optional features
            if hasattr(ctx.settings, "feature_store"):
                delattr(ctx.settings, "feature_store")
            if hasattr(ctx.settings, "serving"):
                delattr(ctx.settings, "serving")

            with patch("src.utils.deps.dependencies.__import__") as mock_import:
                validate_dependencies(ctx.settings)

                # No packages should be required
                mock_import.assert_not_called()

    def test_validate_dependencies_real_import_success(self, component_test_context):
        """실제 import 성공 시나리오 테스트 (mock 없음)"""
        with component_test_context.classification_stack() as ctx:
            # Use basic configuration that shouldn't require special packages
            # Should complete without errors
            validate_dependencies(ctx.settings)
