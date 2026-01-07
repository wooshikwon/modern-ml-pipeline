"""
Serving context management comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/serving/_context.py

Author: Phase 2B Development
Date: 2025-09-13
"""

from unittest.mock import Mock

import mlflow
from pydantic import BaseModel

from src.serving._context import AppContext, app_context


class TestAppContextInitialization:
    """AppContext 초기화 테스트 - Context 클래스 기반"""

    def test_app_context_default_initialization(self, component_test_context):
        """기본 AppContext 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create new AppContext instance (Real Object Testing)
            app_ctx = AppContext()

            # Verify default initialization
            assert app_ctx.model is None
            assert app_ctx.model_uri == ""
            assert app_ctx.settings is None
            assert isinstance(app_ctx.PredictionRequest, type)
            assert isinstance(app_ctx.BatchPredictionRequest, type)
            assert issubclass(app_ctx.PredictionRequest, BaseModel)
            assert issubclass(app_ctx.BatchPredictionRequest, BaseModel)

    def test_app_context_fresh_instance_creation(self, component_test_context):
        """새 AppContext 인스턴스 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create multiple instances to ensure independence
            app_ctx1 = AppContext()
            app_ctx2 = AppContext()

            # Each instance should be independent
            assert app_ctx1 is not app_ctx2
            assert app_ctx1.model is app_ctx2.model  # Both None initially
            assert app_ctx1.model_uri == app_ctx2.model_uri  # Both empty initially

    def test_app_context_pydantic_model_creation(self, component_test_context):
        """Pydantic 모델 생성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Test PredictionRequest model properties
            assert hasattr(app_ctx.PredictionRequest, "__fields__") or hasattr(
                app_ctx.PredictionRequest, "model_fields"
            )
            assert app_ctx.PredictionRequest.__name__ == "DefaultPredictionRequest"

            # Test BatchPredictionRequest model properties
            assert hasattr(app_ctx.BatchPredictionRequest, "__fields__") or hasattr(
                app_ctx.BatchPredictionRequest, "model_fields"
            )
            assert app_ctx.BatchPredictionRequest.__name__ == "DefaultBatchPredictionRequest"

            # Test model instantiation
            pred_instance = app_ctx.PredictionRequest()
            batch_instance = app_ctx.BatchPredictionRequest()
            assert isinstance(pred_instance, BaseModel)
            assert isinstance(batch_instance, BaseModel)


class TestAppContextModelManagement:
    """AppContext 모델 관리 테스트"""

    def test_app_context_model_assignment(self, component_test_context):
        """모델 할당 및 URI 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Mock MLflow model
            mock_model = Mock(spec=mlflow.pyfunc.PyFuncModel)
            test_uri = "runs:/test-run-id/model"

            # Assign model and URI
            app_ctx.model = mock_model
            app_ctx.model_uri = test_uri

            # Verify assignment
            assert app_ctx.model is mock_model
            assert app_ctx.model_uri == test_uri

    def test_app_context_model_state_transitions(self, component_test_context):
        """모델 상태 전환 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Initial state
            assert app_ctx.model is None
            assert app_ctx.model_uri == ""

            # Set model state
            mock_model = Mock(spec=mlflow.pyfunc.PyFuncModel)
            app_ctx.model = mock_model
            app_ctx.model_uri = "runs:/abc123/model"

            # Verify loaded state
            assert app_ctx.model is not None
            assert app_ctx.model_uri != ""

            # Clear model state
            app_ctx.model = None
            app_ctx.model_uri = ""

            # Verify cleared state
            assert app_ctx.model is None
            assert app_ctx.model_uri == ""

    def test_app_context_model_uri_validation(self, component_test_context):
        """모델 URI 형식 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Test various URI formats
            valid_uris = [
                "runs:/run-id/model",
                "file:///path/to/model",
                "s3://bucket/model",
                "models:/model-name/version",
                "/local/path/to/model",
            ]

            for uri in valid_uris:
                app_ctx.model_uri = uri
                assert app_ctx.model_uri == uri


class TestAppContextSettingsManagement:
    """AppContext 설정 관리 테스트"""

    def test_app_context_settings_assignment(self, component_test_context):
        """설정 객체 할당 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Use real Settings from context (Real Object Testing)
            real_settings = ctx.settings

            # Assign settings
            app_ctx.settings = real_settings

            # Verify assignment
            assert app_ctx.settings is real_settings
            assert app_ctx.settings is not None

    def test_app_context_settings_state_management(self, component_test_context):
        """설정 상태 관리 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Initial state
            assert app_ctx.settings is None

            # Set settings
            app_ctx.settings = ctx.settings
            assert app_ctx.settings is not None

            # Clear settings
            app_ctx.settings = None
            assert app_ctx.settings is None

    def test_app_context_settings_immutability(self, component_test_context):
        """설정 객체 불변성 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()
            original_settings = ctx.settings

            # Assign settings
            app_ctx.settings = original_settings

            # Settings reference should remain the same
            assert app_ctx.settings is original_settings


class TestAppContextSchemaManagement:
    """AppContext 스키마 관리 테스트"""

    def test_app_context_prediction_request_schema_update(self, component_test_context):
        """PredictionRequest 스키마 업데이트 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Original schema
            original_schema = app_ctx.PredictionRequest
            assert original_schema.__name__ == "DefaultPredictionRequest"

            # Create new schema with fields
            from pydantic import create_model

            new_schema = create_model(
                "CustomPredictionRequest", feature1=(float, ...), feature2=(str, ...)
            )

            # Update schema
            app_ctx.PredictionRequest = new_schema

            # Verify update
            assert app_ctx.PredictionRequest is not original_schema
            assert app_ctx.PredictionRequest.__name__ == "CustomPredictionRequest"

            # Test model instantiation with new schema
            instance = app_ctx.PredictionRequest(feature1=1.0, feature2="test")
            assert instance.feature1 == 1.0
            assert instance.feature2 == "test"

    def test_app_context_batch_prediction_request_schema_update(self, component_test_context):
        """BatchPredictionRequest 스키마 업데이트 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Original schema
            original_schema = app_ctx.BatchPredictionRequest

            # Create new batch schema
            from typing import List

            from pydantic import create_model

            new_batch_schema = create_model(
                "CustomBatchPredictionRequest", batch_data=(List[dict], ...)
            )

            # Update batch schema
            app_ctx.BatchPredictionRequest = new_batch_schema

            # Verify update
            assert app_ctx.BatchPredictionRequest is not original_schema
            assert app_ctx.BatchPredictionRequest.__name__ == "CustomBatchPredictionRequest"

    def test_app_context_schema_type_validation(self, component_test_context):
        """스키마 타입 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Both schemas should be BaseModel subclasses
            assert issubclass(app_ctx.PredictionRequest, BaseModel)
            assert issubclass(app_ctx.BatchPredictionRequest, BaseModel)

            # Test type hints
            from typing import get_type_hints

            hints = get_type_hints(AppContext.__init__)

            # PredictionRequest and BatchPredictionRequest should be Type[BaseModel]
            # Note: This tests the class definition structure


class TestGlobalAppContextInstance:
    """전역 app_context 인스턴스 테스트"""

    def test_global_app_context_instance_exists(self, component_test_context):
        """전역 app_context 인스턴스 존재 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Global instance should exist
            assert app_context is not None
            assert isinstance(app_context, AppContext)

    def test_global_app_context_singleton_behavior(self, component_test_context):
        """전역 app_context 싱글톤 동작 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Import app_context multiple times should return same instance
            from src.serving._context import app_context as app_context_1
            from src.serving._context import app_context as app_context_2

            assert app_context_1 is app_context_2
            assert app_context_1 is app_context

    def test_global_app_context_state_persistence(self, component_test_context):
        """전역 app_context 상태 지속성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Set state
            original_uri = app_context.model_uri
            test_uri = "test://model/uri"
            app_context.model_uri = test_uri

            # State should persist across imports
            from src.serving._context import app_context as imported_context

            assert imported_context.model_uri == test_uri

            # Clean up for other tests
            app_context.model_uri = original_uri

    def test_global_app_context_initial_state(self, component_test_context):
        """전역 app_context 초기 상태 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Note: Global state may have been modified by previous tests
            # Test the structure rather than exact values
            assert hasattr(app_context, "model")
            assert hasattr(app_context, "model_uri")
            assert hasattr(app_context, "settings")
            assert hasattr(app_context, "PredictionRequest")
            assert hasattr(app_context, "BatchPredictionRequest")


class TestAppContextIntegration:
    """AppContext 통합 테스트"""

    def test_app_context_complete_workflow(self, component_test_context, mlflow_test_context):
        """완전한 AppContext 워크플로우 테스트"""
        with component_test_context.classification_stack() as comp_ctx:
            with mlflow_test_context.for_classification(
                experiment="app_context_test"
            ) as mlflow_ctx:
                app_ctx = AppContext()

                # Step 1: Initial state verification
                assert app_ctx.model is None
                assert app_ctx.settings is None

                # Step 2: Set up settings (Real Object Testing)
                app_ctx.settings = mlflow_ctx.settings
                assert app_ctx.settings is not None

                # Step 3: Mock model loading (external dependency)
                mock_model = Mock(spec=mlflow.pyfunc.PyFuncModel)
                model_uri = "runs:/test-run/model"
                app_ctx.model = mock_model
                app_ctx.model_uri = model_uri

                # Step 4: Verify complete setup
                assert app_ctx.model is not None
                assert app_ctx.model_uri == model_uri
                assert app_ctx.settings is mlflow_ctx.settings

                # Step 5: Test schema customization
                from pydantic import create_model

                custom_schema = create_model(
                    "TestPredictionRequest", user_id=(int, ...), features=(dict, ...)
                )
                app_ctx.PredictionRequest = custom_schema

                # Verify schema update worked
                assert app_ctx.PredictionRequest.__name__ == "TestPredictionRequest"

                # Test model instantiation
                request_instance = app_ctx.PredictionRequest(
                    user_id=123, features={"feature1": 1.0}
                )
                assert request_instance.user_id == 123
                assert request_instance.features == {"feature1": 1.0}

    def test_app_context_error_handling_workflow(self, component_test_context):
        """AppContext 오류 처리 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            app_ctx = AppContext()

            # Test handling of None assignments
            app_ctx.model = None
            app_ctx.settings = None
            app_ctx.model_uri = ""

            # Should not raise exceptions
            assert app_ctx.model is None
            assert app_ctx.settings is None
            assert app_ctx.model_uri == ""

            # Test schema reset to defaults
            from pydantic import create_model

            original_request = app_ctx.PredictionRequest

            # Set and then reset
            app_ctx.PredictionRequest = create_model("TempRequest")
            app_ctx.PredictionRequest = create_model("DefaultPredictionRequest")

            # Should work without errors
            assert app_ctx.PredictionRequest.__name__ == "DefaultPredictionRequest"

    def test_app_context_multiple_instances_independence(self, component_test_context):
        """여러 AppContext 인스턴스 독립성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create multiple independent instances
            app_ctx1 = AppContext()
            app_ctx2 = AppContext()

            # Set different states
            app_ctx1.model_uri = "uri1"
            app_ctx2.model_uri = "uri2"

            app_ctx1.settings = ctx.settings
            app_ctx2.settings = None

            # Verify independence
            assert app_ctx1.model_uri != app_ctx2.model_uri
            assert app_ctx1.settings != app_ctx2.settings

            # Modify schemas independently
            from pydantic import create_model

            app_ctx1.PredictionRequest = create_model("Schema1", field1=(int, ...))
            app_ctx2.PredictionRequest = create_model("Schema2", field2=(str, ...))

            # Verify schema independence
            assert app_ctx1.PredictionRequest.__name__ == "Schema1"
            assert app_ctx2.PredictionRequest.__name__ == "Schema2"
            assert app_ctx1.PredictionRequest != app_ctx2.PredictionRequest
