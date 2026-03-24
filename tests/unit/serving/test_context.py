"""
Serving context management tests for mmp/serving/_context.py.

Tests the public API: initialize(), is_ready, singleton behavior, and schema management.
"""

from unittest.mock import Mock

import mlflow
from pydantic import BaseModel, create_model

from mmp.serving._context import AppContext, app_context


class TestAppContextInitializeAndIsReady:
    """Test initialize() and is_ready property."""

    def test_default_state_not_ready(self):
        """New AppContext is not ready."""
        ctx = AppContext()
        assert not ctx.is_ready
        assert ctx.model is None
        assert ctx.settings is None

    def test_initialize_makes_ready(self, component_test_context):
        """initialize() sets all fields and marks as ready."""
        with component_test_context.classification_stack() as comp_ctx:
            ctx = AppContext()
            mock_model = Mock(spec=mlflow.pyfunc.PyFuncModel)
            pred_schema = create_model("Pred", feature1=(float, ...))
            batch_schema = create_model("BatchPred", items=(list, ...))

            ctx.initialize(
                model=mock_model,
                model_uri="runs:/abc/model",
                settings=comp_ctx.settings,
                prediction_request=pred_schema,
                batch_prediction_request=batch_schema,
            )

            assert ctx.is_ready
            assert ctx.model is mock_model
            assert ctx.model_uri == "runs:/abc/model"
            assert ctx.settings is comp_ctx.settings
            assert ctx.PredictionRequest.__name__ == "Pred"
            assert ctx.BatchPredictionRequest.__name__ == "BatchPred"

    def test_not_ready_without_model(self, component_test_context):
        """is_ready is False when model is None even after partial setup."""
        with component_test_context.classification_stack() as comp_ctx:
            ctx = AppContext()
            ctx.settings = comp_ctx.settings
            ctx._initialized = True
            ctx.model = None
            assert not ctx.is_ready


class TestGlobalAppContextSingleton:
    """Test that app_context is a module-level singleton."""

    def test_singleton_identity(self):
        """Multiple imports return same instance."""
        from mmp.serving._context import app_context as ctx1
        from mmp.serving._context import app_context as ctx2

        assert ctx1 is ctx2
        assert ctx1 is app_context

    def test_state_persistence(self):
        """State changes persist across imports."""
        original_uri = app_context.model_uri
        app_context.model_uri = "test://singleton"

        from mmp.serving._context import app_context as imported

        assert imported.model_uri == "test://singleton"
        app_context.model_uri = original_uri  # cleanup


class TestAppContextSchemaUpdate:
    """Test PredictionRequest schema replacement."""

    def test_schema_update(self, component_test_context):
        """PredictionRequest can be replaced with custom schema."""
        with component_test_context.classification_stack():
            ctx = AppContext()
            original = ctx.PredictionRequest
            assert original.__name__ == "DefaultPredictionRequest"

            new_schema = create_model("Custom", user_id=(int, ...), score=(float, ...))
            ctx.PredictionRequest = new_schema

            assert ctx.PredictionRequest.__name__ == "Custom"
            instance = ctx.PredictionRequest(user_id=1, score=0.95)
            assert instance.user_id == 1

    def test_instances_are_independent(self, component_test_context):
        """Different AppContext instances don't share state."""
        with component_test_context.classification_stack() as comp_ctx:
            ctx1 = AppContext()
            ctx2 = AppContext()

            ctx1.model_uri = "uri1"
            ctx2.model_uri = "uri2"
            ctx1.settings = comp_ctx.settings
            ctx2.settings = None

            assert ctx1.model_uri != ctx2.model_uri
            assert ctx1.settings is not ctx2.settings
