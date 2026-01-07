from __future__ import annotations

from typing import Optional

from fastapi.testclient import TestClient

from src.pipelines.train_pipeline import run_train_pipeline
from src.serving._context import app_context
from src.serving._lifespan import setup_api_context
from src.serving.router import app


class ServingTestContext:
    """
    API Serving 테스트를 위한 컨텍스트 클래스
    tests/README.md 원칙 준수:
    - 퍼블릭 API만 호출 (setup_api_context)
    - 불필요한 모킹 금지 (실제 모델 훈련/로딩)
    - 컨텍스트 중심 설계 (설정/자원 준비 캡슐화)
    """

    def __init__(self, mlflow_context, isolated_temp_directory):
        self.mlflow_context = mlflow_context
        self.temp_dir = isolated_temp_directory

    def with_trained_model(
        self, task: str = "classification", model: str = "RandomForestClassifier"
    ):
        """실제 모델을 훈련하고 serving API를 준비한 컨텍스트 반환"""
        return _ServingContextManager(task, model, self)


class _ServingContextManager:
    def __init__(self, task: str, model: str, context: ServingTestContext):
        self.task = task
        self.model = model
        self.context = context
        self.client: Optional[TestClient] = None
        self.run_id: Optional[str] = None
        self.settings = None

    def __enter__(self) -> "_ServingContextManager":
        # 1. MLflow context로 모델 훈련 (tests/README.md 원칙 준수)
        with self.context.mlflow_context.for_classification(
            f"serving_test_{self.task}", self.model
        ) as mlflow_ctx:
            # 2. 실제 모델 훈련 (퍼블릭 API 호출)
            train_result = run_train_pipeline(mlflow_ctx.settings)
            assert train_result is not None and hasattr(train_result, "run_id")

            self.run_id = train_result.run_id
            self.settings = mlflow_ctx.settings

            # 3. setup_api_context로 실제 모델 로딩 (퍼블릭 API 사용)
            setup_api_context(self.run_id, self.settings)

            # 4. 준비된 TestClient 반환
            self.client = TestClient(app)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 정리 (상태 격리 원칙 준수)"""
        # app_context 정리하여 다른 테스트에 영향 방지
        app_context.model = None
        app_context.settings = None
        app_context.model_uri = ""
        return None

    # ===== 헬퍼 메서드 (관찰 중심) =====
    def is_model_loaded(self) -> bool:
        """모델 로딩 상태 확인"""
        return app_context.model is not None and app_context.settings is not None

    def get_model_info(self) -> dict:
        """모델 정보 반환 (관찰용)"""
        if not self.is_model_loaded():
            return {"loaded": False}

        try:
            wrapped_model = app_context.model.unwrap_python_model()
            return {
                "loaded": True,
                "run_id": self.run_id,
                "model_class": getattr(wrapped_model, "model_class_path", "unknown"),
                "model_uri": app_context.model_uri,
            }
        except Exception:
            return {"loaded": True, "info_error": True}
