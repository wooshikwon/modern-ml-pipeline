import pytest
from fastapi.testclient import TestClient
import mlflow
import shutil

from src.serving.router import app, setup_api_context
from src.settings import Settings
from src.pipelines.train_pipeline import run_training


@pytest.mark.requires_dev_stack
class TestRouterAPIFeatureStore:
    @pytest.fixture(scope="class")
    def trained_model_run_id(self, ensure_dev_stack_or_skip, dev_test_settings: Settings):
        test_tracking_uri = "./test_mlruns_router_api_fs"
        mlflow.set_tracking_uri(test_tracking_uri)
        result_artifact = run_training(settings=dev_test_settings)
        yield result_artifact.run_id
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        mlflow.set_tracking_uri("mlruns")

    @pytest.fixture(scope="class")
    def client(self, ensure_dev_stack_or_skip, dev_test_settings: Settings, trained_model_run_id: str):
        setup_api_context(run_id=trained_model_run_id, settings=dev_test_settings)
        return TestClient(app)

    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_uri" in data

    def test_predict_minimal_response(self, client: TestClient):
        # 엔티티 키만으로도 서빙이 가능해야 함(증강이 선행)
        payload = {"user_id": "u1001", "product_id": "p2001"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"prediction", "model_uri"} 