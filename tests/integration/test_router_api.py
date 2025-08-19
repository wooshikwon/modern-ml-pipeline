import pytest
from fastapi.testclient import TestClient
import mlflow
import shutil

from src.serving.router import app, setup_api_context
from src.settings import Settings
from src.pipelines.train_pipeline import run_training


@pytest.mark.requires_dev_stack
class TestRouterAPI:
    @pytest.fixture(scope="class")
    def trained_model_run_id(self, dev_test_settings: Settings):
        test_tracking_uri = "./test_mlruns_router_api"
        mlflow.set_tracking_uri(test_tracking_uri)
        result_artifact = run_training(settings=dev_test_settings)
        yield result_artifact.run_id
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        mlflow.set_tracking_uri("mlruns")

    @pytest.fixture(scope="class")
    def client(self, dev_test_settings: Settings, trained_model_run_id: str):
        setup_api_context(run_id=trained_model_run_id, settings=dev_test_settings)
        return TestClient(app)

    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_uri" in data

    def test_predict_minimal_response(self, client: TestClient):
        # PredictionRequest는 동적 생성되므로, 입력 키는 loader_sql_snapshot 또는 data_schema 기준
        # 테스트 피처 스토어 환경에서 기본 엔티티 컬럼 예: user_id, product_id
        payload = {"user_id": "u1001", "product_id": "p2001"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"prediction", "model_uri"} 