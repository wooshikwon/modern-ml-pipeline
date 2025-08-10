import pytest
from fastapi.testclient import TestClient
import mlflow
import shutil

from src.serving.router import app, setup_api_context
from src.settings import Settings
from src.pipelines.train_pipeline import run_training


class TestRouterAPIPolicyLocal:
    @pytest.fixture(scope="class")
    def trained_model_run_id(self, local_test_settings: Settings):
        test_tracking_uri = "./test_mlruns_router_api_local"
        mlflow.set_tracking_uri(test_tracking_uri)
        result_artifact = run_training(settings=local_test_settings)
        yield result_artifact.run_id
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        mlflow.set_tracking_uri("mlruns")

    @pytest.fixture(scope="class")
    def client(self, local_test_settings: Settings, trained_model_run_id: str):
        setup_api_context(run_id=trained_model_run_id, settings=local_test_settings)
        return TestClient(app)

    def test_predict_policy_blocked_for_pass_through(self, client: TestClient):
        payload = {"user_id": "u1001", "product_id": "p2001"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 503
        assert "pass_through" in resp.text 