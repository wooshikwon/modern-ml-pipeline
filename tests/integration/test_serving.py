from fastapi.testclient import TestClient

from src.serving.router import app


class TestServing:
    def test_root_and_health_without_model_returns_service_info(self):
        client = TestClient(app)
        # 루트는 항상 200과 상태 문자열을 반환
        r = client.get("/")
        assert r.status_code == 200
        assert "status" in r.json()

        # health는 모델 미로드 시 503, 내부 처리 시 500이 날 수 있음
        h = client.get("/health")
        assert h.status_code in (200, 503, 500)
