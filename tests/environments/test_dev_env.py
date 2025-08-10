"""
DEV 환경 완전 기능 검증 테스트

Blueprint 원칙 9: 환경별 차등적 기능 분리
"모든 기능이 완전히 작동하는 안전한 실험실"

DEV 환경의 철학:
- 완전한 Feature Store 활용
- API 서빙 모든 기능 지원
- 팀 공유 MLflow 서버
- 실제 운영과 동일한 아키텍처
- mmp-local-dev 스택 완전 활용
"""

import pytest
from fastapi.testclient import TestClient

from src.engine.factory import Factory
from src.components.augmenter import Augmenter
from src.settings import Settings
from serving.api import app, setup_api_context
from src.pipelines.train_pipeline import run_training


pytest.skip("Deprecated/outdated test module pending Stage 6 test overhaul (environment tests refactor).", allow_module_level=True)

@pytest.mark.requires_dev_stack
class TestDevEnvironment:
    """
    DEV 환경의 핵심 기능들을 검증하는 통합 테스트.
    Blueprint 원칙 9: "모든 기능이 완전히 작동하는 안전한 실험실"
    """

    def test_dev_env_uses_feature_store_augmenter(self, dev_test_settings: Settings):
        """
        DEV 환경에서는 PassThroughAugmenter가 아닌,
        실제 Feature Store와 연동하는 Augmenter가 생성되는지 검증한다.
        """
        factory = Factory(dev_test_settings)
        augmenter = factory.create_augmenter()
        
        assert isinstance(augmenter, Augmenter)
        # PassThroughAugmenter는 'passthrough'라는 특수 속성을 가질 수 있도록 설계했다고 가정
        assert not getattr(augmenter, 'passthrough', False), \
            "DEV 환경에서 PassThroughAugmenter가 사용되었습니다."

    def test_dev_env_api_serving_enabled(self, dev_test_settings: Settings):
        """
        DEV 환경에서 API 서버가 정상적으로 실행되고 /health 엔드포인트가
        200 OK를 반환하는지 검증한다.
        """
        # API 컨텍스트 설정을 위해 최소한의 학습을 수행하여 run_id 확보
        run_id = run_training(dev_test_settings).run_id
        
        setup_api_context(run_id=run_id, settings=dev_test_settings)
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert run_id in health_data["model_uri"]

    def test_dev_env_loads_correct_configs(self, dev_test_settings: Settings):
        """
        DEV 환경에서 dev.yaml의 설정값들이 올바르게 로드되었는지 검증한다.
        """
        # HPO 활성화 여부 검증
        assert dev_test_settings.hyperparameter_tuning.enabled is True, \
            "DEV 환경에서 하이퍼파라미터 튜닝이 비활성화되어 있습니다."
            
        # MLflow 실험 이름 검증
        assert "Dev" in dev_test_settings.mlflow.experiment_name, \
            "DEV 환경의 MLflow 실험 이름이 올바르지 않습니다."
            
        # 데이터 어댑터 설정 검증 (예시)
        # dev.yaml에서 postgresql을 사용한다고 가정
        adapter_config = dev_test_settings.data_adapters.adapters.get("postgresql")
        assert adapter_config is not None, "DEV 환경에 PostgreSQL 어댑터 설정이 없습니다."
        assert adapter_config.config.get("database") == "mlpipeline", \
            "DEV 환경의 PostgreSQL 데이터베이스 설정이 올바르지 않습니다." 