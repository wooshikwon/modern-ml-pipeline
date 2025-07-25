"""
LOCAL 환경 차등 기능 검증 테스트

Blueprint 원칙 9: 환경별 차등적 기능 분리
"제약은 단순함을 낳고, 단순함은 집중을 낳는다"

LOCAL 환경의 철학:
- 빠른 실험과 디버깅에 집중
- 복잡한 인프라 의존성 제거
- 의도적 제약을 통한 핵심 로직 집중
- PassThroughAugmenter 사용
- API 서빙 시스템적 차단
"""

import pytest
import typer
from typer.testing import CliRunner

from src.core.factory import Factory
from src.core.augmenter import PassThroughAugmenter
from src.settings import Settings
from main import app  # CLI app import

runner = CliRunner()

@pytest.mark.local_env
class TestLocalEnvironment:
    """
    LOCAL 환경의 의도적 제약 기능들을 검증하는 통합 테스트.
    Blueprint 원칙 9: "제약은 단순함을 낳고, 단순함은 집중을 낳는다"
    """

    def test_local_env_uses_passthrough_augmenter(self, local_test_settings: Settings):
        """
        LOCAL 환경에서는 Feature Store를 사용하지 않는
        PassThroughAugmenter가 생성되는지 검증한다.
        """
        factory = Factory(local_test_settings)
        augmenter = factory.create_augmenter()
        
        assert isinstance(augmenter, PassThroughAugmenter), \
            "LOCAL 환경에서 PassThroughAugmenter가 사용되지 않았습니다."

    def test_local_env_api_serving_is_blocked(self):
        """
        LOCAL 환경에서 `serve-api` CLI 명령어가 시스템적으로 차단되는지 검증한다.
        """
        result = runner.invoke(app, ["serve-api", "--run-id", "test-run"])
        
        # typer.Exit(code=1)이 호출되었는지 확인
        assert result.exit_code == 1
        # 에러 메시지가 올바르게 출력되는지 확인
        assert "API Serving이 현재 환경에서 비활성화되어 있습니다." in result.stdout
        assert "현재 환경: local" in result.stdout

    def test_local_env_loads_correct_configs(self, local_test_settings: Settings):
        """
        LOCAL 환경에서 local.yaml 또는 base.yaml의 설정값들이
        올바르게 로드되었는지 검증한다.
        """
        # HPO 비활성화 여부 검증 (base.yaml 기본값)
        assert local_test_settings.hyperparameter_tuning.enabled is False, \
            "LOCAL 환경에서 하이퍼파라미터 튜닝이 활성화되어 있습니다."
            
        # MLflow 실험 이름 검증 (base.yaml 기본값)
        assert "Campaign-Uplift-Modeling" in local_test_settings.mlflow.experiment_name, \
            "LOCAL 환경의 MLflow 실험 이름이 올바르지 않습니다."
            
        # 데이터 어댑터 설정 검증 (base.yaml 기본값)
        adapter_name = local_test_settings.data_adapters.default_loader
        assert adapter_name == "filesystem", \
            "LOCAL 환경의 기본 로더가 'filesystem'이 아닙니다." 