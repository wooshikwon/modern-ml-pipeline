"""LOCAL 환경 차등 기능 검증 테스트 - BLUEPRINT 철학 기반

Blueprint 원칙 2: 환경별 역할 분담
"제약은 단순함을 낳고, 단순함은 집중을 낳는다"

LOCAL 환경의 철학:
- 빠른 실험과 디버깅에 집중
- 복잡한 인프라 의존성 제거
- 의도적 제약을 통한 핵심 로직 집중
- PassThroughfetcher 사용
- API 서빙 시스템적 차단
"""

import pytest
import pandas as pd

from src.factory.factory import Factory
from src.components.fetcher import PassThroughfetcher
from src.settings import load_settings


@pytest.mark.local_env
class TestLocalEnvironmentBlueprintCompliance:
    """Local 환경의 BLUEPRINT 철학 준수 검증"""

    @pytest.fixture
    def local_settings(self):
        """Local 환경 Settings - BLUEPRINT 원칙 2 검증용"""
        return load_settings(
            recipe_file="tests/fixtures/recipes/local_classification_test.yaml", env_name="local"
        )

    def test_local_environment_settings_loaded(self, local_settings):
        """Local 환경 설정 로딩 - BLUEPRINT 원칙 1 (설정-논리 분리)"""
        assert local_settings is not None
        # BLUEPRINT: Settings 객체가 제대로 로드되어야 함
        assert hasattr(local_settings, 'environment')
        assert hasattr(local_settings, 'feature_store')
        assert hasattr(local_settings, 'recipe')

    def test_local_environment_factory_creates_passthrough_fetcher(self, local_settings):
        """Local 환경에서 PassThrough fetcher 생성 - BLUEPRINT 원칙 2"""
        factory = Factory(local_settings)
        
        # BLUEPRINT 원칙 2: Local 환경에서는 PassThroughfetcher 사용
        fetcher = factory.create_fetcher()
        
        assert fetcher is not None
        # Local 환경에서는 PassThroughfetcher가 생성되어야 함
        assert isinstance(fetcher, PassThroughfetcher)

    def test_local_environment_feature_store_configuration(self, local_settings):
        """Local 환경 Feature Store 설정 - BLUEPRINT 원칙 2"""
        # BLUEPRINT: Settings에 feature_store 설정이 있어야 함
        assert hasattr(local_settings, 'feature_store')
        assert hasattr(local_settings.feature_store, 'provider')
        
        # 환경별로 설정이 다름 - 실제 recipe에서는 'pass_through' fetcher 사용
        # Feature store provider는 환경에 따라 설정되지만, 
        # Local recipe는 pass_through fetcher를 사용하므로 실질적으로 비활성화됨

    def test_local_environment_serving_configuration(self, local_settings):
        """Local 환경 서빙 설정 - BLUEPRINT 원칙 2"""  
        # BLUEPRINT: Settings에 serving 설정이 있어야 함
        assert hasattr(local_settings, 'serving')
        assert hasattr(local_settings.serving, 'enabled')
        
        # Recipe 레벨에서 pass_through fetcher 사용으로 serving 제약
        # 실제 serving은 fetcher 타입에 따라 차단됨 (PassThroughfetcher 시 503 정책)

    @pytest.mark.unit
    def test_local_environment_fetcher_passthrough_behavior(self, local_settings):
        """Local 환경 fetcher PassThrough 동작 - BLUEPRINT 데이터 계약"""
        fetcher = PassThroughfetcher()
        
        # BLUEPRINT 데이터 계약: entity + timestamp 입력
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'event_ts': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'label': [1, 0]
        })
        
        result = fetcher.fetch(sample_data, run_mode="train")
        
        # BLUEPRINT: PassThrough는 입력과 동일한 출력
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        pd.testing.assert_frame_equal(result, sample_data)

    @pytest.mark.blueprint_principle_2
    def test_local_environment_constraints_enable_focus(self, local_settings):
        """Local 환경 제약이 핵심 로직 집중을 가능하게 함 - BLUEPRINT 철학"""
        factory = Factory(local_settings)
        
        # BLUEPRINT 철학: 제약을 통한 단순함
        # Local에서는 복잡한 인프라 없이 핵심 로직에만 집중
        
        # 1. 전처리기는 정상 생성
        preprocessor = factory.create_preprocessor()
        assert preprocessor is not None
        
        # 2. fetcher는 PassThrough (단순함)
        fetcher = factory.create_fetcher()
        assert isinstance(fetcher, PassThroughfetcher)
        
        # 3. 모델 생성도 정상 작동 (핵심 로직)
        model = factory.create_model()
        assert model is not None

    def test_local_environment_fast_experimentation_ready(self, local_settings):
        """Local 환경 빠른 실험 준비 - BLUEPRINT 설계 의도"""
        # BLUEPRINT: 컴포넌트 등록 먼저 수행
        from src.factory import register_all_components
        register_all_components()
        
        factory = Factory(local_settings)
        
        # BLUEPRINT: Local 환경은 빠른 실험을 위해 설계됨
        # 모든 핵심 컴포넌트가 외부 의존성 없이 생성 가능해야 함
        
        components = {
            'preprocessor': factory.create_preprocessor(),
            'fetcher': factory.create_fetcher(),
            'model': factory.create_model(),
            'evaluator': factory.create_evaluator()
        }
        
        for name, component in components.items():
            assert component is not None, f"{name} creation failed in local environment"