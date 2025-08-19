"""Serving API 테스트 - BLUEPRINT 철학 기반

Blueprint 원칙 6: 자기 기술 API (Self-Describing API)
- API는 스스로를 설명할 수 있어야 함
- 메타데이터와 스키마 정보 제공
- 환경별 차등 기능 (Local vs DEV)
"""

import pytest

from src.serving.router import app
from src.serving.schemas import PredictionResponse
from src.settings.loaders import load_settings_by_file


class TestServingAPIBlueprintCompliance:
    """Serving API의 BLUEPRINT 철학 준수 검증"""

    @pytest.fixture
    def mock_settings(self):
        """테스트용 Settings - Serving API 검증용"""
        return load_settings_by_file(
            recipe_file="tests/fixtures/recipes/local_classification_test.yaml"
        )

    def test_serving_app_exists(self):
        """Serving App 존재 - BLUEPRINT 서빙 인프라"""
        assert app is not None
        # BLUEPRINT: FastAPI App이 정의되어 있어야 함
        assert hasattr(app, 'router')
        assert hasattr(app.router, 'routes')

    def test_prediction_schema_factory_exists(self):
        """예측 요청 스키마 팩토리 존재 - BLUEPRINT 자기 기술 API"""
        # BLUEPRINT 원칙 6: API는 스키마를 통해 자기를 설명
        from src.serving.schemas import create_dynamic_prediction_request
        
        assert create_dynamic_prediction_request is not None
        
        # 동적 스키마 생성 테스트
        TestRequest = create_dynamic_prediction_request("Test", ["user_id"])
        assert TestRequest is not None
        
        # 생성된 스키마가 Pydantic 모델인지 확인
        from pydantic import BaseModel
        assert issubclass(TestRequest, BaseModel)

    def test_prediction_response_schema_exists(self):
        """예측 응답 스키마 존재 - BLUEPRINT 자기 기술 API"""
        # BLUEPRINT: 응답도 명확한 스키마를 가져야 함
        assert PredictionResponse is not None

    @pytest.mark.unit
    def test_serving_components_importable(self):
        """Serving 컴포넌트들이 import 가능 - BLUEPRINT 모듈화"""
        # BLUEPRINT 원칙 4: 모듈화된 컴포넌트들
        from src.serving import schemas
        from src.serving.router import app
        
        assert app is not None
        assert schemas is not None
        
        # 핵심 serving 컴포넌트들이 존재해야 함
        from src.serving import _endpoints
        from src.serving._context import app_context
        
        assert _endpoints is not None
        assert app_context is not None

    @pytest.mark.blueprint_principle_2
    def test_serving_environment_awareness(self, mock_settings):
        """Serving 환경별 인식 - BLUEPRINT 원칙 2"""
        # BLUEPRINT: Serving은 환경별로 다르게 동작해야 함
        
        # Settings 객체에 serving 설정이 있어야 함
        assert hasattr(mock_settings, 'serving')
        assert hasattr(mock_settings.serving, 'enabled')
        
        # 환경별로 서빙 정책이 다름 (Local=False, DEV/PROD=True)
        env_type = mock_settings.environment.app_env
        if env_type == 'local':
            assert mock_settings.serving.enabled == False
        elif env_type in ['dev', 'prod']:
            # DEV/PROD에서는 serving이 활성화될 수 있음
            assert isinstance(mock_settings.serving.enabled, bool)

    def test_serving_context_exists(self):
        """ServingContext 존재 확인 - BLUEPRINT 설계"""
        from src.serving._context import app_context
        
        # BLUEPRINT: app_context는 서빙 상태를 관리하는 중앙 객체
        assert app_context is not None
        
        # Context 객체가 기본 속성들을 가지고 있는지 확인
        assert hasattr(app_context, 'model')
        assert hasattr(app_context, 'settings')

    def test_serving_policy_enforcement_interface(self):
        """Serving 정책 강제 인터페이스 - BLUEPRINT 보안"""
        # BLUEPRINT: Serving은 환경별 정책을 강제해야 함
        
        # Local 환경에서는 503 응답 정책
        # DEV 환경에서는 Feature Store 필수 정책
        
        # 정책 관련 모듈들이 존재하는지 확인
        from src.serving import _lifespan, _endpoints
        
        assert _lifespan is not None
        assert _endpoints is not None