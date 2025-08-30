"""Factory 컴포넌트 단위 테스트 - BLUEPRINT 철학 기반 TDD 구현

Blueprint 핵심 원칙 검증:
- 원칙 3: 선언적 파이프라인 - Factory를 통한 동적 컴포넌트 조립
- 원칙 4: 모듈화/확장성 - Factory & Registry 패턴 구현
- 원칙 1: 설정-논리 분리 - Settings 기반 컴포넌트 생성

Factory 패턴 적용:
- SettingsFactory로 완전한 Settings 객체 생성 (파일 의존성 제거)
- 실제 Factory 클래스 유지하여 통합 테스트 성격 보장
- Blueprint 정책과 실제 컴포넌트 생성 로직 검증
"""
import pytest
from unittest.mock import patch

from src.engine.factory import Factory
from src.settings import Settings
from tests.factories.settings_factory import SettingsFactory


class TestFactoryBlueprintCompliance:
    """Factory의 Blueprint 철학 준수 테스트 - Factory 패턴 적용"""

    @pytest.fixture
    def mock_settings(self, test_factories):
        """테스트용 Settings 객체 - SettingsFactory 사용으로 파일 의존성 제거"""
        # SettingsFactory로 완전한 설정 딕셔너리 생성
        settings_dict = test_factories['settings'].create_classification_settings("local")
        
        # 실제 Factory가 요구하는 Settings 객체로 변환
        # Factory는 settings.recipe.model.class_path 같은 실제 속성에 접근하므로
        return Settings(**settings_dict)

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.blueprint_principle_1
    def test_factory_initialization_with_settings(self, mock_settings):
        """Factory 초기화 - Settings 객체 주입 (BLUEPRINT 원칙 1)"""
        factory = Factory(mock_settings)
        
        assert factory is not None
        assert factory.settings == mock_settings
        # BLUEPRINT: Settings를 단일 진실 공급원으로 사용
        assert hasattr(factory, 'model_config')

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.blueprint_principle_3
    @pytest.mark.blueprint_principle_4
    def test_factory_dynamic_component_creation_interface(self, mock_settings):
        """Factory 동적 컴포넌트 생성 인터페이스 검증 (BLUEPRINT 원칙 3, 4)"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 4: 모듈화된 컴포넌트 생성 메서드들
        creation_methods = [
            'create_data_adapter',
            'create_augmenter', 
            'create_preprocessor',
            'create_model',
            'create_evaluator'
        ]
        
        for method_name in creation_methods:
            assert hasattr(factory, method_name), f"Factory missing {method_name} method"
            assert callable(getattr(factory, method_name)), f"{method_name} is not callable"

    @pytest.mark.core
    @pytest.mark.unit
    @pytest.mark.blueprint_principle_1
    def test_factory_model_config_property(self, mock_settings):
        """Factory model_config 프로퍼티 - 레시피 논리 접근 (BLUEPRINT 원칙 1)"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT: Recipe(논리)와 Config(인프라) 분리된 설계
        model_config = factory.model_config
        assert model_config is not None
        # 레시피에서 모델 논리 정보를 가져와야 함
        assert hasattr(model_config, 'loader') or hasattr(model_config, 'class_path')

    @pytest.mark.core
    @pytest.mark.unit
    def test_factory_creates_preprocessor_following_blueprint(self, mock_settings):
        """Factory 전처리기 생성 - BLUEPRINT 철학 준수"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 3: 선언적 구성에 따른 컴포넌트 생성
        preprocessor = factory.create_preprocessor()
        
        assert preprocessor is not None
        # BLUEPRINT: 모든 ML 컴포넌트는 sklearn 호환 인터페이스
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')
        assert hasattr(preprocessor, 'fit_transform')

    @pytest.mark.core
    @pytest.mark.unit
    def test_factory_augmenter_creation_environment_driven(self, mock_settings):
        """Factory Augmenter 생성 - 환경별 차등 기능 (BLUEPRINT 원칙 2)"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 2: 환경별 역할 분담
        # Local 환경에서는 PassThroughAugmenter가 생성되어야 함
        augmenter = factory.create_augmenter()
        
        assert augmenter is not None
        # BLUEPRINT: Augmenter는 PIT 조인 인터페이스를 가져야 함
        assert hasattr(augmenter, 'augment')

    def test_factory_error_handling_for_invalid_class_path(self, mock_settings):
        """Factory 오류 처리 - 존재하지 않는 클래스 경로"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT: 명확한 오류 메시지로 디버깅 지원
        with patch.object(factory.settings.recipe.model, 'class_path', 'non.existent.Class'):
            with pytest.raises((ImportError, AttributeError, ValueError)):
                factory.create_model()

    @pytest.mark.blueprint_principle_4
    @pytest.mark.blueprint_principle_4
    def test_factory_extensibility_through_registry(self, mock_settings):
        """Factory 확장성 - Registry 패턴 (BLUEPRINT 원칙 4)"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 4: Registry를 통한 플러그인 형태 확장
        # Registry가 동적으로 컴포넌트를 찾을 수 있어야 함
        try:
            adapter = factory.create_data_adapter()
            # 기본 어댑터가 생성되어야 함
            assert adapter is not None
        except Exception as e:
            # Registry 기반 오류는 명확한 메시지를 제공해야 함
            assert "adapter" in str(e).lower() or "registry" in str(e).lower()
    
    def test_factory_augmenter_selection_policy_local_environment(self, mock_settings):
        """Factory Augmenter 선택 정책 - Local 환경 (BLUEPRINT.md 148-155라인)"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 2: Local 환경에서는 PassThroughAugmenter 사용
        augmenter = factory.create_augmenter(run_mode="train")
        
        assert augmenter is not None
        assert type(augmenter).__name__ == "PassThroughAugmenter"
        # PassThrough는 run_mode에 관계없이 동일하게 동작
        assert hasattr(augmenter, 'augment')
        
    def test_factory_augmenter_selection_policy_serving_restrictions(self, mock_settings):
        """Factory Augmenter Serving 제약 정책 - BLUEPRINT.md 서빙 제약"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT: Serving에서는 PassThrough/SqlFallback 금지
        # Local 환경이라 PassThrough가 선택되지만, serving 모드에서는 예외 발생
        with pytest.raises(TypeError, match="Serving에서는.*금지"):
            factory.create_augmenter(run_mode="serving")
            
    def test_factory_augmenter_selection_policy_comprehensive(self, mock_settings):
        """Factory Augmenter 선택 정책 종합 테스트 - BLUEPRINT.md 정책 전체"""
        factory = Factory(mock_settings)
        
        # 1) Local 환경: PassThrough (run_mode 무관)
        for mode in ["train", "batch"]:
            augmenter = factory.create_augmenter(run_mode=mode)
            assert type(augmenter).__name__ == "PassThroughAugmenter"
            
        # 2) Serving 모드: 예외 발생
        with pytest.raises(TypeError):
            factory.create_augmenter(run_mode="serving")
            
        # 3) 비지원 Augmenter type: 예외 발생  
        # (mock_settings는 local 환경이라 정상 동작하거나, 비지원 설정에 대한 예외 처리 필요)
        
    def test_factory_augmenter_policy_error_messages(self, mock_settings):
        """Factory Augmenter 정책 에러 메시지 검증 - 명확한 디버깅 지원"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT: 명확한 오류 메시지로 디버깅 지원
        try:
            factory.create_augmenter(run_mode="serving")
            assert False, "Serving 모드에서 예외가 발생해야 함"
        except TypeError as e:
            error_message = str(e).lower()
            assert "serving" in error_message
            assert "금지" in str(e) or "feature store" in error_message
            
    def test_factory_model_creation_with_hyperparameters(self, mock_settings):
        """Factory 모델 생성 - 하이퍼파라미터 동적 로딩"""
        factory = Factory(mock_settings)
        
        # BLUEPRINT 원칙 1, 3: Recipe에 정의된 설정으로 동적 모델 생성
        model = factory.create_model()
        
        assert model is not None
        # sklearn 호환 인터페이스 확인
        assert hasattr(model, 'fit')
        
        # Recipe에 정의된 하이퍼파라미터가 적용되어야 함
        # (local_classification_test.yaml: n_estimators: 50, max_depth: 10 등)
        if hasattr(model, 'n_estimators'):
            assert model.n_estimators == 50
        if hasattr(model, 'max_depth'):
            assert model.max_depth == 10
        if hasattr(model, 'random_state'):
            assert model.random_state == 42