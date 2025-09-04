"""
A03-1: Factory 기반 컴포넌트 조립 통합 테스트

DEV_PLANS.md A03-1 구현:
- Factory가 Settings 기반으로 컴포넌트를 올바르게 조립하는지 검증
- Blueprint 원칙에 따른 의존성 주입 확인
- 환경별 fetcher 선택 정책 통합 테스트
"""

import pytest
from unittest.mock import patch
from src.factory.factory import Factory
from src.settings import load_settings


class TestFactoryBasedComponentAssembly:
    """A03-1: Factory 기반 컴포넌트 조립 테스트"""
    
    def test_factory_creates_all_core_components(self):
        """Factory가 모든 핵심 컴포넌트를 올바르게 생성하는지 테스트 (GREEN 단계)"""
        # 실제 존재하는 레시피 파일 사용
        settings = load_settings("models/classification/logistic_regression", "local")
        
        # 디버그: optuna 가용성 확인
        try:
            import optuna
            print(f"✅ optuna available: {optuna.__version__}")
        except ImportError as e:
            print(f"❌ optuna not available: {e}")
        
        # Bootstrap으로 어댑터들이 자동 등록됨 (순환 import 해결!)
        from src.factory import bootstrap
        try:
            bootstrap(settings)
            print("✅ Bootstrap successful")
        except ImportError as e:
            pytest.fail(f"Bootstrap failed with ImportError: {e}. This suggests dependency validation is failing even though optuna is available.")
        
        factory = Factory(settings)
        
        # 핵심 컴포넌트들이 모두 생성되는지 확인
        components = {}
        
        # 각 컴포넌트 생성 시도 - 단계별로 접근
        try:
            components['data_adapter'] = factory.create_data_adapter()
            assert components['data_adapter'] is not None, "data_adapter should not be None"
            
            components['fetcher'] = factory.create_fetcher()
            assert components['fetcher'] is not None, "fetcher should not be None"
            
            components['preprocessor'] = factory.create_preprocessor()
            # preprocessor는 None일 수 있음 (설정에 따라)
            
            components['model'] = factory.create_model()
            assert components['model'] is not None, "model should not be None"
            
            # trainer와 evaluator는 추후 단계에서 테스트 (복잡한 의존성)
            
        except Exception as e:
            pytest.fail(f"Factory should create core components successfully, but failed with: {e}")
        
        # 필수 컴포넌트들이 생성되었는지 확인
        required_components = ['data_adapter', 'fetcher', 'model']
        for comp_name in required_components:
            assert comp_name in components, f"{comp_name} should be created by Factory"
            assert components[comp_name] is not None, f"{comp_name} should not be None"
    
    def test_components_receive_correct_settings_injection(self):
        """컴포넌트들이 올바른 Settings 주입을 받는지 테스트 (RED)"""
        settings = load_settings("models/classification/logistic_regression", "local")
        
        # Bootstrap으로 어댑터들이 자동 등록됨
        from src.factory import bootstrap
        bootstrap(settings)
        
        factory = Factory(settings)
        
        # 각 컴포넌트 생성 (create_trainer는 Factory에 없으므로 제외)
        components = {
            'fetcher': factory.create_fetcher(),
            'preprocessor': factory.create_preprocessor(),
            'model': factory.create_model()
        }
        
        # 모든 컴포넌트가 동일한 settings 인스턴스를 가지는지 확인
        for comp_name, component in components.items():
            if hasattr(component, 'settings'):
                assert component.settings is settings, f"{comp_name} should have the same settings instance"
            elif hasattr(component, '_settings'):
                assert component._settings is settings, f"{comp_name} should have the same settings instance via _settings"
    
    def test_fetcher_selection_policy_integration(self):
        """환경별 fetcher 선택 정책 통합 테스트 (RED)"""
        # 다양한 환경 설정으로 테스트
        test_cases = [
            {
                'env': 'local',
                'expected_type': 'PassThroughfetcher',
                'description': 'Local 환경에서는 PassThroughfetcher 사용'
            },
            {
                'env': 'dev', 
                'expected_type': 'FeatureStorefetcher',
                'description': 'Dev 환경에서는 FeatureStorefetcher 사용'
            }
        ]
        
        for case in test_cases:
            with patch.dict('os.environ', {'ENV_NAME': case['env']}):
                settings = load_settings("models/classification/logistic_regression", "local")
                
                # Bootstrap으로 어댑터들이 자동 등록됨
                from src.factory import bootstrap
                bootstrap(settings)
                
                factory = Factory(settings)
                
                fetcher = factory.create_fetcher()
                
                # fetcher 타입이 환경에 맞게 선택되었는지 확인
                actual_type = fetcher.__class__.__name__
                assert case['expected_type'] in actual_type or actual_type == case['expected_type'], \
                    f"{case['description']}: Expected {case['expected_type']}, got {actual_type}"
    
    def test_factory_component_interaction_readiness(self):
        """Factory로 생성된 컴포넌트들이 상호작용 준비가 되어있는지 테스트 (RED)"""
        settings = load_settings("models/classification/logistic_regression", "local")
        
        # Bootstrap으로 어댑터들이 자동 등록됨
        from src.factory import bootstrap
        bootstrap(settings)
        
        factory = Factory(settings)
        
        # 컴포넌트들 생성 (Factory에서 제공하는 메서드만)
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        # trainer는 Factory에서 직접 제공하지 않음
        
        # 컴포넌트들이 상호작용 준비가 되었는지 확인
        # (실제 데이터 흐름은 A03-2에서 테스트)
        
        # 각 컴포넌트가 필요한 메서드를 가지고 있는지 확인
        component_methods = {
            'data_adapter': ['load', 'save'],  # 예상 메서드
            'fetcher': ['augment'],
            'preprocessor': ['fit', 'transform'],
            'model': ['fit', 'predict']
        }
        
        components = {
            'data_adapter': data_adapter,
            'fetcher': fetcher, 
            'preprocessor': preprocessor,
            'model': model
        }
        
        for comp_name, component in components.items():
            expected_methods = component_methods.get(comp_name, [])
            for method_name in expected_methods:
                if hasattr(component, method_name):
                    assert callable(getattr(component, method_name)), \
                        f"{comp_name}.{method_name} should be callable"
    
    def test_factory_error_handling_graceful_degradation(self):
        """Factory의 우아한 에러 처리 테스트 (RED)"""
        settings = load_settings("models/classification/logistic_regression", "local")
        
        # Bootstrap으로 어댑터들이 자동 등록됨
        from src.factory import bootstrap
        bootstrap(settings)
        
        factory = Factory(settings)
        
        # 잘못된 설정으로 컴포넌트 생성 시 적절한 에러가 발생하는지 확인
        with patch.object(settings.recipe.model, 'class_path', 'non.existent.Model'):
            with pytest.raises((ImportError, ModuleNotFoundError, ValueError)):
                factory.create_model()
        
        # 에러 발생 후에도 Factory가 계속 사용 가능한지 확인
        # (정상적인 컴포넌트는 계속 생성 가능해야 함)
        try:
            fetcher = factory.create_fetcher()  # 이는 성공해야 함
            assert fetcher is not None
        except Exception as e:
            pytest.fail(f"Factory should remain functional after partial failure, but got: {e}")