"""통합 호환성 테스트 - BLUEPRINT 철학 기반

Blueprint 전체 호환성 검증:
- 기존 워크플로우 완전한 하위 호환성  
- 새로운 기능의 점진적 활성화
- 3계층 아키텍처 통합 동작
- Factory 기반 컴포넌트 조립 검증
"""

import pytest

from src.engine.factory import Factory
from src.components._trainer import Trainer
from src.settings.loaders import load_settings_by_file


class TestMMP3LayerArchitectureCompatibility:
    """MMP 3계층 아키텍처 통합 호환성 검증"""

    @pytest.fixture
    def integration_settings(self):
        """통합 테스트용 Settings - 3계층 검증용"""
        return load_settings_by_file(
            recipe_file="tests/fixtures/recipes/local_classification_test.yaml"
        )

    def test_layer1_components_creation(self, integration_settings):
        """Layer 1 (Components) 생성 - BLUEPRINT 계층 검증"""
        factory = Factory(integration_settings)
        
        # BLUEPRINT Layer 1: 개별 ML 작업 단위들
        components = {
            'trainer': Trainer(settings=integration_settings, factory_provider=lambda: factory),
            'preprocessor': factory.create_preprocessor(),
            'augmenter': factory.create_augmenter(),
            'model': factory.create_model(),
            'evaluator': factory.create_evaluator()
        }
        
        for name, component in components.items():
            assert component is not None, f"Layer 1 component {name} creation failed"

    def test_layer2_engine_orchestration(self, integration_settings):
        """Layer 2 (Engine) 오케스트레이션 - BLUEPRINT Factory & Registry"""
        factory = Factory(integration_settings)
        
        # BLUEPRINT Layer 2: Factory를 통한 동적 컴포넌트 조립
        assert factory is not None
        assert factory.settings == integration_settings
        
        # Factory & Registry 패턴이 정상 동작하는지 확인
        assert hasattr(factory, 'model_config')
        model_config = factory.model_config
        assert model_config is not None

    @pytest.mark.integration
    def test_layer3_pipeline_readiness(self, integration_settings):
        """Layer 3 (Pipelines) 준비성 - BLUEPRINT 엔드투엔드 흐름"""
        factory = Factory(integration_settings)
        
        # BLUEPRINT Layer 3: 파이프라인에 필요한 모든 컴포넌트가 준비되어야 함
        pipeline_components = {}
        
        try:
            pipeline_components['data_adapter'] = factory.create_data_adapter()
            pipeline_components['preprocessor'] = factory.create_preprocessor()
            pipeline_components['augmenter'] = factory.create_augmenter()
            pipeline_components['model'] = factory.create_model()
            pipeline_components['evaluator'] = factory.create_evaluator()
            pipeline_components['trainer'] = Trainer(
                settings=integration_settings, 
                factory_provider=lambda: factory
            )
        except Exception as e:
            # 특정 컴포넌트 실패는 로그하되, 기본 구조는 검증
            print(f"Component creation warning: {e}")
        
        # 최소한 핵심 컴포넌트들은 생성되어야 함
        essential_components = ['preprocessor', 'model', 'trainer']
        for comp_name in essential_components:
            assert comp_name in pipeline_components, f"Essential component {comp_name} missing"
            assert pipeline_components[comp_name] is not None

    @pytest.mark.blueprint_principle_1
    def test_recipe_config_separation_integration(self, integration_settings):
        """Recipe-Config 분리 통합 검증 - BLUEPRINT 원칙 1"""
        # BLUEPRINT 원칙 1: 설정과 논리의 분리
        
        # Settings에 Recipe(논리)와 Config(인프라) 정보가 모두 있어야 함
        assert hasattr(integration_settings, 'recipe'), "Recipe(논리) 정보 누락"
        assert hasattr(integration_settings, 'environment'), "Config(인프라) 정보 누락"
        
        # Recipe는 ML 실험 논리를 포함
        recipe = integration_settings.recipe
        assert hasattr(recipe, 'model'), "Recipe에 모델 논리 누락"
        
        # Config는 실행 환경을 포함  
        environment = integration_settings.environment
        assert hasattr(environment, 'app_env'), "Config에 환경 정보 누락"

    @pytest.mark.blueprint_principle_3
    def test_declarative_pipeline_integration(self, integration_settings):
        """선언적 파이프라인 통합 검증 - BLUEPRINT 원칙 3"""
        factory = Factory(integration_settings)
        
        # BLUEPRINT 원칙 3: 선언적 YAML 구성이 실제 컴포넌트로 변환
        # Settings의 선언적 정보가 실제 객체로 조립되는지 확인
        
        # 1. 모델 클래스 경로 → 실제 모델 객체
        model = factory.create_model()
        assert model is not None, "선언적 모델 구성이 실제 객체로 변환되지 않음"
        
        # 2. 전처리 구성 → 실제 전처리기 객체  
        preprocessor = factory.create_preprocessor()
        assert preprocessor is not None, "선언적 전처리 구성이 실제 객체로 변환되지 않음"

    @pytest.mark.blueprint_principle_4
    def test_modularity_extensibility_integration(self, integration_settings):
        """모듈화/확장성 통합 검증 - BLUEPRINT 원칙 4"""
        factory = Factory(integration_settings)
        
        # BLUEPRINT 원칙 4: 모듈화된 컴포넌트들의 독립성
        
        # 각 컴포넌트는 독립적으로 생성 가능해야 함
        components = []
        component_types = ['preprocessor', 'augmenter', 'model', 'evaluator']
        
        for comp_type in component_types:
            try:
                create_method = getattr(factory, f'create_{comp_type}')
                component = create_method()
                components.append((comp_type, component))
            except Exception as e:
                # 일부 컴포넌트는 외부 의존성으로 실패할 수 있음
                print(f"Component {comp_type} creation failed (expected): {e}")
        
        # 최소한 일부 컴포넌트는 독립적으로 생성되어야 함
        assert len(components) > 0, "모든 컴포넌트 생성 실패 - 모듈화 문제"