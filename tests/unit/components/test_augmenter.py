"""Augmenter 컴포넌트 단위 테스트 - BLUEPRINT 철학 기반 TDD 구현

Blueprint 핵심 원칙 검증:
- 원칙 2: 환경별 역할 분담 - Local(PassThrough) vs Dev(FeatureStore)
- Augmenter 선택 정책: 환경/provider/서빙 모드별 차등 동작
- PIT 조인 데이터 계약: entity + timestamp → 피처 추가

Factory 패턴 적용:
- SettingsFactory로 파일 의존성 제거
- TestDataFactory로 표준화된 테스트 데이터
- 실제 컴포넌트 유지하여 Blueprint 검증 보장
"""
import pytest
import pandas as pd

from src.components._augmenter import PassThroughAugmenter, FeatureStoreAugmenter
from tests.factories.settings_factory import SettingsFactory
from tests.factories.test_data_factory import TestDataFactory


class TestAugmenterBlueprintCompliance:
    """Augmenter의 Blueprint 철학 준수 테스트 - Factory 패턴 적용"""

    @pytest.fixture
    def mock_settings_local(self, test_factories):
        """Local 환경 Settings - SettingsFactory 사용"""
        # SettingsFactory로 완전한 Settings 생성 (파일 의존성 제거)
        settings_dict = test_factories['settings'].create_local_settings()
        return test_factories['mocks'].get_settings(settings_dict)

    @pytest.fixture
    def sample_entity_data(self):
        """BLUEPRINT Augmenter 데이터 계약: entity + timestamp - TestDataFactory 사용"""
        # 기존 패턴 유지 (event_ts 컬럼명)
        return TestDataFactory.create_minimal_entity_data(entity_count=3)

    def test_pass_through_augmenter_initialization(self, mock_settings_local):
        """PassThroughAugmenter 초기화 - BLUEPRINT 원칙 2 (Local 환경)"""
        # 실제 API: PassThroughAugmenter는 인자 없이 초기화
        augmenter = PassThroughAugmenter()
        
        assert augmenter is not None
        # BLUEPRINT: Augmenter는 PIT 조인 인터페이스를 가져야 함
        assert hasattr(augmenter, 'augment')
        assert callable(getattr(augmenter, 'augment'))

    def test_pass_through_augmenter_blueprint_contract(self, mock_settings_local, sample_entity_data):
        """PassThroughAugmenter 데이터 계약 - BLUEPRINT PIT 조인 규격"""
        augmenter = PassThroughAugmenter()
        
        # BLUEPRINT: PassThrough는 입력 그대로 반환 (로컬 환경 편의성)
        result = augmenter.augment(sample_entity_data, run_mode="train")
        
        assert isinstance(result, pd.DataFrame)
        # BLUEPRINT: 입력과 동일한 스키마 유지
        assert len(result) == len(sample_entity_data)
        assert list(result.columns) == list(sample_entity_data.columns)
        # BLUEPRINT: 데이터 계약 - entity + timestamp 보존
        assert 'user_id' in result.columns
        assert 'event_ts' in result.columns

    @pytest.mark.unit
    def test_pass_through_augmenter_run_mode_agnostic(self, mock_settings_local, sample_entity_data):
        """PassThroughAugmenter 실행 모드 무관성 - BLUEPRINT 원칙"""
        augmenter = PassThroughAugmenter()
        
        # BLUEPRINT: PassThrough는 train/batch/serving 모드와 무관하게 동일 동작
        train_result = augmenter.augment(sample_entity_data, run_mode="train")
        batch_result = augmenter.augment(sample_entity_data, run_mode="batch")
        serving_result = augmenter.augment(sample_entity_data, run_mode="serving")
        
        # 모든 모드에서 동일한 결과
        pd.testing.assert_frame_equal(train_result, batch_result)
        pd.testing.assert_frame_equal(batch_result, serving_result)

    def test_feature_store_augmenter_interface_compliance(self, mock_settings_local):
        """FeatureStoreAugmenter 인터페이스 준수 - BLUEPRINT 호환성"""
        # FeatureStoreAugmenter는 실제 Feature Store 없이도 인터페이스 검증 가능
        
        # BLUEPRINT: 모든 Augmenter는 동일한 인터페이스를 가져야 함
        assert hasattr(FeatureStoreAugmenter, '__init__')
        
        # init signature 확인 (settings 인자 필요)
        import inspect
        init_sig = inspect.signature(FeatureStoreAugmenter.__init__)
        assert 'settings' in init_sig.parameters

    def test_augmenter_error_handling_invalid_run_mode(self, mock_settings_local, sample_entity_data):
        """Augmenter 오류 처리 - 잘못된 run_mode"""
        augmenter = PassThroughAugmenter()
        
        # BLUEPRINT: 명확한 오류 메시지로 디버깅 지원
        with pytest.raises(ValueError, match="run_mode"):
            augmenter.augment(sample_entity_data, run_mode="invalid_mode")

    @pytest.mark.blueprint_principle_2
    def test_augmenter_environment_driven_behavior(self, mock_settings_local):
        """Augmenter 환경별 차등 동작 - BLUEPRINT 원칙 2"""
        augmenter = PassThroughAugmenter()
        
        # BLUEPRINT 원칙 2: Local 환경에서는 PassThrough 동작
        # 실제 Feature Store 연결 없이도 동작해야 함
        sample_data = TestDataFactory.create_minimal_entity_data(entity_count=1)
        
        result = augmenter.augment(sample_data, run_mode="train")
        assert result is not None
        assert len(result) == 1

    def test_augmenter_data_contract_preservation(self, mock_settings_local):
        """Augmenter 데이터 계약 보존 - BLUEPRINT 핵심 계약"""
        augmenter = PassThroughAugmenter()
        
        # BLUEPRINT: entity + timestamp는 반드시 보존되어야 함
        input_data = TestDataFactory.create_minimal_entity_data(entity_count=2)
        
        result = augmenter.augment(input_data, run_mode="train")
        
        # 핵심 계약: entity와 timestamp 컬럼 보존
        assert 'user_id' in result.columns
        assert 'event_ts' in result.columns
        # Left Join 방식: 입력 행 수 유지
        assert len(result) == len(input_data)