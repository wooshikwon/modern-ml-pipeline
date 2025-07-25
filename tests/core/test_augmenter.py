"""
Augmenter 컴포넌트 테스트 (Blueprint v17.0 현대화)

Blueprint 원칙 검증:
- 원칙 5: 단일 Augmenter, 컨텍스트 주입
- 원칙 9: 환경별 차등적 기능 분리 (LOCAL vs DEV)
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.settings import Settings
from src.core.augmenter import Augmenter, PassThroughAugmenter
from src.core.factory import Factory

class TestAugmenterModernized:
    """Augmenter 컴포넌트 단위 테스트 (Blueprint v17.0, 완전 현대화)"""

    def test_passthrough_augmenter_initialization(self, local_test_settings: Settings):
        """PassThroughAugmenter가 settings 객체로 올바르게 초기화되는지 테스트"""
        augmenter = PassThroughAugmenter(settings=local_test_settings)
        assert augmenter.settings == local_test_settings

    def test_passthrough_augmenter_returns_input_unchanged(self, local_test_settings: Settings):
        """PassThroughAugmenter가 입력을 그대로 반환하는지 테스트"""
        passthrough = PassThroughAugmenter(settings=local_test_settings)
        sample_df = pd.DataFrame({'a': [1], 'b': [2]})
        
        augmented_df = passthrough.augment(sample_df)
        
        pd.testing.assert_frame_equal(sample_df, augmented_df)

    def test_feature_store_augmenter_initialization(self, dev_test_settings: Settings):
        """(DEV) Augmenter가 settings와 factory로 올바르게 초기화되는지 테스트"""
        factory = Factory(dev_test_settings)
        with patch.object(factory, 'create_feature_store_adapter') as mock_create_fs_adapter:
            augmenter = Augmenter(settings=dev_test_settings, factory=factory)
            assert augmenter.settings == dev_test_settings
            assert augmenter.factory == factory
            mock_create_fs_adapter.assert_called_once()

    # 🆕 Blueprint v17.0: 환경별 Factory 동작 검증 (핵심 추가)
    def test_factory_creates_passthrough_augmenter_in_local_env(self, local_test_settings: Settings):
        """
        Factory가 LOCAL 환경에서 PassThroughAugmenter를 생성하는지 검증한다.
        Blueprint 원칙 9: 환경별 차등적 기능 분리
        """
        factory = Factory(local_test_settings)
        augmenter = factory.create_augmenter()
        
        assert isinstance(augmenter, PassThroughAugmenter), \
            "LOCAL 환경에서 Factory는 PassThroughAugmenter를 생성해야 합니다."
        assert augmenter.settings == local_test_settings

    def test_factory_creates_feature_store_augmenter_in_dev_env(self, dev_test_settings: Settings):
        """
        Factory가 DEV 환경에서 FeatureStore 연동 Augmenter를 생성하는지 검증한다.
        Blueprint 원칙 9: 환경별 차등적 기능 분리
        """
        factory = Factory(dev_test_settings)
        with patch.object(factory, 'create_feature_store_adapter') as mock_create_fs_adapter:
            augmenter = factory.create_augmenter()
            
            assert isinstance(augmenter, Augmenter), \
                "DEV 환경에서 Factory는 Augmenter를 생성해야 합니다."
            assert not isinstance(augmenter, PassThroughAugmenter), \
                "DEV 환경에서는 PassThroughAugmenter가 생성되어서는 안 됩니다."
            assert augmenter.settings == dev_test_settings
            mock_create_fs_adapter.assert_called_once()

    @patch('src.utils.adapters.feature_store_adapter.FeatureStoreAdapter')
    def test_augmenter_uses_feature_store_adapter_in_dev(self, MockFeatureStoreAdapter, dev_test_settings: Settings):
        """DEV 환경에서 Augmenter가 FeatureStoreAdapter를 사용하는지 테스트"""
        mock_fs_adapter_instance = MockFeatureStoreAdapter.return_value
        
        # Factory가 Mock Adapter를 반환하도록 설정
        factory = Factory(dev_test_settings)
        factory.create_feature_store_adapter = Mock(return_value=mock_fs_adapter_instance)

        augmenter = Augmenter(settings=dev_test_settings, factory=factory)
        
        # augment 메서드 호출 시 adapter의 read가 호출되는지 검증
        sample_df = pd.DataFrame({'user_id': ['u1']})
        augmenter.augment(sample_df, run_mode="batch")
        
        mock_fs_adapter_instance.read.assert_called_once_with(
            model_input=sample_df,
            run_mode='batch'
        )

    # 🆕 Blueprint v17.0: 컨텍스트 주입 테스트 강화
    def test_augmenter_context_injection_batch_vs_serving(self, dev_test_settings: Settings):
        """
        단일 Augmenter가 run_mode 컨텍스트에 따라 다른 동작을 하는지 검증한다.
        Blueprint 원칙 5: 단일 Augmenter, 컨텍스트 주입
        """
        with patch('src.utils.adapters.feature_store_adapter.FeatureStoreAdapter') as MockFS:
            mock_fs_adapter = MockFS.return_value
            
            factory = Factory(dev_test_settings)
            factory.create_feature_store_adapter = Mock(return_value=mock_fs_adapter)
            
            augmenter = Augmenter(settings=dev_test_settings, factory=factory)
            sample_df = pd.DataFrame({'user_id': ['u1']})
            
            # 1. Batch 모드 테스트
            augmenter.augment(sample_df, run_mode="batch")
            mock_fs_adapter.read.assert_called_with(
                model_input=sample_df,
                run_mode='batch'
            )
            
            # 2. Serving 모드 테스트  
            mock_fs_adapter.reset_mock()
            augmenter.augment(sample_df, run_mode="serving")
            mock_fs_adapter.read.assert_called_with(
                model_input=sample_df,
                run_mode='serving'
            ) 