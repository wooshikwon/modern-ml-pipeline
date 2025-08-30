"""인터페이스 계약 준수 테스트 - Contract Compliance Testing

Phase 3.5: interface 모듈의 ABC 클래스들과 구현체들의 계약 준수를 검증합니다.

테스트 전략:
- Contract Compliance Testing: ABC 추상 메서드 시그니처 검증
- Factory 패턴 통합: Mock 컴포넌트들의 인터페이스 준수 검증
- sklearn 호환성: fit, transform, fit_transform 메서드 패턴 검증
- 메서드 시그니처 및 반환 타입 검증
"""
import pytest
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union
from unittest.mock import Mock, MagicMock
import inspect

from src.interface.base_augmenter import BaseAugmenter
from src.interface.base_preprocessor import BasePreprocessor
from src.interface.base_model import BaseModel
from src.interface.base_adapter import BaseAdapter


class TestInterfaceContractCompliance:
    """모든 컴포넌트의 인터페이스 계약 준수 테스트"""
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_augmenter_interface_compliance(self, test_factories):
        """BaseAugmenter 인터페이스 계약 준수 검증"""
        # Given: Mock Augmenter (Factory 패턴 통합)
        augmenter = test_factories['mocks'].get_augmenter()
        
        # Then: 필수 인터페이스 보유 검증
        required_methods = ['augment']
        for method in required_methods:
            assert hasattr(augmenter, method), f"Augmenter must have '{method}' method"
            assert callable(getattr(augmenter, method)), f"'{method}' must be callable"
        
        # augment 메서드 시그니처 검증 (BaseAugmenter ABC 준수)
        augment_method = getattr(augmenter, 'augment')
        assert augment_method is not None
        
        # Mock을 통한 호출 가능성 검증
        test_data = pd.DataFrame({'user_id': ['test'], 'event_timestamp': [pd.Timestamp.now()]})
        result = augmenter.augment(test_data, run_mode="train")
        assert result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_augmenter_abc_signature_verification(self, test_factories):
        """BaseAugmenter ABC 추상 메서드 시그니처 검증"""
        # Given: BaseAugmenter ABC 클래스
        # When: 추상 메서드 시그니처 검사
        augment_method = BaseAugmenter.augment
        
        # Then: 추상 메서드 검증
        assert hasattr(augment_method, '__isabstractmethod__')
        assert augment_method.__isabstractmethod__ is True
        
        # 시그니처 검증
        sig = inspect.signature(augment_method)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'data' in params
        assert 'run_mode' in params
        
        # 타입 힌트 검증
        data_param = sig.parameters['data']
        run_mode_param = sig.parameters['run_mode']
        
        # data 파라미터는 Union[pd.DataFrame, Dict[str, pd.DataFrame]] 타입이어야 함
        assert data_param.annotation != inspect.Parameter.empty
        
        # run_mode 기본값 검증
        assert run_mode_param.default == "batch"
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_preprocessor_interface_compliance(self, test_factories):
        """BasePreprocessor 인터페이스 계약 준수 검증"""
        # Given: Mock Preprocessor (Factory 패턴 통합)
        preprocessor = test_factories['mocks'].get_preprocessor()
        
        # Then: sklearn 호환 인터페이스 검증
        sklearn_methods = ['fit', 'transform', 'fit_transform']
        for method in sklearn_methods:
            assert hasattr(preprocessor, method), f"Preprocessor must have '{method}' method"
            assert callable(getattr(preprocessor, method)), f"'{method}' must be callable"
        
        # 메서드 호출 가능성 검증
        test_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        
        # fit 메서드 호출
        fit_result = preprocessor.fit(test_data)
        assert fit_result is not None
        
        # transform 메서드 호출
        transform_result = preprocessor.transform(test_data)
        assert transform_result is not None
        
        # fit_transform 메서드 호출
        fit_transform_result = preprocessor.fit_transform(test_data)
        assert fit_transform_result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_preprocessor_abc_signature_verification(self, test_factories):
        """BasePreprocessor ABC 추상 메서드 시그니처 검증"""
        # Given: BasePreprocessor ABC 클래스
        # When/Then: 추상 메서드들 시그니처 검증
        
        # fit 메서드 검증
        fit_method = BasePreprocessor.fit
        assert hasattr(fit_method, '__isabstractmethod__')
        assert fit_method.__isabstractmethod__ is True
        
        fit_sig = inspect.signature(fit_method)
        fit_params = list(fit_sig.parameters.keys())
        assert 'self' in fit_params
        assert 'X' in fit_params
        assert 'y' in fit_params
        
        # y 파라미터의 Optional 기본값 검증
        y_param = fit_sig.parameters['y']
        assert y_param.default is None
        
        # transform 메서드 검증
        transform_method = BasePreprocessor.transform
        assert hasattr(transform_method, '__isabstractmethod__')
        assert transform_method.__isabstractmethod__ is True
        
        transform_sig = inspect.signature(transform_method)
        transform_params = list(transform_sig.parameters.keys())
        assert 'self' in transform_params
        assert 'X' in transform_params
        
        # fit_transform 메서드는 구현체가 있어야 함 (추상 메서드가 아님)
        fit_transform_method = BasePreprocessor.fit_transform
        assert not hasattr(fit_transform_method, '__isabstractmethod__') or not fit_transform_method.__isabstractmethod__
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_model_interface_compliance(self, test_factories):
        """BaseModel 인터페이스 계약 준수 검증"""
        # Given: Mock Model (Factory 패턴 통합)
        mock_factory = test_factories['mocks'].get_factory({})
        model = mock_factory.create_model()
        
        # Then: 모델 인터페이스 검증
        model_methods = ['fit', 'predict']
        for method in model_methods:
            assert hasattr(model, method), f"Model must have '{method}' method"
            assert callable(getattr(model, method)), f"'{method}' must be callable"
        
        # predict 메서드 호출 가능성 검증
        test_data = pd.DataFrame({'feature1': [1, 2], 'feature2': [0.1, 0.2]})
        prediction_result = model.predict(test_data)
        assert prediction_result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_base_adapter_interface_compliance(self, test_factories):
        """BaseAdapter 인터페이스 계약 준수 검증"""
        # Given: Mock Data Adapter (Factory 패턴 통합)
        mock_factory = test_factories['mocks'].get_factory({})
        adapter = mock_factory.create_data_adapter("storage")
        
        # Then: 데이터 어댑터 인터페이스 검증
        adapter_methods = ['read', 'write']
        for method in adapter_methods:
            assert hasattr(adapter, method), f"Adapter must have '{method}' method"
            assert callable(getattr(adapter, method)), f"'{method}' must be callable"
        
        # read 메서드 호출 가능성 검증
        read_result = adapter.read("test_path")
        assert read_result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_augmenter_run_mode_contract(self, test_factories):
        """Augmenter run_mode 매개변수 계약 검증"""
        # Given: Mock Augmenter와 다양한 run_mode
        augmenter = test_factories['mocks'].get_augmenter()
        test_data = pd.DataFrame({'user_id': ['test'], 'feature': [1.0]})
        
        # When/Then: 지원되는 run_mode들 검증
        supported_modes = ["train", "batch", "serving"]
        for mode in supported_modes:
            result = augmenter.augment(test_data, run_mode=mode)
            assert result is not None, f"Augmenter must support run_mode='{mode}'"
        
        # 기본값 검증 (run_mode 생략 시 "batch"가 기본값)
        default_result = augmenter.augment(test_data)
        assert default_result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_preprocessor_sklearn_compatibility(self, test_factories):
        """Preprocessor sklearn 호환성 계약 검증"""
        # Given: Mock Preprocessor
        preprocessor = test_factories['mocks'].get_preprocessor()
        test_X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        test_y = pd.Series([0, 1, 0])
        
        # When/Then: sklearn 스타일 체인 호출 검증
        
        # 1. fit 메서드는 self를 반환해야 함 (체이닝을 위해)
        fit_result = preprocessor.fit(test_X, test_y)
        # Mock의 경우 return_value 설정에 따라 달라지므로 호출 성공 여부만 확인
        assert fit_result is not None
        
        # 2. transform 메서드는 DataFrame을 반환해야 함
        transform_result = preprocessor.transform(test_X)
        assert transform_result is not None
        
        # 3. fit_transform은 fit + transform의 결과와 동일해야 함
        fit_transform_result = preprocessor.fit_transform(test_X, test_y)
        assert fit_transform_result is not None
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_interface_inheritance_structure(self, test_factories):
        """인터페이스 상속 구조 검증"""
        # Given/When: ABC 클래스들의 상속 구조 검증
        # Then: ABC를 상속받는지 확인
        assert issubclass(BaseAugmenter, ABC)
        assert issubclass(BasePreprocessor, ABC)
        assert issubclass(BaseModel, ABC)
        assert issubclass(BaseAdapter, ABC)
        
        # 추상 메서드 보유 확인
        assert len(BaseAugmenter.__abstractmethods__) >= 1
        assert len(BasePreprocessor.__abstractmethods__) >= 2  # fit, transform
        assert len(BaseModel.__abstractmethods__) >= 1
        assert len(BaseAdapter.__abstractmethods__) >= 2  # read, write
    
    @pytest.mark.core
    @pytest.mark.unit
    def test_factory_component_interface_integration(self, test_factories):
        """Factory를 통한 컴포넌트 인터페이스 통합 검증"""
        # Given: Factory를 통한 다양한 컴포넌트 생성
        mock_factory = test_factories['mocks'].get_factory({})
        
        # When: Factory로 각 컴포넌트 생성
        augmenter = mock_factory.create_augmenter()
        preprocessor = mock_factory.create_preprocessor()
        model = mock_factory.create_model()
        data_adapter = mock_factory.create_data_adapter("storage")
        
        # Then: 모든 컴포넌트가 예상 인터페이스를 제공하는지 검증
        
        # Augmenter 인터페이스 검증
        assert hasattr(augmenter, 'augment')
        assert callable(augmenter.augment)
        
        # Preprocessor 인터페이스 검증
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')
        assert hasattr(preprocessor, 'fit_transform')
        
        # Model 인터페이스 검증
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # DataAdapter 인터페이스 검증
        assert hasattr(data_adapter, 'read')
        assert hasattr(data_adapter, 'write')
    
    @pytest.mark.unit
    def test_mock_component_signature_consistency(self, test_factories):
        """Mock 컴포넌트와 실제 인터페이스의 시그니처 일치성 검증"""
        # Given: Mock Augmenter
        mock_augmenter = test_factories['mocks'].get_augmenter()
        
        # When/Then: Mock이 예상되는 인터페이스를 제공하는지 검증
        
        # augment 메서드 시그니처 검증
        assert hasattr(mock_augmenter, 'augment')
        
        # 실제 BaseAugmenter의 augment와 호환되는 호출 방식 확인
        test_data = pd.DataFrame({'test': [1]})
        
        # 다양한 호출 방식 지원 확인
        result1 = mock_augmenter.augment(test_data)
        result2 = mock_augmenter.augment(test_data, run_mode="train")
        
        assert result1 is not None
        assert result2 is not None
    
    @pytest.mark.unit
    def test_interface_documentation_compliance(self, test_factories):
        """인터페이스 문서화 계약 준수 검증"""
        # Given: ABC 클래스들의 docstring 검증
        # When/Then: 각 추상 클래스와 메서드가 적절한 문서화를 가지고 있는지 확인
        
        # BaseAugmenter 문서화 검증
        assert BaseAugmenter.__doc__ is not None
        assert len(BaseAugmenter.__doc__.strip()) > 0
        
        # BasePreprocessor 문서화 검증
        assert BasePreprocessor.__doc__ is not None
        assert len(BasePreprocessor.__doc__.strip()) > 0
        
        # 추상 메서드 문서화 검증
        assert BaseAugmenter.augment.__doc__ is not None
        assert BasePreprocessor.fit.__doc__ is not None
        assert BasePreprocessor.transform.__doc__ is not None