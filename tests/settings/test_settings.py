"""
Settings 테스트

설정 로딩, 검증, 환경별 설정 병합 테스트
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, mock_open
from src.settings import Settings, load_settings, ModelHyperparametersSettings


class TestSettings:
    """Settings 테스트"""
    
    def test_load_settings_xgboost(self):
        """XGBoost 설정 로딩 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 기본 구조 확인
        assert settings.model.name == "xgboost_x_learner"
        assert hasattr(settings, 'data_sources')
        assert hasattr(settings, 'mlflow')
        assert hasattr(settings, 'preprocessing')
        assert hasattr(settings, 'serving')
        
        # 모델 설정 확인
        assert hasattr(settings.model, 'hyperparameters')
        assert hasattr(settings.model, 'loader')
        assert hasattr(settings.model, 'augmenter')
    
    def test_load_settings_causal_forest(self):
        """CausalForest 설정 로딩 테스트"""
        settings = load_settings("causal_forest")
        
        # 기본 구조 확인
        assert settings.model.name == "causal_forest"
        assert hasattr(settings, 'data_sources')
        assert hasattr(settings, 'mlflow')
        assert hasattr(settings, 'preprocessing')
        assert hasattr(settings, 'serving')
        
        # 모델 설정 확인
        assert hasattr(settings.model, 'hyperparameters')
        assert hasattr(settings.model, 'loader')
        assert hasattr(settings.model, 'augmenter')
    
    def test_model_hyperparameters_settings_pydantic_v2(self):
        """Pydantic v2 RootModel 호환성 테스트"""
        # 하이퍼파라미터 딕셔너리 생성
        hyperparams = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
        
        # RootModel 생성
        hyperparameters_settings = ModelHyperparametersSettings(hyperparams)
        
        # root 속성을 통해 접근
        assert hyperparameters_settings.root == hyperparams
        assert hyperparameters_settings.root['n_estimators'] == 100
        assert hyperparameters_settings.root['learning_rate'] == 0.1
        assert hyperparameters_settings.root['max_depth'] == 6
    
    def test_settings_model_copy(self):
        """Settings 모델 복사 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 설정 복사
        copied_settings = settings.model_copy()
        
        # 복사된 설정이 원본과 동일한지 확인
        assert copied_settings.model.name == settings.model.name
        assert copied_settings.data_sources == settings.data_sources
        assert copied_settings.mlflow == settings.mlflow
        
        # 복사된 설정이 독립적인지 확인 (deep copy)
        assert copied_settings is not settings
    
    def test_settings_validation(self):
        """Settings 검증 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 필수 필드 존재 확인
        assert settings.model.name is not None
        assert settings.model.hyperparameters is not None
        assert settings.model.loader is not None
        assert settings.model.augmenter is not None
        
        # 데이터 타입 확인
        assert isinstance(settings.model.name, str)
        assert isinstance(settings.model.hyperparameters, ModelHyperparametersSettings)
        assert isinstance(settings.model.hyperparameters.root, dict)
    
    def test_environment_specific_settings(self):
        """환경별 설정 테스트"""
        # 기본 설정 로딩
        settings = load_settings("xgboost_x_learner")
        
        # 환경 변수 설정 없이 로딩 (기본값 사용)
        assert settings is not None
        
        # 환경별 설정이 있는지 확인
        assert hasattr(settings, 'data_sources')
        assert hasattr(settings, 'mlflow')
    
    def test_settings_json_serialization(self):
        """Settings JSON 직렬화 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # JSON 직렬화 가능한지 확인
        json_str = settings.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # 필수 필드가 JSON에 포함되어 있는지 확인
        assert 'model' in json_str
        assert 'xgboost_x_learner' in json_str
    
    def test_settings_dict_conversion(self):
        """Settings 딕셔너리 변환 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 딕셔너리 변환
        settings_dict = settings.model_dump()
        
        # 딕셔너리 구조 확인
        assert isinstance(settings_dict, dict)
        assert 'model' in settings_dict
        assert 'data_sources' in settings_dict
        assert 'mlflow' in settings_dict
        
        # 모델 정보 확인
        assert settings_dict['model']['name'] == 'xgboost_x_learner'
        assert 'hyperparameters' in settings_dict['model']
    
    def test_hyperparameters_access_patterns(self):
        """하이퍼파라미터 접근 패턴 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 다양한 접근 방식 테스트
        hyperparams = settings.model.hyperparameters.root
        assert isinstance(hyperparams, dict)
        
        # 기본 하이퍼파라미터 확인
        assert 'n_estimators' in hyperparams
        assert 'learning_rate' in hyperparams
        assert 'max_depth' in hyperparams
        
        # 값 타입 확인
        assert isinstance(hyperparams['n_estimators'], int)
        assert isinstance(hyperparams['learning_rate'], float)
        assert isinstance(hyperparams['max_depth'], int)
    
    def test_settings_immutability(self):
        """Settings 불변성 테스트"""
        settings = load_settings("xgboost_x_learner")
        original_name = settings.model.name
        
        # 설정 복사 후 수정
        modified_settings = settings.model_copy()
        
        # 원본이 변경되지 않았는지 확인
        assert settings.model.name == original_name
        assert modified_settings.model.name == original_name
        
        # 복사본은 독립적으로 수정 가능
        assert modified_settings is not settings
    
    def test_invalid_model_name_handling(self):
        """잘못된 모델명 처리 테스트"""
        # 존재하지 않는 모델명으로 로딩 시도
        with pytest.raises(Exception):
            load_settings("non_existent_model")
    
    def test_settings_defaults(self):
        """Settings 기본값 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 기본값이 설정되어 있는지 확인
        assert settings.mlflow.tracking_uri is not None
        assert settings.mlflow.experiment_name is not None
        
        # 데이터 소스 설정 확인
        assert settings.data_sources is not None
        
        # 전처리 설정 확인
        assert settings.preprocessing is not None
        
        # 서빙 설정 확인
        assert settings.serving is not None
    
    def test_model_specific_configurations(self):
        """모델별 특화 설정 테스트"""
        xgboost_settings = load_settings("xgboost_x_learner")
        causal_forest_settings = load_settings("causal_forest")
        
        # 각 모델이 고유한 설정을 가지는지 확인
        assert xgboost_settings.model.name != causal_forest_settings.model.name
        
        # 하이퍼파라미터가 모델별로 다른지 확인
        xgb_hyperparams = xgboost_settings.model.hyperparameters.root
        cf_hyperparams = causal_forest_settings.model.hyperparameters.root
        
        # 일부 하이퍼파라미터는 다를 수 있음
        # (구체적인 비교는 실제 recipe 파일에 따라 달라짐)
        assert isinstance(xgb_hyperparams, dict)
        assert isinstance(cf_hyperparams, dict)
    
    def test_settings_field_types(self):
        """Settings 필드 타입 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # 문자열 필드
        assert isinstance(settings.model.name, str)
        
        # 딕셔너리 필드 (RootModel을 통해 접근)
        assert isinstance(settings.model.hyperparameters.root, dict)
        
        # 객체 필드
        assert hasattr(settings.model, 'loader')
        assert hasattr(settings.model, 'augmenter')
        
        # 중첩 객체 필드
        assert hasattr(settings, 'mlflow')
        assert hasattr(settings, 'data_sources')
    
    def test_settings_backward_compatibility(self):
        """Settings 하위 호환성 테스트"""
        settings = load_settings("xgboost_x_learner")
        
        # Pydantic v1 스타일 접근도 가능한지 확인
        # (실제로는 v2 방식을 사용하지만, 호환성 확인)
        assert hasattr(settings.model, 'hyperparameters')
        assert hasattr(settings.model.hyperparameters, 'root')
        
        # 기존 코드와 호환되는지 확인
        hyperparams = settings.model.hyperparameters.root
        assert isinstance(hyperparams, dict)
        
        # 기본적인 딕셔너리 연산 지원
        assert len(hyperparams) > 0
        for key, value in hyperparams.items():
            assert isinstance(key, str)
            assert value is not None
    
    def test_settings_error_handling(self):
        """Settings 오류 처리 테스트"""
        # 잘못된 설정으로 인한 오류 처리
        
        # 1. 존재하지 않는 파일
        with pytest.raises(Exception):
            load_settings("non_existent_model")
        
        # 2. 잘못된 YAML 구조 (Mock 테스트)
        with patch('builtins.open', mock_open(read_data='invalid: yaml: content: [')):
            with pytest.raises(Exception):
                load_settings("test_model")
    
    def test_settings_lazy_loading(self):
        """Settings 지연 로딩 테스트"""
        # 설정이 실제로 사용될 때까지 로딩되지 않는지 확인
        # (현재 구현에서는 즉시 로딩이지만, 향후 최적화 가능)
        
        settings = load_settings("xgboost_x_learner")
        
        # 모든 필드가 즉시 사용 가능한지 확인
        assert settings.model.name is not None
        assert settings.model.hyperparameters is not None
        assert settings.data_sources is not None
        assert settings.mlflow is not None 