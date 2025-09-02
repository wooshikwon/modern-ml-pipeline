"""SettingsFactory 검증 테스트"""
import pytest

from tests.factories.settings_factory import SettingsFactory


@pytest.mark.core
@pytest.mark.unit
class TestSettingsFactory:
    """SettingsFactory 기능 검증"""
    
    def test_create_base_settings(self):
        """기본 설정 생성 검증"""
        settings = SettingsFactory.create_base_settings()
        
        # 필수 섹션 존재 확인
        assert 'environment' in settings
        assert 'mlflow' in settings
        assert 'serving' in settings
        assert 'artifact_stores' in settings
        
        # 기본값 확인
        assert settings['environment']['env_name'] == 'test'
        assert 'localhost:5002' in settings['mlflow']['tracking_uri']
    
    def test_create_classification_settings(self):
        """분류 설정 생성 검증"""
        settings = SettingsFactory.create_classification_settings()
        
        # Recipe 존재 확인
        assert 'recipe' in settings
        assert 'model' in settings['recipe']
        
        model = settings['recipe']['model']
        
        # 모델 클래스 경로
        assert 'sklearn.ensemble.RandomForestClassifier' in model['class_path']
        
        # 하이퍼파라미터
        assert 'hyperparameters' in model
        assert model['hyperparameters']['random_state'] == 42
        assert model['hyperparameters']['n_estimators'] == 50
        
        # Entity 스키마
        entity_schema = model['loader']['entity_schema']
        assert entity_schema['entity_columns'] == ['user_id']
        assert entity_schema['timestamp_column'] == 'event_timestamp'
        
        # 데이터 인터페이스
        assert model['data_interface']['task_type'] == 'classification'
        assert model['data_interface']['target_column'] == 'target'
        
        # Augmenter
        assert model['augmenter']['type'] == 'pass_through'
    
    def test_create_regression_settings(self):
        """회귀 설정 생성 검증"""
        settings = SettingsFactory.create_regression_settings()
        
        model = settings['recipe']['model']
        
        # 모델 클래스 경로
        assert 'sklearn.linear_model.LinearRegression' in model['class_path']
        
        # 데이터 인터페이스
        assert model['data_interface']['task_type'] == 'regression'
        assert model['data_interface']['target_column'] == 'target'
    
    def test_create_local_settings(self):
        """로컬 환경 설정 검증"""
        settings = SettingsFactory.create_local_settings()
        
        assert settings['environment']['env_name'] == 'local'
        assert 'Campaign-Uplift-Modeling' in settings['mlflow']['experiment_name']
    
    def test_create_dev_settings(self):
        """개발 환경 설정 검증"""
        settings = SettingsFactory.create_dev_settings()
        
        assert settings['environment']['env_name'] == 'dev'
        assert 'feature_store' in settings
        assert settings['feature_store']['provider'] == 'feast'
    
    def test_create_minimal_settings(self):
        """최소 설정 생성 검증"""
        settings = SettingsFactory.create_minimal_settings("classification")
        
        # 핵심 필드만 존재
        assert 'environment' in settings
        assert 'recipe' in settings
        
        model = settings['recipe']['model']
        assert model['data_interface']['task_type'] == 'classification'
        assert model['hyperparameters']['random_state'] == 42
    
    def test_settings_customization_with_overrides(self):
        """설정 커스터마이징 (overrides) 검증"""
        custom_overrides = {
            'environment': {'env_name': 'custom_test'},
            'recipe': {
                'model': {
                    'hyperparameters': {'n_estimators': 100}
                }
            }
        }
        
        settings = SettingsFactory.create_classification_settings(**custom_overrides)
        
        # Override된 값 확인
        assert settings['environment']['env_name'] == 'custom_test'
        assert settings['recipe']['model']['hyperparameters']['n_estimators'] == 100
        
        # 기본값은 유지
        assert settings['recipe']['model']['hyperparameters']['random_state'] == 42
    
    def test_deep_merge_functionality(self):
        """딕셔너리 깊은 병합 기능 검증"""
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        update = {'a': {'b': 10}, 'e': 4}
        
        result = SettingsFactory._deep_merge(base, update)
        
        # 깊은 병합 확인
        assert result['a']['b'] == 10  # 업데이트됨
        assert result['a']['c'] == 2   # 유지됨
        assert result['d'] == 3        # 유지됨
        assert result['e'] == 4        # 추가됨
    
    def test_feature_store_settings(self):
        """Feature Store 설정 검증"""
        settings = SettingsFactory.create_feature_store_settings("feast")
        
        assert settings['environment']['env_name'] == 'dev'
        assert settings['feature_store']['provider'] == 'feast'
        
        # Feature Store Augmenter 설정
        model = settings['recipe']['model']
        assert model['augmenter']['type'] == 'feature_store'
        assert model['augmenter']['provider'] == 'feast'
    
    def test_standard_helpers(self):
        """표준 헬퍼 메서드 검증"""
        entity_schema = SettingsFactory.get_standard_entity_schema()
        
        assert entity_schema['entity_columns'] == ['user_id']
        assert entity_schema['timestamp_column'] == 'event_timestamp'
        
        # Preprocessor 설정
        preprocessor_config = SettingsFactory.get_standard_preprocessor_config("classification")
        
        assert preprocessor_config['name'] == 'simple_scaler'
        assert 'exclude_cols' in preprocessor_config['params']
        
        expected_exclude = ['user_id', 'event_timestamp', 'target', 'approved', 'label']
        assert set(preprocessor_config['params']['exclude_cols']) == set(expected_exclude)
    
    @pytest.mark.extended
    def test_settings_environment_variations(self):
        """다양한 환경 설정 생성 검증"""
        environments = ['local', 'dev', 'test', 'prod']
        
        for env in environments:
            settings = SettingsFactory.create_classification_settings(env=env)
            
            assert settings['recipe']['name'] == f'test_classification_{env}'
            assert 'environment' in settings
            
            # 각 환경별로 고유한 설정이 있는지 확인
            assert isinstance(settings, dict)
            assert len(settings) > 0