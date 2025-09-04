"""Trainer 컴포넌트 종합 테스트 - Blueprint 원칙 기반 TDD 구현

이 테스트는 BLUEPRINT.md의 핵심 설계 철학을 검증합니다:
- 원칙 1: 설정과 논리의 분리 (Settings 기반 동작)
- 원칙 2: 환경별 역할 분담 (local 환경 제약)
- 원칙 3: 선언적 파이프라인 (YAML 설정 기반)
- 원칙 4: 모듈화와 확장성 (Factory 패턴 의존성 주입)
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier

from src.components._trainer import Trainer
from src.interface import Basefetcher, BasePreprocessor, BaseEvaluator


@pytest.mark.unit
@pytest.mark.blueprint_principle_1
@pytest.mark.blueprint_principle_4
class TestTrainerBlueprintCompliance:
    """Trainer Blueprint 원칙 준수 테스트 - Factory 패턴 적용"""

    @pytest.fixture
    def classification_settings(self, test_factories):
        """분류 작업용 Settings 객체 - Factory 패턴 적용"""
        settings_dict = test_factories['settings'].create_classification_settings("local")
        from src.settings import Settings
        return Settings(**settings_dict)

    @pytest.fixture
    def mock_factory_provider(self):
        """Factory provider mock - 의존성 주입 패턴 테스트"""
        def provider():
            factory = Mock()
            
            # Mock 모델 생성
            model = Mock(spec=RandomForestClassifier)
            model.fit = Mock()
            model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
            model.predict_proba = Mock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]]))
            model.set_params = Mock()
            factory.create_model.return_value = model
            
            # Mock 전처리기 생성
            preprocessor = Mock(spec=BasePreprocessor)
            preprocessor.fit = Mock()
            preprocessor.transform = Mock(side_effect=lambda x: x)  # 패스스루
            factory.create_preprocessor.return_value = preprocessor
            
            # Mock 평가기 생성
            evaluator = Mock(spec=BaseEvaluator)
            evaluator.evaluate = Mock(return_value={
                'accuracy': 0.85,
                'precision_weighted': 0.84,
                'recall_weighted': 0.85,
                'f1_weighted': 0.84
            })
            factory.create_evaluator.return_value = evaluator
            
            return factory
        return provider

    @pytest.fixture
    def sample_training_data(self, test_factories):
        """테스트용 학습 데이터 생성 - Factory 패턴 적용"""
        return test_factories['data'].create_classification_data(n_samples=100)

    @pytest.fixture
    def mock_components(self):
        """모든 컴포넌트 Mock 객체"""
        fetcher = Mock(spec=Basefetcher)
        fetcher.fetch = Mock(side_effect=lambda df, **kwargs: df)  # 패스스루
        
        preprocessor = Mock(spec=BasePreprocessor)
        preprocessor.fit = Mock()
        preprocessor.transform = Mock(side_effect=lambda x: x)
        
        evaluator = Mock(spec=BaseEvaluator)
        evaluator.evaluate = Mock(return_value={
            'accuracy': 0.85,
            'precision_weighted': 0.84,
            'recall_weighted': 0.85,
            'f1_weighted': 0.84
        })
        
        model = Mock(spec=RandomForestClassifier)
        model.fit = Mock()
        
        return {
            'fetcher': fetcher,
            'preprocessor': preprocessor, 
            'evaluator': evaluator,
            'model': model
        }

    def test_trainer_initialization_follows_blueprint_principles(self, classification_settings, mock_factory_provider):
        """Trainer 초기화가 Blueprint 원칙을 따르는지 검증"""
        # Given: Settings와 Factory Provider가 주어졌을 때
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # Then: Blueprint 원칙에 맞는 초기화
        assert trainer is not None
        assert trainer.settings == classification_settings  # 원칙 1: 설정 기반 동작
        assert trainer.factory_provider == mock_factory_provider  # 원칙 4: 의존성 주입
        assert trainer.training_results == {}  # 초기 상태
        
    def test_trainer_factory_dependency_injection(self, classification_settings):
        """Factory 의존성 주입 패턴 검증 - Blueprint 원칙 4"""
        # Given: Factory provider 없이 초기화
        trainer = Trainer(settings=classification_settings, factory_provider=None)
        
        # When: Factory가 필요한 작업 수행 시도
        # Then: 명확한 에러 메시지와 함께 실패
        with pytest.raises(RuntimeError, match="Factory provider가 주입되지 않았습니다"):
            trainer._get_factory()

    def test_trainer_train_method_blueprint_contract(self, classification_settings, mock_factory_provider, sample_training_data, mock_components):
        """Trainer.train() 메서드가 Blueprint 계약을 준수하는지 검증"""
        # Given: 모든 필수 컴포넌트가 준비됨
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # When: 학습 수행
        trained_model, fitted_preprocessor, metrics, training_results = trainer.train(
            df=sample_training_data,
            model=mock_components['model'],
            fetcher=mock_components['fetcher'],
            preprocessor=mock_components['preprocessor'],
            evaluator=mock_components['evaluator']
        )
        
        # Then: Blueprint 계약 준수
        assert trained_model is not None  # 학습된 모델 반환
        assert fitted_preprocessor is not None  # 피팅된 전처리기 반환
        assert isinstance(metrics, dict)  # 평가 지표 반환
        assert isinstance(training_results, dict)  # 학습 결과 메타데이터 반환
        
        # 필수 메타데이터 존재 검증
        assert 'evaluation_metrics' in training_results
        assert 'training_methodology' in training_results
        assert 'hyperparameter_optimization' in training_results

    def test_trainer_hyperparameter_tuning_disabled_behavior(self, classification_settings, mock_factory_provider, sample_training_data, mock_components):
        """하이퍼파라미터 튜닝 비활성화 시 동작 검증 - Blueprint 원칙 2 (환경별 제약)"""
        # Given: 하이퍼파라미터 튜닝이 비활성화된 설정
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # When: 학습 수행
        _, _, _, training_results = trainer.train(
            df=sample_training_data,
            model=mock_components['model'],
            fetcher=mock_components['fetcher'], 
            preprocessor=mock_components['preprocessor'],
            evaluator=mock_components['evaluator']
        )
        
        # Then: 하이퍼파라미터 튜닝이 건너뛰어짐
        assert training_results['hyperparameter_optimization']['enabled'] is False
        mock_components['model'].fit.assert_called_once()  # 직접 학습 호출 확인

    def test_trainer_data_split_methodology(self, classification_settings, mock_factory_provider, sample_training_data, mock_components):
        """데이터 분할 방법론 검증 - Blueprint 일관성 원칙"""
        # Given: 충분한 크기의 데이터셋
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # When: 학습 수행
        with patch('src.components._trainer._modules.data_handler.split_data') as mock_split:
            # 80:20 분할 시뮬레이션
            train_size = int(len(sample_training_data) * 0.8)
            mock_split.return_value = (
                sample_training_data.iloc[:train_size],
                sample_training_data.iloc[train_size:]
            )
            
            _, _, _, training_results = trainer.train(
                df=sample_training_data,
                model=mock_components['model'],
                fetcher=mock_components['fetcher'],
                preprocessor=mock_components['preprocessor'],
                evaluator=mock_components['evaluator']
            )
        
        # Then: 분할 방법론 메타데이터 기록
        methodology = training_results['training_methodology']
        assert methodology['train_test_split_method'] == 'stratified'
        assert methodology['train_ratio'] == 0.8
        assert methodology['preprocessing_fit_scope'] == 'train_only'

    def test_trainer_error_handling_invalid_model(self, classification_settings, mock_factory_provider, sample_training_data, mock_components):
        """잘못된 모델 객체에 대한 에러 처리 검증"""
        # Given: 잘못된 모델 객체
        invalid_model = "not_a_model"  # 문자열은 fit 메서드가 없음
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # When & Then: 명확한 에러 메시지와 함께 실패
        with pytest.raises(TypeError, match="BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다"):
            trainer._fit_model(invalid_model, None, None, None)

    def test_trainer_task_type_specific_training(self, classification_settings, mock_factory_provider):
        """Task type별 모델 학습 방식 검증"""
        trainer = Trainer(settings=classification_settings, factory_provider=mock_factory_provider)
        
        # Mock 데이터
        X, y = pd.DataFrame({'feature': [1, 2, 3]}), pd.Series([0, 1, 0])
        additional_data = {'treatment': pd.Series([1, 0, 1])}
        
        # Classification 모델 테스트 (scikit-learn 호환성 확보)
        clf_model = Mock()
        clf_model.fit = Mock()
        # scikit-learn tags 인터페이스 Mock
        clf_model.__sklearn_tags__ = Mock(return_value=Mock(estimator_type="classifier"))
        trainer._fit_model(clf_model, X, y, additional_data)
        clf_model.fit.assert_called_once_with(X, y)
        
        # Clustering 모델 테스트 (y 불필요, scikit-learn 호환성 확보)
        cluster_model = Mock()
        cluster_model.fit = Mock()
        # scikit-learn tags 인터페이스 Mock
        cluster_model.__sklearn_tags__ = Mock(return_value=Mock(estimator_type="clusterer"))
        # 클러스터링용 설정으로 일시 변경
        original_task = classification_settings.recipe.model.data_interface.task_type
        classification_settings.recipe.model.data_interface.task_type = "clustering"
        trainer._fit_model(cluster_model, X, y, additional_data)
        cluster_model.fit.assert_called_once_with(X)
        # 원복
        classification_settings.recipe.model.data_interface.task_type = original_task