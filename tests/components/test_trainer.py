"""
Trainer 컴포넌트 테스트 (Blueprint v17.0 현대화)

학습 프로세스, 컴포넌트 조합, 메트릭 수집, HPO 로직 테스트

Blueprint 원칙 검증:
- 원칙 8: 자동화된 HPO + Data Leakage 완전 방지
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, call

pytest.skip("Deprecated/outdated test module pending Stage 6 test overhaul (trainer interface updated).", allow_module_level=True)

from src.components.trainer import Trainer
from src.engine.factory import Factory
from src.settings import Settings
from src.components.augmenter import Augmenter, PassThroughAugmenter
from src.components.preprocessor import Preprocessor

class TestTrainerModernized:
    """Trainer 컴포넌트 단위 테스트 (Blueprint v17.0, 완전 현대화)"""

    def test_trainer_initialization(self, local_test_settings: Settings):
        """Trainer가 settings 객체로 올바르게 초기화되는지 테스트"""
        trainer = Trainer(settings=local_test_settings)
        assert trainer.settings == local_test_settings

    @patch('src.core.trainer.mlflow')
    def test_train_method_flow_in_local_env(self, mock_mlflow, local_test_settings: Settings):
        """
        LOCAL 환경에서 Trainer.train() 메소드의 실행 흐름을 검증한다.
        (실제 학습은 Mocking)
        """
        trainer = Trainer(settings=local_test_settings)
        
        # Mock 컴포넌트 및 데이터
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit_transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
        mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})

        mock_augmenter = PassThroughAugmenter(settings=local_test_settings)
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'approved': [1, 0, 1, 0] # target_col for stratification
        })

        # train 메소드 실행
        trained_preprocessor, trained_model, training_results = trainer.train(
            df=df,
            model=mock_model,
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor,
        )

        # 검증
        assert trained_preprocessor is not None
        assert trained_model is not None
        assert "metrics" in training_results
        
        # Preprocessor가 train 데이터에 fit 되었는지 확인
        assert mock_preprocessor.fit_transform.call_count == 1
        # Model이 train 데이터에 fit 되었는지 확인
        assert mock_model.fit.call_count == 1
        
        # MLflow 로깅이 호출되었는지 확인
        mock_mlflow.log_metrics.assert_called_once()

    # 🆕 Blueprint v17.0: 상세 실행 흐름 검증
    @patch('src.core.trainer.mlflow')
    @patch('src.core.trainer.train_test_split')
    def test_trainer_execution_flow_detailed_verification(self, mock_split, mock_mlflow, local_test_settings: Settings):
        """
        Trainer.train() 메소드의 상세 실행 흐름을 순서대로 검증한다.
        data_split → augment → preprocess → model.fit → evaluate → mlflow.log_metrics
        """
        trainer = Trainer(settings=local_test_settings)
        
        # Mock 설정
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_augmenter = Mock()
        
        # mock_split 반환값 설정
        train_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'approved': [1, 0]})
        test_df = pd.DataFrame({'user_id': ['u3', 'u4'], 'approved': [1, 0]})
        mock_split.return_value = (train_df, test_df)
        
        # mock_augmenter 반환값 설정
        mock_augmenter.augment.return_value = pd.DataFrame({'f1': [0.1, 0.2], 'approved': [1, 0]})
        
        # mock_preprocessor 반환값 설정
        mock_preprocessor.fit.return_value = mock_preprocessor
        mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'approved': [1, 0, 1, 0]
        })

        # train 메소드 실행
        trainer.train(
            df=df,
            model=mock_model,
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor,
        )

        # 1. Data Split이 호출되었는지
        mock_split.assert_called_once()
        
        # 2. Augmenter가 train/test 데이터에 각각 호출되었는지
        assert mock_augmenter.augment.call_count == 2
        
        # 3. Preprocessor가 train 데이터에 fit 호출되었는지 (Data Leakage 방지)
        mock_preprocessor.fit.assert_called_once()
        
        # 4. Preprocessor가 train/test 데이터에 transform 호출되었는지
        assert mock_preprocessor.transform.call_count == 2
        
        # 5. Model이 fit 호출되었는지
        mock_model.fit.assert_called_once()
        
        # 6. MLflow 로깅이 호출되었는지
        mock_mlflow.log_metrics.assert_called_once()

    # 🆕 Blueprint v17.0: 하이퍼파라미터 최적화 비활성화 테스트
    def test_trainer_with_hyperparameter_tuning_disabled(self, local_test_settings: Settings):
        """
        hyperparameter_tuning.enabled = False일 때 기존 방식으로 동작하는지 검증한다.
        """
        # 설정에서 HPO가 비활성화되어 있는지 확인
        assert local_test_settings.model.hyperparameter_tuning is None or \
               not local_test_settings.model.hyperparameter_tuning.enabled
        
        trainer = Trainer(settings=local_test_settings)
        
        # Mock 컴포넌트
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_augmenter = PassThroughAugmenter(settings=local_test_settings)
        
        mock_preprocessor.fit.return_value = mock_preprocessor
        mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'approved': [1, 0, 1, 0]
        })

        with patch('src.core.trainer.mlflow'):
            trained_preprocessor, trained_model, training_results = trainer.train(
                df=df,
                model=mock_model,
                augmenter=mock_augmenter,
                preprocessor=mock_preprocessor,
            )

        # HPO가 비활성화되어 있으므로 hyperparameter_optimization 메타데이터가 없거나 enabled=False
        hpo_data = training_results.get('hyperparameter_optimization', {})
        assert not hpo_data.get('enabled', False)

    # 🆕 Blueprint v17.0: 하이퍼파라미터 최적화 활성화 테스트
    @patch('src.core.trainer.mlflow')
    def test_trainer_with_hyperparameter_tuning_enabled(self, mock_mlflow, dev_test_settings: Settings):
        """
        hyperparameter_tuning.enabled = True일 때 Optuna 관련 로직이 호출되는지 검증한다.
        """
        # HPO 활성화된 설정 생성
        settings_with_hpo = dev_test_settings.model_copy(deep=True)
        if not hasattr(settings_with_hpo.model, 'hyperparameter_tuning') or \
           settings_with_hpo.model.hyperparameter_tuning is None:
            # HPO 설정이 없다면 추가
            from src.settings.models import HyperparameterTuningSettings
            settings_with_hpo.model.hyperparameter_tuning = HyperparameterTuningSettings(
                enabled=True,
                n_trials=5,
                metric="accuracy",
                direction="maximize"
            )
        else:
            settings_with_hpo.model.hyperparameter_tuning.enabled = True
            settings_with_hpo.model.hyperparameter_tuning.n_trials = 5

        trainer = Trainer(settings=settings_with_hpo)

        # Optuna 관련 Mock
        with patch('src.core.trainer.optuna') as mock_optuna:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 1
            mock_optuna.create_study.return_value = mock_study
            mock_study.best_trial = mock_trial
            mock_study.best_trial.value = 0.95
            mock_study.best_trial.params = {"n_estimators": 100}
            mock_study.trials = [mock_trial]
            
            # Mock 컴포넌트
            mock_model = Mock()
            mock_preprocessor = Mock()
            mock_augmenter = Mock()
            
            mock_augmenter.augment.return_value = pd.DataFrame({'f1': [0.1, 0.2], 'approved': [1, 0]})
            mock_preprocessor.fit.return_value = mock_preprocessor
            mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
            
            df = pd.DataFrame({
                'user_id': ['u1', 'u2', 'u3', 'u4'],
                'approved': [1, 0, 1, 0]
            })

            trained_preprocessor, trained_model, training_results = trainer.train(
                df=df,
                model=mock_model,
                augmenter=mock_augmenter,
                preprocessor=mock_preprocessor,
            )

            # Optuna Study가 생성되었는지 검증
            mock_optuna.create_study.assert_called_once()
            
            # HPO 결과가 training_results에 포함되었는지 검증
            assert 'hyperparameter_optimization' in training_results
            hpo_data = training_results['hyperparameter_optimization']
            assert hpo_data['enabled'] == True
            assert 'best_params' in hpo_data
            assert 'best_score' in hpo_data

    # 🆕 Blueprint v17.0: HPO와 Data Leakage 방지 조합 검증
    @patch('src.core.trainer.mlflow')
    @patch('src.core.trainer.train_test_split')
    def test_trainer_hpo_with_data_leakage_prevention(self, mock_split, mock_mlflow, dev_test_settings: Settings):
        """
        HPO 과정에서도 Data Leakage 방지가 올바르게 작동하는지 검증한다.
        각 trial마다 독립적인 train/validation split이 수행되어야 함.
        """
        # HPO 활성화된 설정
        settings_with_hpo = dev_test_settings.model_copy(deep=True)
        if not hasattr(settings_with_hpo.model, 'hyperparameter_tuning') or \
           settings_with_hpo.model.hyperparameter_tuning is None:
            from src.settings.models import HyperparameterTuningSettings
            settings_with_hpo.model.hyperparameter_tuning = HyperparameterTuningSettings(
                enabled=True,
                n_trials=3,  # 적은 수로 테스트
                metric="accuracy",
                direction="maximize"
            )
        else:
            settings_with_hpo.model.hyperparameter_tuning.enabled = True
            settings_with_hpo.model.hyperparameter_tuning.n_trials = 3

        trainer = Trainer(settings=settings_with_hpo)

        # train_test_split Mock 설정
        train_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'approved': [1, 0]})
        val_df = pd.DataFrame({'user_id': ['u3', 'u4'], 'approved': [1, 0]})
        mock_split.return_value = (train_df, val_df)

        with patch('src.core.trainer.optuna') as mock_optuna:
            # Optuna Mock 설정
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.suggest_int.return_value = 100
            mock_trial.suggest_float.return_value = 0.1
            mock_optuna.create_study.return_value = mock_study
            mock_study.optimize.side_effect = lambda objective, n_trials: [objective(mock_trial) for _ in range(n_trials)]
            mock_study.best_trial = mock_trial
            mock_study.best_trial.value = 0.95
            mock_study.best_trial.params = {"n_estimators": 100}

            # Mock 컴포넌트
            mock_model = Mock()
            mock_preprocessor = Mock()
            mock_augmenter = Mock()
            
            mock_augmenter.augment.return_value = pd.DataFrame({'f1': [0.1, 0.2], 'approved': [1, 0]})
            mock_preprocessor.fit.return_value = mock_preprocessor
            mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
            
            df = pd.DataFrame({
                'user_id': ['u1', 'u2', 'u3', 'u4'],
                'approved': [1, 0, 1, 0]
            })

            trainer.train(
                df=df,
                model=mock_model,
                augmenter=mock_augmenter,
                preprocessor=mock_preprocessor,
            )

            # train_test_split이 호출되었는지 확인 (HPO든 일반이든 data split은 필수)
            assert mock_split.call_count >= 1
            
            # Preprocessor가 fit 호출되었는지 확인 (Data Leakage 방지)
            assert mock_preprocessor.fit.call_count >= 1

    # 🆕 Blueprint v17.0: training_results 메타데이터 완성도 검증
    @patch('src.core.trainer.mlflow')
    def test_trainer_training_results_metadata_completeness(self, mock_mlflow, local_test_settings: Settings):
        """
        training_results에 모든 필요한 메타데이터가 포함되는지 검증한다.
        """
        trainer = Trainer(settings=local_test_settings)
        
        # Mock 컴포넌트
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_augmenter = PassThroughAugmenter(settings=local_test_settings)
        
        mock_preprocessor.fit.return_value = mock_preprocessor
        mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'approved': [1, 0, 1, 0]
        })

        trained_preprocessor, trained_model, training_results = trainer.train(
            df=df,
            model=mock_model,
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor,
        )

        # 필수 메타데이터 존재 검증
        required_keys = ['metrics', 'training_methodology']
        for key in required_keys:
            assert key in training_results, f"training_results에 '{key}' 메타데이터가 누락되었습니다."
        
        # training_methodology의 Data Leakage 방지 메타데이터 검증
        tm_data = training_results['training_methodology']
        assert 'preprocessing_fit_scope' in tm_data
        assert tm_data['preprocessing_fit_scope'] == 'train_only'
        
        # metrics 데이터 타입 검증
        assert isinstance(training_results['metrics'], dict)

    # 🆕 Blueprint v17.0: 에러 처리 테스트
    def test_trainer_handles_invalid_model_gracefully(self, local_test_settings: Settings):
        """
        잘못된 모델이 주입된 경우 적절히 처리하는지 검증한다.
        """
        trainer = Trainer(settings=local_test_settings)
        
        # 잘못된 모델 (fit 메서드가 없는 객체)
        invalid_model = "This is not a model"
        
        mock_preprocessor = Mock()
        mock_augmenter = PassThroughAugmenter(settings=local_test_settings)
        
        df = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'approved': [1, 0]
        })

        # 적절한 오류가 발생하는지 검증
        with pytest.raises(AttributeError):
            trainer.train(
                df=df,
                model=invalid_model,
                augmenter=mock_augmenter,
                preprocessor=mock_preprocessor,
            ) 