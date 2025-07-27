from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
from datetime import datetime

from sklearn.model_selection import train_test_split

from src.settings import Settings
from src.utils.system.logger import logger
from src.interface.base_trainer import BaseTrainer
from src.utils.system.schema_utils import validate_schema

if TYPE_CHECKING:
    from src.components.augmenter import BaseAugmenter
    from src.components.preprocessor import BasePreprocessor


class Trainer(BaseTrainer):
    """
    모델 학습 및 평가 전체 과정을 관장하는 클래스.
    현대화된 Recipe 구조 전용 (settings.recipe.model)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Trainer가 초기화되었습니다.")

    def train(self, df, model, augmenter, preprocessor, context_params=None):
        """
        학습 진입점. 하이퍼파라미터 튜닝 활성화 여부에 따라 경로 분기.
        """
        tuning_config = self.settings.recipe.model.hyperparameter_tuning
        if tuning_config and tuning_config.enabled:
            return self._train_with_hyperparameter_optimization(df, model, augmenter, preprocessor, context_params)
        else:
            return self._train_with_fixed_hyperparameters(df, model, augmenter, preprocessor, context_params)

    def _train_with_hyperparameter_optimization(self, df, model, augmenter, preprocessor, context_params):
        """Optuna 기반 자동 최적화 (내부 메서드)"""
        logger.info("🚀 하이퍼파라미터 자동 최적화 모드 시작")
        
        # 기본 설정 검증 및 데이터 분할
        self.settings.recipe.model.data_interface.validate_required_fields()  # 🔄 수정: entity_schema → data_interface
        train_df, test_df = self._split_data(df)
        
        # 피처 증강
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
            test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)
        
        # Optuna 관련 컴포넌트 생성
        from src.engine.factory import Factory
        factory = Factory(self.settings)
        
        try:
            optuna_integration = factory.create_optuna_integration()
            tuning_utils = factory.create_tuning_utils()
        except (ValueError, ImportError) as e:
            logger.warning(f"Optuna 컴포넌트 생성 실패, 고정 하이퍼파라미터로 진행: {e}")
            return self._train_with_fixed_hyperparameters(df, model, augmenter, preprocessor, context_params)
        
        # Optuna Study 생성
        study = optuna_integration.create_study(
            direction=self.settings.recipe.model.hyperparameter_tuning.direction,
            study_name=f"study_{self.settings.recipe.model.computed['run_name']}"
        )
        
        start_time = datetime.now()
        
        # Optuna 최적화 실행
        try:
            def objective(trial):
                # 하이퍼파라미터 샘플링
                params = tuning_utils.suggest_hyperparameters_from_recipe(
                    trial, self.settings.recipe.model.hyperparameters
                )
                
                # 단일 학습 iteration 실행
                result = self._single_training_iteration(train_df, params, trial.number)
                return result['score']
            
            # Study 실행
            study.optimize(
                objective, 
                n_trials=self.settings.recipe.model.hyperparameter_tuning.n_trials,
                timeout=getattr(self.settings.hyperparameter_tuning, 'timeout', None)
            )
            
            # 최적 결과로 최종 모델 학습
            best_params = study.best_params
            logger.info(f"✅ 최적 하이퍼파라미터: {best_params}")
            
            # 최종 모델 생성 및 학습
            final_result = self._single_training_iteration(train_df, best_params, seed=42)
            
            # 최종 테스트 평가
            trained_model = final_result['model']
            trained_preprocessor = final_result['preprocessor']
            
            # Test 데이터로 최종 평가
            evaluator = factory.create_evaluator()
            X_test, y_test, _ = self._prepare_training_data(test_df)
            
            if trained_preprocessor:
                X_test_processed = trained_preprocessor.transform(X_test)
            else:
                X_test_processed = X_test
            
            final_metrics = evaluator.evaluate(trained_model, X_test_processed, y_test, test_df)
            
            # 결과 준비
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()
            
            training_results = {
                'hyperparameter_optimization': {
                    'enabled': True,
                    'engine': 'optuna',
                    'best_params': best_params,
                    'best_score': study.best_value,
                    'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
                    'total_trials': len(study.trials),
                    'pruned_trials': len([t for t in study.trials if t.state.name == 'PRUNED']),
                    'optimization_time': optimization_time,
                    'search_space': tuning_utils.extract_search_space_from_recipe(self.settings.recipe.model.hyperparameters)
                },
                'training_methodology': {
                    'train_test_split_method': 'stratified',
                    'train_ratio': 0.8,
                    'validation_strategy': 'train_validation_split_per_trial',
                    'random_state': 42,
                    'preprocessing_fit_scope': 'train_only'
                }
            }
            
            logger.info(f"🎉 하이퍼파라미터 최적화 완료! 최고 점수: {study.best_value:.4f} ({optimization_time:.1f}초)")
            return trained_model, trained_preprocessor, final_metrics, training_results
            
        except Exception as e:
            logger.error(f"하이퍼파라미터 최적화 실행 중 오류: {e}")
            # Fallback: 고정 하이퍼파라미터로 진행
            return self._train_with_fixed_hyperparameters(df, model, augmenter, preprocessor, context_params)

    def _train_with_fixed_hyperparameters(self, df, model, augmenter, preprocessor, context_params):
        """기존 고정 하이퍼파라미터 방식 (기존 로직 재사용)"""
        logger.info("고정 하이퍼파라미터 모드 (기존 방식)")
        
        # 기존 train 메서드의 로직을 그대로 사용
        self.settings.recipe.model.data_interface.validate_required_fields()
        task_type = self.settings.recipe.model.data_interface.task_type
        
        # 데이터 분할
        train_df, test_df = self._split_data(df)
        
        # 피처 증강
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
            test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)
            logger.info("피처 증강 완료.")
        else:
            logger.info("Augmenter가 주입되지 않아 피처 증강을 건너뜁니다.")

        # 동적 데이터 준비
        X_train, y_train, additional_data = self._prepare_training_data(train_df)
        X_test, y_test, _ = self._prepare_training_data(test_df)
        
        # 원본 데이터 스키마 검증 (entity, timestamp, target 컬럼 있는지)
        validate_schema(train_df, self.settings, for_training=False)

        # 전처리기 학습 및 변환 (주입받은 Preprocessor 사용)
        if preprocessor:
            logger.info("전처리기 학습을 시작합니다.")
            preprocessor.fit(X_train)  # ← ✅ Data Leakage 방지
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            logger.info("전처리기 학습 및 변환 완료.")
        else:
            X_train_processed = X_train
            X_test_processed = X_test
            logger.info("Preprocessor가 주입되지 않아 전처리를 건너뜁니다.")

        # 스키마 검증 (모델 학습용 데이터)
        validate_schema(X_train_processed, self.settings, for_training=True)

        # 동적 모델 학습
        logger.info(f"'{self.settings.recipe.model.class_path}' 모델 학습을 시작합니다.")
        self._fit_model(model, X_train_processed, y_train, additional_data)
        logger.info("모델 학습 완료.")

        # 평가자로 평가 수행
        from src.engine.factory import Factory
        factory = Factory(self.settings)
        evaluator = factory.create_evaluator()
        
        metrics = evaluator.evaluate(model, X_test_processed, y_test, test_df)
        
        # 기본 training_results (HPO 없음)
        training_results = {
            'hyperparameter_optimization': None,
            'training_methodology': {
                'train_test_split_method': 'stratified',
                'train_ratio': 0.8,
                'validation_strategy': 'train_test_split',
                'random_state': 42,
                'preprocessing_fit_scope': 'train_only'
            }
        }
        
        return model, preprocessor, metrics, training_results

    def _single_training_iteration(self, train_df, params, seed):
        """핵심: Data Leakage 방지 + 단일 학습 로직"""
        
        # 1. Train/Validation Split (Data Leakage 방지)
        train_data, val_data = train_test_split(
            train_df, test_size=0.2, random_state=seed, 
            stratify=self._get_stratify_column_data(train_df)
        )
        
        # 2. 동적 데이터 준비
        X_train, y_train, additional_data = self._prepare_training_data(train_data)
        X_val, y_val, _ = self._prepare_training_data(val_data)
        
        # 3. Preprocessor fit (Train only) ← ✅ Data Leakage 방지
        from src.engine.factory import Factory
        factory = Factory(self.settings)
        preprocessor = factory.create_preprocessor()
        
        if preprocessor:
            preprocessor.fit(X_train)  # Train 데이터에만 fit
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        
        # 4. Model 생성 및 학습 (동적 하이퍼파라미터 적용)
        tuning_utils = factory.create_tuning_utils()
        model_instance = tuning_utils.create_model_with_params(self.settings.recipe.model.class_path, params)
        self._fit_model(model_instance, X_train_processed, y_train, additional_data)
        
        # 5. 평가
        evaluator = factory.create_evaluator()
        metrics = evaluator.evaluate(model_instance, X_val_processed, y_val, val_data)
        
        # 주요 메트릭 추출 (tuning에 사용)
        score = tuning_utils.extract_optimization_score(
            metrics, self.settings.recipe.model.hyperparameter_tuning.metric
        )
        
        return {
            'model': model_instance,
            'preprocessor': preprocessor,
            'score': score,
            'metrics': metrics,
            'params': params
        }

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """동적 데이터 준비 (task_type에 따라 다름)"""
        data_interface = self.settings.recipe.model.data_interface
        task_type = data_interface.task_type
        
        # 공통: Feature와 Target 분리
        if task_type in ["classification", "regression"]:
            target_col = data_interface.target_column
            X = df.drop(columns=[target_col])
            y = df[target_col]
            additional_data = {}
            
        elif task_type == "clustering":
            # Clustering: target 없음
            X = df.copy()
            y = None
            additional_data = {}
            
        elif task_type == "causal":
            # Causal: treatment와 target 모두 필요
            target_col = data_interface.target_column
            treatment_col = data_interface.treatment_column
            X = df.drop(columns=[target_col, treatment_col])
            y = df[target_col]
            additional_data = {
                'treatment': df[treatment_col],
                'treatment_value': data_interface.treatment_value
            }
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")
        
        return X, y, additional_data

    def _fit_model(self, model, X, y, additional_data):
        """동적 모델 학습 (task_type별 처리)"""
        data_interface = self.settings.recipe.model.data_interface
        task_type = data_interface.task_type
        
        if task_type in ["classification", "regression"]:
            model.fit(X, y)
        elif task_type == "clustering":
            model.fit(X)  # y 없음
        elif task_type == "causal":
            # CausalML 모델들: X, y, treatment 모두 필요
            treatment = additional_data['treatment']
            model.fit(X, treatment, y)
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 분할 (stratify 지원)"""
        data_interface = self.settings.recipe.model.data_interface
        stratify_data = self._get_stratify_column_data(df)
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=stratify_data
        )
        return train_df, test_df

    def _get_stratify_column_data(self, df: pd.DataFrame):
        """Stratify용 컬럼 데이터 추출"""
        data_interface = self.settings.recipe.model.data_interface
        task_type = data_interface.task_type
        
        if task_type == "classification":
            # 분류: target 컬럼으로 stratify
            target_col = data_interface.target_column
            return df[target_col] if target_col in df.columns else None
        elif task_type == "causal":
            # 인과추론: treatment 컬럼으로 stratify
            treatment_col = data_interface.treatment_column
            return df[treatment_col] if treatment_col in df.columns else None
        else:
            # 회귀, 클러스터링: stratify 없음
            return None