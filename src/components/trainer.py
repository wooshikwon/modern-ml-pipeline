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
    모든 의존성은 외부(주로 Factory)로부터 주입받는다.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Trainer가 초기화되었습니다.")

    def train(
        self,
        df: pd.DataFrame,
        model,
        augmenter: Optional[BaseAugmenter] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        context_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[BasePreprocessor], Any, Dict[str, Any]]:
        """
        기존 인터페이스 유지하면서 내부에서 하이퍼파라미터 최적화 처리
        """
        logger.info("모델 학습 프로세스 시작...")
        context_params = context_params or {}

        # 🆕 하이퍼파라미터 튜닝 여부 확인
        hyperparameter_tuning_config = self.settings.model.hyperparameter_tuning
        is_tuning_enabled = (
            hyperparameter_tuning_config and 
            hyperparameter_tuning_config.enabled and
            self.settings.hyperparameter_tuning and
            self.settings.hyperparameter_tuning.enabled
        )

        if is_tuning_enabled:
            return self._train_with_hyperparameter_optimization(
                df, model, augmenter, preprocessor, context_params
            )
        else:
            return self._train_with_fixed_hyperparameters(
                df, model, augmenter, preprocessor, context_params
            )
    
    def _train_with_hyperparameter_optimization(self, df, model, augmenter, preprocessor, context_params):
        """Optuna 기반 자동 최적화 (내부 메서드)"""
        logger.info("🚀 하이퍼파라미터 자동 최적화 모드 시작")
        
        # 기본 설정 검증 및 데이터 분할
        self.settings.model.data_interface.validate_required_fields()
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
            direction=self.settings.model.hyperparameter_tuning.direction,
            study_name=f"study_{self.settings.model.computed['run_name']}"
        )
        
        start_time = datetime.now()
        
        def objective(trial):
            # 하이퍼파라미터 샘플링
            params = optuna_integration.suggest_hyperparameters(
                trial, self.settings.model.hyperparameters.root
            )
            
            # 단일 학습 실행 (Data Leakage 방지)
            result = self._single_training_iteration(
                train_df, params, seed=trial.number
            )
            
            # Pruning 지원
            trial.report(result['score'], step=trial.number)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
                
            return result['score']
        
        # 최적화 실행 (실험 논리 + 인프라 제약)
        study.optimize(
            objective,
            n_trials=self.settings.model.hyperparameter_tuning.n_trials,
            timeout=self.settings.hyperparameter_tuning.timeout
        )
        
        end_time = datetime.now()
        
        # 최적 파라미터로 최종 학습
        best_params = study.best_params
        final_result = self._single_training_iteration(
            train_df, best_params, seed=42
        )
        
        # 최종 테스트 평가
        final_model = final_result['model']
        final_preprocessor = final_result['preprocessor']
        
        # 테스트 데이터 평가
        X_test, y_test, _ = self._prepare_training_data(test_df)
        if final_preprocessor:
            X_test_processed = final_preprocessor.transform(X_test)
        else:
            X_test_processed = X_test
        
        evaluator = factory.create_evaluator()
        final_metrics = evaluator.evaluate(final_model, X_test_processed, y_test, test_df)
        
        # 🆕 최적화 메타데이터 포함
        results = {
            'metrics': final_metrics,
            'hyperparameter_optimization': tuning_utils.create_optimization_metadata(
                study, start_time, end_time, best_params
            ),
            'training_methodology': {
                'train_test_split_method': 'stratified',
                'preprocessing_fit_scope': 'train_only',  # Data Leakage 방지 증명
                'optimization_trials': len(study.trials)
            }
        }
        
        logger.info(f"🎯 하이퍼파라미터 최적화 완료! 최고 점수: {study.best_value}, 총 {len(study.trials)}회 시도")
        
        # 기존 인터페이스와 호환되는 반환값
        return final_preprocessor, final_model, results
    
    def _train_with_fixed_hyperparameters(self, df, model, augmenter, preprocessor, context_params):
        """기존 고정 하이퍼파라미터 방식 (기존 로직 재사용)"""
        logger.info("고정 하이퍼파라미터 모드 (기존 방식)")
        
        # 기존 train 메서드의 로직을 그대로 사용
        self.settings.model.data_interface.validate_required_fields()
        task_type = self.settings.model.data_interface.task_type
        
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

        # 스키마 검증
        validate_schema(X_train_processed, self.settings)

        # 동적 모델 학습
        logger.info(f"'{self.settings.model.class_path}' 모델 학습을 시작합니다.")
        self._fit_model(model, X_train_processed, y_train, additional_data)
        logger.info("모델 학습 완료.")

        # 동적 평가
        from src.engine.factory import Factory
        factory = Factory(self.settings)
        evaluator = factory.create_evaluator()
        metrics = evaluator.evaluate(model, X_test_processed, y_test, test_df)

        results = {
            "metrics": metrics, 
            "hyperparameter_optimization": {"enabled": False},  # 🆕 일관성 유지
            "training_methodology": {
                "train_test_split_method": "stratified",
                "preprocessing_fit_scope": "train_only"  # Data Leakage 방지 보장
            }
        }
        logger.info("모델 학습 프로세스가 성공적으로 완료되었습니다.")

        return preprocessor, model, results
    
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
        model_instance = tuning_utils.create_model_with_params(self.settings.model.class_path, params)
        self._fit_model(model_instance, X_train_processed, y_train, additional_data)
        
        # 5. 평가
        evaluator = factory.create_evaluator()
        metrics = evaluator.evaluate(model_instance, X_val_processed, y_val, val_data)
        
        # 주요 메트릭 추출 (tuning에 사용)
        score = tuning_utils.extract_optimization_score(
            metrics, self.settings.model.hyperparameter_tuning.metric
        )
        
        return {
            'model': model_instance,
            'preprocessor': preprocessor,
            'score': score,
            'metrics': metrics,
            'params': params
        }
    
    def _get_stratify_column_data(self, df):
        """Data Leakage 방지용 stratify 컬럼 데이터 반환"""
        from src.utils.system.tuning_utils import TuningUtils
        stratify_col = TuningUtils.get_stratify_column(
            df, 
            self.settings.model.data_interface.task_type,
            self.settings.model.data_interface
        )
        
        if stratify_col:
            return df[stratify_col]
        else:
            return None

    # 기존 메서드들 유지 (변경 없음)
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
        """task_type에 따른 동적 데이터 준비"""
        task_type = self.settings.model.data_interface.task_type
        data_interface = self.settings.model.data_interface
        
        # 제외할 컬럼들 동적 결정
        exclude_cols = []
        if data_interface.target_col:
            exclude_cols.append(data_interface.target_col)
        if data_interface.treatment_col:
            exclude_cols.append(data_interface.treatment_col)
        
        X = df.drop(columns=exclude_cols, errors="ignore")
        
        if task_type == "clustering":
            logger.info("클러스터링 모델: target 데이터 없이 진행")
            return X, None, {}
        
        y = df[data_interface.target_col]
        
        additional_data = {}
        if task_type == "causal":
            additional_data["treatment"] = df[data_interface.treatment_col]
            logger.info("인과추론 모델: treatment 데이터 추가")
        elif task_type == "regression" and data_interface.sample_weight_col:
            additional_data["sample_weight"] = df[data_interface.sample_weight_col]
            logger.info(f"회귀 모델: sample_weight 컬럼 사용 ({data_interface.sample_weight_col})")
        
        return X, y, additional_data

    def _fit_model(self, model, X: pd.DataFrame, y: Optional[pd.Series], additional_data: Dict[str, Any]):
        """task_type에 따른 동적 모델 학습"""
        task_type = self.settings.model.data_interface.task_type
        
        if task_type == "clustering":
            model.fit(X)
        elif task_type == "causal":
            model.fit(X, y, additional_data["treatment"])
        elif task_type == "regression" and "sample_weight" in additional_data:
            model.fit(X, y, sample_weight=additional_data["sample_weight"])
        else:
            # classification, regression (without sample_weight)
            model.fit(X, y)

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터를 학습/테스트 세트로 분할합니다. task_type에 따라 적절한 stratify 컬럼을 선택합니다."""
        task_type = self.settings.model.data_interface.task_type
        data_interface = self.settings.model.data_interface
        test_size = 0.2
        
        # task_type에 따라 stratify 컬럼 결정
        stratify_col = None
        if task_type == "causal" and data_interface.treatment_col:
            stratify_col = data_interface.treatment_col
        elif task_type == "classification" and data_interface.target_col:
            stratify_col = data_interface.target_col
        
        logger.info(f"데이터 분할 (테스트 사이즈: {test_size}, 기준: {stratify_col})")
        
        # stratify가 가능한지 확인
        if stratify_col and stratify_col in df.columns and df[stratify_col].nunique() > 1:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df[stratify_col]
            )
            logger.info(f"'{stratify_col}' 컬럼 기준 계층화 분할 수행")
        else:
            if stratify_col:
                logger.warning(f"'{stratify_col}' 컬럼으로 계층화 분할을 할 수 없어 랜덤 분할합니다.")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        logger.info(f"분할 완료: 학습셋 {len(train_df)} 행, 테스트셋 {len(test_df)} 행")
        return train_df, test_df