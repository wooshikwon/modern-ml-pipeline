import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split

from src.settings.settings import Settings
from src.utils.system.logger import logger
from src.core.augmenter import BaseAugmenter
from src.core.preprocessor import BasePreprocessor
# BaseModel import 제거: 외부 라이브러리 직접 사용으로 전환
from src.interface.base_trainer import BaseTrainer
from src.utils.system.schema_utils import validate_schema


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
        데이터 분할, 피처 증강, 전처리, 모델 학습, 평가의 전체 파이프라인을 실행합니다.
        task_type에 따라 동적으로 데이터를 준비하고 적절한 evaluator를 사용합니다.
        """
        logger.info("모델 학습 프로세스 시작...")
        context_params = context_params or {}

        # 1. 설정 검증
        self.settings.model.data_interface.validate_required_fields()
        task_type = self.settings.model.data_interface.task_type
        logger.info(f"Task Type: {task_type}")

        # 2. 데이터 분할
        train_df, test_df = self._split_data(df)

        # 3. 피처 증강 (주입받은 Augmenter 사용)
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
            test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)
            logger.info("피처 증강 완료.")
        else:
            logger.info("Augmenter가 주입되지 않아 피처 증강을 건너뜁니다.")

        # 4. 동적 데이터 준비
        X_train, y_train, additional_data = self._prepare_training_data(train_df)
        X_test, y_test, _ = self._prepare_training_data(test_df)

        # 5. 전처리기 학습 및 변환 (주입받은 Preprocessor 사용)
        if preprocessor:
            logger.info("전처리기 학습을 시작합니다.")
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            logger.info("전처리기 학습 및 변환 완료.")
        else:
            X_train_processed = X_train
            X_test_processed = X_test
            logger.info("Preprocessor가 주입되지 않아 전처리를 건너뜁니다.")

        # 6. 스키마 검증
        validate_schema(X_train_processed, self.settings)

        # 7. 동적 모델 학습
        logger.info(f"'{self.settings.model.class_path}' 모델 학습을 시작합니다.")
        self._fit_model(model, X_train_processed, y_train, additional_data)
        logger.info("모델 학습 완료.")

        # 8. 동적 평가
        from src.core.factory import Factory
        factory = Factory(self.settings)
        evaluator = factory.create_evaluator()
        metrics = evaluator.evaluate(model, X_test_processed, y_test, test_df)

        results = {"metrics": metrics, "metadata": {}}
        logger.info("모델 학습 프로세스가 성공적으로 완료되었습니다.")

        return preprocessor, model, results

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