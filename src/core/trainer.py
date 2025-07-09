# library import
import pandas as pd
import numpy as np
import mlflow
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# module import
from config.settings import Settings
from src.utils.logger import logger
from src.core.factory import Factory
from src.core.augmenter import Augmenter
from src.core.preprocessor import Preprocessor
from src.interface.base_model import BaseModel


class Trainer:
    """
    모델 학습 및 평가 전체 과정을 관장하는 클래스.
    loader에서 받은 데이터를 기반으로 피처 증강, 피처 전처리, 모델 학습, 평가, MLflow 로깅을 순차적으로 실행.
    """

    def __init__(self, settings: Settings):
        """
        Trainer를 초기화하고 Factory를 주입받습니다.
        """
        self.settings = settings
        self.factory = Factory(self.settings)
        self.augmenter: Augmenter | None = None
        self.preprocessor: Preprocessor | None = None
        self.model: BaseModel | None = None
        self.training_results: Dict[str, Any] = {}
        logger.info("Trainer 초기화 완료.")

    def train(self, df: pd.DataFrame) -> Tuple[Augmenter, Preprocessor, BaseModel, Dict[str, Any]]:
        """
        데이터 분할, 피처 증강, 피처 전처리, 모델 학습 및 평가, MLflow 로깅의 전체 파이프라인을 실행합니다.
        """
        logger.info("모델 학습 프로세스 시작...")

        # 1. 데이터 분할
        train_df, test_df = self._split_data(df)

        # 2. 피처 증강기(Augmenter) 생성 및 학습/적용
        logger.info("피처 증강 시작...")
        self.augmenter = Augmenter(config=self.settings.pipeline.augmenter)
        self.augmenter.fit(train_df)
        train_df_aug = self.augmenter.augment(train_df)
        test_df_aug = self.augmenter.augment(test_df)
        logger.info("피처 증강 완료.")

        # 3. 데이터 전처리기(Preprocessor) 생성 및 학습/적용
        logger.info("데이터 전처리 시작...")
        self.preprocessor = Preprocessor(config=self.settings.pipeline.preprocessor)
        # 증강된 학습 데이터로 전처리기를 학습시킵니다.
        self.preprocessor.fit(train_df_aug)
        X_train = self.preprocessor.process(train_df_aug)
        X_test = self.preprocessor.process(test_df_aug)
        logger.info(f"데이터 전처리 완료. 학습 피처 수: {len(X_train.columns)}")

        # 4. 모델 생성 및 학습
        self.model = self.factory.create_model()
        
        treatment_col = self.settings.model.data_interface.treatment_col
        target_col = self.settings.model.data_interface.target_col
        
        # treatment와 target은 원본 데이터프레임에서 추출합니다.
        treatment_train = train_df[treatment_col]
        y_train = train_df[target_col]

        self.model.fit(
            X=X_train,
            treatment=treatment_train,
            y=y_train
        )

        # 5. 모델 평가
        # 평가는 원본 데이터와 변환된 데이터를 모두 사용합니다.
        metrics = self._evaluate(
            train_orig=train_df, X_train=X_train,
            test_orig=test_df, X_test=X_test
        )
        self.training_results = {'metrics': metrics, 'metadata': {}}

        # 6. MLflow 모델 로깅 (래퍼 사용)
        logger.info("MLflow 모델 로깅 시작...")
        pyfunc_wrapper = self.factory.create_pyfunc_wrapper(
            model=self.model,
            augmenter=self.augmenter,
            preprocessor=self.preprocessor
        )
        mlflow.pyfunc.log_model(artifact_path="model", python_model=pyfunc_wrapper)
        logger.info("MLflow 모델 로깅 완료.")

        logger.info("모델 학습 프로세스 성공적으로 완료.")
        return self.augmenter, self.preprocessor, self.model, self.training_results

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터를 학습/테스트 세트로 분할합니다."""
        treatment_col = self.settings.model.data_interface.treatment_col
        test_size = 0.2

        logger.info(f"데이터 분할 (테스트 사이즈: {test_size}, 기준: {treatment_col})")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df[treatment_col]
        )
        logger.info(f"분할 완료: 학습셋 {len(train_df)} 행, 테스트셋 {len(test_df)} 행")
        return train_df, test_df

    def _evaluate(self, train_orig: pd.DataFrame, X_train: pd.DataFrame,
                  test_orig: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """학습된 모델의 성능을 종합적으로 평가합니다."""
        logger.info("모델 성능 평가 시작...")

        treatment_col = self.settings.model.data_interface.treatment_col
        target_col = self.settings.model.data_interface.target_col
        metrics = {}

        # Causal Model은 predict 대신 predict_uplift와 같은 커스텀 메서드를 가질 수 있습니다.
        train_uplift = self.model.predict(X_train)
        test_uplift = self.model.predict(X_test)

        # ATE (Average Treatment Effect) 계산
        for prefix, df, uplift in [('train', train_orig, train_uplift),
                                   ('test', test_orig, test_uplift)]:
            # 실제 ATE는 원본 데이터에서 계산
            treatment_mask = df[treatment_col] == self.settings.model.data_interface.treatment_value
            actual_ate = df.loc[treatment_mask, target_col].mean() - df.loc[~treatment_mask, target_col].mean()

            metrics[f'{prefix}_actual_ate'] = actual_ate
            # 예측된 ATE는 모델의 예측값 평균으로 추정
            metrics[f'{prefix}_predicted_ate'] = uplift.mean()

        logger.info(f"모델 성능 평가 완료: {metrics}")
        return metrics