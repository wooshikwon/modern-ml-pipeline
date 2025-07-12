import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split

from src.settings.settings import Settings
from src.utils.logger import logger
from src.core.augmenter import BaseAugmenter
from src.core.preprocessor import BasePreprocessor
from src.interface.base_model import BaseModel
from src.interface.base_trainer import BaseTrainer
from src.utils.schema_utils import validate_schema


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
        model: BaseModel,
        augmenter: Optional[BaseAugmenter] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        context_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[BasePreprocessor], BaseModel, Dict[str, Any]]:
        """
        데이터 분할, 피처 증강, 전처리, 모델 학습, 평가의 전체 파이프라인을 실행합니���.
        """
        logger.info("모델 학습 프로세스 시작...")
        context_params = context_params or {}

        # 1. 데이터 분할
        train_df, test_df = self._split_data(df)

        # 2. 피처 증강 (주입받은 Augmenter 사용)
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            train_df = augmenter.augment(train_df, context_params)
            test_df = augmenter.augment(test_df, context_params)
            logger.info("피처 증강 완료.")
        else:
            logger.info("Augmenter가 주입되지 않아 피처 증강을 건너뜁니다.")

        # 3. 데이터 준비
        X_train = train_df.drop(
            columns=[
                self.settings.model.data_interface.target_col,
                self.settings.model.data_interface.treatment_col,
            ],
            errors="ignore",
        )
        y_train = train_df[self.settings.model.data_interface.target_col]
        treatment_train = train_df[self.settings.model.data_interface.treatment_col]

        X_test = test_df.drop(
            columns=[
                self.settings.model.data_interface.target_col,
                self.settings.model.data_interface.treatment_col,
            ],
            errors="ignore",
        )

        # 4. 전처리기 ���습 및 변환 (주입받은 Preprocessor 사용)
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

        # 5. 스키마 검증
        validate_schema(X_train_processed, self.settings)

        # 6. 모델 학습 (주입받은 Model 사용)
        logger.info(f"'{self.settings.model.name}' 모델 학습을 시작합니다.")
        model.fit(X_train_processed, y_train, treatment_train)
        logger.info("모델 학습 완료.")

        # 7. 평가
        metrics = self._evaluate(model, train_df, X_train_processed, test_df, X_test_processed)

        results = {"metrics": metrics, "metadata": {}}
        logger.info("모델 학습 프로세스가 성공적으로 완료되었습니다.")

        return preprocessor, model, results

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터를 학습/��스트 세트로 분할합니다."""
        treatment_col = self.settings.model.data_interface.treatment_col
        test_size = 0.2
        logger.info(f"데이터 분할 (테스트 사이즈: {test_size}, 기준: {treatment_col})")
        # stratify가 가능한지 확인
        if treatment_col in df.columns and df[treatment_col].nunique() > 1:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df[treatment_col]
            )
        else:
            logger.warning(
                f"'{treatment_col}' 컬럼으로 계층화 분할을 할 수 없어 랜덤 분할합니다."
            )
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        logger.info(f"분할 완료: 학습셋 {len(train_df)} 행, 테스트셋 {len(test_df)} 행")
        return train_df, test_df

    def _evaluate(
        self,
        model: BaseModel,
        train_orig: pd.DataFrame,
        X_train: pd.DataFrame,
        test_orig: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """학습된 모델의 성능을 종합적으로 평가합니다."""
        logger.info("모델 성능 평가 시작...")
        treatment_col = self.settings.model.data_interface.treatment_col
        target_col = self.settings.model.data_interface.target_col
        treatment_value = self.settings.model.data_interface.treatment_value

        metrics = {}

        # 예측
        train_uplift = model.predict(X_train)
        test_uplift = model.predict(X_test)

        # ATE 계산
        for prefix, df, uplift in [
            ("train", train_orig, train_uplift),
            ("test", test_orig, test_uplift),
        ]:
            treatment_mask = df[treatment_col] == treatment_value
            control_mask = ~treatment_mask

            # 그룹별 샘플이 하나 이상 있는지 확인
            if treatment_mask.sum() > 0 and control_mask.sum() > 0:
                actual_ate = (
                    df.loc[treatment_mask, target_col].mean()
                    - df.loc[control_mask, target_col].mean()
                )
            else:
                actual_ate = float("nan")
                logger.warning(
                    f"'{prefix}' 데이터셋에 처치 또는 통제 그룹 중 하나가 없어 ATE를 계산할 수 없습니다."
                )

            metrics[f"{prefix}_actual_ate"] = actual_ate
            metrics[f"{prefix}_predicted_ate"] = uplift.mean()

        logger.info(f"모델 성능 평가 완료: {metrics}")
        return metrics