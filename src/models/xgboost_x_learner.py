import pandas as pd
import numpy as np
from causalml.inference.meta import XGBTRegressor
from sklearn.exceptions import NotFittedError

# settings.py와 BaseModel의 경로는 실제 프로젝트 구조에 맞게 조정해주세요.
from config.settings import Settings
from src.interface.base_model import BaseModel
from src.utils.logger import logger

class XGBoostXLearner(BaseModel):
    """
    CausalML의 XGBTRegressor를 래핑한 X-Learner 모델.
    """
    def __init__(self, settings: Settings):
        """XGBoostXLearner를 초기화합니다."""
        self.settings = settings
        self.params = self.settings.model.hyperparameters
        self.model_name = self.settings.model.name

        self.model = XGBTRegressor(**self.params)
        self._is_fitted = False
        logger.info(f"XGBoostXLearner 초기화 완료. Parameters: {self.params}")

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series) -> 'XGBoostXLearner':
        """모델을 학습시킵니다."""
        logger.info(f"{self.model_name} 모델 학습 시작...")
        self.model.fit(
            X=X.values,
            treatment=treatment.values,
            y=y.values
        )
        self._is_fitted = True
        logger.info(f"{self.model_name} 모델 학습 완료.")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """개별 처치 효과(Uplift)를 예측합니다."""
        if not self._is_fitted:
            raise NotFittedError("모델이 학습되지 않았습니다. `fit`을 먼저 호출해주세요.")

        logger.info(f"{len(X)}개 샘플에 대한 Uplift 예측 시작...")
        uplift = self.model.predict(X.values)
        logger.info(f"Uplift 예측 완료. 평균 Uplift: {uplift.mean():.4f}")
        return pd.Series(uplift.flatten(), name="uplift_score", index=X.index)