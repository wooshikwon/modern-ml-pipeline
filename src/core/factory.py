import logging
import pandas as pd
import mlflow
from typing import Union, Dict, Optional

from src.settings.settings import Settings
from src.interface.base_model import BaseModel
from src.models import XGBoostXLearner, CausalForestModel
from src.core.augmenter import BatchAugmenter, RealtimeAugmenter
from src.core.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    실행 컨텍스트(실시간/배치)를 인지하여 적절한 Augmenter를 동적으로 선택하는
    MLflow Pyfunc 모델 래퍼.
    """
    def __init__(self, model: BaseModel, preprocessor: Preprocessor, settings: Settings):
        self.model = model
        self.preprocessor = preprocessor
        self.settings = settings
        # Augmenter는 predict 시점에 동적으로 생성되므로 여기서는 None으로 초기화
        self._batch_augmenter: Optional[BatchAugmenter] = None
        self._realtime_augmenter: Optional[RealtimeAugmenter] = None

    @property
    def batch_augmenter(self) -> BatchAugmenter:
        if self._batch_augmenter is None:
            self._batch_augmenter = BatchAugmenter(
                config=self.settings.augmenter.batch, settings=self.settings
            )
        return self._batch_augmenter

    @property
    def realtime_augmenter(self) -> RealtimeAugmenter:
        if self._realtime_augmenter is None:
            self._realtime_augmenter = RealtimeAugmenter(config=self.settings.augmenter.realtime)
        return self._realtime_augmenter

    def predict(
        self, context, model_input: pd.DataFrame, params: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        params에 'run_mode'가 있으면 해당 모드로, 없으면 'serving' 모드로 동작합니다.
        """
        run_mode = params.get("run_mode", "serving") if params else "serving"
        logger.info(f"PyfuncWrapper 실행 모드: {run_mode}")

        if run_mode == "batch":
            augmenter = self.batch_augmenter
        else:
            augmenter = self.realtime_augmenter
        
        augmented_data = augmenter.augment(model_input.copy())
        preprocessed_data = self.preprocessor.transform(augmented_data)
        predictions = self.model.predict(preprocessed_data)
        
        return pd.DataFrame(predictions)


class Factory:
    """
    설정 파일(`recipe`)에 기반하여 특정 모델의 인���턴스를 생성하는 팩토리 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Factory가 초기화되었습니다.")

    def create_model(self) -> BaseModel:
        """
        설정에 명시된 모델 이름에 따라 해당 모델의 인스턴스를 생성합니다.
        """
        model_name = self.settings.model.name
        logger.info(f"'{model_name}' 모델 인스턴스 생성을 시작합니다.")

        if model_name == "xgboost_x_learner":
            return XGBoostXLearner(settings=self.settings)
        elif model_name == "causal_forest":
            return CausalForestModel(settings=self.settings)
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: '{model_name}'")

    def create_pyfunc_wrapper(
        self, model: BaseModel, preprocessor: Preprocessor
    ) -> PyfuncWrapper:
        """
        학습된 모델과 데이터 처리기들을 MLflow Pyfunc 래퍼로 감쌉니다.
        Augmenter는 Wrapper 내부에서 동적으로 생성되므로 전달받지 않습니다.
        """
        logger.info("컨텍스트 인지 Pyfunc 래퍼를 생성합니다.")
        return PyfuncWrapper(model=model, preprocessor=preprocessor, settings=self.settings)
