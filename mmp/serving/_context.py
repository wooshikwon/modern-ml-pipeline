# mmp/serving/context.py

import threading
from typing import Type

import mlflow
from pydantic import BaseModel, create_model

from mmp.settings import Settings


class AppContext:
    """
    API 서버의 전역 상태를 관리하는 컨텍스트 클래스입니다.
    서버 생명주기 동안 모델, 설정 등의 객체를 보관합니다.

    초기화는 lifespan에서 한 번만 수행되며, 이후 요청 처리 중에는 읽기 전용입니다.
    _lock은 초기화 시점의 동시성 보호를 위해 존재합니다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_uri: str = ""
        self.settings: Settings | None = None
        self.PredictionRequest: Type[BaseModel] = create_model("DefaultPredictionRequest")
        self.BatchPredictionRequest: Type[BaseModel] = create_model("DefaultBatchPredictionRequest")

    def initialize(
        self,
        model: mlflow.pyfunc.PyFuncModel,
        model_uri: str,
        settings: Settings,
        prediction_request: Type[BaseModel],
        batch_prediction_request: Type[BaseModel],
    ) -> None:
        """스레드 안전한 초기화. lifespan에서 한 번만 호출."""
        with self._lock:
            self.model = model
            self.model_uri = model_uri
            self.settings = settings
            self.PredictionRequest = prediction_request
            self.BatchPredictionRequest = batch_prediction_request
            self._initialized = True

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None


# 전역 컨텍스트 인스턴스
app_context = AppContext()
