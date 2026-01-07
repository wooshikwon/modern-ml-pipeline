# src/serving/context.py

from typing import Type

import mlflow
from pydantic import BaseModel, create_model

from src.settings import Settings


class AppContext:
    """
    API 서버의 전역 상태를 관리하는 컨텍스트 클래스입니다.
    서버 생명주기 동안 모델, 설정 등의 객체를 보관합니다.
    """

    def __init__(self):
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_uri: str = ""
        self.settings: Settings | None = None
        self.PredictionRequest: Type[BaseModel] = create_model("DefaultPredictionRequest")
        self.BatchPredictionRequest: Type[BaseModel] = create_model("DefaultBatchPredictionRequest")


# 전역 컨텍스트 인스턴스
app_context = AppContext()
