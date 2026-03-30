# mmp/serving/context.py

import logging
import threading
from typing import Type

import mlflow
from pydantic import BaseModel, create_model

from mmp.settings import Settings

logger = logging.getLogger(__name__)


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

        # Startup 캐시: 매 요청마다 reflection 대신 사전 계산된 값 사용
        self.data_interface_schema: dict | None = None
        self.signature_type_map: dict[str, str] = {}  # col_name -> target dtype ("float64", "int64")
        self.required_columns: set[str] = set()
        self.feature_columns: set[str] = set()
        self.has_feature_store_fetcher: bool = False

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
            self._build_serving_cache(model)
            self._initialized = True

    def _build_serving_cache(self, model: mlflow.pyfunc.PyFuncModel) -> None:
        """모델 로드 시 한 번 계산하여 매 요청의 reflection을 제거."""
        try:
            wrapped = model.unwrap_python_model()
            self.data_interface_schema = getattr(wrapped, "data_interface_schema", {}) or {}

            # Feature Store fetcher 유무
            trained_fetcher = getattr(wrapped, "trained_fetcher", None)
            self.has_feature_store_fetcher = (
                trained_fetcher is not None and hasattr(trained_fetcher, "_fetcher_config")
            )

            # Required columns
            di = self.data_interface_schema
            if self.has_feature_store_fetcher:
                self.required_columns = set(di.get("entity_columns", []))
            else:
                self.required_columns = set(di.get("required_columns", []) or [])

            # Feature columns
            self.feature_columns = set(di.get("feature_columns") or [])

            # Signature type mapping
            sig = getattr(getattr(model, "metadata", None), "signature", None)
            schema_inputs = getattr(sig, "inputs", None)
            if schema_inputs and hasattr(schema_inputs, "inputs"):
                for col_spec in getattr(schema_inputs, "inputs", []) or []:
                    name = getattr(col_spec, "name", None)
                    col_type = str(getattr(col_spec, "type", "")).lower()
                    if "." in col_type:
                        col_type = col_type.split(".")[-1]
                    if name and col_type:
                        self.signature_type_map[name] = col_type
                    if name:
                        self.feature_columns.add(name)

            # Fallback: required_columns가 비어있으면 signature에서 보충
            if not self.required_columns and self.signature_type_map:
                self.required_columns = set(self.signature_type_map.keys())
        except Exception as e:
            logger.warning(f"[API] 서빙 캐시 빌드 실패 (매 요청 reflection으로 폴백): {type(e).__name__}: {e}")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None


# 전역 컨텍스트 인스턴스
app_context = AppContext()
