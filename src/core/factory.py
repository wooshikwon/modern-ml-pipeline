import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import mlflow
import pandas as pd

from src.core.augmenter import Augmenter, LocalFileAugmenter, BaseAugmenter
from src.core.preprocessor import BasePreprocessor, Preprocessor
from src.interface.base_model import BaseModel
from src.interface.base_adapter import BaseAdapter
from src.interface.base_data_adapter import BaseDataAdapter
from src.models import CausalForestModel, XGBoostXLearner
from src.settings.settings import Settings
from src.utils.logger import logger
from src.utils.data_adapters.file_system_adapter import FileSystemAdapter
from src.utils.data_adapters.bigquery_adapter import BigQueryAdapter
from src.utils.data_adapters.gcs_adapter import GCSAdapter
from src.utils.data_adapters.s3_adapter import S3Adapter

# Redis는 선택적 의존성으로 처리
try:
    from src.utils.data_adapters.redis_adapter import RedisAdapter
    HAS_REDIS = True
except ImportError:
    RedisAdapter = None
    HAS_REDIS = False

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    순수 로직 컴포넌트와 로직의 스냅샷만 내장하는 '완전 독립형' 모델 래퍼.
    """
    def __init__(
        self,
        augmenter: BaseAugmenter,
        preprocessor: Optional[BasePreprocessor],
        model: BaseModel,
        loader_uri: str,
        recipe_snapshot: Dict[str, Any],
        augmenter_sql_snapshot: str,
    ):
        self.augmenter = augmenter
        self.preprocessor = preprocessor
        self.model = model
        self.loader_uri = loader_uri
        self.recipe_snapshot = recipe_snapshot
        self.augmenter_sql_snapshot = augmenter_sql_snapshot

    def predict(
        self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        params = params or {}
        run_mode = params.get("run_mode", "serving")
        context_params = params.get("context_params", {})
        feature_store_config = params.get("feature_store_config")
        return_intermediate = params.get("return_intermediate", False)

        logger.info(f"PyfuncWrapper.predict 실행 시작 (모드: {run_mode})")

        augmented_df = self.augmenter.augment(
            data=model_input,
            run_mode=run_mode,
            context_params=context_params,
            feature_store_config=feature_store_config,
        )
        preprocessed_df = (
            self.preprocessor.transform(augmented_df) if self.preprocessor else augmented_df
        )
        predictions = self.model.predict(preprocessed_df)
        
        results_df = model_input.merge(
            pd.DataFrame(predictions, index=model_input.index, columns=["uplift_score"]),
            left_index=True,
            right_index=True,
        )
        logger.info("PyfuncWrapper.predict 실행 완료.")

        if return_intermediate and run_mode == "batch":
            return {
                "final_results": results_df,
                "augmented_data": augmented_df,
                "preprocessed_data": preprocessed_df,
            }
        else:
            return results_df

class Factory:
    """
    설정(settings)과 URI 스킴(scheme)에 기반하여 모든 핵심 컴포넌트의 인스턴스를 생성하는 중앙 팩토리 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Factory가 초기화되었습니다.")

    def create_data_adapter(self, scheme: str) -> BaseDataAdapter:
        logger.info(f"'{scheme}' 스킴에 대한 데이터 어댑터를 생성합니다.")
        if scheme == 'file':
            return FileSystemAdapter(self.settings)
        elif scheme == 'bq':
            return BigQueryAdapter(self.settings)
        elif scheme == 'gs':
            return GCSAdapter(self.settings)
        elif scheme == 's3':
            return S3Adapter(self.settings)
        else:
            raise ValueError(f"지원하지 않는 데이터 어댑터 스킴입니다: {scheme}")

    def create_redis_adapter(self):
        if not HAS_REDIS:
            logger.warning("Redis 라이브러리가 설치되지 않아 Redis 어댑터를 생성할 수 없습니다.")
            raise ImportError("Redis 라이브러리가 필요합니다. `pip install redis`로 설치하세요.")
        
        logger.info("Redis 어댑터를 생성합니다.")
        return RedisAdapter(self.settings.serving.realtime_feature_store)

    def create_augmenter(self) -> BaseAugmenter:
        augmenter_config = self.settings.model.augmenter
        if not augmenter_config:
            raise ValueError("Augmenter 설정이 레시피에 없습니다.")
        is_local = self.settings.environment.app_env == "local"
        if is_local and augmenter_config.local_override_uri:
            return LocalFileAugmenter(uri=augmenter_config.local_override_uri)
        return Augmenter(
            source_uri=augmenter_config.source_uri,
            settings=self.settings,
        )

    def create_preprocessor(self) -> Optional[BasePreprocessor]:
        preprocessor_config = self.settings.model.preprocessor
        if not preprocessor_config: return None
        return Preprocessor(config=preprocessor_config, settings=self.settings)

    def create_model(self) -> BaseModel:
        model_name = self.settings.model.name
        if model_name == "xgboost_x_learner":
            return XGBoostXLearner(settings=self.settings)
        elif model_name == "causal_forest":
            return CausalForestModel(settings=self.settings)
        raise ValueError(f"지원하지 않는 모델 타입입니다: '{model_name}'")

    def create_pyfunc_wrapper(
        self, trained_model: BaseModel, trained_preprocessor: Optional[BasePreprocessor]
    ) -> PyfuncWrapper:
        logger.info("순수 로직 PyfuncWrapper 생성을 시작합니다.")
        augmenter = self.create_augmenter()
        
        augmenter_sql_snapshot = ""
        if isinstance(augmenter, Augmenter):
            augmenter_sql_snapshot = augmenter.sql_template_str
        
        recipe_path = Path(__file__).resolve().parent.parent.parent / "recipes" / f"{self.settings.model.name}.yaml"
        recipe_snapshot = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

        return PyfuncWrapper(
            augmenter=augmenter,
            preprocessor=trained_preprocessor,
            model=trained_model,
            loader_uri=self.settings.model.loader.source_uri,
            recipe_snapshot=recipe_snapshot,
            augmenter_sql_snapshot=augmenter_sql_snapshot,
        )
