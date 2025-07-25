from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, List
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger

try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

if TYPE_CHECKING:
    from src.settings import Settings


class FeastAdapter(BaseAdapter):
    """
    Feast 라이브러리를 직접 사용하는 가벼운 Feature Store 어댑터.
    복잡한 임시 파일 생성 로직을 제거하고, Feast의 네이티브 기능을 활용합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        if not FEAST_AVAILABLE:
            raise ImportError("Feast SDK is not installed. Please install with `pip install feast`.")
        
        self.settings = settings
        self.store = self._init_feature_store()

    def _init_feature_store(self) -> FeatureStore:
        """설정(Settings) 객체로부터 Feast FeatureStore를 초기화합니다."""
        try:
            feast_config_dict = self.settings.feature_store.feast_config.dict()
            logger.info(f"Initializing Feast FeatureStore with config: {feast_config_dict}")
            # repo_path는 Feast가 feature_store.yaml을 찾기 위해 필요할 수 있습니다.
            # 일반적으로는 config만으로 충분합니다.
            return FeatureStore(config=feast_config_dict)
        except Exception as e:
            logger.error(f"Failed to initialize Feast FeatureStore: {e}", exc_info=True)
            raise

    def get_historical_features(self, entity_df: pd.DataFrame, features: List[str], **kwargs) -> pd.DataFrame:
        """오프라인 스토어에서 과거 시점의 피처를 가져옵니다."""
        logger.info(f"Getting historical features for {len(entity_df)} entities.")
        try:
            retrieval_job = self.store.get_historical_features(
                entity_df=entity_df,
                features=features,
            )
            return retrieval_job.to_df()
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}", exc_info=True)
            raise

    def get_online_features(self, entity_rows: List[Dict[str, Any]], features: List[str], **kwargs) -> pd.DataFrame:
        """온라인 스토어에서 실시간 피처를 가져옵니다."""
        logger.info(f"Getting online features for {len(entity_rows)} entities.")
        try:
            retrieval_job = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows,
            )
            return retrieval_job.to_df()
        except Exception as e:
            logger.error(f"Failed to get online features: {e}", exc_info=True)
            raise

    def read(self, **kwargs) -> pd.DataFrame:
        """BaseAdapter 호환성을 위한 read 메서드. get_historical_features를 호출합니다."""
        entity_df = kwargs.get("entity_df")
        features = kwargs.get("features")
        if entity_df is None or features is None:
            raise ValueError("'entity_df' and 'features' must be provided for read operation.")
        return self.get_historical_features(entity_df, features, **kwargs)

    def write(self, df: pd.DataFrame, **kwargs):
        """Feast는 주로 읽기용이므로, write는 기본적으로 지원하지 않습니다."""
        raise NotImplementedError("Write operation is not supported by the FeastAdapter.") 