from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, List
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from pydantic import BaseModel

try:
    from feast import FeatureStore
    from feast.repo_config import RepoConfig
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
        
        # FeastAdapter는 복잡한 설정 구조로 인해 별도의 feature_store 섹션을 사용
        logger.info("FeastAdapter 초기화 중. feature_store 설정 섹션 사용.")
        self.store = self._init_feature_store()

    def _init_feature_store(self) -> FeatureStore:
        """Initializes the Feast FeatureStore object."""
        try:
            # FeastAdapter는 settings.feature_store.feast_config에서 설정을 읽음
            # (다른 어댑터와 달리 복잡한 Feast 설정 구조로 인해 별도 섹션 사용)
            config_data = self.settings.feature_store.feast_config
            logger.info(f"Feast 설정 로드됨. project: {config_data.get('project', 'unknown')}")

            if isinstance(config_data, dict):
                # Convert dict to RepoConfig object before passing to FeatureStore
                repo_config = RepoConfig(**config_data)
                fs = FeatureStore(config=repo_config)
            elif isinstance(config_data, BaseModel): # Should be RepoConfig, but check BaseModel for safety
                # If it's already a Pydantic model, use it directly
                fs = FeatureStore(config=config_data)
            else:
                raise TypeError(f"Unsupported config type for Feast: {type(config_data)}")
            
            logger.info("Feature Store adapter initialized successfully.")
            return fs
        except Exception as e:
            logger.error(f"Failed to initialize Feast FeatureStore: {e}", exc_info=True)
            return None

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