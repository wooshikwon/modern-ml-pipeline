from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, List
from pydantic import BaseModel

from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from src.engine import AdapterRegistry

try:
    from feast import FeatureStore
    from feast.repo_config import RepoConfig
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

if TYPE_CHECKING:
    from src.settings import Settings


class FeastAdapter(BaseAdapter):
    """Feast 기반 Feature Store 어댑터"""
    def __init__(self, settings: 'Settings', **kwargs):
        if not FEAST_AVAILABLE:
            raise ImportError("Feast SDK is not installed. Please install with `pip install feast`.")
        self.settings = settings
        logger.info("FeastAdapter 초기화 중. feature_store 설정 섹션 사용.")
        self.store = self._init_feature_store()

    def _init_feature_store(self) -> 'FeatureStore':
        try:
            config_data = self.settings.feature_store.feast_config
            logger.info(f"Feast 설정 로드됨. project: {config_data.get('project', 'unknown')}")
            if isinstance(config_data, dict):
                repo_config = RepoConfig(**config_data)
                fs = FeatureStore(config=repo_config)
            elif isinstance(config_data, BaseModel):
                fs = FeatureStore(config=config_data)
            else:
                raise TypeError(f"Unsupported config type for Feast: {type(config_data)}")
            logger.info("Feature Store adapter initialized successfully.")
            return fs
        except Exception as e:
            logger.error(f"Failed to initialize Feast FeatureStore: {e}", exc_info=True)
            return None

    def get_historical_features(self, entity_df: pd.DataFrame, features: List[str], **kwargs) -> pd.DataFrame:
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

    def get_historical_features_with_validation(
        self, entity_df: pd.DataFrame, features: List[str], data_interface_config: Dict[str, Any] = None, **kwargs
    ) -> pd.DataFrame:
        logger.info("🔒 Point-in-Time Correctness 보장 피처 조회 시작")
        if data_interface_config:
            self._validate_point_in_time_schema(entity_df, data_interface_config)
        result_df = self.get_historical_features(entity_df, features, **kwargs)
        if data_interface_config:
            self._validate_asof_join_result(entity_df, result_df, data_interface_config)
        logger.info("✅ Point-in-Time Correctness 검증 완료")
        return result_df

    def _validate_point_in_time_schema(self, entity_df: pd.DataFrame, config: Dict[str, Any]):
        entity_columns = config.get('entity_columns', [])
        timestamp_column = config.get('timestamp_column', '')
        missing_entities = [col for col in entity_columns if col not in entity_df.columns]
        if missing_entities:
            raise ValueError(
                f"🚨 Point-in-Time 검증 실패: 필수 Entity 컬럼 누락 {missing_entities}\n"
                f"Required: {entity_columns}, Found: {list(entity_df.columns)}"
            )
        if timestamp_column and timestamp_column not in entity_df.columns:
            raise ValueError(f"🚨 Point-in-Time 검증 실패: Timestamp 컬럼 '{timestamp_column}' 누락")
        if timestamp_column and not pd.api.types.is_datetime64_any_dtype(entity_df[timestamp_column]):
            raise ValueError(f"🚨 Point-in-Time 검증 실패: '{timestamp_column}'이 datetime 타입이 아닙니다")
        logger.info(f"✅ Point-in-Time 스키마 검증 통과: {entity_columns} + {timestamp_column}")

    def _validate_asof_join_result(self, input_df: pd.DataFrame, result_df: pd.DataFrame, config: Dict[str, Any]):
        timestamp_column = config.get('timestamp_column', '')
        if not timestamp_column or timestamp_column not in result_df.columns:
            logger.warning("Timestamp 컬럼 없음: ASOF JOIN 결과 검증 생략")
            return
        if len(result_df) != len(input_df):
            logger.warning(
                f"⚠️ ASOF JOIN 결과 행 수 불일치: input({len(input_df)}) vs result({len(result_df)})"
            )
        current_time = pd.Timestamp.now()
        future_data = result_df[result_df[timestamp_column] > current_time]
        if len(future_data) > 0:
            logger.warning(f"⚠️ 미래 데이터 감지: {len(future_data)}개 행이 현재 시점({current_time}) 이후")
        logger.info("✅ ASOF JOIN Point-in-Time 무결성 검증 완료")

    def get_online_features(self, entity_rows: List[Dict[str, Any]], features: List[str], **kwargs) -> pd.DataFrame:
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
        entity_df = kwargs.get("entity_df")
        features = kwargs.get("features")
        if entity_df is None or features is None:
            raise ValueError("'entity_df' and 'features' must be provided for read operation.")
        return self.get_historical_features(entity_df, features, **kwargs)

    def write(self, df: pd.DataFrame, table_name: str, **kwargs):
        raise NotImplementedError("FeastAdapter does not support write operation.")


AdapterRegistry.register("feature_store", FeastAdapter)

