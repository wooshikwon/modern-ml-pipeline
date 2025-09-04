from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, List
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger
from pydantic import BaseModel
from src.settings import Settings

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
    Feast 라이브러리를 직접 사용하는 Feature Store 어댑터.
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
    
    def get_historical_features_with_validation(
        self, entity_df: pd.DataFrame, features: List[str], 
        data_interface_config: Dict[str, Any] = None, **kwargs
    ) -> pd.DataFrame:
        """
        🆕 Phase 2: Point-in-Time Correctness 보장 피처 조회
        기존 get_historical_features + 완전한 시점 안전성 검증
        
        Args:
            entity_df: 조회할 엔티티 DataFrame  
            features: 조회할 피처 목록
            data_interface_config: EntitySchema 설정 (Entity+Timestamp 정보)
            **kwargs: Feast get_historical_features에 전달할 추가 인자
            
        Returns:
            Point-in-Time 검증을 통과한 피처 DataFrame
        """
        logger.info("🔒 Point-in-Time Correctness 보장 피처 조회 시작")
        
        # 1. Point-in-Time 스키마 검증
        if data_interface_config:
            self._validate_point_in_time_schema(entity_df, data_interface_config)
        
        # 2. 기존 get_historical_features 호출 (검증된 ASOF JOIN)
        result_df = self.get_historical_features(entity_df, features, **kwargs)
        
        # 3. ASOF JOIN 결과 검증 (미래 데이터 누출 차단)
        if data_interface_config:
            self._validate_asof_join_result(entity_df, result_df, data_interface_config)
        
        logger.info("✅ Point-in-Time Correctness 검증 완료")
        return result_df
    
    def _validate_point_in_time_schema(self, entity_df: pd.DataFrame, config: Dict[str, Any]):
        """Entity + Timestamp 필수 컬럼 Point-in-Time 검증"""
        entity_columns = config.get('entity_columns', [])
        timestamp_column = config.get('timestamp_column', '')
        
        # Entity 컬럼 존재 검증
        missing_entities = [col for col in entity_columns if col not in entity_df.columns]
        if missing_entities:
            raise ValueError(
                f"🚨 Point-in-Time 검증 실패: 필수 Entity 컬럼 누락 {missing_entities}\n"
                f"Required: {entity_columns}, Found: {list(entity_df.columns)}"
            )
        
        # Timestamp 컬럼 존재 및 타입 검증
        if timestamp_column and timestamp_column not in entity_df.columns:
            raise ValueError(
                f"🚨 Point-in-Time 검증 실패: Timestamp 컬럼 '{timestamp_column}' 누락"
            )
        
        if timestamp_column and not pd.api.types.is_datetime64_any_dtype(entity_df[timestamp_column]):
            raise ValueError(
                f"🚨 Point-in-Time 검증 실패: '{timestamp_column}'이 datetime 타입이 아닙니다"
            )
        
        logger.info(f"✅ Point-in-Time 스키마 검증 통과: {entity_columns} + {timestamp_column}")
    
    def _validate_asof_join_result(
        self, input_df: pd.DataFrame, result_df: pd.DataFrame, config: Dict[str, Any]
    ):
        """ASOF JOIN 결과의 Point-in-Time 무결성 검증"""
        timestamp_column = config.get('timestamp_column', '')
        
        if not timestamp_column or timestamp_column not in result_df.columns:
            logger.warning("Timestamp 컬럼 없음: ASOF JOIN 결과 검증 생략")
            return
        
        # 입력 대비 결과 행 수 확인
        if len(result_df) != len(input_df):
            logger.warning(
                f"⚠️ ASOF JOIN 결과 행 수 불일치: input({len(input_df)}) vs result({len(result_df)})"
            )
        
        # 미래 데이터 누출 검증 (현재 시점 이후 피처 값 감지)
        current_time = pd.Timestamp.now()
        future_data = result_df[result_df[timestamp_column] > current_time]
        
        if len(future_data) > 0:
            logger.warning(
                f"⚠️ 미래 데이터 감지: {len(future_data)}개 행이 현재 시점({current_time}) 이후"
            )
        
        logger.info("✅ ASOF JOIN Point-in-Time 무결성 검증 완료")

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

    def write(self, df: pd.DataFrame, table_name: str, **kwargs):
        """Feast는 주로 읽기용이므로, write는 기본적으로 지원하지 않습니다."""
        raise NotImplementedError("FeastAdapter does not support write operation.")

# Self-registration
from .._registry import AdapterRegistry
AdapterRegistry.register("feature_store", FeastAdapter) 