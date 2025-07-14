from typing import Dict, Any, List, Optional
from src.interface.base_adapter import BaseAdapter
from src.settings.settings import Settings
from src.utils.system.logger import logger


class FeatureStoreAdapter(BaseAdapter):
    """환경별 Feature Store 통합 어댑터 (Blueprint v17.0 - 선언적 Feature Store 지원)"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.feature_store_config = settings.feature_store
        self._init_connections()
    
    def _init_connections(self):
        """환경별 연결 초기화"""
        # 기존 Redis 어댑터 활용 (연결 실패 시에도 계속 진행)
        from src.core.factory import Factory
        factory = Factory(self.settings)
        
        try:
            self.redis_adapter = factory.create_redis_adapter()
            logger.info("Feature Store: Redis 어댑터 연결 성공")
        except (ImportError, Exception) as e:
            logger.warning(f"Feature Store: Redis 어댑터 연결 실패 (정상적인 개발 환경 상황): {e}")
            self.redis_adapter = None
        
        # 배치 모드를 위한 데이터 어댑터 (오프라인 스토어 시뮬레이션)
        try:
            self.batch_adapter = factory.create_data_adapter('bq')
            logger.info("Feature Store: 배치 어댑터 연결 성공")
        except Exception as e:
            logger.warning(f"Feature Store: 배치 어댑터 연결 실패 (정상적인 개발 환경 상황): {e}")
            self.batch_adapter = None
    
    def get_features_from_config(self, entity_df, feature_config: List[Dict[str, Any]], run_mode: str = "batch"):
        """
        🆕 선언적 Feature Store 설정에서 피처 조회
        feature_config 예시:
        [
            {"feature_namespace": "user_demographics", "features": ["age", "gender"]},
            {"feature_namespace": "user_behavior", "features": ["ltv"]}
        ]
        """
        logger.info(f"Feature Store에서 피처 조회 시작 (모드: {run_mode})")
        
        if run_mode == "batch":
            return self._get_features_batch_mode(entity_df, feature_config)
        else:
            return self._get_features_realtime_mode(entity_df, feature_config)
    
    def _get_features_batch_mode(self, entity_df, feature_config: List[Dict[str, Any]]):
        """배치 모드: 오프라인 스토어에서 대량 피처 조회"""
        logger.info("배치 모드 Feature Store 조회")
        
        # 모든 피처 네임스페이스에서 필요한 피처들을 수집
        all_features = []
        feature_namespaces = []
        
        for namespace_config in feature_config:
            namespace = namespace_config["feature_namespace"]
            features = namespace_config["features"]
            feature_namespaces.append(namespace)
            all_features.extend([f"{namespace}.{feature}" for feature in features])
        
        logger.info(f"조회할 피처 네임스페이스: {feature_namespaces}")
        logger.info(f"조회할 전체 피처: {all_features}")
        
        # 오프라인 스토어 시뮬레이션 (실제 구현에서는 Feature Store SDK 사용)
        result_df = self._simulate_offline_feature_store(entity_df, all_features)
        
        return result_df
    
    def _get_features_realtime_mode(self, entity_df, feature_config: List[Dict[str, Any]]):
        """실시간 모드: 온라인 스토어(Redis)에서 피처 조회"""
        logger.info("실시간 모드 Feature Store 조회")
        
        if "member_id" not in entity_df.columns:
            raise ValueError("'member_id' 컬럼이 없어 실시간 피처 조회를 할 수 없습니다.")
        
        entity_keys = entity_df["member_id"].tolist()
        
        # 모든 피처 네임스페이스에서 필요한 피처들을 수집
        all_features = []
        for namespace_config in feature_config:
            namespace = namespace_config["feature_namespace"]
            features = namespace_config["features"]
            all_features.extend([f"{namespace}.{feature}" for feature in features])
        
        logger.info(f"{len(entity_keys)}개 엔티티에 대한 실시간 피처 조회: {all_features}")
        
        if self.redis_adapter:
            feature_map = self.redis_adapter.get_features(entity_keys, all_features)
        else:
            logger.warning("Redis 어댑터가 없어 빈 결과 반환")
            feature_map = {}
        
        # Redis 결과를 DataFrame으로 변환
        if not feature_map:
            logger.warning("조회된 실시간 피처가 없습니다.")
            import pandas as pd
            result_df = pd.DataFrame(columns=["member_id"] + all_features)
        else:
            import pandas as pd
            result_df = pd.DataFrame.from_records(list(feature_map.values()))
            result_df["member_id"] = list(feature_map.keys())
        
        # 누락된 피처 컬럼 추가
        for feature in all_features:
            if feature not in result_df.columns:
                result_df[feature] = None
        
        return result_df
    
    def _simulate_offline_feature_store(self, entity_df, all_features):
        """오프라인 Feature Store 시뮬레이션 (개발/테스트용)"""
        import pandas as pd
        import numpy as np
        
        logger.info("오프라인 Feature Store 시뮬레이션 실행")
        
        # 기존 entity_df를 기반으로 Mock 피처 생성
        result_df = entity_df.copy()
        
        # Mock 피처 데이터 생성 (실제 구현에서는 Feature Store에서 조회)
        np.random.seed(42)  # 재현성을 위한 시드
        
        for feature in all_features:
            if "demographics.gender" in feature:
                result_df[feature.split('.')[-1]] = np.random.choice(["M", "F"], size=len(result_df))
            elif "demographics.age_group" in feature:
                result_df[feature.split('.')[-1]] = np.random.choice(["20s", "30s", "40s", "50s"], size=len(result_df))
            elif "behavior" in feature and "days" in feature:
                result_df[feature.split('.')[-1]] = np.random.randint(1, 365, size=len(result_df))
            elif "purchase" in feature:
                result_df[feature.split('.')[-1]] = np.random.randint(1, 1000, size=len(result_df))
            elif "session" in feature:
                result_df[feature.split('.')[-1]] = np.random.randint(30, 3600, size=len(result_df))
            else:
                # 기본 숫자 피처
                result_df[feature.split('.')[-1]] = np.random.rand(len(result_df))
        
        logger.info(f"Mock 피처 {len(all_features)}개 생성 완료")
        return result_df

    def get_historical_features(self, entity_df, features):
        """기존 호환성: 배치 모드 Feature Store 조회 (레거시)"""
        logger.info("레거시 배치 모드 Feature Store 조회")
        # 기존 방식과 호환성 유지 (임시 구현)
        return entity_df

    def get_online_features(self, entity_keys, features):
        """기존 호환성: 실시간 모드 Feature Store 조회 (레거시)"""
        if self.redis_adapter:
            logger.info(f"{len(entity_keys)}개 엔티티에 대한 레거시 실시간 피처 조회")
            return self.redis_adapter.get_features(entity_keys, features)
        else:
            logger.warning("Redis 어댑터가 없어 빈 결과 반환")
            return {}
    
    # BaseAdapter 인터페이스 구현
    def read(self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs):
        """BaseAdapter 호환성을 위한 read 메서드"""
        params = params or {}
        entity_df = params.get('entity_df')
        features = params.get('features')
        
        if entity_df is not None and features is not None:
            return self.get_historical_features(entity_df, features)
        else:
            logger.warning("entity_df 또는 features가 없어 빈 DataFrame 반환")
            import pandas as pd
            return pd.DataFrame()
    
    def write(self, df, target: str, options: Optional[Dict[str, Any]] = None, **kwargs):
        """BaseAdapter 호환성을 위한 write 메서드 (추후 구현)"""
        logger.info(f"Feature Store write 요청: {target}")
        pass 