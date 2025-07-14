from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger

# 🆕 Feast 완전 활용을 위한 import
try:
    from feast import FeatureStore
    from feast.errors import FeatureStoreException
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger.warning("Feast 라이브러리를 찾을 수 없습니다. 시뮬레이션 모드로 실행됩니다.")


class FeatureStoreAdapter(BaseAdapter):
    """환경별 Feature Store 통합 어댑터 (Blueprint v17.0 - 선언적 Feature Store 지원)"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.feature_store_config = settings.feature_store
        self._init_connections()
    
    def _init_connections(self):
        """환경별 연결 초기화 (🆕 Feast 완전 활용)"""
        # 🆕 Feast 연결 시도
        self.feast_store = None
        if FEAST_AVAILABLE:
            try:
                self.feast_store = self._create_feast_store()
                logger.info("Feature Store: Feast 연결 성공")
            except Exception as e:
                logger.warning(f"Feature Store: Feast 연결 실패, 시뮬레이션 모드로 전환: {e}")
                self.feast_store = None
        
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
    
    def _create_feast_store(self) -> FeatureStore:
        """
        🆕 Config 기반 Feast 스토어 생성
        Blueprint 원칙 1: 설정은 인프라 - config/*.yaml의 feast_config 완전 활용
        """
        if not FEAST_AVAILABLE:
            raise ImportError("Feast 라이브러리가 설치되지 않았습니다")
        
        feast_config = self.settings.feature_store.feast_config
        logger.info(f"Feast 설정 로드: project={feast_config.project}, provider={feast_config.provider}")
        
        # 🎯 Blueprint 원칙: config 기반 완전 설정
        # 임시 feature_store.yaml 생성 (런타임에서 Config 기반으로)
        import tempfile
        import yaml
        import os
        
        # Config를 Feast 설정 형식으로 변환
        feast_yaml_config = {
            "project": feast_config.project,
            "provider": feast_config.provider,
        }
        
        # Registry 설정
        if hasattr(feast_config, 'registry'):
            if hasattr(feast_config.registry, 'registry_type'):
                feast_yaml_config["registry"] = {
                    "registry_type": feast_config.registry.registry_type,
                    "path": feast_config.registry.path
                }
            else:
                feast_yaml_config["registry"] = feast_config.registry
        
        # Offline Store 설정
        if hasattr(feast_config, 'offline_store'):
            feast_yaml_config["offline_store"] = feast_config.offline_store.__dict__
        
        # Online Store 설정
        if hasattr(feast_config, 'online_store'):
            feast_yaml_config["online_store"] = feast_config.online_store.__dict__
        
        # 기타 설정
        if hasattr(feast_config, 'entity_key_serialization_version'):
            feast_yaml_config["entity_key_serialization_version"] = feast_config.entity_key_serialization_version
        
        # 임시 파일에 Feast 설정 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(feast_yaml_config, f)
            temp_config_path = f.name
        
        try:
            # Feast 스토어 생성
            # 현재 디렉토리를 임시로 변경 (Feast는 현재 디렉토리에서 feature_store.yaml 찾음)
            current_dir = os.getcwd()
            temp_dir = os.path.dirname(temp_config_path)
            
            # feature_store.yaml로 복사
            feature_store_yaml_path = os.path.join(temp_dir, "feature_store.yaml")
            import shutil
            shutil.copy2(temp_config_path, feature_store_yaml_path)
            
            # 임시 디렉토리로 이동하여 Feast 스토어 생성
            os.chdir(temp_dir)
            feast_store = FeatureStore(repo_path=temp_dir)
            
            # 원래 디렉토리로 복귀
            os.chdir(current_dir)
            
            logger.info("Feast 스토어 생성 완료")
            return feast_store
            
        except Exception as e:
            os.chdir(current_dir)  # 예외 발생 시에도 원래 디렉토리로 복귀
            raise e
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            if os.path.exists(feature_store_yaml_path):
                os.unlink(feature_store_yaml_path)
    
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
        """배치 모드: 오프라인 스토어에서 대량 피처 조회 (🆕 Feast 완전 활용)"""
        logger.info("배치 모드 Feature Store 조회")
        
        # 모든 피처 네임스페이스에서 필요한 피처들을 수집
        all_features = []
        feature_namespaces = []
        
        for namespace_config in feature_config:
            namespace = namespace_config["feature_namespace"]
            features = namespace_config["features"]
            feature_namespaces.append(namespace)
            all_features.extend([f"{namespace}:{feature}" for feature in features])  # Feast 형식
        
        logger.info(f"조회할 피처 네임스페이스: {feature_namespaces}")
        logger.info(f"조회할 전체 피처: {all_features}")
        
        # 🆕 실제 Feast 오프라인 스토어 조회 시도
        if self.feast_store:
            try:
                result_df = self._get_features_from_feast_offline(entity_df, all_features)
                logger.info("Feast 오프라인 스토어에서 피처 조회 완료")
                return result_df
            except Exception as e:
                logger.warning(f"Feast 오프라인 스토어 조회 실패, 시뮬레이션으로 전환: {e}")
        
        # 오프라인 스토어 시뮬레이션 (Feast 실패 시 또는 비활성화 시)
        result_df = self._simulate_offline_feature_store(entity_df, all_features)
        
        return result_df
    
    def _get_features_from_feast_offline(self, entity_df, features: List[str]) -> pd.DataFrame:
        """
        🆕 실제 Feast 오프라인 스토어에서 피처 조회
        Point-in-time join을 통한 안전한 피처 조회
        """
        try:
            # 🎯 Feast Point-in-time join 수행
            # event_timestamp가 필요하므로 확인
            if "event_timestamp" not in entity_df.columns:
                logger.warning("event_timestamp 컬럼이 없어 현재 시간으로 설정")
                from datetime import datetime
                entity_df = entity_df.copy()
                entity_df["event_timestamp"] = datetime.now()
            
            # Feast feature service를 통한 피처 조회
            logger.info(f"Feast 오프라인 스토어에서 {len(features)}개 피처 조회")
            
            # get_historical_features를 통한 Point-in-time join
            historical_features = self.feast_store.get_historical_features(
                entity_df=entity_df,
                features=features,
            )
            
            # Pandas DataFrame으로 변환
            result_df = historical_features.to_df()
            
            logger.info(f"Feast 오프라인 조회 완료: {len(result_df)} 행, {len(result_df.columns)} 컬럼")
            return result_df
            
        except Exception as e:
            logger.error(f"Feast 오프라인 스토어 조회 중 오류: {e}")
            raise e
    
    def _get_features_realtime_mode(self, entity_df, feature_config: List[Dict[str, Any]]):
        """실시간 모드: 온라인 스토어(Redis)에서 피처 조회 (🆕 Feast 완전 활용)"""
        logger.info("실시간 모드 Feature Store 조회")
        
        if "member_id" not in entity_df.columns:
            raise ValueError("'member_id' 컬럼이 없어 실시간 피처 조회를 할 수 없습니다.")
        
        entity_keys = entity_df["member_id"].tolist()
        
        # 모든 피처 네임스페이스에서 필요한 피처들을 수집
        all_features = []
        for namespace_config in feature_config:
            namespace = namespace_config["feature_namespace"]
            features = namespace_config["features"]
            all_features.extend([f"{namespace}:{feature}" for feature in features])  # Feast 형식
        
        logger.info(f"{len(entity_keys)}개 엔티티에 대한 실시간 피처 조회: {all_features}")
        
        # 🆕 실제 Feast 온라인 스토어 조회 시도
        if self.feast_store:
            try:
                result_df = self._get_features_from_feast_online(entity_keys, all_features)
                logger.info("Feast 온라인 스토어에서 피처 조회 완료")
                return result_df
            except Exception as e:
                logger.warning(f"Feast 온라인 스토어 조회 실패, Redis 어댑터로 전환: {e}")
        
        # 기존 Redis 어댑터 사용 (Feast 실패 시 또는 비활성화 시)
        if self.redis_adapter:
            feature_map = self.redis_adapter.get_features(entity_keys, all_features)
        else:
            logger.warning("Redis 어댑터가 없어 빈 결과 반환")
            feature_map = {}
        
        # Redis 결과를 DataFrame으로 변환
        if not feature_map:
            logger.warning("조회된 실시간 피처가 없습니다.")
            result_df = pd.DataFrame(columns=["member_id"] + all_features)
        else:
            result_df = pd.DataFrame.from_records(list(feature_map.values()))
            result_df["member_id"] = list(feature_map.keys())
        
        # 누락된 피처 컬럼 추가
        for feature in all_features:
            if feature not in result_df.columns:
                result_df[feature] = None
        
        return result_df
    
    def _get_features_from_feast_online(self, entity_keys: List[str], features: List[str]) -> pd.DataFrame:
        """
        🆕 실제 Feast 온라인 스토어에서 피처 조회
        실시간 저지연 피처 조회를 통한 API 서빙 지원
        """
        try:
            # 🎯 Feast 온라인 스토어 조회
            logger.info(f"Feast 온라인 스토어에서 {len(entity_keys)}개 엔티티, {len(features)}개 피처 조회")
            
            # Entity 데이터 준비 (Feast 온라인 조회 형식)
            entity_dict = {}
            for i, entity_key in enumerate(entity_keys):
                entity_dict[i] = {"member_id": entity_key}  # member_id는 주요 entity
            
            # get_online_features를 통한 실시간 조회
            online_features = self.feast_store.get_online_features(
                features=features,
                entity_rows=list(entity_dict.values())
            )
            
            # Pandas DataFrame으로 변환
            result_df = online_features.to_df()
            
            logger.info(f"Feast 온라인 조회 완료: {len(result_df)} 행, {len(result_df.columns)} 컬럼")
            return result_df
            
        except Exception as e:
            logger.error(f"Feast 온라인 스토어 조회 중 오류: {e}")
            raise e
    
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
            # 🆕 Feast 형식 (namespace:feature) 지원
            if ":" in feature:
                namespace, feature_name = feature.split(":", 1)
                column_name = feature_name  # 컬럼명은 feature_name만 사용
            else:
                # 기존 형식 (namespace.feature) 지원
                feature_name = feature.split('.')[-1]
                column_name = feature_name
            
            if "demographics" in feature and "gender" in feature:
                result_df[column_name] = np.random.choice(["M", "F"], size=len(result_df))
            elif "demographics" in feature and "age" in feature:
                result_df[column_name] = np.random.choice(["20s", "30s", "40s", "50s"], size=len(result_df))
            elif "behavior" in feature and "days" in feature:
                result_df[column_name] = np.random.randint(1, 365, size=len(result_df))
            elif "purchase" in feature:
                result_df[column_name] = np.random.randint(1, 1000, size=len(result_df))
            elif "session" in feature:
                result_df[column_name] = np.random.randint(30, 3600, size=len(result_df))
            else:
                # 기본 숫자 피처
                result_df[column_name] = np.random.rand(len(result_df))
        
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