import pandas as pd
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from pathlib import Path

from src.interface.base_augmenter import BaseAugmenter
from src.utils.system.logger import logger
from src.settings.settings import Settings
from src.utils.system import sql_utils
# Factory는 dynamic import로 사용하여 순환 참조 방지

class LocalFileAugmenter(BaseAugmenter):
    """로컬 피처 파일과 조인하여 데이터를 증강하는 클래스. (개발용)"""
    def __init__(self, uri: str):
        parsed_uri = urlparse(uri)
        self.feature_path = Path(parsed_uri.path.lstrip('/'))
        if not self.feature_path.is_absolute():
            self.feature_path = Path(__file__).resolve().parent.parent.parent / self.feature_path

    def augment(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info(f"로컬 피처 파일로 증강을 시작합니다: {self.feature_path}")
        if not self.feature_path.exists():
            raise FileNotFoundError(f"로컬 피처 파일을 찾을 수 없습니다: {self.feature_path}")
        feature_df = pd.read_parquet(self.feature_path)
        return pd.merge(data, feature_df, on="member_id", how="left")

class Augmenter(BaseAugmenter):
    """
    실행 컨텍스트(run_mode)에 따라 배치 또는 실시간으로 동작하는 단일 증강기 클래스.
    🆕 Blueprint v17.0: Feature Store 선언적 방식과 기존 SQL 방식 모두 지원
    """
    def __init__(self, source_uri: Optional[str] = None, settings: Optional[Settings] = None, augmenter_config: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.augmenter_config = augmenter_config or {}
        
        # 🆕 Blueprint v17.0: type 기반 동작 방식 결정
        self.augmenter_type = self.augmenter_config.get('type', 'sql')  # 기본값: 기존 SQL 방식
        
        if self.augmenter_type == 'feature_store':
            # 🆕 새로운 Feature Store 방식
            self._init_feature_store_mode()
        else:
            # 🔄 기존 SQL 방식 (완전 호환성 유지)
            self._init_sql_mode(source_uri)
    
    def _init_feature_store_mode(self):
        """🆕 Feature Store 모드 초기화"""
        logger.info("🆕 Feature Store 모드로 Augmenter 초기화")
        
        self.source_uri = None
        self.sql_template_str = None
        self.realtime_features_list = []
        
        # Feature Store 설정 파싱
        self.feature_config = self.augmenter_config.get('features', [])
        
        # 모든 피처 목록 추출 (실시간 조회용)
        for namespace_config in self.feature_config:
            features = namespace_config.get('features', [])
            namespace = namespace_config.get('feature_namespace', '')
            self.realtime_features_list.extend([f"{namespace}.{feature}" for feature in features])
        
        logger.info(f"Feature Store 피처 설정: {len(self.feature_config)}개 네임스페이스, {len(self.realtime_features_list)}개 피처")
        
        # Feature Store 어댑터 초기화
        self._init_feature_store_adapters()
    
    def _init_sql_mode(self, source_uri: str):
        """🔄 기존 SQL 모드 초기화 (완전 호환성 유지)"""
        logger.info("🔄 기존 SQL 모드로 Augmenter 초기화")
        
        self.source_uri = source_uri
        self.sql_template_str = self._load_sql_template()
        self.realtime_features_list = sql_utils.get_selected_columns(self.sql_template_str)
        
        # 기존 어댑터 초기화
        self._init_sql_adapters()
    
    def _init_feature_store_adapters(self):
        """Feature Store 어댑터 초기화"""
        from src.core.factory import Factory
        factory = Factory(self.settings)
        
        try:
            self.feature_store_adapter = factory.create_feature_store_adapter()
            logger.info("Feature Store 어댑터 초기화 성공")
        except (ValueError, Exception) as e:
            logger.warning(f"Feature Store 어댑터 초기화 실패: {e}")
            self.feature_store_adapter = None
        
        # Redis 어댑터 (실시간용) - 연결 실패 시에도 계속 진행
        try:
            self.redis_adapter = factory.create_redis_adapter()
            logger.info("Redis 어댑터 초기화 성공")
        except (ImportError, Exception) as e:
            logger.warning(f"Redis 어댑터 초기화 실패 (정상적인 개발 환경 상황): {e}")
            self.redis_adapter = None
    
    def _init_sql_adapters(self):
        """기존 SQL 어댑터 초기화 (완전 호환성 유지)"""
        from src.core.factory import Factory
        factory = Factory(self.settings)
        self.batch_adapter = factory.create_data_adapter('bq')  # Augmenter는 항상 BigQuery를 사용
        
        # 실시간 모드를 위한 Redis 어댑터 (선택적)
        try:
            self.redis_adapter = factory.create_redis_adapter()
        except ImportError:
            logger.warning("Redis가 설치되지 않아 실시간 피처 조회 기능이 비활성화됩니다.")
            self.redis_adapter = None

    def _load_sql_template(self) -> str:
        """기존 SQL 템플릿 로딩 (완전 호환성 유지)"""
        parsed_uri = urlparse(self.source_uri)
        path = Path(parsed_uri.path.lstrip('/'))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Augmenter SQL 템플릿 파일을 찾을 수 없습니다: {path}")
        return path.read_text(encoding="utf-8")

    def augment(
        self,
        data: pd.DataFrame,
        run_mode: str,
        context_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        🔄 기존 인터페이스 100% 유지하면서 Feature Store 방식 지원 추가
        """
        if self.augmenter_type == 'feature_store':
            # 🆕 Feature Store 방식
            return self._augment_feature_store(data, run_mode, context_params, **kwargs)
        else:
            # 🔄 기존 SQL 방식 (완전 호환성 유지)
            if run_mode == "batch":
                return self._augment_batch(data, context_params)
            elif run_mode == "serving":
                return self._augment_realtime(data, kwargs.get("feature_store_config"))
            else:
                raise ValueError(f"지원하지 않는 Augmenter 실행 모드입니다: {run_mode}")
    
    def _augment_feature_store(self, data: pd.DataFrame, run_mode: str, context_params: Optional[Dict[str, Any]], **kwargs):
        """🆕 Feature Store 방식 피처 증강"""
        logger.info(f"🆕 Feature Store 방식 피처 증강 시작 (모드: {run_mode})")
        
        if not self.feature_store_adapter:
            logger.warning("Feature Store 어댑터가 없어 원본 데이터 반환")
            return data
        
        try:
            # FeatureStoreAdapter의 새로운 메서드 사용
            augmented_df = self.feature_store_adapter.get_features_from_config(
                entity_df=data,
                feature_config=self.feature_config,
                run_mode=run_mode
            )
            
            logger.info(f"Feature Store 피처 증강 완료: {len(augmented_df.columns)}개 컬럼")
            return augmented_df
            
        except Exception as e:
            logger.error(f"Feature Store 피처 증강 실패: {e}")
            # 안전한 fallback: 원본 데이터 반환
            return data

    def _augment_batch(
        self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """🔄 기존 배치 모드 피처 증강 (완전 호환성 유지)"""
        logger.info(f"🔄 기존 배치 모드 피처 증강을 시작합니다. (URI: {self.source_uri})")
        
        # 미리 생성된 배치 어댑터 사용 (Factory 생성 로직 제거)
        feature_df = self.batch_adapter.read(self.source_uri, params=context_params)
        return pd.merge(data, feature_df, on="member_id", how="left")

    def _augment_realtime(
        self, data: pd.DataFrame, feature_store_config: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """🔄 기존 실시간 모드 피처 증강 (완전 호환성 유지)"""
        if not feature_store_config:
            raise ValueError("실시간 증강을 위해서는 feature_store_config가 필요합니다.")
        if "member_id" not in data.columns:
            raise ValueError("'member_id' 컬럼이 없어 실시간 피처 조회를 할 수 없습니다.")

        user_ids = data["member_id"].tolist()
        logger.info(f"🔄 기존 방식: {len(user_ids)}명의 사용자에 대한 실시간 피처 조회를 시작합니다.")

        store_type = feature_store_config.get("store_type")
        
        if store_type == "redis":
            if self.redis_adapter is None:
                logger.warning("Redis가 설치되지 않아 실시간 피처 조회를 건너뜁니다.")
                feature_map = {}
            else:
                # 미리 생성된 Redis 어댑터 사용 (Factory 생성 로직 제거)
                feature_map = self.redis_adapter.get_features(user_ids, self.realtime_features_list)
        else:
            raise NotImplementedError(f"지원하지 않는 실시간 스토어 타입입니다: {store_type}")

        if not feature_map:
            logger.warning("조회된 실시간 피처가 없습니다.")
            feature_df = pd.DataFrame(columns=["member_id"] + self.realtime_features_list)
        else:
            feature_df = pd.DataFrame.from_records(list(feature_map.values()))
            feature_df["member_id"] = list(feature_map.keys())

        for col in self.realtime_features_list:
            if col not in feature_df.columns:
                feature_df[col] = None
        
        return pd.merge(data, feature_df, on="member_id", how="left")

    # 🔄 기존 메서드들 완전 유지 (Blueprint v13.0 호환성)
    def augment_batch(
        self, data: pd.DataFrame, sql_snapshot: str, context_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        🔄 배치 모드 피처 증강 (Blueprint v13.0 호환성 유지)
        SQL 스냅샷을 사용하여 배치 추론 시 피처 증강
        """
        logger.info(f"🔄 기존 배치 모드 피처 증강을 시작합니다. (SQL 스냅샷 사용)")
        
        # SQL 스냅샷을 직접 실행
        if sql_snapshot and hasattr(self, 'batch_adapter') and self.batch_adapter:
            feature_df = self.batch_adapter.read(sql_snapshot, params=context_params)
            return pd.merge(data, feature_df, on="member_id", how="left")
        else:
            logger.warning("SQL 스냅샷이 없거나 배치 어댑터가 없어 피처 증강을 건너뜁니다.")
            return data

    def augment_realtime(
        self, 
        data: pd.DataFrame, 
        sql_snapshot: str,
        feature_store_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        🔄 실시간 모드 피처 증강 (Blueprint v13.0 호환성 유지)
        SQL 스냅샷을 파싱하여 Feature Store 조회로 변환
        """
        if not feature_store_config:
            raise ValueError("실시간 증강을 위해서는 feature_store_config가 필요합니다.")
        if "member_id" not in data.columns:
            raise ValueError("'member_id' 컬럼이 없어 실시간 피처 조회를 할 수 없습니다.")

        user_ids = data["member_id"].tolist()
        logger.info(f"🔄 기존 실시간 방식: {len(user_ids)}명의 사용자에 대한 실시간 피처 조회를 시작합니다.")

        # SQL 스냅샷에서 피처 컬럼 추출
        if not feature_columns:
            from src.utils.system.sql_utils import get_selected_columns
            feature_columns = get_selected_columns(sql_snapshot)
        
        store_type = feature_store_config.get("store_type")
        
        if store_type == "redis":
            if self.redis_adapter is None:
                logger.warning("Redis가 설치되지 않아 실시간 피처 조회를 건너뜁니다.")
                feature_map = {}
            else:
                # SQL 스냅샷 기반 Feature Store 조회
                feature_map = self.redis_adapter.get_features(user_ids, feature_columns)
        else:
            raise NotImplementedError(f"지원하지 않는 실시간 스토어 타입입니다: {store_type}")

        if not feature_map:
            logger.warning("조회된 실시간 피처가 없습니다.")
            feature_df = pd.DataFrame(columns=["member_id"] + feature_columns)
        else:
            feature_df = pd.DataFrame.from_records(list(feature_map.values()))
            feature_df["member_id"] = list(feature_map.keys())

        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = None
        
        return pd.merge(data, feature_df, on="member_id", how="left")
