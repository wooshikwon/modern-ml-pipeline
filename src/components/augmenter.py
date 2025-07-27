import pandas as pd
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from pathlib import Path

from src.interface.base_augmenter import BaseAugmenter
from src.utils.system.logger import logger
from src.settings import Settings
from src.utils.system import sql_utils
# Factory는 dynamic import로 사용하여 순환 참조 방지

class PassThroughAugmenter(BaseAugmenter):
    """
    Blueprint 원칙 9 구현: LOCAL 환경의 의도적 제약
    "제약은 단순함을 낳고, 단순함은 집중을 낳는다"
    
    LOCAL 환경에서 사용되는 Augmenter로, 데이터를 변경 없이 그대로 반환하여
    Feature Store나 복잡한 피처 증강 없이 빠른 실험과 디버깅에 집중할 수 있게 합니다.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info(f"LOCAL 환경: PassThroughAugmenter 초기화 (환경: {self.settings.environment.app_env})")
    
    def augment(
        self, 
        data: pd.DataFrame, 
        run_mode: str = "batch",
        context_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """데이터를 변경 없이 그대로 반환 (의도된 설계)"""
        logger.info("LOCAL 환경: Augmenter Pass-Through 모드 - 피처 증강 건너뛰기 (Blueprint 철학)")
        logger.info(f"입력 데이터: {len(data)} 행, {len(data.columns)} 컬럼")
        return data

    def augment_batch(
        self, data: pd.DataFrame, sql_snapshot: str, context_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """배치 모드에서도 데이터를 그대로 반환"""
        logger.info("LOCAL 환경: 배치 모드 Pass-Through")
        return data

    def augment_realtime(
        self, 
        data: pd.DataFrame, 
        sql_snapshot: str,
        feature_store_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """실시간 모드에서도 데이터를 그대로 반환"""
        logger.info("LOCAL 환경: 실시간 모드 Pass-Through")
        return data

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
    🆕 Phase 2: Type 기반 통합 Augmenter (Pass-through + Feature Store)
    
    단일 클래스가 settings.recipe.model.augmenter.type에 따라 
    적절한 증강 모드로 동작하는 현대화된 구조
    """
    def __init__(self, settings: Settings, factory: "Factory"):
        self.settings = settings
        self.factory = factory
        
        # Phase 1 경로: settings.recipe.model.augmenter.type 사용
        self.augmenter_type = self.settings.recipe.model.augmenter.type if hasattr(self.settings.recipe.model, 'augmenter') and self.settings.recipe.model.augmenter else "pass_through"
        
        logger.info(f"🔄 Augmenter 초기화: type={self.augmenter_type}")
        
        # type 기반 초기화
        if self.augmenter_type == "feature_store":
            try:
                # Factory에서 data_adapter("feature_store") 생성
                self.feast_adapter = factory.create_data_adapter("feature_store")
                self.feature_config = self.settings.recipe.model.augmenter.features or []
                logger.info("✅ Feature Store 어댑터 초기화 성공")
            except Exception as e:
                logger.warning(f"⚠️ Feature Store 어댑터 초기화 실패: {e}")
                logger.info("🔄 Pass-through 모드로 자동 전환")
                self.augmenter_type = "pass_through"
                self.feast_adapter = None
        else:
            # pass_through는 별도 초기화 불필요
            self.feast_adapter = None
            self.feature_config = []

    def augment(
        self,
        spine_df: pd.DataFrame,
        run_mode: str = "batch",
        **kwargs,
    ) -> pd.DataFrame:
        """
        🆕 Phase 2: Type 기반 명확한 분기로 증강 수행
        
        Args:
            spine_df: Entity+Timestamp 스파인 DataFrame
            run_mode: "batch" 또는 "realtime" 
            **kwargs: 추가 매개변수
            
        Returns:
            증강된 DataFrame (type에 따라 원본 또는 Feature Store 증강)
        """
        logger.info(f"🔄 피처 증강 시작: type={self.augmenter_type}, mode={run_mode}")
        
        if self.augmenter_type == "pass_through":
            # Blueprint 원칙 9: LOCAL 환경 의도적 제약
            logger.info("✅ Pass-through 모드: Feature Store 없이 학습 (Blueprint 철학)")
            logger.info(f"   입력 데이터: {len(spine_df)} 행, {len(spine_df.columns)} 컬럼")
            return spine_df
            
        elif self.augmenter_type == "feature_store" and self.feast_adapter:
            # Phase 2: Point-in-Time Correctness 보장 피처 증강
            logger.info("🔒 Feature Store 모드: Point-in-Time 안전성 보장")
            
            try:
                # Phase 1 EntitySchema 정보 활용
                data_interface_config = self._get_data_interface_config()
                features = self._build_feature_list()
                
                # 🆕 Phase 2: 검증 강화된 피처 조회
                augmented_df = self.feast_adapter.get_historical_features_with_validation(
                    entity_df=spine_df,
                    features=features,
                    data_interface_config=data_interface_config
                )
                
                logger.info(f"✅ Feature Store 증강 완료: {len(augmented_df)} 행, {len(augmented_df.columns)} 컬럼")
                return augmented_df
                
            except Exception as e:
                logger.error(f"❌ Feature Store 증강 실패: {e}")
                logger.info("🔄 안전한 fallback: 원본 데이터 반환")
                return spine_df
        else:
            # fallback: pass_through로 동작
            logger.warning("⚠️ 알 수 없는 augmenter type, pass-through로 동작")
            return spine_df
    
    def _get_data_interface_config(self) -> Dict[str, Any]:
        """Phase 1 EntitySchema + Data Interface 설정 추출 (27개 Recipe 대응)"""
        try:
            # Entity + Timestamp는 entity_schema에서
            entity_schema = self.settings.recipe.model.loader.entity_schema
            # ML 설정들은 data_interface에서
            data_interface = self.settings.recipe.model.data_interface
            
            return {
                'entity_columns': entity_schema.entity_columns,
                'timestamp_column': entity_schema.timestamp_column,
                'target_column': data_interface.target_column,  # 🔄 수정: data_interface에서 가져옴
                'task_type': data_interface.task_type           # 🔄 수정: data_interface에서 가져옴
            }
        except Exception as e:
            logger.warning(f"Schema 정보 추출 실패: {e}")
            return {}
    
    def _build_feature_list(self) -> List[str]:
        """Feature Store 피처 목록 생성"""
        try:
            features = []
            for feature_group in self.feature_config:
                namespace = feature_group.get('feature_namespace', '')
                feature_names = feature_group.get('features', [])
                
                for feature_name in feature_names:
                    features.append(f"{namespace}:{feature_name}")
            
            logger.info(f"🎯 Feature 목록 생성: {len(features)}개 피처")
            return features
            
        except Exception as e:
            logger.warning(f"Feature 목록 생성 실패: {e}")
            return []
