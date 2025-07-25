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
    Feature Store와 연동하여 데이터를 증강하는 Augmenter.
    DEV, PROD 등 Feature Store를 사용하는 환경을 위해 사용됩니다.
    """
    def __init__(self, settings: Settings, factory: "Factory"):
        self.settings = settings
        self.factory = factory
        self.feature_config = self.settings.model.augmenter.features or []
        
        # Feature Store 어댑터 초기화
        try:
            self.feature_store_adapter = self.factory.create_feature_store_adapter()
            logger.info("Feature Store 어댑터 초기화 성공")
        except (ValueError, Exception) as e:
            logger.warning(f"Feature Store 어댑터 초기화 실패: {e}")
            self.feature_store_adapter = None

    def augment(
        self,
        data: pd.DataFrame,
        run_mode: str = "batch",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Feature Store를 통해 피처를 증강합니다.
        """
        logger.info(f"Feature Store 피처 증강 시작 (모드: {run_mode})")
        
        if not self.feature_store_adapter:
            logger.warning("Feature Store 어댑터가 없어 원본 데이터를 반환합니다.")
            return data
        
        try:
            # FeatureStoreAdapter를 통해 피처 조회
            augmented_df = self.feature_store_adapter.read(
                model_input=data,
                run_mode=run_mode
            )
            
            logger.info(f"Feature Store 피처 증강 완료: {len(augmented_df.columns)}개 컬럼")
            return augmented_df
            
        except Exception as e:
            logger.error(f"Feature Store 피처 증강 실패: {e}")
            # 안전한 fallback: 원본 데이터 반환
            return data
