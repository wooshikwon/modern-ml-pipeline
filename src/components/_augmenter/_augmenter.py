from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING
from src.interface import BaseAugmenter
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings
    from src.engine import Factory


class Augmenter(BaseAugmenter):
    """
    Feature Store (Feast)를 사용하여 피처를 증강하는 클래스.
    DEV/PROD 환경에서 사용됩니다.
    """
    def __init__(self, settings: Settings, factory: Factory):
        self.settings = settings
        self.factory = factory
        self.feature_store_adapter = self.factory.create_feature_store_adapter()

    def augment(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info("Feature Store를 통해 피처 증강을 시작합니다.")
        
        # ... (피처 증강 로직) ...
        # 이 부분은 실제 FeastAdapter 호출 로직으로 채워져야 합니다.
        # 지금은 예시로 원본 DataFrame을 반환합니다.
        
        logger.info("피처 증강 완료.")
        return df
