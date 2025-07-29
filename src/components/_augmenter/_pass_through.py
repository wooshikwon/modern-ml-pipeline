# src/components/_augmenter/_pass_through.py
from src.interface import BaseAugmenter
from src.settings import Settings
import pandas as pd

class PassThroughAugmenter(BaseAugmenter):
    """
    LOCAL 환경에서 사용되는 Augmenter.
    데이터를 수정하지 않고 그대로 반환합니다.
    """
    def __init__(self, settings: Settings):
        self.settings = settings

    def augment(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df 