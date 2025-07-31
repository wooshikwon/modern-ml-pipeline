# src/components/_augmenter/_pass_through.py
import pandas as pd

from src.interface import BaseAugmenter
from src.utils.system.logger import logger


class PassThroughAugmenter(BaseAugmenter):
    """
    피처 증강을 수행하지 않고 원본 데이터를 그대로 통과시키는 Augmenter.
    로컬 환경이나 피처 증강이 필요 없는 시나리오에서 사용됩니다.
    """

    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        'passthrough' 모드가 활성화되었거나 레시피에 augmenter가 정의되지 않았음을
        알리는 로그를 남기고, 입력 DataFrame을 수정 없이 그대로 반환합니다.
        """
        logger.info("피처 증강을 건너뜁니다. ('passthrough' 모드 또는 레시피에 augmenter 미정의)")
        return df 