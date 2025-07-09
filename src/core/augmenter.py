from typing import Dict, Union

import pandas as pd

from src.interface.base_augmenter import BaseAugmenter
from src.utils.logger import logger


class Augmenter(BaseAugmenter):
    """
    프로젝트를 위한 커스텀 데이터 증강기.
    loader에서 받은 데이터를 feature store에서 추출한 데이터와 조인하여 증강하는 역할.
    """
    def augment(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        입력 데이터를 그대로 반환. 향후 여러 테이블 조인 등의 로직이 추가될 수 있음.
        """
        logger.info("Augmenter is in pass-through mode.")
        if isinstance(data, dict):
            raise NotImplementedError("Handling multiple dataframes is not implemented yet.")
        return data