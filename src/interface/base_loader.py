
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import pandas as pd


class BaseLoader(ABC):
    """
    데이터 로딩을 위한 추상 기본 클래스(ABC).

    모든 로더 구현체는 이 클래스를 상속받아 `load` 메서드를 구현해야 합니다.
    이를 통해 파이프라인에서 일관된 방식으로 데이터를 로드할 수 있습니다.
    """

    @abstractmethod
    def load(self, dataset_name: str, settings: Settings) -> pd.DataFrame:
        """
        데이터 소스에서 데이터를 로드하여 pandas DataFrame으로 반환합니다.

        Args:
            dataset_name (str):
                Loader 내부의 여러개 데이터셋 경로 중 '특정 sql' 파일을 선택하도록 전달하는 이름.
            settings (Settings):
                환경 설정 등을 포함한 utils/ 도구에 전달할 객체.

        Returns:
            pd.DataFrame: 로드된 데이터.
        """
        pass
