from abc import abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

from src.interface.base_adapter import BaseAdapter

class BaseDataAdapter(BaseAdapter):
    """
    데이터의 읽기/쓰기를 모두 지원하는 데이터 어댑터의 표준 인터페이스.
    """

    @abstractmethod
    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        지정된 소스에서 파라미터를 사용하여 데이터를 읽어옵니다.
        """
        raise NotImplementedError

    @abstractmethod
    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        지정된 목적지에 데이터프레임을 씁니다.
        """
        raise NotImplementedError