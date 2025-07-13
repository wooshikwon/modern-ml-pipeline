from abc import ABC, abstractmethod
from typing import Any

class BaseAdapter(ABC):
    """
    모든 어댑터의 최상위 추상 클래스.
    설정 객체를 받고, 클라이언트를 초기화하는 기본 구조를 정의한다.
    """
    def __init__(self, settings: Any):
        self.settings = settings
        # 클라이언트 초기화는 서브클래스에서 처리
        # self.client = self._get_client()

    @abstractmethod
    def _get_client(self) -> Any:
        """
        각 기술에 맞는 클라이언트 객체를 생성하고 반환합니다.
        (e.g., BigQuery Client, GCS Client, Redis Client)
        """
        raise NotImplementedError
