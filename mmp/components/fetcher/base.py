"""
BaseFetcher - 데이터 증강 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import pandas as pd


class BaseFetcher(ABC):
    """데이터 증강을 위한 추상 기본 클래스(ABC)."""

    def __init__(self, settings: Any = None, **kwargs: Any):
        """
        Fetcher 초기화

        Args:
            settings: 프로젝트 설정 객체 (선택사항, 다른 Base 클래스와 인터페이스 통일)
            **kwargs: 추가 키워드 인자
        """
        self.settings = settings

    @abstractmethod
    def fetch(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], run_mode: str = "batch"
    ) -> pd.DataFrame:
        """
        데이터를 증강하는 추상 메서드입니다.
        단일 또는 여러 개의 데이터프레임을 입력받아, 조인, 정제 등의 처리를 거친
        하나의 증강된 데이터프레임을 반환합니다.

        Args:
            data: 입력 데이터프레임 또는 이름 있는 데이터프레임 딕셔너리
            run_mode: "train" | "batch" | "serving" 중 하나로 동작 모드 지정
        """
        raise NotImplementedError
