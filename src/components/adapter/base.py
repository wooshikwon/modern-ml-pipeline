"""
BaseAdapter - 데이터 어댑터 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseAdapter(ABC):
    """
    통합된 데이터 어댑터의 표준 계약

    모든 어댑터는 이 클래스를 상속받아 외부 시스템과의 데이터 읽기/쓰기를 처리합니다.
    각 어댑터는 자체적으로 클라이언트를 초기화하고 관리합니다.
    """

    def __init__(self, settings: Any):
        """
        어댑터 초기화

        Args:
            settings: 프로젝트 설정 객체
        """
        self.settings = settings

    @abstractmethod
    def read(self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
        """
        데이터 읽기 표준 인터페이스

        Args:
            source: 데이터 소스 (URI, 테이블명, 파일 경로 등)
            params: 읽기 파라미터 (SQL 템플릿 변수, 필터 조건 등)
            **kwargs: 추가 옵션

        Returns:
            pandas.DataFrame: 읽어온 데이터
        """
        raise NotImplementedError

    @abstractmethod
    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        데이터 쓰기 표준 인터페이스

        Args:
            df: 저장할 데이터프레임
            target: 저장 대상 (URI, 테이블명, 파일 경로 등)
            options: 쓰기 옵션 (write_mode, partition_by 등)
            **kwargs: 추가 옵션
        """
        raise NotImplementedError
