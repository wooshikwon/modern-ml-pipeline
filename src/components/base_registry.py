"""
BaseRegistry - 모든 Registry의 공통 기반 클래스

각 컴포넌트 Registry(AdapterRegistry, EvaluatorRegistry 등)가 상속받아
일관된 인터페이스를 제공한다.

사용 예시:
    class AdapterRegistry(BaseRegistry[BaseAdapter]):
        _registry: Dict[str, Type[BaseAdapter]] = {}
        _base_class = BaseAdapter
"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Type, TypeVar

from src.utils.core.logger import logger

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """
    모든 Registry의 공통 기반 클래스.

    Generic[T]를 통해 각 Registry가 관리하는 컴포넌트 타입을 명시한다.
    하위 클래스는 반드시 _registry와 _base_class를 정의해야 한다.

    Attributes:
        _registry: 등록된 컴포넌트를 저장하는 딕셔너리. 하위 클래스에서 반드시 정의.
        _base_class: 등록 가능한 컴포넌트의 기반 클래스. None이면 타입 검증 생략.
    """

    _registry: Dict[str, Type[T]]
    _base_class: Type[T] = None

    @classmethod
    def register(cls, key: str, klass: Type[T]) -> None:
        """
        컴포넌트를 Registry에 등록한다.

        Args:
            key: 등록 키 (예: 'storage', 'classification')
            klass: 등록할 클래스

        Raises:
            TypeError: _base_class가 정의되어 있고 klass가 그 하위 클래스가 아닌 경우
        """
        if cls._base_class is not None:
            if not issubclass(klass, cls._base_class):
                raise TypeError(
                    f"{klass.__name__}은(는) {cls._base_class.__name__}의 하위 클래스여야 합니다"
                )
        cls._registry[key] = klass

    @classmethod
    def get_class(cls, key: str) -> Type[T]:
        """
        등록된 클래스를 반환한다.

        Args:
            key: 조회할 키

        Returns:
            등록된 클래스

        Raises:
            KeyError: 등록되지 않은 키인 경우
        """
        if key not in cls._registry:
            available = cls.list_keys()
            raise KeyError(f"알 수 없는 키: '{key}'. 사용 가능: {available}")
        return cls._registry[key]

    @classmethod
    def create(cls, key: str, *args: Any, **kwargs: Any) -> T:
        """
        등록된 클래스의 인스턴스를 생성한다.

        Args:
            key: 생성할 컴포넌트의 키
            *args: 생성자에 전달할 위치 인자
            **kwargs: 생성자에 전달할 키워드 인자

        Returns:
            생성된 인스턴스
        """
        klass = cls.get_class(key)
        return klass(*args, **kwargs)

    @classmethod
    def list_keys(cls) -> List[str]:
        """
        등록된 모든 키 목록을 반환한다.

        Returns:
            등록된 키 목록
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """테스트용: 등록된 항목 모두 제거"""
        cls._registry.clear()
