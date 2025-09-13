"""
Fixtures Module - Example Data and Test Fixtures

이 모듈은 Pydantic 모델의 예제 데이터와 테스트용 fixtures를 제공합니다.
원래 model_config에 하드코딩되어 있던 예제 데이터들을 분리하여 관리합니다.
"""

from .recipe_examples import RECIPE_EXAMPLES
from .config_examples import CONFIG_EXAMPLES

__all__ = [
    "RECIPE_EXAMPLES",
    "CONFIG_EXAMPLES",
]