# Import base class
from .base import BaseModel

__all__ = ["BaseModel"]

# torch-extras 모델은 torch 설치 시에만 노출 (custom/__init__.py와 동일 패턴)
try:
    from .custom.ft_transformer import FTTransformerClassifier, FTTransformerRegressor

    __all__ += ["FTTransformerClassifier", "FTTransformerRegressor"]
except ImportError:
    pass
