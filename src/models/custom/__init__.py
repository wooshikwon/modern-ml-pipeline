"""
Custom Models Module

sklearn interface를 제공하는 커스텀 모델들을 포함합니다.
- FT-Transformer: Deep learning tabular model
- Timeseries Wrappers: ARIMA, ExponentialSmoothing sklearn 호환 wrapper들
"""

# FT-Transformer 모델들
try:
    from .ft_transformer import FTTransformerWrapperBase
    __all__ = ['FTTransformerWrapperBase']
except ImportError:
    __all__ = []

# Timeseries wrapper 모델들  
try:
    from .timeseries_wrappers import ARIMAWrapper, ExponentialSmoothingWrapper
    __all__.extend(['ARIMAWrapper', 'ExponentialSmoothingWrapper'])
except ImportError:
    pass