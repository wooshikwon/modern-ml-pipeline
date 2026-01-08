# DataHandler 컴포넌트 - Task별 데이터 처리 특화
from .base import BaseDataHandler

# Import modules to trigger self-registration
from .modules.sequence_handler import SequenceDataHandler
from .modules.tabular_handler import TabularDataHandler
from .modules.timeseries_handler import TimeseriesDataHandler
from .registry import DataHandlerRegistry

__all__ = [
    "BaseDataHandler",
    "DataHandlerRegistry",
    "TabularDataHandler",
    "TimeseriesDataHandler",
    "SequenceDataHandler",
]
