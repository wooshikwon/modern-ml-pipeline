# DataHandler 컴포넌트 - Task별 데이터 처리 특화
from .registry import DataHandlerRegistry
from src.interface import BaseDataHandler

# Import modules to trigger self-registration
from .modules.tabular_handler import TabularDataHandler
from .modules.timeseries_handler import TimeseriesDataHandler

__all__ = [
    "DataHandlerRegistry",
    "BaseDataHandler", 
    "TabularDataHandler",
    "TimeseriesDataHandler"
]