# Import base class
from .base import BaseFetcher

# Import all fetcher modules to trigger self-registration
from .modules.feature_store_fetcher import FeatureStoreFetcher
from .modules.pass_through_fetcher import PassThroughFetcher

# Import the registry for external use
from .registry import FetcherRegistry

__all__ = [
    "BaseFetcher",
    "FeatureStoreFetcher",
    "PassThroughFetcher",
    "FetcherRegistry",
]
