# Import all fetcher modules to trigger self-registration
from ._modules.feature_store_fetcher import FeatureStoreAugmenter
from ._modules.pass_through_fetcher import PassThroughAugmenter

# Import the registry for external use
from ._registry import FetcherRegistry

__all__ = [
    "FeatureStoreAugmenter",
    "PassThroughAugmenter", 
    "FetcherRegistry",
]