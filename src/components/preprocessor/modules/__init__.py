# Import core preprocessor modules to ensure they register themselves
# All modules should be imported to ensure proper registration
from . import imputer
from . import scaler
from . import discretizer
from . import feature_generator
from . import missing
from . import encoder  # Required for one_hot_encoder and label_encoder

__all__ = [
    'imputer',
    'scaler',
    'discretizer',
    'feature_generator',
    'missing',
    'encoder'
]