# Import core preprocessor modules to ensure they register themselves
# Only importing modules that don't have external dependencies
from . import imputer
from . import scaler
from . import discretizer
from . import feature_generator

# Skip encoder for now due to category_encoders dependency
# from . import encoder  

__all__ = [
    'imputer',
    'scaler', 
    'discretizer',
    'feature_generator'
]