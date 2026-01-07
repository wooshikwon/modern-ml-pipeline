# Import core preprocessor modules to ensure they register themselves
# All modules should be imported to ensure proper registration
from . import encoder  # Required for one_hot_encoder and label_encoder
from . import discretizer, feature_generator, imputer, missing, scaler

__all__ = ["imputer", "scaler", "discretizer", "feature_generator", "missing", "encoder"]
