# Import trainer modules to trigger self-registration
from ._modules.trainer import Trainer

# Import the registry for external use (optional)
from ._registry import TrainerRegistry

# Also import utility modules
from ._modules.data_handler import split_data, prepare_training_data
from ._modules.optimizer import OptunaOptimizer

__all__ = [
    "Trainer",
    "TrainerRegistry", 
    "split_data",
    "prepare_training_data",
    "OptunaOptimizer",
]