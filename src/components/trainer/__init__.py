# Import trainer modules to trigger self-registration
from .modules.trainer import Trainer

# Import the registry for external use (optional)
from .registry import TrainerRegistry

# Also import utility modules
from .modules.data_handler import split_data, prepare_training_data
from .modules.optimizer import OptunaOptimizer

__all__ = [
    "Trainer",
    "TrainerRegistry", 
    "split_data",
    "prepare_training_data",
    "OptunaOptimizer",
]