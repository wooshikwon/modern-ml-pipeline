# Import trainer modules to trigger self-registration
from .trainer import Trainer

# Import the registry for external use (optional)
from .registry import TrainerRegistry

# Also import utility modules
from .modules.optimizer import OptunaOptimizer

__all__ = [
    "Trainer",
    "TrainerRegistry", 
    "OptunaOptimizer",
]