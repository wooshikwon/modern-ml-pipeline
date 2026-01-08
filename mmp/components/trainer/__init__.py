# Import base class
from .base import BaseTrainer

# Import the registry for external use
from .registry import TrainerRegistry

# Import trainer modules to trigger self-registration
from .trainer import Trainer

__all__ = [
    "BaseTrainer",
    "Trainer",
    "TrainerRegistry",
]
