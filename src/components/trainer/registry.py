"""
Trainer Registry - Self-registration pattern for trainers.
"""

from __future__ import annotations

from typing import Dict, Type

from src.components.base_registry import BaseRegistry

from .base import BaseTrainer


class TrainerRegistry(BaseRegistry[BaseTrainer]):
    """Trainer 전용 Registry."""

    _registry: Dict[str, Type[BaseTrainer]] = {}
    _base_class = BaseTrainer
