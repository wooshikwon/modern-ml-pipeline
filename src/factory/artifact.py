"""Compatibility shim for PyfuncWrapper import path.

This module re-exports PyfuncWrapper from the new location
`src.utils.integrations.pyfunc_wrapper` to maintain backward compatibility
with existing imports like `from src.factory.artifact import PyfuncWrapper`.
"""

from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper  # noqa: F401

__all__ = ["PyfuncWrapper"]
