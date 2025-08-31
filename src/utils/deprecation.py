"""
Deprecation utilities for marking legacy code
Phase 4: Deprecation warnings implementation
"""

import warnings
import functools
from typing import Callable, Optional, Any


def deprecated(
    reason: str,
    version: str = "2.0",
    alternative: Optional[str] = None
) -> Callable:
    """
    Mark a function or method as deprecated.
    
    Args:
        reason: Reason for deprecation
        version: Version when it will be removed
        alternative: Alternative method to use
    
    Returns:
        Decorated function that shows deprecation warning
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__}() is deprecated and will be removed in v{version}. {reason}"
            if alternative:
                msg += f" Use {alternative} instead."
            
            warnings.warn(
                msg,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class DeprecatedProperty:
    """Mark a property as deprecated."""
    
    def __init__(self, prop: property, reason: str, version: str = "2.0", alternative: Optional[str] = None):
        """
        Initialize deprecated property wrapper.
        
        Args:
            prop: Original property
            reason: Reason for deprecation
            version: Version when it will be removed
            alternative: Alternative to use
        """
        self.prop = prop
        self.reason = reason
        self.version = version
        self.alternative = alternative
    
    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Get property value with deprecation warning."""
        if obj is None:
            return self
        
        msg = f"Property '{self.prop.fget.__name__}' is deprecated and will be removed in v{self.version}. {self.reason}"
        if self.alternative:
            msg += f" Use {self.alternative} instead."
            
        warnings.warn(
            msg,
            category=DeprecationWarning,
            stacklevel=2
        )
        return self.prop.fget(obj)
    
    def __set__(self, obj: Any, value: Any) -> None:
        """Set property value with deprecation warning."""
        if self.prop.fset is None:
            raise AttributeError("can't set attribute")
        
        msg = f"Setting property '{self.prop.fget.__name__}' is deprecated and will be removed in v{self.version}. {self.reason}"
        if self.alternative:
            msg += f" Use {self.alternative} instead."
            
        warnings.warn(
            msg,
            category=DeprecationWarning,
            stacklevel=2
        )
        self.prop.fset(obj, value)


def show_deprecation_warning(
    feature: str,
    version: str = "2.0",
    alternative: Optional[str] = None,
    critical: bool = False
) -> None:
    """
    Show a deprecation warning message.
    
    Args:
        feature: Feature that is deprecated
        version: Version when it will be removed
        alternative: Alternative to use
        critical: Whether this is a critical deprecation
    """
    msg = f"⚠️ {feature} is deprecated and will be removed in v{version}."
    if alternative:
        msg += f" Please use {alternative} instead."
    
    if critical:
        msg = f"🔴 CRITICAL: {msg} This will stop working soon!"
    
    warnings.warn(
        msg,
        category=DeprecationWarning,
        stacklevel=2
    )