"""
Utility functions and exception classes for hidden-regime package.

Provides common utilities, custom exceptions, and helper functions
used across the package.
"""

from .exceptions import (
    DataLoadError,
    DataQualityError,
    HiddenRegimeError,
    ValidationError,
)

__all__ = ["HiddenRegimeError", "DataLoadError", "DataQualityError", "ValidationError"]
