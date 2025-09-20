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

from .state_mapping import (
    percent_change_to_log_return,
    log_return_to_percent_change,
    map_states_to_financial_regimes,
    get_regime_characteristics,
)

__all__ = [
    "HiddenRegimeError",
    "DataLoadError",
    "DataQualityError",
    "ValidationError",
    "percent_change_to_log_return",
    "log_return_to_percent_change",
    "map_states_to_financial_regimes",
    "get_regime_characteristics",
]