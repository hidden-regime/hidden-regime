"""
Historical Market Analysis Module

Provides curated datasets and analysis functions for major historical market events,
enabling validation of regime detection algorithms and creation of compelling
content comparing HMM performance against known market periods.
"""

from .datasets import (
    MAJOR_MARKET_EVENTS,
    load_crisis_2008,
    load_covid_crash_2020,
    load_dotcom_bubble,
    load_historical_period,
)
from .validation import (
    validate_historical_detection,
    validate_regime_accuracy,
)

__all__ = [
    # Dataset functions
    "load_historical_period",
    "load_crisis_2008", 
    "load_covid_crash_2020",
    "load_dotcom_bubble",
    "MAJOR_MARKET_EVENTS",
    # Validation functions
    "validate_historical_detection",
    "validate_regime_accuracy",
]