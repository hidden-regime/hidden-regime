"""
Observation generation components for hidden-regime pipeline.

Provides base and specialized observation generators for transforming raw data
into features suitable for model training and prediction.
"""

from .base import BaseObservationGenerator
from .financial import FinancialObservationGenerator

__all__ = [
    "BaseObservationGenerator",
    "FinancialObservationGenerator",
]
