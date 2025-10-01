"""
Financial-first regime analysis components.

This module provides financial market-focused components that understand
financial concepts natively, rather than treating them as generic abstractions.
"""

from .analysis import FinancialAnalysisResult, FinancialRegimeAnalysis
from .config import FinancialRegimeConfig
from .regime_characterizer import FinancialRegimeCharacterizer, RegimeProfile

__all__ = [
    "FinancialRegimeCharacterizer",
    "RegimeProfile",
    "FinancialRegimeConfig",
    "FinancialRegimeAnalysis",
    "FinancialAnalysisResult",
]
