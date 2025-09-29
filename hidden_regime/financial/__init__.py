"""
Financial-first regime analysis components.

This module provides financial market-focused components that understand
financial concepts natively, rather than treating them as generic abstractions.
"""

from .regime_characterizer import FinancialRegimeCharacterizer, RegimeProfile
from .config import FinancialRegimeConfig
from .analysis import FinancialRegimeAnalysis, FinancialAnalysisResult

__all__ = [
    'FinancialRegimeCharacterizer',
    'RegimeProfile',
    'FinancialRegimeConfig',
    'FinancialRegimeAnalysis',
    'FinancialAnalysisResult'
]