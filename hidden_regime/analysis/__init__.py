"""
Analysis components for hidden-regime pipeline.

Provides analysis components that implement the AnalysisComponent interface
for interpreting model outputs with domain knowledge and generating insights.
"""

from .financial import FinancialAnalysis
from .indicator_comparison import IndicatorPerformanceComparator
from .performance import RegimePerformanceAnalyzer

__all__ = [
    "FinancialAnalysis",
    "RegimePerformanceAnalyzer",
    "IndicatorPerformanceComparator",
]
