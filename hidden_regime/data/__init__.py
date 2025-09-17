"""
Data loading components for hidden-regime pipeline.

Provides data loading components that implement the DataComponent interface
for various data sources including financial market data.
"""

from .financial import FinancialDataLoader

__all__ = [
    "FinancialDataLoader",
]