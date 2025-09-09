"""
Data loading, preprocessing, and validation module for hidden-regime.

This module provides functionality for:
- Loading stock market data from multiple sources
- Data preprocessing including log returns calculation
- Data quality validation and anomaly detection
"""

from .loader import DataLoader
from .preprocessing import DataPreprocessor
from .validation import DataValidator

__all__ = ["DataLoader", "DataPreprocessor", "DataValidator"]
