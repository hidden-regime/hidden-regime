"""
Configuration management for hidden-regime package.

Provides settings and configuration classes for data loading,
preprocessing, and validation parameters.
"""

from .settings import DataConfig, PreprocessingConfig, ValidationConfig

__all__ = ["DataConfig", "ValidationConfig", "PreprocessingConfig"]
