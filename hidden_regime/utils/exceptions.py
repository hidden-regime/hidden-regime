"""
Custom exception classes for hidden-regime package.

Provides specific exception types for different error conditions
that can occur during data loading, processing, and analysis.
"""


class HiddenRegimeError(Exception):
    """Base exception class for hidden-regime package."""

    pass


class DataLoadError(HiddenRegimeError):
    """
    Raised when data loading fails.

    This includes network errors, invalid tickers, API failures,
    or insufficient data for the requested time period.
    """

    pass


class DataQualityError(HiddenRegimeError):
    """
    Raised when data quality issues are detected.

    This includes excessive missing values, price anomalies,
    or data that fails validation checks.
    """

    pass


class ValidationError(HiddenRegimeError):
    """
    Raised when data validation fails.

    This includes invalid date ranges, malformed input parameters,
    or data that doesn't meet minimum quality requirements.
    """

    pass


class ConfigurationError(HiddenRegimeError):
    """
    Raised when configuration is invalid.

    This includes missing required settings, invalid parameter ranges,
    or incompatible configuration combinations.
    """

    pass


class HMMTrainingError(HiddenRegimeError):
    """
    Raised when HMM training fails.

    This includes convergence failures, numerical instabilities,
    or insufficient data for model training.
    """

    pass


class HMMInferenceError(HiddenRegimeError):
    """
    Raised when HMM inference fails.

    This includes prediction failures, invalid observations,
    or model state errors during inference.
    """

    pass