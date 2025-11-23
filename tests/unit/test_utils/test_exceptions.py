"""
Unit tests for utils/exceptions.py

Tests exception hierarchy and error handling.
"""

import pytest
from hidden_regime.utils.exceptions import (
    HiddenRegimeError,
    DataLoadError,
    DataQualityError,
    ValidationError,
    ConfigurationError,
    HMMTrainingError,
    HMMInferenceError,
)


def test_hidden_regime_error_hierarchy():
    """Test that all exceptions inherit from HiddenRegimeError."""
    assert issubclass(DataLoadError, HiddenRegimeError)
    assert issubclass(DataQualityError, HiddenRegimeError)
    assert issubclass(ValidationError, HiddenRegimeError)
    assert issubclass(ConfigurationError, HiddenRegimeError)
    assert issubclass(HMMTrainingError, HiddenRegimeError)
    assert issubclass(HMMInferenceError, HiddenRegimeError)


def test_data_load_error_raised():
    """Test DataLoadError can be raised."""
    with pytest.raises(DataLoadError):
        raise DataLoadError("Failed to load data")


def test_data_quality_error_with_message():
    """Test DataQualityError with custom message."""
    with pytest.raises(DataQualityError) as exc_info:
        raise DataQualityError("Data contains NaN values")

    assert "NaN values" in str(exc_info.value)


def test_validation_error_context():
    """Test ValidationError with context."""
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError("Invalid configuration parameter: n_states must be >= 2")

    assert "n_states" in str(exc_info.value)


def test_configuration_error_details():
    """Test ConfigurationError with detailed message."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Missing required configuration key: 'observed_signal'")

    assert "observed_signal" in str(exc_info.value)


def test_hmm_training_error():
    """Test HMMTrainingError."""
    with pytest.raises(HMMTrainingError) as exc_info:
        raise HMMTrainingError("Failed to converge after 100 iterations")

    assert "converge" in str(exc_info.value)


def test_hmm_inference_error():
    """Test HMMInferenceError."""
    with pytest.raises(HMMInferenceError) as exc_info:
        raise HMMInferenceError("Cannot predict on unfitted model")

    assert "unfitted" in str(exc_info.value)


def test_analysis_error_propagation():
    """Test exception chaining and propagation."""
    try:
        try:
            raise DataLoadError("Original error")
        except DataLoadError as e:
            raise ValidationError("Validation failed") from e
    except ValidationError as exc_info:
        assert exc_info.__cause__ is not None
        assert isinstance(exc_info.__cause__, DataLoadError)
