"""
Unit tests for config/ module serialization and validation.

Tests configuration validation, JSON/dict serialization, copy methods,
and factory patterns across all config types.
"""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import replace

from hidden_regime.config import (
    HMMConfig,
    FinancialDataConfig,
    FinancialObservationConfig,
    InterpreterConfiguration,
    SignalGenerationConfiguration,
    ReportConfig,
)
from hidden_regime.utils.exceptions import ValidationError, ConfigurationError


# ============================================================================
# HMM CONFIG TESTS (5 tests)
# ============================================================================


def test_hmm_config_validation_valid():
    """Test HMM configuration with valid parameters."""
    config = HMMConfig(
        n_states=3,
        max_iterations=100,
        tolerance=1e-4,
        random_seed=42,
    )

    assert config.n_states == 3
    assert config.max_iterations == 100
    assert config.tolerance == 1e-4


def test_hmm_config_validation_invalid():
    """Test HMM configuration rejects invalid parameters."""
    # Invalid n_states
    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        HMMConfig(n_states=0)

    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        HMMConfig(n_states=-1)


def test_hmm_config_serialization_json():
    """Test HMM config JSON serialization roundtrip."""
    config = HMMConfig(n_states=3, max_iterations=100, random_seed=42)

    # Convert to dict using dataclass method
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["n_states"] == 3

    # Should be JSON serializable
    json_str = json.dumps(config_dict, default=str)
    assert isinstance(json_str, str)


def test_hmm_config_serialization_dict():
    """Test HMM config dict conversion."""
    config = HMMConfig(n_states=2, random_seed=42)

    config_dict = config.to_dict()

    # Reconstruct from dict
    config2 = HMMConfig.from_dict(config_dict)

    assert config2.n_states == config.n_states
    assert config2.random_seed == config.random_seed


def test_hmm_config_copy_with_updates():
    """Test HMM config copy with parameter updates."""
    config = HMMConfig(n_states=2, max_iterations=100)

    # Use .copy() method from BaseConfig
    config2 = config.copy(n_states=3)

    assert config2.n_states == 3
    assert config2.max_iterations == 100
    assert config.n_states == 2  # Original unchanged


# ============================================================================
# DATA CONFIG TESTS (5 tests)
# ============================================================================


def test_data_config_validation_valid():
    """Test FinancialDataConfig with valid parameters."""
    config = FinancialDataConfig(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    assert config.ticker == "AAPL"
    assert config.start_date == "2023-01-01"


def test_data_config_validation_invalid():
    """Test FinancialDataConfig rejects invalid parameters."""
    # Empty ticker
    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        FinancialDataConfig(ticker="")


def test_data_config_serialization_json():
    """Test FinancialDataConfig JSON serialization."""
    config = FinancialDataConfig(ticker="SPY", start_date="2023-01-01")

    config_dict = config.to_dict()
    json_str = json.dumps(config_dict, default=str)

    assert isinstance(json_str, str)


def test_data_config_serialization_dict():
    """Test FinancialDataConfig dict conversion."""
    config = FinancialDataConfig(ticker="MSFT")

    config_dict = config.to_dict()
    config2 = FinancialDataConfig.from_dict(config_dict)

    assert config2.ticker == config.ticker


def test_data_config_copy_with_updates():
    """Test FinancialDataConfig copy method."""
    config = FinancialDataConfig(ticker="AAPL")

    config2 = config.copy(ticker="GOOGL")

    assert config2.ticker == "GOOGL"
    assert config.ticker == "AAPL"


# ============================================================================
# OBSERVATION CONFIG TESTS (5 tests)
# ============================================================================


def test_observation_config_validation_valid():
    """Test FinancialObservationConfig with valid parameters."""
    config = FinancialObservationConfig(
        generators=["log_return", "volatility"],
        price_column="close",
    )

    assert "log_return" in config.generators
    assert config.price_column == "close"


def test_observation_config_validation_invalid():
    """Test FinancialObservationConfig rejects invalid parameters."""
    # Invalid generator
    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        FinancialObservationConfig(generators=["invalid_generator"])


def test_observation_config_serialization_json():
    """Test FinancialObservationConfig JSON serialization."""
    config = FinancialObservationConfig(generators=["log_return"])

    config_dict = config.to_dict()
    json_str = json.dumps(config_dict, default=str)

    assert isinstance(json_str, str)


def test_observation_config_serialization_dict():
    """Test FinancialObservationConfig dict conversion."""
    config = FinancialObservationConfig(generators=["log_return", "rsi"])

    config_dict = config.to_dict()
    config2 = FinancialObservationConfig.from_dict(config_dict)

    assert config2.generators == config.generators


def test_observation_config_copy_with_updates():
    """Test FinancialObservationConfig copy method."""
    config = FinancialObservationConfig(price_column="close")

    config2 = config.copy(price_column="open")

    assert config2.price_column == "open"
    assert config.price_column == "close"


# ============================================================================
# INTERPRETER CONFIG TESTS (5 tests)
# ============================================================================


def test_interpreter_config_validation_valid():
    """Test InterpreterConfiguration with valid parameters."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven",
    )

    assert config.n_states == 3
    assert config.interpretation_method == "data_driven"


def test_interpreter_config_validation_invalid():
    """Test InterpreterConfiguration rejects invalid parameters."""
    # Invalid n_states
    with pytest.raises((ValidationError, ValueError)):
        InterpreterConfiguration(n_states=1)


def test_interpreter_config_serialization_json():
    """Test InterpreterConfiguration JSON serialization."""
    config = InterpreterConfiguration(n_states=2)

    config_dict = config.to_dict()
    json_str = json.dumps(config_dict, default=str)

    assert isinstance(json_str, str)


def test_interpreter_config_serialization_dict():
    """Test InterpreterConfiguration dict conversion."""
    config = InterpreterConfiguration(n_states=3)

    config_dict = config.to_dict()
    config2 = InterpreterConfiguration.from_dict(config_dict)

    assert config2.n_states == config.n_states


def test_interpreter_config_copy_with_updates():
    """Test InterpreterConfiguration copy method."""
    config = InterpreterConfiguration(n_states=2)

    # InterpreterConfiguration is a plain dataclass, use replace()
    config2 = replace(config, n_states=4)

    assert config2.n_states == 4
    assert config.n_states == 2


# ============================================================================
# SIGNAL GENERATION CONFIG TESTS (5 tests)
# ============================================================================


def test_signal_config_validation_valid():
    """Test SignalGenerationConfiguration with valid parameters."""
    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 1.0),
    )

    assert config.strategy_type == "regime_following"
    assert config.position_size_range == (0.0, 1.0)


def test_signal_config_validation_invalid():
    """Test SignalGenerationConfiguration rejects invalid parameters."""
    # Invalid position size range
    with pytest.raises((ValidationError, ValueError)):
        SignalGenerationConfiguration(position_size_range=(-0.1, 1.0))


def test_signal_config_serialization_json():
    """Test SignalGenerationConfiguration JSON serialization."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")

    config_dict = config.to_dict()
    json_str = json.dumps(config_dict, default=str)

    assert isinstance(json_str, str)


def test_signal_config_serialization_dict():
    """Test SignalGenerationConfiguration dict conversion."""
    config = SignalGenerationConfiguration(strategy_type="confidence_weighted")

    config_dict = config.to_dict()
    config2 = SignalGenerationConfiguration.from_dict(config_dict)

    assert config2.strategy_type == config.strategy_type


def test_signal_config_copy_with_updates():
    """Test SignalGenerationConfiguration copy method."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")

    # SignalGenerationConfiguration is a plain dataclass, use replace()
    config2 = replace(config, strategy_type="regime_contrarian")

    assert config2.strategy_type == "regime_contrarian"
    assert config.strategy_type == "regime_following"
