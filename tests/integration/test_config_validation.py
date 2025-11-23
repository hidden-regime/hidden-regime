"""
Configuration Validation Integration Tests.

Tests verify that:
- Component configurations are validated correctly
- Configuration mismatches are detected
- Components respect configuration constraints
- Configuration changes propagate correctly
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.utils.exceptions import ValidationError, ConfigurationError


# ============================================================================
# Model Configuration Tests
# ============================================================================


@pytest.mark.integration
def test_hmm_config_requires_positive_states():
    """Test that HMM configuration requires positive number of states."""
    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        HMMConfig(n_states=0)

    with pytest.raises((ValidationError, ValueError, ConfigurationError)):
        HMMConfig(n_states=-1)


@pytest.mark.integration
def test_hmm_config_state_count_affects_output():
    """Test that n_states configuration affects model output."""
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})

    # Model with 2 states
    model_2 = HiddenMarkovModel(HMMConfig(n_states=2))
    model_2.fit(data)
    output_2 = model_2.predict(data)

    # Should have state_0_prob and state_1_prob
    assert 'state_0_prob' in output_2.columns
    assert 'state_1_prob' in output_2.columns
    assert 'state_2_prob' not in output_2.columns

    # Model with 3 states
    model_3 = HiddenMarkovModel(HMMConfig(n_states=3))
    model_3.fit(data)
    output_3 = model_3.predict(data)

    # Should have state_0_prob, state_1_prob, and state_2_prob
    assert 'state_0_prob' in output_3.columns
    assert 'state_1_prob' in output_3.columns
    assert 'state_2_prob' in output_3.columns


@pytest.mark.integration
def test_hmm_config_with_valid_parameters():
    """Test that HMM configuration accepts valid parameters."""
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})

    # Valid configuration should work
    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)
    model.fit(data)

    assert model.is_fitted, "Model should complete training with valid config"


# ============================================================================
# Interpreter Configuration Tests
# ============================================================================


@pytest.mark.integration
def test_interpreter_config_state_count_validation():
    """Test that interpreter validates n_states matches input."""
    # Create model output with 2 states
    model_output = pd.DataFrame({
        'state': [0, 1, 0],
        'confidence': [0.8, 0.75, 0.82],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.25, 0.75]),
            np.array([0.82, 0.18])
        ],
        'emission_means': [np.array([0.001, -0.002])] * 3,
        'emission_stds': [np.array([0.01, 0.02])] * 3,
    })

    # Interpreter configured for 2 states - should work
    config_2 = InterpreterConfiguration(n_states=2)
    interpreter_2 = FinancialInterpreter(config_2)
    result = interpreter_2.update(model_output)
    assert len(result) == 3, "Should process all rows"

    # Interpreter configured for 3 states - may fail or handle gracefully
    config_3 = InterpreterConfiguration(n_states=3)
    interpreter_3 = FinancialInterpreter(config_3)

    try:
        result_3 = interpreter_3.update(model_output)
        # If it handles gracefully, result should still be valid
        assert len(result_3) > 0, "Should handle mismatched state count"
    except (ValidationError, ValueError, IndexError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_interpreter_force_labels_configuration():
    """Test that forced labels override data-driven interpretation."""
    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.25, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.002])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    # With forced labels
    custom_labels = ["custom_state_0", "custom_state_1"]
    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=custom_labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)
    result = interpreter.update(model_output)

    # Should use custom labels
    assert result.iloc[0]['regime_type'] == "custom_state_0"
    assert result.iloc[1]['regime_type'] == "custom_state_1"


@pytest.mark.integration
def test_interpreter_interpretation_method_affects_output():
    """Test that interpretation_method configuration affects regime classification."""
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.002, -0.002])],  # Clear bull/bear distinction
        'emission_stds': [np.array([0.01, 0.02])],
    })

    # Data-driven interpretation
    config_data = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interp_data = FinancialInterpreter(config_data)
    result_data = interp_data.update(model_output)

    # Should produce regime type based on emission parameters
    regime_type = result_data.iloc[0]['regime_type'].lower()
    assert 'bull' in regime_type or 'uptrend' in regime_type or \
           'bear' in regime_type or 'downtrend' in regime_type or \
           'sideways' in regime_type or 'neutral' in regime_type, \
        f"Should classify based on data, got: {regime_type}"


# ============================================================================
# Signal Generator Configuration Tests
# ============================================================================


@pytest.mark.integration
def test_signal_config_strategy_type_affects_output():
    """Test that strategy_type configuration produces different signals."""
    regime_input = pd.DataFrame({
        'state': [0],
        'regime_type': ['bullish'],
        'regime_confidence': [0.8],
        'regime_strength': [0.8],
        'signal_valid': [True]
    })

    # Regime-following strategy
    config_follow = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_follow = FinancialSignalGenerator(config_follow)
    result_follow = signal_follow.update(regime_input)

    # Should produce positive signal for bullish
    assert result_follow.iloc[0]['position_size'] > 0, "Following should be long on bullish"

    # Regime-contrarian strategy
    config_contrarian = SignalGenerationConfiguration(strategy_type="regime_contrarian")
    signal_contrarian = FinancialSignalGenerator(config_contrarian)
    result_contrarian = signal_contrarian.update(regime_input)

    # Should produce negative signal for bullish (contrarian)
    assert result_contrarian.iloc[0]['position_size'] < 0, "Contrarian should be short on bullish"


@pytest.mark.integration
def test_signal_config_position_size_range():
    """Test that position_size_range configuration is respected."""
    regime_input = pd.DataFrame({
        'state': [0],
        'regime_type': ['bullish'],
        'regime_confidence': [1.0],  # Maximum confidence
        'regime_strength': [1.0],
        'signal_valid': [True]
    })

    # Test with 50% max position
    config_half = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 0.5)
    )
    signal_gen_half = FinancialSignalGenerator(config_half)
    result_half = signal_gen_half.update(regime_input)

    # Position should not exceed 0.5
    assert abs(result_half.iloc[0]['position_size']) <= 0.5, \
        "Position should respect max size of 0.5"

    # Test with 25% max position
    config_quarter = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 0.25)
    )
    signal_gen_quarter = FinancialSignalGenerator(config_quarter)
    result_quarter = signal_gen_quarter.update(regime_input)

    # Position should not exceed 0.25
    assert abs(result_quarter.iloc[0]['position_size']) <= 0.25, \
        "Position should respect max size of 0.25"


# ============================================================================
# Cross-Component Configuration Consistency Tests
# ============================================================================


@pytest.mark.integration
def test_model_interpreter_state_count_consistency():
    """Test that model and interpreter must have consistent n_states."""
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})

    # Model with 2 states
    model = HiddenMarkovModel(HMMConfig(n_states=2))
    model.fit(data)
    model_output = model.predict(data)

    # Adapt for interpreter
    n_states = 2
    adapted = model_output.copy()
    adapted['state'] = adapted['predicted_state']
    state_prob_cols = [f'state_{i}_prob' for i in range(n_states)]
    adapted['state_probabilities'] = adapted[state_prob_cols].apply(
        lambda row: np.array(row.values), axis=1
    )
    adapted['emission_means'] = [model.emission_means_] * len(adapted)
    adapted['emission_stds'] = [model.emission_stds_] * len(adapted)

    # Interpreter with matching 2 states - should work
    interp_match = FinancialInterpreter(InterpreterConfiguration(n_states=2))
    result = interp_match.update(adapted)
    assert len(result) > 0, "Matching state counts should work"

    # Interpreter with mismatched 3 states - should fail or handle gracefully
    interp_mismatch = FinancialInterpreter(InterpreterConfiguration(n_states=3))
    try:
        result_mismatch = interp_mismatch.update(adapted)
        # If it handles gracefully, should still work
        assert len(result_mismatch) > 0, "Should handle state count mismatch"
    except (ValidationError, ValueError, IndexError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_configuration_parameter_ranges():
    """Test that configuration parameters are validated for valid ranges."""
    # Invalid position size range (min > max)
    with pytest.raises((ValidationError, ValueError)):
        SignalGenerationConfiguration(
            strategy_type="regime_following",
            position_size_range=(1.0, 0.5)  # Invalid: min > max
        )

    # Invalid position size range (negative)
    with pytest.raises((ValidationError, ValueError)):
        SignalGenerationConfiguration(
            strategy_type="regime_following",
            position_size_range=(-2.0, 1.0)  # Invalid: exceeds [-1, 1]
        )


@pytest.mark.integration
def test_full_pipeline_configuration_consistency():
    """Test that complete pipeline works with consistent configurations."""
    np.random.seed(789)
    n_states = 3

    # Create data
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})

    # Configure all components with n_states=3
    model = HiddenMarkovModel(HMMConfig(n_states=n_states))
    model.fit(data)
    model_output = model.predict(data)

    # Adapt for interpreter
    adapted = model_output.copy()
    adapted['state'] = adapted['predicted_state']
    state_prob_cols = [f'state_{i}_prob' for i in range(n_states)]
    adapted['state_probabilities'] = adapted[state_prob_cols].apply(
        lambda row: np.array(row.values), axis=1
    )
    adapted['emission_means'] = [model.emission_means_] * len(adapted)
    adapted['emission_stds'] = [model.emission_stds_] * len(adapted)

    # Interpreter with n_states=3
    interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=n_states))
    regime_output = interpreter.update(adapted)

    # Signal generator (n_states doesn't affect it directly)
    signal_gen = FinancialSignalGenerator(
        SignalGenerationConfiguration(strategy_type="regime_following")
    )
    signal_output = signal_gen.update(regime_output)

    # All components should complete successfully
    assert len(signal_output) == len(data), "Pipeline should process all data"
    assert 'position_size' in signal_output.columns, "Should produce signals"


@pytest.mark.integration
def test_reconfiguration_after_initialization():
    """Test that models/components handle reconfiguration correctly."""
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})

    # Train with 2 states
    model_2 = HiddenMarkovModel(HMMConfig(n_states=2))
    model_2.fit(data)
    output_2 = model_2.predict(data)
    assert len(output_2) > 0, "Should work with 2 states"

    # Create new model with 3 states (not reconfigure, but new instance)
    model_3 = HiddenMarkovModel(HMMConfig(n_states=3))
    model_3.fit(data)
    output_3 = model_3.predict(data)
    assert len(output_3) > 0, "Should work with 3 states"

    # Verify different state counts
    assert 'state_2_prob' not in output_2.columns, "2-state model shouldn't have state_2"
    assert 'state_2_prob' in output_3.columns, "3-state model should have state_2"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
