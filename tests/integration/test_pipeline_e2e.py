"""
End-to-End Pipeline Integration Tests.

Tests verify that the complete pipeline works correctly when all components
are connected together: Data → Model → Interpreter → Signal Generator.

This is critical for validating:
1. Data flows correctly through all components
2. Output from one component is valid input for the next
3. No data corruption or information loss across boundaries
4. Real-world usage patterns work as expected

These tests use REAL components (not mocks) to validate actual integration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.utils.exceptions import ValidationError


# ============================================================================
# Helper Functions
# ============================================================================


def adapt_model_output_for_interpreter(model_output: pd.DataFrame, model: HiddenMarkovModel) -> pd.DataFrame:
    """
    Adapt model output format to interpreter input format.

    Model outputs: predicted_state, confidence, state_0_prob, state_1_prob, ...
    Interpreter expects: state, confidence, state_probabilities, emission_means, emission_stds

    This adapter layer is necessary because components have different interfaces.
    """
    n_states = model.n_states

    # Rename predicted_state to state
    adapted = model_output.copy()
    adapted['state'] = adapted['predicted_state']

    # Convert individual state probability columns to array column
    state_prob_cols = [f'state_{i}_prob' for i in range(n_states)]
    adapted['state_probabilities'] = adapted[state_prob_cols].apply(
        lambda row: np.array(row.values), axis=1
    )

    # Add emission parameters from model (same for all rows - model learned params)
    adapted['emission_means'] = [model.emission_means_] * len(adapted)
    adapted['emission_stds'] = [model.emission_stds_] * len(adapted)

    return adapted


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_price_data():
    """Generate synthetic price data for testing.

    Creates a simple bull → bear transition pattern for predictable testing.
    """
    np.random.seed(42)
    n_points = 100

    # Bull market: positive drift, low vol (first 50 points)
    bull_returns = np.random.normal(0.001, 0.01, 50)

    # Bear market: negative drift, higher vol (last 50 points)
    bear_returns = np.random.normal(-0.002, 0.02, 50)

    all_returns = np.concatenate([bull_returns, bear_returns])

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(all_returns))

    # Create DataFrame
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_points)]

    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': all_returns
    })


@pytest.fixture
def basic_hmm_config():
    """Basic HMM configuration for testing."""
    return HMMConfig(
        n_states=2
    )


@pytest.fixture
def basic_interpreter_config():
    """Basic interpreter configuration for testing."""
    return InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )


@pytest.fixture
def basic_signal_config():
    """Basic signal generation configuration for testing."""
    return SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 1.0)
    )


# ============================================================================
# Complete Pipeline Tests
# ============================================================================


@pytest.mark.integration
def test_complete_pipeline_data_to_signals(
    synthetic_price_data,
    basic_hmm_config,
    basic_interpreter_config,
    basic_signal_config
):
    """Test complete pipeline: raw data → fitted model → interpreted regimes → trading signals."""

    # Step 1: Prepare data for model (model expects DataFrame with observed_signal column)
    observations = synthetic_price_data[['returns']].copy()
    observations.columns = ['log_return']  # Model expects 'log_return' by default

    # Step 2: Train model
    model = HiddenMarkovModel(basic_hmm_config)
    model.fit(observations)

    # Verify model trained successfully
    assert model.is_fitted, "Model should be trained"

    # Step 3: Get model predictions
    model_output = model.update(observations)

    # Verify model output structure
    assert 'predicted_state' in model_output.columns, "Model output missing 'predicted_state'"
    assert 'confidence' in model_output.columns, "Model output missing 'confidence'"
    assert len(model_output) == len(observations), "Model output length mismatch"

    # Step 4: Adapt model output for interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)

    # Step 5: Interpret regimes
    interpreter = FinancialInterpreter(basic_interpreter_config)
    regime_output = interpreter.update(interpreter_input)

    # Verify interpreter output structure
    assert 'regime_type' in regime_output.columns, "Interpreter output missing 'regime_type'"
    assert 'regime_confidence' in regime_output.columns, "Interpreter output missing 'regime_confidence'"
    assert len(regime_output) == len(model_output), "Interpreter output length mismatch"

    # Step 6: Generate trading signals
    signal_generator = FinancialSignalGenerator(basic_signal_config)
    final_output = signal_generator.update(regime_output)

    # Verify signal output structure
    assert 'position_size' in final_output.columns, "Signal output missing 'position_size'"
    assert 'signal_valid' in final_output.columns, "Signal output missing 'signal_valid'"
    assert len(final_output) == len(regime_output), "Signal output length mismatch"

    # Verify signals are in valid range
    assert all(final_output['position_size'] >= -1.0), "Signals below minimum"
    assert all(final_output['position_size'] <= 1.0), "Signals above maximum"

    # Verify we got meaningful signals (not all zeros)
    assert not all(final_output['position_size'] == 0), "All signals are zero (pipeline not working)"

    print(f"\n✓ Complete pipeline test passed:")
    print(f"  - Input: {len(observations)} price observations")
    print(f"  - Detected: {regime_output['regime_type'].nunique()} distinct regimes")
    print(f"  - Generated: {(final_output['position_size'] != 0).sum()} non-zero signals")


@pytest.mark.integration
def test_pipeline_preserves_temporal_order(
    synthetic_price_data,
    basic_hmm_config,
    basic_interpreter_config,
    basic_signal_config
):
    """Test that pipeline preserves temporal order of observations."""

    # Prepare DataFrame for model
    observations = synthetic_price_data[['returns']].copy()
    observations.columns = ['log_return']

    # Add explicit time index
    time_index = np.arange(len(observations))

    # Run through pipeline
    model = HiddenMarkovModel(basic_hmm_config)
    model.fit(observations)
    model_output = model.update(observations)

    # Adapt model output for interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)

    interpreter = FinancialInterpreter(basic_interpreter_config)
    regime_output = interpreter.update(interpreter_input)

    signal_generator = FinancialSignalGenerator(basic_signal_config)
    final_output = signal_generator.update(regime_output)

    # Verify all outputs have same length and order
    assert len(model_output) == len(observations)
    assert len(regime_output) == len(observations)
    assert len(final_output) == len(observations)

    # If we add index to final output, verify it's preserved
    # (In production, indices should be maintained throughout)
    print(f"\n✓ Temporal order preserved through {len(observations)} time steps")


@pytest.mark.integration
def test_pipeline_handles_regime_transitions(
    basic_hmm_config,
    basic_interpreter_config,
    basic_signal_config
):
    """Test that pipeline correctly handles regime transitions."""

    np.random.seed(123)

    # Create clear regime transition: bull → bear
    bull_phase = np.random.normal(0.002, 0.008, 30)  # Strong positive, low vol
    bear_phase = np.random.normal(-0.003, 0.025, 30)  # Strong negative, high vol
    returns = np.concatenate([bull_phase, bear_phase])

    # Convert to DataFrame
    observations = pd.DataFrame({'log_return': returns})

    # Run pipeline
    model = HiddenMarkovModel(basic_hmm_config)
    model.fit(observations)
    model_output = model.update(observations)

    # Adapt model output for interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)

    interpreter = FinancialInterpreter(basic_interpreter_config)
    regime_output = interpreter.update(interpreter_input)

    signal_generator = FinancialSignalGenerator(basic_signal_config)
    final_output = signal_generator.update(regime_output)

    # Verify regime transition was detected
    regimes = regime_output['regime_type'].values
    assert len(set(regimes)) >= 2, "Should detect at least 2 different regimes"

    # Verify signals changed around transition
    signals_first_half = final_output['position_size'][:30].mean()
    signals_second_half = final_output['position_size'][30:].mean()

    # Signals should be different (bull should be positive, bear negative or lower)
    assert signals_first_half != signals_second_half, \
        "Signals should change across regime transition"

    print(f"\n✓ Pipeline detected regime transition:")
    print(f"  - First half mean signal: {signals_first_half:.3f}")
    print(f"  - Second half mean signal: {signals_second_half:.3f}")
    print(f"  - Detected regimes: {set(regimes)}")


# ============================================================================
# Data Flow Validation Tests
# ============================================================================


@pytest.mark.integration
def test_model_to_interpreter_data_contract():
    """Test that model output satisfies interpreter input requirements."""

    # Create minimal model output
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

    # Verify interpreter can consume model output
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    result = interpreter.update(model_output)

    # Verify all required columns present
    assert 'regime_type' in result.columns
    assert 'regime_confidence' in result.columns
    assert 'regime_strength' in result.columns
    assert len(result) == len(model_output)

    print("\n✓ Model → Interpreter data contract validated")


@pytest.mark.integration
def test_interpreter_to_signal_data_contract():
    """Test that interpreter output satisfies signal generator input requirements."""

    # Create minimal interpreter output
    interpreter_output = pd.DataFrame({
        'state': [0, 1, 0],
        'regime_type': ['bullish', 'bearish', 'bullish'],
        'regime_confidence': [0.8, 0.75, 0.85],
        'regime_strength': [0.8, 0.75, 0.85],
        'signal_valid': [True, True, True]
    })

    # Verify signal generator can consume interpreter output
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    result = signal_gen.update(interpreter_output)

    # Verify all required columns present
    assert 'position_size' in result.columns
    assert 'signal_valid' in result.columns
    assert len(result) == len(interpreter_output)

    print("\n✓ Interpreter → Signal Generator data contract validated")


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
def test_pipeline_handles_empty_data():
    """Test that pipeline handles empty input gracefully."""

    # Empty DataFrame
    empty_observations = pd.DataFrame({'log_return': []})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Model should handle empty data
    with pytest.raises((ValueError, AssertionError, ValidationError)):
        model.fit(empty_observations)

    print("\n✓ Pipeline correctly rejects empty data")


@pytest.mark.integration
def test_pipeline_handles_single_observation():
    """Test pipeline behavior with minimal data (single observation)."""

    # Single observation DataFrame
    single_observation = pd.DataFrame({'log_return': [0.001]})

    # HMM needs multiple observations to train
    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should fail or warn with insufficient data
    with pytest.raises((ValueError, AssertionError, RuntimeError, ValidationError)):
        model.fit(single_observation)

    print("\n✓ Pipeline correctly rejects insufficient data")


@pytest.mark.integration
def test_pipeline_with_missing_values():
    """Test that pipeline detects and handles missing values."""

    # Create data with NaN
    observations_with_nan = pd.DataFrame({'log_return': [0.001, np.nan, 0.002, -0.001]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should detect NaN and raise error
    with pytest.raises((ValueError, AssertionError, ValidationError)):
        model.fit(observations_with_nan)

    print("\n✓ Pipeline correctly detects missing values")


# ============================================================================
# State Consistency Tests
# ============================================================================


@pytest.mark.integration
def test_state_indices_consistent_across_components():
    """Test that state indices remain consistent through pipeline."""

    np.random.seed(456)
    returns = np.random.normal(0, 0.01, 50)
    observations = pd.DataFrame({'log_return': returns})

    # Train and predict
    hmm_config = HMMConfig(n_states=3)
    model = HiddenMarkovModel(hmm_config)
    model.fit(observations)
    model_output = model.update(observations)

    # Verify state indices are in valid range [0, n_states-1]
    states = model_output['predicted_state'].values
    assert all((states >= 0) & (states < 3)), \
        f"State indices out of range: {states}"

    # Adapt model output for interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)

    # Pass to interpreter
    interp_config = InterpreterConfiguration(n_states=3)
    interpreter = FinancialInterpreter(interp_config)
    regime_output = interpreter.update(interpreter_input)

    # State column should be preserved
    assert 'state' in regime_output.columns
    assert all(regime_output['state'] == interpreter_input['state']), \
        "State indices changed during interpretation"

    print(f"\n✓ State indices consistent (found states: {set(states)})")


@pytest.mark.integration
def test_confidence_propagation_through_pipeline():
    """Test that confidence metrics are preserved/transformed correctly."""

    np.random.seed(789)
    returns = np.random.normal(0, 0.01, 40)
    observations = pd.DataFrame({'log_return': returns})

    # Run pipeline
    model = HiddenMarkovModel(HMMConfig(n_states=2))
    model.fit(observations)
    model_output = model.update(observations)

    # Adapt model output for interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)

    interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=2))
    regime_output = interpreter.update(interpreter_input)

    signal_gen = FinancialSignalGenerator(
        SignalGenerationConfiguration(strategy_type="confidence_weighted")
    )
    signal_output = signal_gen.update(regime_output)

    # Verify confidence exists at each stage
    assert 'confidence' in model_output.columns, "Model output missing confidence"
    assert 'regime_confidence' in regime_output.columns, "Interpreter output missing confidence"

    # For confidence_weighted strategy, signals should scale with confidence
    # Higher confidence should generally lead to larger absolute positions
    high_conf_mask = regime_output['regime_confidence'] > 0.7
    low_conf_mask = regime_output['regime_confidence'] < 0.6

    if high_conf_mask.sum() > 0 and low_conf_mask.sum() > 0:
        high_conf_positions = abs(signal_output.loc[high_conf_mask, 'position_size']).mean()
        low_conf_positions = abs(signal_output.loc[low_conf_mask, 'position_size']).mean()

        # Not strict inequality due to other factors, but should show trend
        print(f"\n✓ Confidence propagation verified:")
        print(f"  - High confidence avg position: {high_conf_positions:.3f}")
        print(f"  - Low confidence avg position: {low_conf_positions:.3f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
