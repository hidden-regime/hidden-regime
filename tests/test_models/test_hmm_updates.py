"""
Tests for HMM update strategies (static, incremental, adaptive_refit).

Tests the three update strategies for handling streaming data and persistent service scenarios.
"""

import warnings
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel


def generate_regime_data(
    n_obs: int = 100,
    regime: str = "bull",
    random_seed: int = None,
) -> np.ndarray:
    """
    Generate synthetic data with regime characteristics.

    Args:
        n_obs: Number of observations
        regime: Regime type ('bear', 'sideways', 'bull', 'crisis')
        random_seed: Random seed for reproducibility

    Returns:
        Array of log returns with regime characteristics
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    regime_params = {
        'bear': {'mean': -0.015, 'std': 0.025},
        'sideways': {'mean': 0.001, 'std': 0.015},
        'bull': {'mean': 0.012, 'std': 0.020},
        'crisis': {'mean': -0.035, 'std': 0.040},
    }

    params = regime_params.get(regime, {'mean': 0.0, 'std': 0.02})
    return np.random.normal(params['mean'], params['std'], n_obs)


def generate_regime_switching_data(
    regimes: list,
    obs_per_regime: int = 50,
    random_seed: int = None,
) -> np.ndarray:
    """
    Generate data with multiple regime switches.

    Args:
        regimes: List of regime names in order
        obs_per_regime: Observations per regime
        random_seed: Random seed

    Returns:
        Array of log returns with regime switches
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []
    for regime in regimes:
        regime_data = generate_regime_data(obs_per_regime, regime)
        data.append(regime_data)

    return np.concatenate(data)


class TestStaticUpdateStrategy:
    """Tests for static update strategy (no parameter updates)."""

    def test_static_strategy_no_parameter_change(self):
        """Static strategy should never change parameters."""
        config = HMMConfig(
            n_states=2,
            update_strategy='static',
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Initial fit on bull regime
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        df1 = pd.DataFrame({'log_return': data1}, index=dates1)
        hmm.fit(df1)

        # Store parameters after initial fit
        params_after_fit = {
            'means': hmm.emission_means_.copy(),
            'stds': hmm.emission_stds_.copy(),
            'transitions': hmm.transition_matrix_.copy(),
            'initial': hmm.initial_probs_.copy(),
        }

        # Multiple updates with dramatically different data (bear regime)
        for i in range(5):
            data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100+i)
            dates2 = pd.date_range(f"2023-04-{i+1:02d}", periods=50, freq="D")
            df2 = pd.DataFrame({'log_return': data2}, index=dates2)
            result = hmm.update(df2)

            # Check that predictions are returned
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 50
            assert 'predicted_state' in result.columns

        # Parameters should be EXACTLY identical (no updates)
        assert np.allclose(hmm.emission_means_, params_after_fit['means'])
        assert np.allclose(hmm.emission_stds_, params_after_fit['stds'])
        assert np.allclose(hmm.transition_matrix_, params_after_fit['transitions'])
        assert np.allclose(hmm.initial_probs_, params_after_fit['initial'])

    def test_static_strategy_update_history_empty(self):
        """Static strategy should not log any updates."""
        config = HMMConfig(
            n_states=2,
            update_strategy='static',
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit and update
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        # Check update history is empty
        assert len(hmm.training_history_['update_history']) == 0
        assert len(hmm.training_history_['refit_history']) == 0

    def test_static_strategy_first_update_triggers_fit(self):
        """First update on unfitted model should trigger fit."""
        config = HMMConfig(
            n_states=2,
            update_strategy='static',
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # First update should fit
        assert not hmm.is_fitted
        data = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        result = hmm.update(pd.DataFrame({'log_return': data}, index=dates))

        assert hmm.is_fitted
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

        # Check fit observations are stored
        assert hmm.training_history_['fit_observations'] is not None
        assert len(hmm.training_history_['fit_observations']) == 100


class TestIncrementalUpdateStrategy:
    """Tests for incremental update strategy (smooth parameter updates)."""

    def test_incremental_strategy_parameter_updates(self):
        """Incremental strategy should smoothly update parameters."""
        config = HMMConfig(
            n_states=2,
            update_strategy='incremental',
            incremental_learning_rate=0.1,
            incremental_min_observations=10,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Initial fit on bull regime
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        params_after_fit = hmm.emission_means_.copy()

        # Update with bear regime data (opposite characteristics)
        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        params_after_update = hmm.emission_means_.copy()

        # Parameters should have changed (learned from new data)
        assert not np.allclose(params_after_fit, params_after_update)

        # Check update was logged
        assert len(hmm.training_history_['update_history']) > 0
        last_update = hmm.training_history_['update_history'][-1]
        assert 'timestamp' in last_update
        assert 'observation_count' in last_update
        assert 'parameter_changes' in last_update

    def test_incremental_strategy_learning_rate_effect(self):
        """Higher learning rate should lead to larger parameter changes."""
        # Low learning rate
        config_low = HMMConfig(
            n_states=2,
            update_strategy='incremental',
            incremental_learning_rate=0.01,
            incremental_min_observations=10,
            max_iterations=10,
            random_seed=42,
        )
        hmm_low = HiddenMarkovModel(config_low)

        # High learning rate
        config_high = HMMConfig(
            n_states=2,
            update_strategy='incremental',
            incremental_learning_rate=0.2,
            incremental_min_observations=10,
            max_iterations=10,
            random_seed=42,
        )
        hmm_high = HiddenMarkovModel(config_high)

        # Fit both on same data
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        df1 = pd.DataFrame({'log_return': data1}, index=dates1)
        hmm_low.fit(df1)
        hmm_high.fit(df1)

        params_low_before = hmm_low.emission_means_.copy()
        params_high_before = hmm_high.emission_means_.copy()

        # Update both with same new data
        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        df2 = pd.DataFrame({'log_return': data2}, index=dates2)
        hmm_low.update(df2)
        hmm_high.update(df2)

        # Calculate parameter change magnitudes
        change_low = np.linalg.norm(hmm_low.emission_means_ - params_low_before)
        change_high = np.linalg.norm(hmm_high.emission_means_ - params_high_before)

        # High learning rate should have larger changes
        assert change_high > change_low

    def test_incremental_strategy_min_observations_threshold(self):
        """Updates should only occur when min_observations is met."""
        config = HMMConfig(
            n_states=2,
            update_strategy='incremental',
            incremental_learning_rate=0.1,
            incremental_min_observations=50,  # High threshold
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Initial fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        params_after_fit = hmm.emission_means_.copy()

        # Update with fewer observations than threshold (30 < 50)
        data2 = generate_regime_data(n_obs=30, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=30, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        # Parameters should NOT have changed (below threshold)
        assert np.allclose(hmm.emission_means_, params_after_fit)

        # Update with sufficient observations (60 >= 50)
        data3 = generate_regime_data(n_obs=60, regime='bear', random_seed=200)
        dates3 = pd.date_range("2023-05-01", periods=60, freq="D")
        hmm.update(pd.DataFrame({'log_return': data3}, index=dates3))

        # Parameters SHOULD have changed (threshold met)
        assert not np.allclose(hmm.emission_means_, params_after_fit)

    def test_incremental_strategy_observation_counter(self):
        """Test that last_update_observation counter is maintained."""
        config = HMMConfig(
            n_states=2,
            update_strategy='incremental',
            incremental_learning_rate=0.1,
            incremental_min_observations=10,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        initial_count = hmm.training_history_['last_update_observation']

        # Update with 50 observations
        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        # Counter should have increased by 50
        assert hmm.training_history_['last_update_observation'] == initial_count + 50


class TestAdaptiveRefitStrategy:
    """Tests for adaptive refit strategy (trigger-based refitting)."""

    def test_adaptive_refit_quality_tracking(self):
        """Adaptive refit should track quality metrics over time."""
        config = HMMConfig(
            n_states=2,
            update_strategy='adaptive_refit',
            refit_trigger_mode='quality',
            quality_degradation_threshold=0.15,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        assert len(hmm.training_history_['quality_metrics']) == 0

        # Update multiple times
        for i in range(3):
            data = generate_regime_data(n_obs=50, regime='bull', random_seed=100+i)
            dates = pd.date_range(f"2023-04-{i+1:02d}", periods=50, freq="D")
            hmm.update(pd.DataFrame({'log_return': data}, index=dates))

        # Quality metrics should be tracked
        assert len(hmm.training_history_['quality_metrics']) == 3

        # Each metric should have required fields
        for metric in hmm.training_history_['quality_metrics']:
            assert 'log_likelihood_per_obs' in metric
            assert 'timestamp' in metric

    def test_adaptive_refit_observation_counter(self):
        """Test observations_since_refit counter."""
        config = HMMConfig(
            n_states=2,
            update_strategy='adaptive_refit',
            refit_trigger_mode='time',
            refit_interval_observations=200,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        assert hmm.training_history_['observations_since_refit'] == 0

        # Update with 50 observations
        data2 = generate_regime_data(n_obs=50, regime='bull', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        assert hmm.training_history_['observations_since_refit'] == 50

        # Update with another 50 observations
        data3 = generate_regime_data(n_obs=50, regime='bull', random_seed=200)
        dates3 = pd.date_range("2023-05-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data3}, index=dates3))

        assert hmm.training_history_['observations_since_refit'] == 100

    def test_adaptive_refit_time_trigger(self):
        """Test time-based refit trigger."""
        config = HMMConfig(
            n_states=2,
            update_strategy='adaptive_refit',
            refit_trigger_mode='time',
            refit_interval_observations=100,
            refit_use_recent_window=True,
            refit_window_observations=150,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        params_after_fit = hmm.emission_means_.copy()

        # Update with 50 observations (below threshold, no refit)
        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))

        # Should not have refitted yet
        assert len(hmm.training_history_['refit_history']) == 0

        # Update with 60 more observations (total 110 >= 100, triggers refit)
        data3 = generate_regime_data(n_obs=60, regime='bear', random_seed=200)
        dates3 = pd.date_range("2023-05-01", periods=60, freq="D")

        # Capture output to check for refit message
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        hmm.update(pd.DataFrame({'log_return': data3}, index=dates3))

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Should have triggered refit
        assert 'refit' in output.lower()
        assert len(hmm.training_history_['refit_history']) == 1

        # Counter should reset
        assert hmm.training_history_['observations_since_refit'] == 0

        # Parameters should have changed due to refit
        assert not np.allclose(hmm.emission_means_, params_after_fit)

    def test_adaptive_refit_stores_diagnostics(self):
        """Test that refit events store diagnostics for comparison."""
        config = HMMConfig(
            n_states=2,
            update_strategy='adaptive_refit',
            refit_trigger_mode='time',
            refit_interval_observations=100,
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        # Trigger refit with enough observations
        data2 = generate_regime_data(n_obs=120, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=120, freq="D")

        # Suppress output
        import io
        import sys
        sys.stdout = io.StringIO()
        hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))
        sys.stdout = sys.__stdout__

        # Check refit history
        assert len(hmm.training_history_['refit_history']) == 1
        refit_event = hmm.training_history_['refit_history'][0]

        # Check required fields
        assert 'timestamp' in refit_event
        assert 'reason' in refit_event
        assert 'observations_since_last' in refit_event
        assert 'old_diagnostics' in refit_event
        assert 'new_diagnostics' in refit_event

        # Reason should mention observation limit
        assert 'observation' in refit_event['reason'].lower() or 'limit' in refit_event['reason'].lower()


class TestUpdateStrategyIntegration:
    """Integration tests for update strategies."""

    def test_strategy_switching_via_config(self):
        """Test that different strategies can be selected via config."""
        strategies = ['static', 'incremental', 'adaptive_refit']

        for strategy in strategies:
            config = HMMConfig(
                n_states=2,
                update_strategy=strategy,
                max_iterations=10,
                random_seed=42,
            )
            hmm = HiddenMarkovModel(config)

            # Fit and update
            data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
            dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
            hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

            data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
            dates2 = pd.date_range("2023-04-01", periods=50, freq="D")

            # Suppress output for adaptive_refit
            import io
            import sys
            sys.stdout = io.StringIO()
            result = hmm.update(pd.DataFrame({'log_return': data2}, index=dates2))
            sys.stdout = sys.__stdout__

            # All should return valid predictions
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 50
            assert 'predicted_state' in result.columns

    def test_regime_switching_data_handling(self):
        """Test update strategies with regime-switching data."""
        # Test all strategies on same regime-switching data
        strategies = ['static', 'incremental', 'adaptive_refit']
        results = {}

        for strategy in strategies:
            config = HMMConfig(
                n_states=3,
                update_strategy=strategy,
                incremental_learning_rate=0.1 if strategy == 'incremental' else 0.05,
                refit_interval_observations=150 if strategy == 'adaptive_refit' else None,
                max_iterations=15,
                random_seed=42,
            )
            hmm = HiddenMarkovModel(config)

            # Initial fit on bull regime
            data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
            dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
            hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

            # Simulate regime switches: bull -> sideways -> bear
            regime_sequence = ['sideways', 'bear', 'bull']
            all_predictions = []

            for i, regime in enumerate(regime_sequence):
                data = generate_regime_data(n_obs=60, regime=regime, random_seed=100+i)
                dates = pd.date_range(f"2023-{4+i:02d}-01", periods=60, freq="D")

                # Suppress output
                import io
                import sys
                sys.stdout = io.StringIO()
                preds = hmm.update(pd.DataFrame({'log_return': data}, index=dates))
                sys.stdout = sys.__stdout__

                all_predictions.append(preds)

            results[strategy] = {
                'predictions': pd.concat(all_predictions),
                'final_params': hmm.emission_means_.copy(),
                'update_count': len(hmm.training_history_['update_history']),
                'refit_count': len(hmm.training_history_['refit_history']),
            }

        # Static should have no updates or refits
        assert results['static']['update_count'] == 0
        assert results['static']['refit_count'] == 0

        # Incremental should have updates but no refits
        assert results['incremental']['update_count'] > 0
        assert results['incremental']['refit_count'] == 0

        # Adaptive refit might have refits (depends on triggers)
        assert results['adaptive_refit']['update_count'] == 0  # No incremental updates
        # Refit count depends on whether threshold was reached


class TestUpdateStrategyEdgeCases:
    """Test edge cases and error handling for update strategies."""

    def test_first_update_on_unfitted_model(self):
        """First update on unfitted model should trigger fit for all strategies."""
        strategies = ['static', 'incremental', 'adaptive_refit']

        for strategy in strategies:
            config = HMMConfig(
                n_states=2,
                update_strategy=strategy,
                max_iterations=10,
                random_seed=42,
            )
            hmm = HiddenMarkovModel(config)

            assert not hmm.is_fitted

            data = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
            dates = pd.date_range("2023-01-01", periods=100, freq="D")

            # Suppress output
            import io
            import sys
            sys.stdout = io.StringIO()
            result = hmm.update(pd.DataFrame({'log_return': data}, index=dates))
            sys.stdout = sys.__stdout__

            assert hmm.is_fitted
            assert hmm.training_history_['fit_observations'] is not None

    def test_update_with_missing_signal(self):
        """Update should raise error if observed_signal missing."""
        config = HMMConfig(
            n_states=2,
            update_strategy='static',
            observed_signal='log_return',
            max_iterations=10,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Fit with correct signal
        data1 = generate_regime_data(n_obs=100, regime='bull', random_seed=42)
        dates1 = pd.date_range("2023-01-01", periods=100, freq="D")
        hmm.fit(pd.DataFrame({'log_return': data1}, index=dates1))

        # Try to update with wrong signal
        data2 = generate_regime_data(n_obs=50, regime='bear', random_seed=100)
        dates2 = pd.date_range("2023-04-01", periods=50, freq="D")
        df_wrong = pd.DataFrame({'wrong_signal': data2}, index=dates2)

        with pytest.raises(ValueError) as exc_info:
            hmm.update(df_wrong)

        assert 'log_return' in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
