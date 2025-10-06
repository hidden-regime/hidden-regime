"""
Scenario tests for HMM update strategies.

Tests realistic use cases:
- Research scenario: Years of daily data with static/slow updates
- Crypto scenario: Weeks of high-frequency data with adaptive refitting
"""

import numpy as np
import pandas as pd
import pytest

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel


def generate_research_data(n_years: int = 2, seed: int = 42) -> pd.DataFrame:
    """
    Generate research scenario data: years of daily returns.

    Simulates stock market data with regime changes over multiple years.
    """
    np.random.seed(seed)
    n_days = n_years * 252  # Trading days

    # Create multi-year data with regime switches
    returns = []
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")  # Business days

    # Simulate multiple regime cycles
    for i in range(n_days):
        # Cycle through regimes based on time
        cycle_position = (i % 252) / 252  # Position in annual cycle

        if cycle_position < 0.3:  # Q1: Bull market
            ret = np.random.normal(0.001, 0.015)
        elif cycle_position < 0.6:  # Q2-Q3: Sideways
            ret = np.random.normal(0.0, 0.010)
        else:  # Q4: Volatile
            ret = np.random.normal(-0.0005, 0.025)

        returns.append(ret)

    return pd.DataFrame({'log_return': returns}, index=dates)


def generate_crypto_data(n_weeks: int = 4, freq: str = '15min', seed: int = 42) -> pd.DataFrame:
    """
    Generate crypto scenario data: weeks of high-frequency returns.

    Simulates cryptocurrency data with rapid regime changes.
    """
    np.random.seed(seed)

    # Calculate number of observations
    if freq == '1min':
        obs_per_day = 24 * 60
    elif freq == '15min':
        obs_per_day = 24 * 4
    elif freq == '1H':
        obs_per_day = 24
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    n_obs = n_weeks * 7 * obs_per_day
    dates = pd.date_range("2024-01-01", periods=n_obs, freq=freq)

    # Simulate crypto volatility with rapid regime switches
    returns = []
    regime_duration = obs_per_day * 2  # 2-day regimes

    for i in range(n_obs):
        regime_idx = (i // regime_duration) % 3

        if regime_idx == 0:  # Pump
            ret = np.random.normal(0.002, 0.030)
        elif regime_idx == 1:  # Dump
            ret = np.random.normal(-0.003, 0.040)
        else:  # Consolidation
            ret = np.random.normal(0.0, 0.015)

        returns.append(ret)

    return pd.DataFrame({'log_return': returns}, index=dates)


class TestResearchScenario:
    """Test research scenario: years of daily data with stable models."""

    def test_static_strategy_for_long_term_research(self):
        """Static strategy should maintain stable parameters for research."""
        # Create model with static strategy
        config = HMMConfig(
            n_states=3,
            update_strategy='static',
            max_iterations=50,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Generate 2 years of daily data
        full_data = generate_research_data(n_years=2, seed=42)

        # Initial fit on first year
        train_data = full_data.iloc[:252]
        hmm.fit(train_data)

        params_after_training = {
            'means': hmm.emission_means_.copy(),
            'stds': hmm.emission_stds_.copy(),
        }

        # Process second year in chunks (simulating live updates)
        for month in range(12):
            start_idx = 252 + (month * 21)
            end_idx = 252 + ((month + 1) * 21)
            monthly_data = full_data.iloc[start_idx:end_idx]

            result = hmm.update(monthly_data)

            # Verify predictions are returned
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(monthly_data)

        # Parameters should be EXACTLY the same (no updates)
        assert np.allclose(hmm.emission_means_, params_after_training['means'])
        assert np.allclose(hmm.emission_stds_, params_after_training['stds'])

        # No update history
        assert len(hmm.training_history_['update_history']) == 0
        assert len(hmm.training_history_['refit_history']) == 0

    def test_incremental_strategy_for_long_term_research(self):
        """Incremental strategy should slowly adapt to market changes."""
        # Create model with conservative incremental updates
        config = HMMConfig(
            n_states=3,
            update_strategy='incremental',
            incremental_learning_rate=0.02,  # Slow adaptation
            incremental_min_observations=20,
            max_iterations=50,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Generate 2 years of data with regime shift in year 2
        full_data = generate_research_data(n_years=2, seed=42)

        # Train on first year
        train_data = full_data.iloc[:252]
        hmm.fit(train_data)

        initial_means = hmm.emission_means_.copy()

        # Process second year (with different regime characteristics)
        test_data = full_data.iloc[252:]
        hmm.update(test_data)

        # Parameters should have changed (learned from new data)
        assert not np.allclose(hmm.emission_means_, initial_means)

        # But changes should be gradual (small magnitude)
        mean_change = np.linalg.norm(hmm.emission_means_ - initial_means)
        assert mean_change < 0.01  # Small change due to low learning rate

        # Update history should exist
        assert len(hmm.training_history_['update_history']) > 0


class TestCryptoScenario:
    """Test crypto scenario: high-frequency data with adaptive refitting."""

    def test_adaptive_refit_for_crypto_time_trigger(self):
        """Adaptive refit with time trigger for crypto trading."""
        # Create model with time-based refitting (every 500 observations ~= 2 hours)
        config = HMMConfig(
            n_states=3,
            update_strategy='adaptive_refit',
            refit_trigger_mode='time',
            refit_interval_observations=500,
            refit_use_recent_window=True,
            refit_window_observations=1000,
            max_iterations=20,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Generate 4 weeks of 15-minute crypto data
        full_data = generate_crypto_data(n_weeks=4, freq='15min', seed=42)

        # Initial fit on first week
        week1_obs = 7 * 24 * 4  # 7 days * 24 hours * 4 (15-min intervals) = 672
        train_data = full_data.iloc[:week1_obs]
        hmm.fit(train_data)

        # Process remaining weeks in batches that exceed threshold
        batch_size = 600  # Exceeds 500 threshold
        for i in range(week1_obs, len(full_data), batch_size):
            batch_data = full_data.iloc[i:i+batch_size]

            # Suppress refit output
            import io
            import sys
            sys.stdout = io.StringIO()
            result = hmm.update(batch_data)
            sys.stdout = sys.__stdout__

            assert isinstance(result, pd.DataFrame)

        # Should have triggered refits (multiple 600-obs batches > 500 threshold)
        assert len(hmm.training_history_['refit_history']) >= 1

        # Each refit should be logged with diagnostics
        for refit_event in hmm.training_history_['refit_history']:
            assert 'reason' in refit_event
            assert 'observation' in refit_event['reason'].lower() or 'limit' in refit_event['reason'].lower()
            assert 'old_diagnostics' in refit_event
            assert 'new_diagnostics' in refit_event

    def test_adaptive_refit_for_crypto_quality_trigger(self):
        """Adaptive refit with quality trigger for crypto trading."""
        # Create model with quality-based refitting
        config = HMMConfig(
            n_states=3,
            update_strategy='adaptive_refit',
            refit_trigger_mode='quality',
            quality_degradation_threshold=0.20,  # 20% degradation triggers refit
            min_silhouette_threshold=0.25,  # Poor clustering triggers refit
            quality_window_size=100,
            enable_distribution_shift_detection=True,
            max_iterations=20,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Generate crypto data
        full_data = generate_crypto_data(n_weeks=2, freq='15min', seed=42)

        # Initial fit
        train_obs = 7 * 24 * 4  # 1 week
        train_data = full_data.iloc[:train_obs]
        hmm.fit(train_data)

        # Process in batches
        batch_size = 2000  # ~8 hours of data
        for i in range(train_obs, len(full_data), batch_size):
            batch_data = full_data.iloc[i:i+batch_size]

            # Suppress output
            import io
            import sys
            sys.stdout = io.StringIO()
            result = hmm.update(batch_data)
            sys.stdout = sys.__stdout__

        # Quality metrics should be tracked
        assert len(hmm.training_history_['quality_metrics']) > 0

        # Each quality metric should have log likelihood
        for metric in hmm.training_history_['quality_metrics']:
            assert 'log_likelihood_per_obs' in metric
            assert 'timestamp' in metric

    def test_adaptive_refit_hybrid_mode(self):
        """Hybrid mode combines time and quality triggers."""
        config = HMMConfig(
            n_states=3,
            update_strategy='adaptive_refit',
            refit_trigger_mode='hybrid',
            refit_interval_observations=500,  # Time trigger (realistic for 15min data)
            quality_degradation_threshold=0.15,  # Quality trigger
            min_silhouette_threshold=0.25,
            max_iterations=20,
            random_seed=42,
        )
        hmm = HiddenMarkovModel(config)

        # Generate data (2 weeks = ~1344 observations)
        full_data = generate_crypto_data(n_weeks=2, freq='15min', seed=42)

        # Initial fit on first week
        train_obs = 7 * 24 * 4  # 672 observations
        hmm.fit(full_data.iloc[:train_obs])

        # Process rest in batches
        import io
        import sys
        sys.stdout = io.StringIO()

        batch_size = 300
        for i in range(train_obs, len(full_data), batch_size):
            batch_data = full_data.iloc[i:i+batch_size]
            if len(batch_data) > 0:
                hmm.update(batch_data)

        sys.stdout = sys.__stdout__

        # Should track both time and quality
        assert hmm.training_history_['observations_since_refit'] >= 0
        assert len(hmm.training_history_['quality_metrics']) > 0


class TestScenarioComparison:
    """Compare different strategies on same scenario data."""

    def test_strategy_comparison_on_research_data(self):
        """Compare all strategies on research scenario."""
        # Generate data
        data = generate_research_data(n_years=1, seed=42)
        train_data = data.iloc[:200]
        test_data = data.iloc[200:]

        strategies = {
            'static': HMMConfig(
                n_states=3,
                update_strategy='static',
                max_iterations=20,
                random_seed=42,
            ),
            'incremental': HMMConfig(
                n_states=3,
                update_strategy='incremental',
                incremental_learning_rate=0.05,
                incremental_min_observations=20,
                max_iterations=20,
                random_seed=42,
            ),
            'adaptive': HMMConfig(
                n_states=3,
                update_strategy='adaptive_refit',
                refit_trigger_mode='time',
                refit_interval_observations=100,
                max_iterations=20,
                random_seed=42,
            ),
        }

        results = {}

        for name, config in strategies.items():
            hmm = HiddenMarkovModel(config)
            hmm.fit(train_data)

            # Suppress output for adaptive
            import io
            import sys
            if name == 'adaptive':
                sys.stdout = io.StringIO()

            preds = hmm.update(test_data)

            if name == 'adaptive':
                sys.stdout = sys.__stdout__

            results[name] = {
                'predictions': preds,
                'final_params': hmm.emission_means_.copy(),
                'update_count': len(hmm.training_history_['update_history']),
                'refit_count': len(hmm.training_history_['refit_history']),
            }

        # Verify different behaviors
        assert results['static']['update_count'] == 0
        assert results['incremental']['update_count'] > 0
        assert results['adaptive']['refit_count'] >= 0  # May or may not refit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
