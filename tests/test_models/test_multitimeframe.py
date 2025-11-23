"""
Tests for MultiTimeframeRegime component.

Tests multi-timeframe regime detection including alignment scoring,
resampling, and signal filtering.
"""

import warnings
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.multitimeframe import MultiTimeframeRegime
from hidden_regime.utils.timeframe_resampling import (
    resample_to_weekly,
    resample_to_monthly,
    validate_timeframe_data,
)


class TestMultiTimeframeRegimeInitialization:
    """Test MultiTimeframeRegime initialization and setup."""

    def test_initialization_with_default_config(self):
        """Test basic initialization with default config."""
        model = MultiTimeframeRegime()

        assert model.n_states == 3
        assert hasattr(model, "models")
        assert "daily" in model.models
        assert "weekly" in model.models
        assert "monthly" in model.models

    def test_initialization_with_custom_config(self):
        """Test initialization with custom HMM config."""
        custom_config = HMMConfig(n_states=2, max_iterations=20)
        model = MultiTimeframeRegime(config=custom_config, n_states=2)

        assert model.n_states == 2
        assert model.models["daily"].n_states == 2
        assert model.models["weekly"].n_states == 2
        assert model.models["monthly"].n_states == 2

    def test_models_are_independent(self):
        """Test that daily, weekly, monthly models are independent."""
        config = HMMConfig(n_states=3, random_seed=42)
        model = MultiTimeframeRegime(config=config)

        # All models should be distinct instances
        assert model.models["daily"] is not model.models["weekly"]
        assert model.models["weekly"] is not model.models["monthly"]
        assert model.models["daily"] is not model.models["monthly"]

        # But all should have same config settings
        assert (
            model.models["daily"].n_states
            == model.models["weekly"].n_states
            == model.models["monthly"].n_states
        )


class TestTimeframeResampling:
    """Test timeframe resampling utilities."""

    @pytest.fixture
    def daily_data_with_index(self):
        """Create sample daily data with proper datetime index."""
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        data = pd.DataFrame(
            {
                "close": np.random.randn(250).cumsum() + 100,
                "high": np.random.randn(250).cumsum() + 101,
                "low": np.random.randn(250).cumsum() + 99,
                "volume": np.random.randint(1000000, 10000000, 250),
            },
            index=dates,
        )
        return data

    def test_resample_to_weekly(self, daily_data_with_index):
        """Test resampling to weekly frequency."""
        weekly = resample_to_weekly(daily_data_with_index)

        assert len(weekly) > 0
        assert isinstance(weekly.index, pd.DatetimeIndex)
        # 250 days from Jan 1 to Sep 6 is ~9 months, should have ~35-40 weeks
        assert 30 < len(weekly) < 50

    def test_resample_to_monthly(self, daily_data_with_index):
        """Test resampling to monthly frequency."""
        monthly = resample_to_monthly(daily_data_with_index)

        assert len(monthly) > 0
        assert isinstance(monthly.index, pd.DatetimeIndex)
        # 250 days from Jan 1 to Sep 6 is ~9 months
        assert 8 < len(monthly) < 10

    def test_resampling_preserves_close_prices(self, daily_data_with_index):
        """Test that resampling uses last close of period."""
        original_last_close = daily_data_with_index["close"].iloc[-1]

        monthly = resample_to_monthly(daily_data_with_index)
        resampled_last_close = monthly["close"].iloc[-1]

        # The last monthly close should match original last close
        # (within floating point precision)
        assert np.isclose(original_last_close, resampled_last_close)

    def test_resample_with_missing_date_column(self):
        """Test resampling with date column instead of index."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {"date": dates, "close": np.random.randn(100).cumsum() + 100}
        )

        weekly = resample_to_weekly(data)
        assert len(weekly) > 0

    def test_validate_timeframe_data_sufficient(self, daily_data_with_index):
        """Test validation with sufficient data."""
        is_valid, message = validate_timeframe_data(daily_data_with_index)

        assert is_valid == True
        assert "passed" in message.lower()

    def test_validate_timeframe_data_insufficient(self):
        """Test validation with insufficient data."""
        # Only 30 days of data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        data = pd.DataFrame(
            {"close": np.random.randn(30).cumsum() + 100}, index=dates
        )

        is_valid, message = validate_timeframe_data(
            data, min_daily_observations=100
        )

        assert is_valid == False
        assert "insufficient" in message.lower()


class TestAlignmentScoring:
    """Test multi-timeframe alignment scoring."""

    @pytest.fixture
    def sample_mtf_output(self):
        """Create sample MultiTimeframeRegime output."""
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        return pd.DataFrame(
            {
                "daily_predicted_state": np.random.randint(0, 3, 250),
                "weekly_predicted_state": np.random.randint(0, 3, 250),
                "monthly_predicted_state": np.random.randint(0, 3, 250),
            },
            index=dates,
        )

    def test_alignment_score_perfect_match(self):
        """Test alignment score when all timeframes match."""
        # All states are 0 (perfect alignment)
        model = MultiTimeframeRegime(n_states=3)

        row = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": 0,
                "monthly_predicted_state": 0,
            }
        )

        score = model._compute_alignment_score(row)

        # Perfect alignment (3 matches) should give 1.0
        assert np.isclose(score, 1.0)

    def test_alignment_score_two_matches(self):
        """Test alignment score when two pairs of timeframes match."""
        model = MultiTimeframeRegime(n_states=3)

        row = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": 0,
                "monthly_predicted_state": 0,
            }
        )

        # All three match means 3 pairs match (daily==weekly, daily==monthly, weekly==monthly)
        score = model._compute_alignment_score(row)

        # Three matching pairs should give 1.0
        assert np.isclose(score, 1.0)

        # Now test exactly 2 pairs matching
        row2 = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": 0,
                "monthly_predicted_state": 1,
            }
        )

        score2 = model._compute_alignment_score(row2)
        # daily==weekly is true, but both are != monthly, so 1 match = 0.6
        assert np.isclose(score2, 0.6)

    def test_alignment_score_one_match(self):
        """Test alignment score when only one pair matches."""
        model = MultiTimeframeRegime(n_states=3)

        row = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": 0,
                "monthly_predicted_state": 1,
            }
        )

        score = model._compute_alignment_score(row)

        # At least one match (daily/weekly agree)
        assert 0.6 <= score <= 0.8

    def test_alignment_score_no_matches(self):
        """Test alignment score when timeframes misaligned."""
        model = MultiTimeframeRegime(n_states=3)

        row = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": 1,
                "monthly_predicted_state": 2,
            }
        )

        score = model._compute_alignment_score(row)

        # No matches (all different) should give 0.3
        assert np.isclose(score, 0.3)

    def test_alignment_score_range(self):
        """Test that alignment scores are always in valid range."""
        model = MultiTimeframeRegime(n_states=3)

        # Test many combinations
        for d in range(3):
            for w in range(3):
                for m in range(3):
                    row = pd.Series(
                        {
                            "daily_predicted_state": d,
                            "weekly_predicted_state": w,
                            "monthly_predicted_state": m,
                        }
                    )

                    score = model._compute_alignment_score(row)

                    # Score should be one of: 0.3, 0.6, 0.8, or 1.0
                    assert score in [0.3, 0.6, 0.8, 1.0]


class TestMultiTimeframeUpdate:
    """Test MultiTimeframeRegime update/training."""

    @pytest.fixture
    def sample_daily_data(self):
        """Create realistic daily OHLC data."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=730, freq="D")

        # Create returns with regime changes (2 years of data)
        returns = np.concatenate(
            [
                np.random.randn(240) * 0.01 - 0.005,  # Bear regime
                np.random.randn(240) * 0.008,  # Sideways
                np.random.randn(250) * 0.012 + 0.003,  # Bull regime
            ]
        )

        prices = (1 + returns).cumprod() * 100

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(730) * 0.001),
                "high": prices * (1 + np.abs(np.random.randn(730)) * 0.005),
                "low": prices * (1 - np.abs(np.random.randn(730)) * 0.005),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 730),
            },
            index=dates,
        )

    def test_update_with_sufficient_data(self, sample_daily_data):
        """Test update with sufficient data for all timeframes."""
        config = HMMConfig(n_states=3, max_iterations=10, random_seed=42)
        model = MultiTimeframeRegime(config=config)

        # Create observations from close prices (log returns)
        close_prices = sample_daily_data["close"]
        log_returns = np.log(close_prices.pct_change() + 1)
        obs_df = pd.DataFrame(
            {"observation": log_returns}, index=sample_daily_data.index
        )

        # This should not raise an error
        result, metadata = model.update(obs_df)

        assert result is not None
        assert len(result) > 0
        assert "daily_predicted_state" in result.columns
        assert "weekly_predicted_state" in result.columns
        assert "monthly_predicted_state" in result.columns

    def test_update_adds_alignment_columns(self, sample_daily_data):
        """Test that update adds alignment score columns."""
        config = HMMConfig(n_states=3, max_iterations=5, random_seed=42)
        model = MultiTimeframeRegime(config=config)

        # Create observations
        close_prices = sample_daily_data["close"]
        log_returns = np.log(close_prices.pct_change() + 1)
        obs_df = pd.DataFrame(
            {"observation": log_returns}, index=sample_daily_data.index
        )

        result, metadata = model.update(obs_df)

        assert "alignment_score" in result.columns
        assert "alignment_label" in result.columns

    def test_update_alignment_scores_valid(self, sample_daily_data):
        """Test that alignment scores are valid."""
        config = HMMConfig(n_states=3, max_iterations=5, random_seed=42)
        model = MultiTimeframeRegime(config=config)

        # Create observations
        close_prices = sample_daily_data["close"]
        log_returns = np.log(close_prices.pct_change() + 1)
        obs_df = pd.DataFrame(
            {"observation": log_returns}, index=sample_daily_data.index
        )

        result, metadata = model.update(obs_df)

        # All scores should be in [0.3, 0.6, 0.8, 1.0]
        assert result["alignment_score"].notna().all()
        for score in result["alignment_score"].unique():
            assert score in [0.3, 0.6, 0.8, 1.0]

    def test_update_insufficient_data_weekly(self):
        """Test update with insufficient data for weekly resampling."""
        # Only 20 days (too few for meaningful weekly analysis)
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        close_prices = np.random.randn(20).cumsum() + 100

        # Create observations from close prices
        log_returns = np.log(np.array(close_prices[1:]) / np.array(close_prices[:-1]))
        obs_df = pd.DataFrame(
            {"observation": [0] + list(log_returns)}, index=dates
        )

        config = HMMConfig(n_states=3, random_seed=42)
        model = MultiTimeframeRegime(config=config)

        # Should handle gracefully (may return partial results or raise)
        try:
            result, metadata = model.update(obs_df)
            # If it returns, result should be valid
            assert result is not None
        except ValueError:
            # Acceptable to raise ValueError for insufficient data
            pass


class TestSignalGenerationIntegration:
    """Test signal generation with multi-timeframe alignment."""

    @pytest.fixture
    def aligned_data(self):
        """Create data where timeframes are perfectly aligned."""
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        return pd.DataFrame(
            {
                "signal_valid": True,
                "regime_type": "Bullish",
                "daily_predicted_state": 2,
                "weekly_predicted_state": 2,
                "monthly_predicted_state": 2,
                "timeframe_alignment": 1.0,
                "regime_strength": 0.8,
            },
            index=dates,
        )

    @pytest.fixture
    def misaligned_data(self):
        """Create data where timeframes are misaligned."""
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        return pd.DataFrame(
            {
                "signal_valid": True,
                "regime_type": "Bullish",
                "daily_predicted_state": 2,
                "weekly_predicted_state": 1,
                "monthly_predicted_state": 0,
                "timeframe_alignment": 0.3,
                "regime_strength": 0.8,
            },
            index=dates,
        )

    def test_signal_with_perfect_alignment(self, aligned_data):
        """Test signal generation with perfect alignment."""
        from hidden_regime.config.signal_generation import (
            SignalGenerationConfiguration,
        )
        from hidden_regime.signal_generation.financial import (
            FinancialSignalGenerator,
        )

        config = SignalGenerationConfiguration(strategy_type="multi_timeframe")
        generator = FinancialSignalGenerator(config)

        for _, row in aligned_data.iterrows():
            signal = generator._calculate_base_signal(row)

            # Perfect alignment with bullish regime should give strong signal
            if row["signal_valid"]:
                assert signal > 0.5  # Should be positive (bullish)

    def test_signal_with_misalignment(self, misaligned_data):
        """Test signal generation with misaligned timeframes."""
        from hidden_regime.config.signal_generation import (
            SignalGenerationConfiguration,
        )
        from hidden_regime.signal_generation.financial import (
            FinancialSignalGenerator,
        )

        config = SignalGenerationConfiguration(
            strategy_type="multi_timeframe",
            position_size_range=[0.0, 1.0],
        )
        generator = FinancialSignalGenerator(config)

        for _, row in misaligned_data.iterrows():
            signal = generator._calculate_base_signal(row)

            # Misaligned timeframes should give weak or zero signal
            if row["signal_valid"]:
                # Alignment < 0.7 should filter signal to 0
                assert signal == 0.0


class TestBackwardCompatibility:
    """Test backward compatibility with single-timeframe models."""

    @pytest.fixture
    def daily_data(self):
        """Create sample daily data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        returns = np.random.randn(200) * 0.01
        prices = (1 + returns).cumprod() * 100

        return pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 200),
            },
            index=dates,
        )

    def test_single_hmm_still_works(self, daily_data):
        """Test that single-timeframe HMM still works independently."""
        from hidden_regime.models.hmm import HiddenMarkovModel

        config = HMMConfig(n_states=3, max_iterations=10, random_seed=42)
        model = HiddenMarkovModel(config)

        # Create observations
        obs_df = pd.DataFrame(
            {"log_return": np.log(daily_data["close"].pct_change() + 1)},
            index=daily_data.index,
        )

        # Should still work normally
        model.fit(obs_df)
        assert model.is_fitted

    def test_pipeline_config_backward_compatible(self):
        """Test that pipeline config works without multitimeframe_config."""
        from hidden_regime.config.pipeline import PipelineConfiguration

        # Create config without specifying multitimeframe_config
        config = PipelineConfiguration()

        # Should work fine
        assert config.multitimeframe_config is None

        # to_dict should work
        config_dict = config.to_dict()
        assert "multitimeframe_config" in config_dict
        assert config_dict["multitimeframe_config"] is None

        # from_dict should work
        restored = PipelineConfiguration.from_dict(config_dict)
        assert restored.multitimeframe_config is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_alignment_score_with_missing_values(self):
        """Test alignment scoring with missing state values."""
        model = MultiTimeframeRegime(n_states=3)

        # Missing a predicted state
        row = pd.Series(
            {
                "daily_predicted_state": 0,
                "weekly_predicted_state": np.nan,
                "monthly_predicted_state": 0,
            }
        )

        # Should handle gracefully
        try:
            score = model._compute_alignment_score(row)
            # If it computes, score should be valid
            assert 0 <= score <= 1
        except (TypeError, ValueError):
            # Acceptable to raise on missing data
            pass

    def test_resample_empty_data(self):
        """Test resampling with empty data."""
        empty_data = pd.DataFrame()

        weekly = resample_to_weekly(empty_data)
        assert len(weekly) == 0

        monthly = resample_to_monthly(empty_data)
        assert len(monthly) == 0

    def test_validate_none_data(self):
        """Test validation with None data."""
        is_valid, message = validate_timeframe_data(None)

        assert is_valid == False
        assert "no data" in message.lower()
