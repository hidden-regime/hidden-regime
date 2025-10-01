"""
Tests for flexible threshold-based regime classification.

Validates that the new approach correctly handles:
1. Outlier-constrained kmeans initialization
2. Flexible threshold-based state classification
3. Proper handling of any number of states (2-5+)
4. Financial domain constraints and validation
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import hidden_regime as hr
from hidden_regime.models.utils import initialize_parameters_kmeans
from hidden_regime.utils.state_mapping import (
    _classify_regime_by_return_threshold,
    _resolve_duplicate_regime_names,
    apply_regime_mapping_to_analysis,
    map_states_to_financial_regimes,
)


class TestFlexibleRegimeClassification:
    """Test flexible threshold-based regime classification."""

    def test_threshold_based_classification(self):
        """Test that thresholds correctly classify regimes."""
        test_cases = [
            (-0.05, "Crisis"),  # -5% daily
            (-0.01, "Bear"),  # -1% daily
            (0.005, "Sideways"),  # 0.5% daily
            (0.02, "Bull"),  # 2% daily
            (0.08, "Euphoric"),  # 8% daily
        ]

        for return_pct, expected_regime in test_cases:
            regime = _classify_regime_by_return_threshold(return_pct)
            assert (
                regime == expected_regime
            ), f"Return {return_pct:.1%} should be {expected_regime}, got {regime}"

    def test_flexible_state_mapping_no_forced_categories(self):
        """Test that state mapping doesn't force arbitrary categories."""
        # Test case: NVDA-like scenario with no bear market
        emission_means_log = np.array(
            [-0.00505901, 0.00921712, 0.02]
        )  # Fixed realistic third state
        mapping = map_states_to_financial_regimes(emission_means_log, 3)

        # Convert to check percentage space
        means_pct = np.exp(emission_means_log) - 1

        # Validate mapping based on actual returns, not forced Bear/Sideways/Bull
        assert mapping[0] == "Bear"  # -0.5% daily is legitimately bear
        assert mapping[1] == "Sideways"  # +0.9% daily is sideways
        assert mapping[2] == "Bull"  # +2.0% daily is bull

        # All should be reasonable regimes
        for state_idx, regime_name in mapping.items():
            return_pct = means_pct[state_idx]
            assert (
                abs(return_pct) <= 0.05
            ), f"State {state_idx} ({regime_name}) has extreme return: {return_pct:.2%}"

    def test_duplicate_regime_handling(self):
        """Test handling of multiple states in same regime category."""
        # Two bull-type regimes
        emission_means_log = np.array([0.005, 0.01, 0.035])  # Sideways, Bull, Bull
        regime_classifications = [
            (0, "Sideways", 0.005),
            (1, "Bull", 0.01),
            (2, "Bull", 0.035),
        ]

        mapping = _resolve_duplicate_regime_names(regime_classifications)

        assert mapping[0] == "Sideways"
        assert mapping[1] == "Weak Bull"  # Lower return bull
        assert mapping[2] == "Strong Bull"  # Higher return bull

    def test_variable_state_counts(self):
        """Test that mapping works correctly for different numbers of states."""

        # 2-state case: No bear market
        emission_means_2 = np.array([0.005, 0.02])  # Sideways, Bull
        mapping_2 = map_states_to_financial_regimes(emission_means_2, 2)

        assert mapping_2[0] == "Sideways"
        assert mapping_2[1] == "Bull"

        # 4-state case: Full range
        emission_means_4 = np.array(
            [-0.01, 0.001, 0.015, 0.08]
        )  # Bear, Sideways, Bull, Euphoric
        mapping_4 = map_states_to_financial_regimes(emission_means_4, 4)

        assert mapping_4[0] == "Bear"
        assert mapping_4[1] == "Sideways"
        assert mapping_4[2] == "Bull"
        assert mapping_4[3] == "Euphoric"

        # 5-state case: Multiple bulls
        emission_means_5 = np.array([-0.02, 0.002, 0.01, 0.025, 0.06])
        mapping_5 = map_states_to_financial_regimes(emission_means_5, 5)

        assert mapping_5[0] == "Bear"
        assert mapping_5[1] == "Sideways"
        assert "Bull" in mapping_5[2]  # Some type of bull
        assert "Bull" in mapping_5[3]  # Some type of bull
        assert mapping_5[4] == "Euphoric"

    def test_constrained_kmeans_prevents_extreme_centers(self):
        """Test that constrained kmeans prevents unrealistic regime centers."""
        # Create data with extreme outliers
        np.random.seed(42)
        normal_returns = np.random.normal(0.001, 0.015, 200)  # Normal market data
        outliers = np.array([0.25, -0.20, 0.30])  # Extreme outliers (+25%, -20%, +30%)

        # Combine normal and outlier data
        all_returns_pct = np.concatenate([normal_returns, outliers])
        all_returns_log = np.log(all_returns_pct + 1)

        # Initialize with constrained kmeans
        initial_probs, transition_matrix, emission_params = (
            initialize_parameters_kmeans(
                n_states=3, returns=all_returns_log, random_seed=42
            )
        )

        # Check that no emission mean is extreme in percentage space
        means_pct = np.exp(emission_params[:, 0]) - 1

        for i, mean_pct in enumerate(means_pct):
            assert (
                abs(mean_pct) <= 0.08
            ), f"State {i} has extreme constrained mean: {mean_pct:.2%}"

        # Check that means are ordered and reasonable
        sorted_means = sorted(means_pct)

        # Lowest should be negative but not extreme
        assert sorted_means[0] >= -0.08 and sorted_means[0] <= -0.001

        # Highest should be positive but not extreme
        assert sorted_means[-1] >= 0.001 and sorted_means[-1] <= 0.05

    def test_integration_with_pipeline(self):
        """Test that the new approach works in full pipeline integration."""
        # Create mock data with known regime structure
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        np.random.seed(42)

        # Create realistic regime structure: bear, sideways, bull periods
        prices = []
        current_price = 100.0

        for i in range(150):
            if i < 30:  # Bear period
                daily_return = np.random.normal(-0.008, 0.025)  # -0.8% mean, 2.5% vol
            elif i < 90:  # Sideways period
                daily_return = np.random.normal(0.002, 0.012)  # +0.2% mean, 1.2% vol
            else:  # Bull period
                daily_return = np.random.normal(0.018, 0.020)  # +1.8% mean, 2.0% vol

            current_price *= 1 + daily_return
            prices.append(current_price)

        mock_data = pd.DataFrame(
            {
                "Open": [p * (1 + np.random.normal(0, 0.002)) for p in prices],
                "High": [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, 150),
                "Adj Close": prices,
            },
            index=dates,
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data

            # Create pipeline with new constrained approach
            pipeline = hr.create_financial_pipeline("TEST", n_states=3)
            result = pipeline.update()

            # Validate results
            assert isinstance(result, str)
            assert len(result) > 0

            # Get analysis results
            analysis_output = pipeline.get_component_output("analysis")
            model_output = pipeline.get_component_output("model")

            # Check that regime names make sense
            regime_names = analysis_output["regime_name"].unique()
            meaningful_names = {
                "Bear",
                "Sideways",
                "Bull",
                "Crisis",
                "Euphoric",
                "Weak Bull",
                "Strong Bull",
                "Weak Bear",
                "Strong Bear",
            }

            for name in regime_names:
                assert any(
                    meaningful in name for meaningful in meaningful_names
                ), f"Unexpected regime name: {name}"

            # Check that model has reasonable emission means
            if (
                hasattr(model_output, "emission_means_")
                and model_output.emission_means_ is not None
            ):
                means_pct = np.exp(model_output.emission_means_) - 1

                # No extreme regimes
                for i, mean_pct in enumerate(means_pct):
                    assert (
                        abs(mean_pct) <= 0.08
                    ), f"State {i} has extreme emission mean: {mean_pct:.2%}"

                # States should be ordered by return
                assert np.all(
                    np.diff(means_pct) >= -0.01
                ), "States not properly ordered by return"

    def test_financial_domain_warnings(self):
        """Test that financial domain validation produces appropriate warnings."""
        # Test extreme regime detection
        extreme_means = np.array(
            [-0.001, 0.001, 0.15]
        )  # Normal, Normal, Extreme (15% daily)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapping = map_states_to_financial_regimes(extreme_means, 3)

            # Should warn about extreme regime
            warning_messages = [str(warning.message) for warning in w]
            assert any("extreme" in msg.lower() for msg in warning_messages)

        # Test inconsistent regime names
        inconsistent_means = np.array(
            [0.02, -0.01, 0.001]
        )  # Bull, Bear, Sideways (wrong order)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapping = map_states_to_financial_regimes(inconsistent_means, 3)

            # Mapping should still work, just with appropriate regime names
            means_pct = np.exp(inconsistent_means) - 1
            assert mapping[0] == "Bull"  # 2% daily
            assert mapping[1] == "Bear"  # -1% daily
            assert mapping[2] == "Sideways"  # 0.1% daily

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # All positive returns (growth stock scenario)
        all_positive = np.array([0.005, 0.015, 0.035])
        mapping_positive = map_states_to_financial_regimes(all_positive, 3)

        # Should get Sideways, Bull-type, Strong Bull-type (no forced Bear)
        # Note: 0.035 log return = 3.56% daily, which is Bull range, not Euphoric (>5%)
        assert mapping_positive[0] == "Sideways"
        assert "Bull" in mapping_positive[1]  # Could be "Bull" or "Weak Bull"
        assert (
            "Bull" in mapping_positive[2]
        )  # Could be "Bull" or "Strong Bull" (3.56% < 5% threshold)

        # All negative returns (bear market scenario)
        all_negative = np.array([-0.04, -0.015, -0.002])
        mapping_negative = map_states_to_financial_regimes(all_negative, 3)

        # Should get Crisis, Bear, Bear or similar
        assert "Crisis" in mapping_negative[0] or "Bear" in mapping_negative[0]
        assert "Bear" in mapping_negative[1] or "Bear" in mapping_negative[2]

    def test_analysis_integration(self):
        """Test that analysis integration uses actual regime characteristics."""
        # Mock analysis data
        analysis_data = pd.DataFrame(
            {
                "predicted_state": [0, 1, 2, 1, 0, 2],
                "confidence": [0.8, 0.9, 0.7, 0.85, 0.75, 0.95],
            },
            index=pd.date_range("2023-01-01", periods=6),
        )

        # Mock emission means
        emission_means = np.array([-0.008, 0.005, 0.025])  # Bear, Sideways, Bull

        # Apply mapping
        result = apply_regime_mapping_to_analysis(analysis_data, emission_means, 3)

        # Check that regime names are applied
        assert "regime_name" in result.columns
        assert "regime_type" in result.columns
        assert "expected_return_pct" in result.columns

        # Check that actual characteristics are used
        bear_rows = result[result["predicted_state"] == 0]
        bull_rows = result[result["predicted_state"] == 2]

        if not bear_rows.empty:
            assert (
                bear_rows["expected_return_pct"].iloc[0] < 0
            )  # Bear should be negative

        if not bull_rows.empty:
            assert (
                bull_rows["expected_return_pct"].iloc[0] > 0.01
            )  # Bull should be meaningfully positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
