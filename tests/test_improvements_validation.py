"""
Validation tests for the hidden-regime improvements.

Tests that the three critical improvements work correctly:
1. Data-driven state interpretation
2. Kmeans initialization as default
3. Automatic financial data pipeline
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import hidden_regime as hr
from hidden_regime.config.model import HMMConfig
from hidden_regime.utils.state_mapping import (
    log_return_to_percent_change,
    map_states_to_financial_regimes,
    percent_change_to_log_return,
)


class TestImprovementsValidation:
    """Test that all improvements work correctly."""

    @pytest.fixture
    def mock_financial_data(self):
        """Create mock financial data with known characteristics."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create price data with clear regimes
        prices = []
        current_price = 100.0

        for i in range(100):
            if i < 30:  # Bear market (declining)
                daily_return = np.random.normal(-0.01, 0.02)  # -1% mean, 2% vol
            elif i < 60:  # Sideways market (flat)
                daily_return = np.random.normal(0.0, 0.01)  # 0% mean, 1% vol
            else:  # Bull market (rising)
                daily_return = np.random.normal(0.015, 0.015)  # 1.5% mean, 1.5% vol

            current_price *= 1 + daily_return
            prices.append(current_price)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, 100),
                "Adj Close": prices,
            },
            index=dates,
        )

        return data

    def test_automatic_financial_pipeline(self, mock_financial_data):
        """Test that financial data pipeline automatically calculates required columns."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_financial_data

            # Create pipeline
            pipeline = hr.create_financial_pipeline("TEST", n_states=3)

            # Get data component output
            data_output = pipeline.data.get_all_data()

            # Validate mandatory columns exist
            mandatory_columns = ["price", "pct_change", "log_return"]
            for col in mandatory_columns:
                assert col in data_output.columns, f"Missing mandatory column: {col}"

            # Validate price calculation (OHLC average)
            expected_price = 0.25 * (
                mock_financial_data["Open"]
                + mock_financial_data["Close"]
                + mock_financial_data["High"]
                + mock_financial_data["Low"]
            )
            # Account for the first row being dropped due to NaN pct_change
            np.testing.assert_array_almost_equal(
                data_output["price"].values,
                expected_price.iloc[1:].values,  # Skip first row
                decimal=6,
            )

            # Validate pct_change calculation
            expected_pct_change = expected_price.pct_change().dropna()
            np.testing.assert_array_almost_equal(
                data_output["pct_change"].values, expected_pct_change.values, decimal=6
            )

            # Validate log_return calculation
            expected_log_return = np.log(expected_pct_change + 1.0)
            np.testing.assert_array_almost_equal(
                data_output["log_return"].values, expected_log_return.values, decimal=6
            )

    def test_kmeans_initialization_default(self):
        """Test that kmeans is now the default initialization method."""
        # Test default config
        config = HMMConfig()
        assert config.initialization_method == "kmeans"

        # Test preset configs
        conservative = HMMConfig.create_conservative()
        assert conservative.initialization_method == "kmeans"

        aggressive = HMMConfig.create_aggressive()
        assert aggressive.initialization_method == "kmeans"

        balanced = HMMConfig.create_balanced()
        assert balanced.initialization_method == "kmeans"

    def test_data_driven_state_interpretation(self, mock_financial_data):
        """Test that state interpretation matches actual emission characteristics."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_financial_data

            # Create and run pipeline
            pipeline = hr.create_financial_pipeline("TEST", n_states=3)
            result = pipeline.update()

            # Get model component to check emission means
            model_component = pipeline.model
            if (
                hasattr(model_component, "emission_means_")
                and model_component.emission_means_ is not None
            ):
                emission_means = model_component.emission_means_

                # Test data-driven state mapping
                state_mapping = map_states_to_financial_regimes(emission_means, 3)

                # Validate mapping makes sense
                sorted_indices = np.argsort(emission_means)

                # Lowest mean should be Bear
                assert state_mapping[sorted_indices[0]] == "Bear"
                # Middle mean should be Sideways
                assert state_mapping[sorted_indices[1]] == "Sideways"
                # Highest mean should be Bull
                assert state_mapping[sorted_indices[2]] == "Bull"

                # Get analysis results
                analysis_output = pipeline.get_component_output("analysis")

                # Check that regime names exist and are meaningful
                assert "regime_name" in analysis_output.columns
                assert "regime_type" in analysis_output.columns

                # Check that regime names match expectations
                unique_regimes = set(analysis_output["regime_name"].unique())
                expected_regimes = {"Bear", "Sideways", "Bull"}
                assert unique_regimes.issubset(expected_regimes)

    def test_financial_utility_functions(self):
        """Test the financial utility conversion functions."""
        # Test percentage to log return conversion
        pct_changes = [0.05, -0.03, 0.0, 0.10, -0.05]

        for pct in pct_changes:
            log_ret = percent_change_to_log_return(pct)
            # Convert back
            pct_back = log_return_to_percent_change(log_ret)

            # Should roundtrip accurately
            np.testing.assert_almost_equal(pct, pct_back, decimal=10)

        # Test specific values
        assert abs(percent_change_to_log_return(0.05) - np.log(1.05)) < 1e-10
        assert abs(log_return_to_percent_change(np.log(1.05)) - 0.05) < 1e-10

    def test_state_mapping_consistency(self):
        """Test that state mapping uses threshold-based classification consistently."""
        # Test case 1: Clear bear, sideways, bull pattern
        emission_means_1 = np.array([-0.01, 0.0001, 0.01])  # Bear, Sideways, Bull
        mapping_1 = map_states_to_financial_regimes(emission_means_1, 3)

        assert mapping_1[0] == "Bear"  # -1% daily = Bear
        assert mapping_1[1] == "Sideways"  # +0.01% daily = Sideways
        assert mapping_1[2] == "Bull"  # +1% daily = Bull

        # Test case 2: Different order - should use thresholds, not position
        emission_means_2 = np.array([0.015, -0.008, 0.0002])  # Bull, Bear, Sideways
        mapping_2 = map_states_to_financial_regimes(emission_means_2, 3)

        # Should map by actual threshold values, not by index
        assert "Bear" in mapping_2[1]  # -0.8% daily = Bear (< -0.5% threshold)
        assert "Sideways" in mapping_2[2]  # +0.02% daily = Sideways
        assert "Bull" in mapping_2[0]  # +1.5% daily = Bull

    def test_integration_with_mock_data(self, mock_financial_data):
        """Test complete integration with realistic mock data."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_financial_data

            # Create pipeline with all improvements
            pipeline = hr.create_financial_pipeline("TEST", n_states=3)

            # Run complete pipeline
            result = pipeline.update()

            # Validate we get meaningful output
            assert isinstance(result, str)
            assert len(result) > 0

            # Validate all components have outputs
            for component_name in ["data", "observations", "model", "analysis"]:
                output = pipeline.get_component_output(component_name)
                assert output is not None
                assert len(output) > 0

            # Validate data has mandatory financial columns
            data_output = pipeline.get_component_output("data")
            mandatory_columns = ["price", "pct_change", "log_return"]
            for col in mandatory_columns:
                assert col in data_output.columns

            # Validate analysis has regime interpretation
            analysis_output = pipeline.get_component_output("analysis")
            assert "regime_name" in analysis_output.columns
            assert "regime_type" in analysis_output.columns

            # Validate regime names are meaningful
            regime_names = analysis_output["regime_name"].unique()
            meaningful_names = {"Bear", "Sideways", "Bull", "Crisis", "Euphoric"}
            assert all(name in meaningful_names for name in regime_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
