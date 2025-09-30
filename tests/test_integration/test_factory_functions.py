"""
Integration tests for factory functions and high-level user API.

Tests the actual user experience with the hidden-regime package,
validating factory functions, configuration creation, and end-to-end workflows.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import hidden_regime as hr
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.pipeline.core import Pipeline
from hidden_regime.pipeline.temporal import TemporalController


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
            "Adj Close": prices,
        },
        index=dates,
    )

    return data


class TestFactoryFunctions:
    """Test high-level factory functions that users interact with."""

    def test_create_financial_pipeline_basic(self):
        """Test basic financial pipeline creation with minimal parameters."""
        pipeline = hr.create_financial_pipeline("AAPL", n_states=3)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.model.config.n_states == 3
        assert pipeline.data.config.ticker == "AAPL"
        assert pipeline.data.config.source == "yfinance"

    def test_create_financial_pipeline_custom_config(self):
        """Test financial pipeline creation with custom configuration."""
        pipeline = hr.create_financial_pipeline(
            "SPY",
            n_states=4,
            model_config_overrides={
                "observed_signal": "close_price",
                "initialization_method": "kmeans",
            },
        )

        assert pipeline.model.config.n_states == 4
        assert pipeline.data.config.ticker == "SPY"
        assert pipeline.model.config.observed_signal == "close_price"
        assert pipeline.model.config.initialization_method == "kmeans"

    def test_create_financial_pipeline_date_range(self):
        """Test financial pipeline creation with specific date range."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        pipeline = hr.create_financial_pipeline(
            "NVDA", start_date=start_date, end_date=end_date, n_states=3
        )

        assert pipeline.data.config.start_date == start_date
        assert pipeline.data.config.end_date == end_date

    def test_create_hmm_config_presets(self):
        """Test HMM configuration preset creation."""
        conservative = HMMConfig.create_conservative()
        aggressive = HMMConfig.create_aggressive()
        balanced = HMMConfig.create_balanced()

        # Conservative should have higher tolerance and smoothing
        assert conservative.tolerance == 1e-8
        assert conservative.smoothing_weight == 0.9
        assert conservative.enable_change_detection == False

        # Aggressive should have lower tolerance and faster adaptation
        assert aggressive.tolerance == 1e-4
        assert aggressive.adaptation_rate == 0.1
        assert aggressive.change_detection_threshold == 2.0

        # Balanced should be middle ground
        assert balanced.tolerance == 1e-6
        assert balanced.adaptation_rate == 0.05
        assert balanced.enable_change_detection == True

    def test_pipeline_with_preset_config(self):
        """Test pipeline creation using preset configurations."""
        # Test using aggressive preset config overrides
        pipeline = hr.create_financial_pipeline(
            "TSLA",
            model_config_overrides={
                "tolerance": 1e-4,
                "adaptation_rate": 0.1,
                "forgetting_factor": 0.95,
                "change_detection_threshold": 2.0,
            },
        )

        assert pipeline.model.config.tolerance == 1e-4
        assert pipeline.model.config.adaptation_rate == 0.1
        assert pipeline.model.config.forgetting_factor == 0.95
        assert pipeline.model.config.change_detection_threshold == 2.0


class TestUserWorkflows:
    """Test complete user workflows from start to finish."""

    @patch("yfinance.download")
    def test_complete_analysis_workflow(self, mock_download, mock_yfinance_data):
        """Test complete analysis workflow from pipeline creation to results."""
        mock_download.return_value = mock_yfinance_data

        # Create pipeline
        pipeline = hr.create_financial_pipeline("AAPL", n_states=3, period="1y")

        # Run analysis
        report_output = pipeline.update()

        # Verify report output is a string
        assert isinstance(report_output, str)
        assert len(report_output) > 0

        # Verify component outputs are stored
        model_output = pipeline.get_component_output("model")
        analysis_output = pipeline.get_component_output("analysis")
        data_output = pipeline.get_component_output("data")

        assert model_output is not None
        assert analysis_output is not None
        assert data_output is not None

        # Verify data output structure
        assert isinstance(data_output, pd.DataFrame)
        assert len(data_output) > 0
        assert "close" in data_output.columns  # Standardized lowercase column names

        # Verify model output has expected attributes
        # (exact structure depends on HMM implementation)
        assert hasattr(model_output, "__len__") or hasattr(model_output, "shape")

    @patch("yfinance.download")
    def test_temporal_analysis_workflow(self, mock_download, mock_yfinance_data):
        """Test temporal analysis workflow for backtesting."""
        mock_download.return_value = mock_yfinance_data

        # Create pipeline and temporal controller
        pipeline = hr.create_financial_pipeline("SPY", n_states=3)

        # Standardize mock data column names to match what FinancialDataLoader produces
        standardized_data = mock_yfinance_data.copy()
        standardized_data.columns = [col.lower() for col in standardized_data.columns]
        standardized_data = standardized_data.rename(columns={"adj close": "adj_close"})

        temporal = TemporalController(pipeline, standardized_data)

        # Set up temporal analysis
        start_analysis = datetime(2023, 6, 1)
        end_analysis = datetime(2023, 8, 1)

        # Step through time period
        try:
            results_list = temporal.step_through_time(
                start_analysis.strftime("%Y-%m-%d"),
                end_analysis.strftime("%Y-%m-%d"),
                freq="W",  # Weekly steps for faster testing
            )

            # Convert results to expected format
            results_history = []
            for i, (date_str, report_output) in enumerate(
                results_list[:3]
            ):  # Limit to 3 for testing
                if report_output:
                    # Get model output from pipeline
                    model_output = pipeline.get_component_output("model")
                    if model_output is not None:
                        results_history.append(
                            {
                                "step": i,
                                "report": report_output,
                                "model_output": model_output,
                            }
                        )
        except Exception:
            # Some steps may fail, which is acceptable for temporal testing
            results_history = []

        # Verify we got some results
        assert len(results_history) > 0

        # Verify structure of results
        for result in results_history:
            assert "report" in result
            assert isinstance(result["report"], str)
            assert len(result["report"]) > 0

    @patch("yfinance.download")
    def test_configuration_validation_workflow(self, mock_download, mock_yfinance_data):
        """Test configuration validation during pipeline creation."""
        mock_download.return_value = mock_yfinance_data

        # Test valid configuration passes
        pipeline = hr.create_financial_pipeline(
            "MSFT", n_states=3, max_iterations=100, tolerance=1e-6, period="6mo"
        )
        assert pipeline is not None

        # Test invalid n_states raises error
        with pytest.raises(Exception):  # Should raise ConfigurationError
            hr.create_financial_pipeline("MSFT", n_states=1)

        # Test invalid tolerance raises error
        with pytest.raises(Exception):  # Should raise ConfigurationError
            hr.create_financial_pipeline("MSFT", n_states=3, tolerance=-1.0)

        # Test invalid forgetting factor raises error
        with pytest.raises(Exception):  # Should raise ConfigurationError
            hr.create_financial_pipeline("MSFT", n_states=3, forgetting_factor=1.5)


class TestPipelineIntegration:
    """Test pipeline component integration and data flow."""

    @pytest.fixture
    def sample_pipeline(self):
        """Create a sample pipeline for testing."""
        return hr.create_financial_pipeline("AAPL", n_states=3, period="6mo")

    @patch("yfinance.download")
    def test_data_loading_integration(
        self, mock_download, sample_pipeline, mock_yfinance_data
    ):
        """Test data loading component integration."""
        mock_download.return_value = mock_yfinance_data

        # Test data loading
        data_component = sample_pipeline.data
        data = data_component.get_all_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "close" in data.columns  # Standardized lowercase column names

        # Test get_all_data method
        all_data = data_component.get_all_data()
        assert isinstance(all_data, pd.DataFrame)
        assert len(all_data) == len(data)

    @patch("yfinance.download")
    def test_observation_generation_integration(
        self, mock_download, sample_pipeline, mock_yfinance_data
    ):
        """Test observation component integration."""
        mock_download.return_value = mock_yfinance_data[
            :100
        ]  # Smaller dataset for speed

        # Load data first
        data_component = sample_pipeline.data
        data = data_component.get_all_data()

        # Generate observations
        obs_component = sample_pipeline.observation
        observations = obs_component.update(data)

        assert isinstance(observations, (pd.DataFrame, pd.Series, np.ndarray))
        assert len(observations) > 0

        # Verify observations have reasonable structure (allowing NaN in initial periods)
        if isinstance(observations, pd.DataFrame):
            # Check that we have mostly non-NaN values (at least 50% valid data)
            total_values = observations.size
            non_nan_values = observations.notna().sum().sum()
            assert (
                non_nan_values > total_values * 0.5
            ), "Too many NaN values in observations"

            # Check that we have valid data in later periods
            if len(observations) > 20:
                last_20_rows = observations.tail(20)
                assert (
                    not last_20_rows.isnull().all().any()
                ), "All values are NaN in recent observations"
        elif isinstance(observations, pd.Series):
            # Allow some NaN values but ensure we have mostly valid data
            assert observations.notna().sum() > len(observations) * 0.5
        else:  # numpy array
            # For numpy arrays, check that we have mostly valid data
            valid_count = np.sum(~np.isnan(observations))
            assert valid_count > observations.size * 0.5

    @patch("yfinance.download")
    def test_model_training_integration(
        self, mock_download, sample_pipeline, mock_yfinance_data
    ):
        """Test model component integration and training."""
        mock_download.return_value = mock_yfinance_data[
            :100
        ]  # Smaller dataset for speed

        # Run pipeline to train model
        report_output = sample_pipeline.update()

        # Verify pipeline ran successfully
        assert isinstance(report_output, str)
        assert len(report_output) > 0

        # Verify model was trained
        model_component = sample_pipeline.model
        model_output = sample_pipeline.get_component_output("model")
        assert model_output is not None

        # Verify we can get component outputs
        data_output = sample_pipeline.get_component_output("data")
        obs_output = sample_pipeline.get_component_output("observations")
        analysis_output = sample_pipeline.get_component_output("analysis")

        assert data_output is not None
        assert obs_output is not None
        assert analysis_output is not None


class TestErrorHandling:
    """Test error handling in factory functions and workflows."""

    def test_invalid_ticker_handling(self):
        """Test handling of invalid ticker symbols."""
        # Test that invalid ticker raises ConfigurationError during pipeline creation
        with pytest.raises(Exception):  # Should raise ConfigurationError
            hr.create_financial_pipeline("INVALID_TICKER", n_states=3)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        with patch("yfinance.download") as mock_download:
            # Mock very small dataset
            dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
            small_data = pd.DataFrame(
                {
                    "Open": [100, 101, 102, 103, 104],
                    "High": [101, 102, 103, 104, 105],
                    "Low": [99, 100, 101, 102, 103],
                    "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                    "Volume": [1000000] * 5,
                    "Adj Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                },
                index=dates,
            )

            mock_download.return_value = small_data

            pipeline = hr.create_financial_pipeline("TEST", n_states=3)

            # Should handle insufficient data gracefully
            with pytest.raises(Exception):  # Should raise training error
                pipeline.run()

    def test_configuration_error_propagation(self):
        """Test that configuration errors are properly propagated."""
        # Test configuration validation in factory function
        with pytest.raises(Exception):
            hr.create_financial_pipeline("AAPL", n_states=0)  # Invalid n_states

        with pytest.raises(Exception):
            hr.create_financial_pipeline(
                "AAPL", n_states=3, tolerance=-1.0
            )  # Invalid tolerance


if __name__ == "__main__":
    pytest.main([__file__])
