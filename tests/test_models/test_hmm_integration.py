"""
Integration tests for HMM with data pipeline.

Tests end-to-end workflows combining data loading, validation,
preprocessing with HMM training and inference.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import hidden_regime as hr
from hidden_regime.models import HiddenMarkovModel, HMMConfig
from hidden_regime.data import DataLoader, DataValidator, DataPreprocessor
from hidden_regime.config import DataConfig, ValidationConfig, PreprocessingConfig


class TestHMMIntegration:
    """Test HMM integration with data pipeline."""

    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Generate regime-like price data
        prices = [100.0]  # Starting price
        regime_means = [-0.02, 0.005, 0.015]  # Bear, Sideways, Bull
        regime_stds = [0.025, 0.015, 0.02]

        # Simulate regime switching
        current_regime = 1  # Start in sideways
        regime_duration = 0

        for i in range(1, 100):
            # Switch regime occasionally
            regime_duration += 1
            if regime_duration > np.random.poisson(10):  # Average 10-day regimes
                current_regime = np.random.choice([0, 1, 2])
                regime_duration = 0

            # Generate return based on current regime
            return_val = np.random.normal(
                regime_means[current_regime], regime_stds[current_regime]
            )

            new_price = prices[-1] * (1 + return_val)
            prices.append(new_price)

        # Create DataFrame
        data = pd.DataFrame({"date": dates, "price": prices})

        # Add OHLC columns (simplified)
        data["Open"] = data["price"] * (1 + np.random.normal(0, 0.001, len(data)))
        data["High"] = data["price"] * (
            1 + np.abs(np.random.normal(0, 0.005, len(data)))
        )
        data["Low"] = data["price"] * (
            1 - np.abs(np.random.normal(0, 0.005, len(data)))
        )
        data["Close"] = data["price"]
        data["Volume"] = np.random.randint(1000000, 10000000, len(data))

        return data

    def test_convenience_function_detect_regimes(self, sample_market_data):
        """Test detect_regimes convenience function with realistic data."""
        # Calculate returns manually for testing
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()

        # Test basic usage
        states = hr.detect_regimes(returns, n_states=3)

        assert len(states) == len(returns)
        assert np.all(states >= 0)
        assert np.all(states < 3)
        assert len(np.unique(states)) <= 3  # At most 3 distinct states

        # Test with return_model=True
        states2, model = hr.detect_regimes(returns, n_states=3, return_model=True)

        assert np.array_equal(states, states2)
        assert isinstance(model, HiddenMarkovModel)
        assert model.is_fitted

    def test_convenience_function_analyze_regime_transitions(self, sample_market_data):
        """Test analyze_regime_transitions convenience function."""
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()
        dates = sample_market_data["date"][1:]  # Match returns length

        analysis = hr.analyze_regime_transitions(
            returns, dates, n_states=3, verbose=False
        )

        # Check structure
        assert "model_info" in analysis
        assert "regime_parameters" in analysis
        assert "regime_interpretations" in analysis
        assert "regime_statistics" in analysis
        assert "state_sequence" in analysis
        assert "state_probabilities" in analysis

        # Verify model info
        assert analysis["model_info"]["n_states"] == 3
        assert analysis["model_info"]["n_observations"] == len(returns)

        # Verify regime statistics
        regime_stats = analysis["regime_statistics"]
        assert regime_stats["n_observations"] == len(returns)
        assert regime_stats["n_states"] == 3

        # Check that all regimes have statistics
        for regime in range(3):
            assert regime in regime_stats["regime_stats"]
            stats = regime_stats["regime_stats"][regime]
            assert "frequency" in stats
            assert "mean_return" in stats
            assert "avg_duration" in stats

    @patch("yfinance.Ticker")
    def test_end_to_end_pipeline_with_hmm(self, mock_ticker, sample_market_data):
        """Test complete end-to-end pipeline from data loading to HMM analysis."""
        # Mock yfinance response
        mock_stock = MagicMock()
        mock_stock.history.return_value = sample_market_data.set_index("date")[
            ["Open", "High", "Low", "Close", "Volume"]
        ]
        mock_ticker.return_value = mock_stock

        # Step 1: Load data
        data_config = DataConfig(
            use_ohlc_average=True, include_volume=True, cache_enabled=False
        )

        loader = DataLoader(data_config)
        raw_data = loader.load_stock_data("TEST", "2023-01-01", "2023-04-10")

        # Step 2: Validate data
        validation_config = ValidationConfig(
            outlier_method="iqr", max_daily_return=0.2, max_consecutive_missing=3
        )

        validator = DataValidator(validation_config)
        validation_result = validator.validate_data(raw_data, "TEST")

        assert validation_result.is_valid
        assert validation_result.quality_score > 0.7  # Should be good quality

        # Step 3: Preprocess data
        preprocessing_config = PreprocessingConfig(
            return_method="log", calculate_volatility=True, apply_smoothing=False
        )

        preprocessor = DataPreprocessor(preprocessing_config, validation_config)
        processed_data = preprocessor.process_data(raw_data, "TEST")

        # Step 4: Train HMM
        hmm_config = HMMConfig(
            n_states=3, max_iterations=50, random_seed=42, early_stopping=True
        )

        hmm = HiddenMarkovModel(config=hmm_config)
        hmm.fit(processed_data["log_return"], verbose=False)

        assert hmm.is_fitted
        assert hmm.training_history_["iterations"] > 0
        assert hmm.training_history_["converged"] in [True, False]

        # Step 5: Perform inference
        states = hmm.predict(processed_data["log_return"])
        probabilities = hmm.predict_proba(processed_data["log_return"])

        assert len(states) == len(processed_data)
        assert probabilities.shape == (len(processed_data), 3)

        # Step 6: Real-time regime detection
        for i in range(min(5, len(processed_data))):
            regime_info = hmm.update_with_observation(
                processed_data["log_return"].iloc[i]
            )

            assert "most_likely_regime" in regime_info
            assert "confidence" in regime_info
            assert 0 <= regime_info["confidence"] <= 1

    def test_hmm_with_different_data_qualities(self):
        """Test HMM behavior with different data quality levels."""
        np.random.seed(42)

        # High quality data
        good_returns = np.random.normal(0, 0.02, 200)  # Clean, sufficient data

        hmm_good = HiddenMarkovModel(n_states=3, config=HMMConfig(random_seed=42))
        hmm_good.fit(good_returns, verbose=False)

        assert hmm_good.is_fitted
        good_likelihood = hmm_good.score(good_returns)

        # Lower quality data (more volatile, some outliers)
        noisy_returns = np.concatenate(
            [
                np.random.normal(0, 0.02, 180),  # Most data is normal
                np.random.normal(0, 0.1, 20),  # Some very volatile periods
            ]
        )

        hmm_noisy = HiddenMarkovModel(n_states=3, config=HMMConfig(random_seed=42))
        hmm_noisy.fit(noisy_returns, verbose=False)

        assert hmm_noisy.is_fitted
        noisy_likelihood = hmm_noisy.score(noisy_returns)

        # Both should work, but good data should generally have better likelihood per observation
        # (Note: direct comparison is tricky due to different data, this just ensures both work)
        assert np.isfinite(good_likelihood)
        assert np.isfinite(noisy_likelihood)

    def test_hmm_with_preprocessor_features(self, sample_market_data):
        """Test HMM using additional features from preprocessor."""
        # Prepare data with preprocessing features
        raw_data = sample_market_data.copy()
        # Use the existing 'Close' column or the 'price' column if 'Close' doesn't exist
        if "Close" not in raw_data.columns:
            raw_data = raw_data.rename(columns={"price": "Close"})

        # Calculate log returns using the Close column
        close_series = raw_data["Close"]
        raw_data["log_return"] = np.log(close_series / close_series.shift(1))
        raw_data = raw_data.dropna().reset_index(drop=True)

        preprocessing_config = PreprocessingConfig(
            calculate_volatility=True, volatility_window=10
        )

        preprocessor = DataPreprocessor(preprocessing_config)
        processed_data = preprocessor.process_data(raw_data)

        # Verify additional features were created
        assert "volatility" in processed_data.columns
        assert "abs_return" in processed_data.columns

        # Train HMM on basic returns
        hmm = HiddenMarkovModel(n_states=3, config=HMMConfig(random_seed=42))
        hmm.fit(processed_data["log_return"], verbose=False)

        # Should work normally
        assert hmm.is_fitted
        states = hmm.predict(processed_data["log_return"])
        assert len(states) == len(processed_data)

    def test_hmm_model_persistence_with_pipeline_data(self, sample_market_data):
        """Test saving and loading HMM trained on pipeline data."""
        # Prepare data through pipeline
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()

        # Train model
        original_hmm = HiddenMarkovModel(n_states=3, config=HMMConfig(random_seed=42))
        original_hmm.fit(returns, verbose=False)

        # Get some predictions for comparison
        original_states = original_hmm.predict(returns)
        original_likelihood = original_hmm.score(returns)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            original_hmm.save(filepath)
            loaded_hmm = HiddenMarkovModel.load(filepath)

            # Compare predictions
            loaded_states = loaded_hmm.predict(returns)
            loaded_likelihood = loaded_hmm.score(returns)

            assert np.array_equal(original_states, loaded_states)
            assert np.isclose(original_likelihood, loaded_likelihood)

        finally:
            if filepath.exists():
                filepath.unlink()

    def test_hmm_config_validation_with_pipeline_data(self, sample_market_data):
        """Test HMM configuration validation with different data sizes."""
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()

        # Test with appropriate configuration
        good_config = HMMConfig(n_states=3, max_iterations=50)
        good_config.validate_for_data(len(returns))  # Should not raise

        hmm = HiddenMarkovModel(config=good_config)
        hmm.fit(returns, verbose=False)
        assert hmm.is_fitted

        # Test with too many states for data size
        bad_config = HMMConfig(n_states=6)  # Too many states for ~100 observations

        with pytest.warns(UserWarning, match="Limited data"):
            bad_config.validate_for_data(len(returns))

        # Should still work but might not converge well
        hmm_bad = HiddenMarkovModel(config=bad_config)
        hmm_bad.fit(returns, max_iterations=20, verbose=False)  # Reduce iterations
        # Don't assert convergence as it might not converge with too many states

    def test_real_time_regime_updates(self, sample_market_data):
        """Test real-time regime detection with streaming data simulation."""
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()

        # Train on first 80% of data
        train_size = int(len(returns) * 0.8)
        train_returns = returns[:train_size]
        test_returns = returns[train_size:]

        # Train HMM
        hmm = HiddenMarkovModel(n_states=3, config=HMMConfig(random_seed=42))
        hmm.fit(train_returns, verbose=False)

        # Simulate real-time updates
        regime_history = []

        for new_return in test_returns:
            regime_info = hmm.update_with_observation(new_return)
            regime_history.append(
                {
                    "return": new_return,
                    "regime": regime_info["most_likely_regime"],
                    "confidence": regime_info["confidence"],
                    "probabilities": regime_info["regime_probabilities"].copy(),
                }
            )

        # Verify results
        assert len(regime_history) == len(test_returns)

        # Check that regime changes are reasonable (not too frequent)
        regime_changes = sum(
            1
            for i in range(1, len(regime_history))
            if regime_history[i]["regime"] != regime_history[i - 1]["regime"]
        )

        # Should have some but not excessive regime changes
        assert regime_changes < len(test_returns) * 0.5  # Less than 50% of periods

        # Check that all confidences are valid
        for entry in regime_history:
            assert 0 <= entry["confidence"] <= 1
            assert len(entry["probabilities"]) == 3
            assert np.allclose(sum(entry["probabilities"]), 1.0)

    def test_hmm_with_validation_edge_cases(self):
        """Test HMM behavior with edge case data from validation."""
        # Test with data that has outliers (should be handled by preprocessor)
        outlier_data = np.concatenate(
            [
                np.random.normal(0, 0.02, 90),
                [0.15, -0.12, 0.08],  # Outliers
                np.random.normal(0, 0.02, 7),
            ]
        )

        # Use preprocessor to handle outliers
        preprocessing_config = PreprocessingConfig(return_method="log")
        validation_config = ValidationConfig(
            outlier_method="iqr", outlier_threshold=2.0
        )

        preprocessor = DataPreprocessor(preprocessing_config, validation_config)

        # Create minimal DataFrame for preprocessor
        test_df = pd.DataFrame(
            {
                "price": np.exp(np.cumsum(outlier_data)),  # Convert returns to prices
                "log_return": outlier_data,
            }
        )

        processed_data = preprocessor.process_data(test_df)

        # Train HMM on processed data
        hmm = HiddenMarkovModel(
            n_states=3, config=HMMConfig(random_seed=42, max_iterations=30)
        )
        hmm.fit(processed_data["log_return"], verbose=False)

        assert hmm.is_fitted

        # Should handle the data without issues
        states = hmm.predict(processed_data["log_return"])
        assert len(states) == len(processed_data)

    def test_multiple_hmm_configurations(self, sample_market_data):
        """Test HMM with different configurations on same data."""
        returns = np.log(
            sample_market_data["price"] / sample_market_data["price"].shift(1)
        ).dropna()

        configurations = [
            HMMConfig.for_market_data(),
            HMMConfig.for_market_data(conservative=True),
            HMMConfig(n_states=2, initialization_method="kmeans", random_seed=42),
            HMMConfig(n_states=4, max_iterations=30, random_seed=42),
        ]

        results = []

        for i, config in enumerate(configurations):
            try:
                hmm = HiddenMarkovModel(config=config)
                hmm.fit(returns, verbose=False)

                likelihood = hmm.score(returns)
                states = hmm.predict(returns)

                results.append(
                    {
                        "config_idx": i,
                        "n_states": config.n_states,
                        "likelihood": likelihood,
                        "n_unique_states": len(np.unique(states)),
                        "converged": hmm.training_history_["converged"],
                    }
                )

            except Exception as e:
                results.append({"config_idx": i, "error": str(e)})

        # At least some configurations should work
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) >= 2

        # All successful results should have reasonable likelihoods
        for result in successful_results:
            assert np.isfinite(result["likelihood"])
            # Note: Log-likelihood can be positive for small datasets with good model fit
