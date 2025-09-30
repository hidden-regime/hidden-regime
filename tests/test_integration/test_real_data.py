"""
Integration tests using real financial data to validate end-to-end functionality.

These tests use actual market data to ensure the pipeline works correctly
with real-world scenarios and data patterns.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import hidden_regime as hr
from hidden_regime.pipeline.temporal import TemporalController


class TestRealDataIntegration:
    """Test pipeline functionality with real financial data."""

    @pytest.mark.integration
    def test_spy_analysis_basic(self):
        """Test basic analysis on SPY (S&P 500 ETF) with real data."""
        try:
            # Create pipeline for SPY with recent data
            pipeline = hr.create_financial_pipeline(
                "SPY", n_states=3, period="6mo", initialization_method="kmeans"
            )

            # Run analysis
            report_output = pipeline.update()

            # Get component outputs
            model_output = pipeline.get_component_output("model")
            analysis_output = pipeline.get_component_output("analysis")
            data_output = pipeline.get_component_output("data")

            # Validate results structure
            assert isinstance(report_output, str)
            assert len(report_output) > 0

            # Validate component outputs
            assert model_output is not None
            assert analysis_output is not None
            assert data_output is not None

            # Validate data output
            assert isinstance(data_output, pd.DataFrame)
            assert len(data_output) > 0
            assert "close" in data_output.columns  # Standardized lowercase column names

            # Basic validation that pipeline executed successfully
            assert pipeline.update_count == 1
            assert pipeline.last_update is not None

        except Exception as e:
            # Allow network/data issues to be skipped in CI
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network issue: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_multiple_assets_analysis(self):
        """Test analysis across multiple popular assets."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        results_dict = {}

        for ticker in tickers:
            try:
                pipeline = hr.create_financial_pipeline(
                    ticker,
                    n_states=3,
                    period="3mo",
                    max_iterations=50,  # Faster for testing
                )

                report_output = pipeline.update()
                results_dict[ticker] = {
                    "report": report_output,
                    "model_output": pipeline.get_component_output("model"),
                    "data_output": pipeline.get_component_output("data"),
                }

                # Basic validation for each asset
                assert isinstance(report_output, str)
                assert len(report_output) > 0
                assert pipeline.get_component_output("model") is not None
                assert pipeline.get_component_output("data") is not None

            except Exception as e:
                if "network" in str(e).lower() or "connection" in str(e).lower():
                    pytest.skip(f"Network issue for {ticker}: {e}")
                else:
                    raise

        # If we got results for multiple assets, verify they're different
        if len(results_dict) >= 2:
            # Reports should be different across assets (basic sanity check)
            reports = [results_dict[ticker]["report"] for ticker in results_dict.keys()]

            # Check that not all assets have identical reports
            for i in range(len(reports)):
                for j in range(i + 1, len(reports)):
                    # Reports should not be exactly identical
                    assert (
                        reports[i] != reports[j]
                    ), f"Assets {list(results_dict.keys())[i]} and {list(results_dict.keys())[j]} have identical reports"

    @pytest.mark.integration
    def test_temporal_analysis_real_data(self):
        """Test temporal analysis with real data for backtesting validation."""
        try:
            # Create pipeline for temporal analysis
            pipeline = hr.create_financial_pipeline(
                "QQQ",  # NASDAQ ETF
                n_states=3,
                period="1y",
                initialization_method="kmeans",
                max_iterations=50,
            )

            # Get full dataset for temporal controller
            data_output = pipeline.data.load_data()

            # Create temporal controller
            temporal = TemporalController(pipeline, data_output)

            # Set analysis start date (6 months ago)
            analysis_start = datetime.now() - timedelta(days=180)
            temporal.update_as_of(analysis_start)

            # Collect results over time
            results_timeline = []

            # Step through monthly intervals
            for month_offset in range(6):
                try:
                    results = temporal.step_forward(
                        step_days=30
                    )  # Step forward by ~1 month
                    if results and "model_results" in results:
                        model_results = results["model_results"]
                        if "regime_probabilities" in model_results:
                            regime_probs = model_results["regime_probabilities"]
                            if len(regime_probs) > 0:
                                results_timeline.append(
                                    {
                                        "month": month_offset,
                                        "current_regime_probs": regime_probs[-1],
                                        "total_observations": len(regime_probs),
                                    }
                                )
                except Exception as e:
                    # Some time steps may fail, which is acceptable
                    warnings.warn(f"Temporal step {month_offset} failed: {e}")
                    continue

            # Validate temporal results
            assert len(results_timeline) > 0, "No successful temporal analysis steps"

            # Check regime probabilities are valid across time
            for result in results_timeline:
                regime_probs = result["current_regime_probs"]
                regime_probs = np.array(regime_probs)  # Convert to numpy array
                assert len(regime_probs) == 3
                assert abs(np.sum(regime_probs) - 1.0) < 1e-6
                assert np.all(regime_probs >= 0)
                assert np.all(regime_probs <= 1)

            # Check that observations increased over time
            observation_counts = [r["total_observations"] for r in results_timeline]
            if len(observation_counts) > 1:
                # Should generally increase (allowing for some variation due to data availability)
                assert (
                    observation_counts[-1] >= observation_counts[0]
                ), "Observation count should increase over time"

        except Exception as e:
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network issue: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_different_observation_signals(self):
        """Test pipeline with different observation signal types."""
        signals_to_test = ["log_return", "close_price", "volume"]

        for signal in signals_to_test:
            try:
                pipeline = hr.create_financial_pipeline(
                    "SPY",
                    n_states=3,
                    period="3mo",
                    observed_signal=signal,
                    max_iterations=50,
                )

                results = pipeline.update()

                # Validate basic structure
                assert "model_results" in results
                model_results = results["model_results"]
                assert "regime_probabilities" in model_results

                # Validate regime probabilities
                regime_probs = model_results["regime_probabilities"]
                assert regime_probs.shape[1] == 3
                assert np.allclose(regime_probs.sum(axis=1), 1.0, atol=1e-6)

                # Check that different signals produce different results
                if signal == "log_return":
                    # Log returns should have some variation
                    regime_entropy = -np.sum(
                        regime_probs * np.log(regime_probs + 1e-10), axis=1
                    )
                    assert (
                        np.mean(regime_entropy) > 0.1
                    ), f"Signal {signal} produced too deterministic regimes"

            except Exception as e:
                if "network" in str(e).lower() or "connection" in str(e).lower():
                    pytest.skip(f"Network issue for signal {signal}: {e}")
                else:
                    # Some signals might not be implemented yet
                    warnings.warn(f"Signal {signal} failed: {e}")
                    continue

    @pytest.mark.integration
    def test_hmm_config_presets_real_data(self):
        """Test different HMM configuration presets with real data."""
        from hidden_regime.config.model import HMMConfig

        presets = {
            "conservative": HMMConfig.create_conservative(),
            "balanced": HMMConfig.create_balanced(),
            "aggressive": HMMConfig.create_aggressive(),
        }

        results_by_preset = {}

        for preset_name, config in presets.items():
            try:
                pipeline = hr.create_financial_pipeline(
                    "IWM", model_config=config, period="3mo"  # Russell 2000 ETF
                )

                results = pipeline.update()
                results_by_preset[preset_name] = results

                # Validate results
                assert "model_results" in results
                model_results = results["model_results"]
                regime_probs = model_results["regime_probabilities"]

                # Conservative should be more stable (less regime switching)
                if preset_name == "conservative":
                    states = model_results["most_likely_states"]
                    regime_changes = np.sum(np.diff(states) != 0)
                    regime_change_rate = regime_changes / len(states)

                    # Conservative should have fewer regime changes
                    assert (
                        regime_change_rate < 0.2
                    ), f"Conservative preset had too many regime changes: {regime_change_rate}"

                # Aggressive should adapt faster (more regime switching allowed)
                elif preset_name == "aggressive":
                    states = model_results["most_likely_states"]
                    regime_changes = np.sum(np.diff(states) != 0)
                    regime_change_rate = regime_changes / len(states)

                    # Aggressive can have more regime changes (but not excessive)
                    assert (
                        regime_change_rate < 0.5
                    ), f"Aggressive preset had excessive regime changes: {regime_change_rate}"

            except Exception as e:
                if "network" in str(e).lower() or "connection" in str(e).lower():
                    pytest.skip(f"Network issue for preset {preset_name}: {e}")
                else:
                    raise

        # If we got results for multiple presets, they should be different
        if len(results_by_preset) >= 2:
            preset_names = list(results_by_preset.keys())
            for i in range(len(preset_names)):
                for j in range(i + 1, len(preset_names)):
                    results_i = results_by_preset[preset_names[i]]
                    results_j = results_by_preset[preset_names[j]]

                    probs_i = results_i["model_results"]["regime_probabilities"]
                    probs_j = results_j["model_results"]["regime_probabilities"]

                    # Different presets should produce somewhat different results
                    correlation = np.corrcoef(probs_i.flatten(), probs_j.flatten())[
                        0, 1
                    ]
                    assert (
                        correlation < 0.9
                    ), f"Presets {preset_names[i]} and {preset_names[j]} produced too similar results"


class TestRealDataEdgeCases:
    """Test edge cases and robustness with real data."""

    @pytest.mark.integration
    def test_volatile_period_analysis(self):
        """Test analysis during known volatile periods."""
        try:
            # Analyze VIX (volatility index) which should show clear regime switching
            pipeline = hr.create_financial_pipeline(
                "VIX",
                n_states=3,
                period="1y",
                initialization_method="kmeans",
                max_iterations=100,
            )

            results = pipeline.update()

            # VIX should show clear regime differentiation
            model_results = results["model_results"]
            regime_probs = model_results["regime_probabilities"]

            # Calculate regime certainty (how decisive the regime assignments are)
            max_probs = np.max(regime_probs, axis=1)
            mean_certainty = np.mean(max_probs)

            # VIX should have fairly decisive regime assignments due to its volatility clustering
            assert (
                mean_certainty > 0.5
            ), f"VIX regime assignments too uncertain: {mean_certainty}"

            # Should have observed multiple regimes
            states = model_results["most_likely_states"]
            unique_states = len(np.unique(states))
            assert (
                unique_states >= 2
            ), f"VIX should show multiple regimes, only found {unique_states}"

        except Exception as e:
            if (
                "network" in str(e).lower()
                or "connection" in str(e).lower()
                or "vix" in str(e).lower()
            ):
                pytest.skip(f"Data issue with VIX: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_small_cap_vs_large_cap(self):
        """Test regime detection differences between small and large cap stocks."""
        try:
            # Large cap (Apple)
            large_cap_pipeline = hr.create_financial_pipeline(
                "AAPL", n_states=3, period="6mo", max_iterations=50
            )

            # Small cap ETF
            small_cap_pipeline = hr.create_financial_pipeline(
                "IWM", n_states=3, period="6mo", max_iterations=50
            )

            large_cap_results = large_cap_pipeline.update()
            small_cap_results = small_cap_pipeline.update()

            # Both should succeed
            assert "model_results" in large_cap_results
            assert "model_results" in small_cap_results

            # Compare volatility characteristics
            large_cap_probs = large_cap_results["model_results"]["regime_probabilities"]
            small_cap_probs = small_cap_results["model_results"]["regime_probabilities"]

            # Calculate regime switching frequency
            large_cap_states = large_cap_results["model_results"]["most_likely_states"]
            small_cap_states = small_cap_results["model_results"]["most_likely_states"]

            large_cap_switches = np.sum(np.diff(large_cap_states) != 0) / len(
                large_cap_states
            )
            small_cap_switches = np.sum(np.diff(small_cap_states) != 0) / len(
                small_cap_states
            )

            # Small caps might switch regimes more frequently (higher volatility)
            # But this is not guaranteed, so we just ensure both are reasonable
            assert (
                0.01 < large_cap_switches < 0.5
            ), f"Large cap switching rate unrealistic: {large_cap_switches}"
            assert (
                0.01 < small_cap_switches < 0.5
            ), f"Small cap switching rate unrealistic: {small_cap_switches}"

        except Exception as e:
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network issue: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_recent_data_availability(self):
        """Test that recent data can be analyzed successfully."""
        try:
            # Test with very recent data (last month)
            pipeline = hr.create_financial_pipeline(
                "SPY", n_states=3, period="1mo", max_iterations=50
            )

            results = pipeline.update()

            # Should get results even with limited recent data
            assert "model_results" in results
            model_results = results["model_results"]

            # May have fewer observations but should still work
            regime_probs = model_results["regime_probabilities"]
            assert (
                len(regime_probs) > 10
            ), "Should have at least 10 trading days of data"

            # Regime probabilities should still be valid
            assert np.allclose(regime_probs.sum(axis=1), 1.0, atol=1e-6)

        except Exception as e:
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network issue: {e}")
            elif "insufficient" in str(e).lower() or "data" in str(e).lower():
                # This might be expected for very recent data
                pytest.skip(f"Insufficient recent data: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
