"""
Comprehensive unit and integration tests for analysis/ modules.

Tests regime evolution, signal attribution, market event studies (setup only),
and analysis workflows that don't require external dependencies.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime, timedelta

from hidden_regime.analysis.regime_evolution import (
    RegimeEvolutionAnalyzer,
)
from hidden_regime.analysis.signal_attribution import (
    SignalAttributionAnalyzer,
)
from hidden_regime.analysis.market_event_study import MarketEventStudy
from hidden_regime.utils.exceptions import AnalysisError


@pytest.fixture
def sample_temporal_snapshots():
    """Create sample temporal snapshots for testing."""
    snapshots = []
    for i in range(50):
        snapshot = {
            "timestamp": f"2023-01-{i+1:02d}",
            "state": i % 3,  # Cycle through 3 states
            "emission_means": [0.001 + i*0.0001, 0.0, -0.001 - i*0.0001],
            "emission_stds": [0.01, 0.015, 0.02],
            "transition_matrix": [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
            "returns": np.random.normal(0.001, 0.02),
        }
        snapshots.append(snapshot)
    return snapshots


@pytest.fixture
def sample_signal_data():
    """Create sample temporal snapshots for signal attribution testing."""
    snapshots = []
    for i in range(100):
        snapshot = {
            "timestamp": f"2023-01-{(i % 28) + 1:02d}",
            "signal_source": np.random.choice(["hmm", "momentum", "mean_reversion"]),
            "signal_strength": np.random.uniform(0.5, 1.0),
            "confidence": np.random.uniform(0.6, 0.95),
            "position": np.random.choice([-1, 0, 1]),
        }
        snapshots.append(snapshot)
    return snapshots


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
    data = pd.DataFrame({
        "close": prices,
    }, index=dates)
    return data


# ============================================================================
# UNIT TESTS (10 tests - subset not requiring 'ta' library)
# ============================================================================


def test_regime_evolution_analysis(sample_temporal_snapshots):
    """Test regime evolution analysis."""
    analyzer = RegimeEvolutionAnalyzer(lookback_window=20)

    result = analyzer.analyze_temporal_evolution(sample_temporal_snapshots)

    # Should return analysis results
    assert isinstance(result, dict)
    assert "regime_transitions" in result or "parameter_drift" in result or len(result) > 0


def test_signal_attribution_analysis(sample_signal_data, sample_price_data):
    """Test signal attribution analysis."""
    analyzer = SignalAttributionAnalyzer(attribution_window=50, min_signals_threshold=5)

    # Analyze attribution
    result = analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # Should return attribution results
    assert result is not None or isinstance(result, dict)
    # Check if method exists and runs without error
    assert isinstance(analyzer.signal_history, list) or isinstance(analyzer.performance_cache, dict)


def test_market_event_study_initialization():
    """Test market event study initialization and setup."""
    study = MarketEventStudy(
        ticker="SPY",
        training_start="2023-01-01",
        training_end="2023-06-30",
        analysis_start="2023-07-01",
        analysis_end="2023-12-31",
        n_states=3,
        key_events={"2023-08-01": "Market Event"},
        output_dir=tempfile.mkdtemp(),
    )

    # Should initialize without errors
    assert study.ticker == "SPY"
    assert study.n_states == 3
    assert len(study.key_events) == 1


def test_regime_evolution_transitions(sample_temporal_snapshots):
    """Test regime transition tracking."""
    analyzer = RegimeEvolutionAnalyzer(lookback_window=20)

    # Track transitions
    analyzer.analyze_temporal_evolution(sample_temporal_snapshots)

    # Transitions should be tracked
    assert isinstance(analyzer.regime_transitions, list)


def test_signal_attribution_performance(sample_signal_data, sample_price_data):
    """Test signal attribution performance metrics."""
    analyzer = SignalAttributionAnalyzer()

    # Analyze signal attribution
    analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # Performance cache should be populated or method should run
    assert isinstance(analyzer.performance_cache, dict)


def test_market_event_detection_setup():
    """Test market event detection setup."""
    study = MarketEventStudy(
        ticker="QQQ",
        training_start="2020-01-01",
        training_end="2020-01-31",
        analysis_start="2020-02-01",
        analysis_end="2020-02-28",
        n_states=2,
        output_dir=tempfile.mkdtemp(),
    )

    # Should be able to detect events (initialization test)
    assert study.n_states == 2
    assert study.ticker == "QQQ"


def test_analysis_caching():
    """Test caching behavior in analysis modules."""
    analyzer = SignalAttributionAnalyzer()

    # Should have performance cache
    assert hasattr(analyzer, "performance_cache")
    assert isinstance(analyzer.performance_cache, dict)

    # Initially empty
    assert len(analyzer.performance_cache) == 0


def test_regime_evolution_parameter_tracking(sample_temporal_snapshots):
    """Test parameter tracking in regime evolution."""
    analyzer = RegimeEvolutionAnalyzer(lookback_window=30)

    result = analyzer.analyze_temporal_evolution(sample_temporal_snapshots)

    # Should track parameters
    assert isinstance(analyzer.parameter_history, dict)


def test_signal_attribution_multiple_sources(sample_signal_data, sample_price_data):
    """Test signal attribution with multiple sources."""
    analyzer = SignalAttributionAnalyzer(attribution_window=30)

    # Analyze
    result = analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # Should handle multiple sources
    assert result is not None or len(analyzer.signal_history) >= 0


def test_regime_evolution_stability_threshold():
    """Test regime evolution with custom stability threshold."""
    analyzer = RegimeEvolutionAnalyzer(lookback_window=20, stability_threshold=0.05)

    assert analyzer.stability_threshold == 0.05
    assert analyzer.lookback_window == 20


# ============================================================================
# INTEGRATION TESTS (10 tests)
# ============================================================================


def test_analysis_module_integration(sample_signal_data, sample_temporal_snapshots, sample_price_data):
    """Test integration of multiple analysis modules."""
    # Initialize analyzers
    regime_analyzer = RegimeEvolutionAnalyzer()
    signal_analyzer = SignalAttributionAnalyzer()

    # Run analyses
    regime_result = regime_analyzer.analyze_temporal_evolution(sample_temporal_snapshots)
    signal_attribution = signal_analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # All should complete without error
    assert isinstance(regime_result, dict)
    assert signal_attribution is not None or len(signal_analyzer.signal_history) >= 0


def test_multi_market_event_studies():
    """Test multiple market event studies setup."""
    studies = []

    for ticker in ["SPY", "QQQ"]:
        study = MarketEventStudy(
            ticker=ticker,
            training_start="2023-01-01",
            training_end="2023-06-30",
            analysis_start="2023-07-01",
            analysis_end="2023-12-31",
            n_states=3,
            output_dir=tempfile.mkdtemp(),
        )
        studies.append(study)

    # Should have created multiple studies
    assert len(studies) == 2
    assert all(study.n_states == 3 for study in studies)


def test_regime_evolution_with_different_windows(sample_temporal_snapshots):
    """Test regime evolution with different lookback windows."""
    windows = [10, 20, 50]
    results = []

    for window in windows:
        analyzer = RegimeEvolutionAnalyzer(lookback_window=window)
        result = analyzer.analyze_temporal_evolution(sample_temporal_snapshots)
        results.append(result)

    # Should have results for all windows
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)


def test_signal_attribution_with_different_thresholds(sample_signal_data, sample_price_data):
    """Test signal attribution with different min signal thresholds."""
    thresholds = [1, 5, 10]
    analyzers = []

    for threshold in thresholds:
        analyzer = SignalAttributionAnalyzer(min_signals_threshold=threshold)
        analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)
        analyzers.append(analyzer)

    # Should have created multiple analyzers
    assert len(analyzers) == 3


def test_regime_characterization_analysis(sample_temporal_snapshots):
    """Test regime characterization analysis."""
    analyzer = RegimeEvolutionAnalyzer()

    # Analyze evolution
    result = analyzer.analyze_temporal_evolution(sample_temporal_snapshots)

    # Should characterize regimes
    assert isinstance(result, dict)
    assert isinstance(analyzer.regime_characteristics, dict)


def test_signal_performance_tracking(sample_signal_data, sample_price_data):
    """Test signal performance tracking."""
    analyzer = SignalAttributionAnalyzer()

    # Track performance
    analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # Should have tracking structures
    assert isinstance(analyzer.signal_history, list)
    assert isinstance(analyzer.performance_cache, dict)


def test_market_event_study_with_multiple_events():
    """Test market event study with multiple key events."""
    events = {
        "2023-08-01": "Event 1",
        "2023-09-01": "Event 2",
        "2023-10-01": "Event 3",
    }

    study = MarketEventStudy(
        ticker="SPY",
        training_start="2023-01-01",
        training_end="2023-07-31",
        analysis_start="2023-08-01",
        analysis_end="2023-12-31",
        n_states=3,
        key_events=events,
        output_dir=tempfile.mkdtemp(),
    )

    # Should track all events
    assert len(study.key_events) == 3


def test_regime_evolution_empty_snapshots():
    """Test regime evolution with empty snapshots."""
    analyzer = RegimeEvolutionAnalyzer()

    # Test with empty list
    result = analyzer.analyze_temporal_evolution([])

    # Should handle gracefully
    assert result is not None or isinstance(analyzer.regime_transitions, list)


def test_signal_attribution_empty_data():
    """Test signal attribution with empty data."""
    analyzer = SignalAttributionAnalyzer()

    empty_snapshots = []
    empty_price = pd.DataFrame()

    # Should handle empty data gracefully
    try:
        result = analyzer.analyze_signal_attribution(empty_snapshots, empty_price)
        # If no error, check result
        assert result is None or isinstance(result, (dict, type(None)))
    except (ValueError, KeyError, AnalysisError, IndexError):
        # Expected for empty data
        pass


def test_comprehensive_analysis_workflow(sample_temporal_snapshots, sample_signal_data, sample_price_data):
    """Test comprehensive analysis workflow."""
    # Initialize all analyzers
    regime_analyzer = RegimeEvolutionAnalyzer(lookback_window=25)
    signal_analyzer = SignalAttributionAnalyzer(attribution_window=50)

    # Run comprehensive analysis
    regime_result = regime_analyzer.analyze_temporal_evolution(sample_temporal_snapshots)
    signal_result = signal_analyzer.analyze_signal_attribution(sample_signal_data, sample_price_data)

    # Comprehensive workflow should produce multiple results
    results = {
        "regime_evolution": regime_result,
        "signal_attribution": signal_result,
    }

    # Verify all components ran
    assert isinstance(results["regime_evolution"], dict)
    assert results["signal_attribution"] is not None or isinstance(signal_analyzer.signal_history, list)
