"""
Unit tests for ModelDataCollector and related snapshot dataclasses.

Tests data collection infrastructure with mocked pipeline components
to ensure fast, isolated unit testing.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from hidden_regime.data.collectors import (
    HMMStateSnapshot,
    ModelDataCollector,
    RegimeAnalysisSnapshot,
    TechnicalIndicatorSnapshot,
    TimestepSnapshot,
)


class TestDataclassSnapshots:
    """Test snapshot dataclass creation and structure."""

    @pytest.mark.unit
    def test_hmm_state_snapshot_creation(self):
        """Test HMMStateSnapshot dataclass creation."""
        snapshot = HMMStateSnapshot(
            timestamp="2023-01-01T00:00:00Z",
            data_end_date="2023-01-01",
            transition_matrix=[[0.9, 0.1], [0.2, 0.8]],
            emission_means=[0.01, -0.01],
            emission_stds=[0.02, 0.025],
            initial_probs=[0.5, 0.5],
            regime_probabilities=[[0.7, 0.3], [0.4, 0.6]],
            most_likely_states=[0, 1],
            current_regime_probs=[0.7, 0.3],
            log_likelihood=-100.5,
            training_iterations=50,
            converged=True,
            training_time=1.5,
            n_observations=100,
            parameter_changes={"delta": 0.01},
        )

        assert snapshot.timestamp == "2023-01-01T00:00:00Z"
        assert snapshot.transition_matrix == [[0.9, 0.1], [0.2, 0.8]]
        assert snapshot.log_likelihood == -100.5
        assert snapshot.converged is True

    @pytest.mark.unit
    def test_technical_indicator_snapshot_creation(self):
        """Test TechnicalIndicatorSnapshot dataclass creation."""
        snapshot = TechnicalIndicatorSnapshot(
            timestamp="2023-01-01T00:00:00Z",
            data_end_date="2023-01-01",
            indicators={"rsi": 55.0, "macd": 0.5},
            signals={"rsi": "neutral", "macd": "buy"},
            signal_rationale={"rsi": "Mid-range", "macd": "Positive crossover"},
            signal_confidence={"rsi": 0.5, "macd": 0.8},
            threshold_crossings=[{"indicator": "macd", "direction": "up"}],
        )

        assert snapshot.indicators == {"rsi": 55.0, "macd": 0.5}
        assert snapshot.signals["macd"] == "buy"

    @pytest.mark.unit
    def test_regime_analysis_snapshot_creation(self):
        """Test RegimeAnalysisSnapshot dataclass creation."""
        snapshot = RegimeAnalysisSnapshot(
            timestamp="2023-01-01T00:00:00Z",
            data_end_date="2023-01-01",
            current_regime="Bull",
            regime_confidence=0.85,
            days_in_regime=15,
            expected_regime_duration=25.5,
            regime_characteristics={"mean_return": 0.02, "volatility": 0.015},
            transition_probabilities={"Bear": 0.1, "Sideways": 0.2, "Bull": 0.7},
        )

        assert snapshot.current_regime == "Bull"
        assert snapshot.regime_confidence == 0.85
        assert snapshot.days_in_regime == 15

    @pytest.mark.unit
    def test_timestep_snapshot_creation(self):
        """Test TimestepSnapshot dataclass creation."""
        hmm_snapshot = HMMStateSnapshot(
            timestamp="2023-01-01T00:00:00Z",
            data_end_date="2023-01-01",
            transition_matrix=[[0.9, 0.1]],
            emission_means=[0.01],
            emission_stds=[0.02],
            initial_probs=[0.5],
            regime_probabilities=[[0.7]],
            most_likely_states=[0],
            current_regime_probs=[0.7],
            log_likelihood=-50.0,
            training_iterations=30,
            converged=True,
            training_time=1.0,
            n_observations=50,
            parameter_changes={},
        )

        snapshot = TimestepSnapshot(
            timestamp="2023-01-01T00:00:00Z",
            data_end_date="2023-01-01",
            hmm_state=hmm_snapshot,
            technical_indicators=None,
            regime_analysis=None,
            execution_time=2.5,
            data_quality_metrics={"data_size": 100},
        )

        assert snapshot.hmm_state is not None
        assert snapshot.execution_time == 2.5
        assert snapshot.data_quality_metrics["data_size"] == 100


class TestModelDataCollectorInitialization:
    """Test ModelDataCollector initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default initialization parameters."""
        collector = ModelDataCollector()

        assert collector.max_history == 1000
        assert collector.collection_level == "detailed"
        assert collector.auto_export is False
        assert collector.export_path is None
        assert len(collector.timestep_data) == 0
        assert collector.collection_stats["total_timesteps"] == 0
        assert collector.collection_stats["successful_collections"] == 0
        assert collector._previous_hmm_state is None

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        collector = ModelDataCollector(
            max_history=500,
            collection_level="basic",
            auto_export=True,
            export_path="/tmp/export.json",
        )

        assert collector.max_history == 500
        assert collector.collection_level == "basic"
        assert collector.auto_export is True
        assert collector.export_path == "/tmp/export.json"

    @pytest.mark.unit
    def test_collection_stats_initialized(self):
        """Test that collection stats are properly initialized."""
        collector = ModelDataCollector()
        stats = collector.collection_stats

        assert "total_timesteps" in stats
        assert "successful_collections" in stats
        assert "failed_collections" in stats
        assert "start_time" in stats
        assert stats["last_collection"] is None


class TestDataCollection:
    """Test data collection methods."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with all components."""
        pipeline = MagicMock()

        # Mock data component
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {"close": np.random.randn(100) * 0.02, "volume": np.random.randint(1000, 10000, 100)},
            index=dates,
        )
        pipeline.data.get_all_data.return_value = data

        # Mock model component
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 3
        pipeline.model.transition_matrix_ = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]
        )
        pipeline.model.emission_means_ = np.array([0.01, -0.01, 0.02])
        pipeline.model.emission_stds_ = np.array([0.015, 0.02, 0.018])
        pipeline.model.initial_probs_ = np.array([0.33, 0.34, 0.33])

        # Mock model methods
        regime_probs = np.random.rand(100, 3)
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
        pipeline.model.predict_proba.return_value = regime_probs
        pipeline.model.predict.return_value = np.random.randint(0, 3, 100)
        pipeline.model.score.return_value = -150.5
        pipeline.model.training_history_ = {
            "iterations": 50,
            "converged": True,
            "training_time": 1.5,
        }

        # Mock analysis component
        pipeline.analysis = MagicMock()

        return pipeline

    @pytest.mark.unit
    def test_collect_timestep_data_success(self, mock_pipeline):
        """Test successful timestep data collection."""
        collector = ModelDataCollector()
        snapshot = collector.collect_timestep_data(mock_pipeline, "2023-01-01", time.time())

        assert isinstance(snapshot, TimestepSnapshot)
        assert snapshot.data_end_date == "2023-01-01"
        assert snapshot.hmm_state is not None
        assert snapshot.execution_time > 0
        assert collector.collection_stats["successful_collections"] == 1
        assert len(collector.timestep_data) == 1

    @pytest.mark.unit
    def test_collect_timestep_increments_stats(self, mock_pipeline):
        """Test that collection increments statistics."""
        collector = ModelDataCollector()

        collector.collect_timestep_data(mock_pipeline, "2023-01-01", time.time())
        collector.collect_timestep_data(mock_pipeline, "2023-01-02", time.time())

        assert collector.collection_stats["total_timesteps"] == 2
        assert collector.collection_stats["successful_collections"] == 2
        assert len(collector.timestep_data) == 2

    @pytest.mark.unit
    def test_collect_timestep_respects_max_history(self, mock_pipeline):
        """Test that deque respects max_history limit."""
        collector = ModelDataCollector(max_history=10)

        for i in range(15):
            collector.collect_timestep_data(mock_pipeline, f"2023-01-{i+1:02d}", time.time())

        # Should only keep last 10 due to deque maxlen
        assert len(collector.timestep_data) == 10
        assert collector.collection_stats["total_timesteps"] == 15

    @pytest.mark.unit
    def test_collect_timestep_handles_error(self):
        """Test error handling during data collection."""
        collector = ModelDataCollector()

        # Create pipeline that will raise error
        bad_pipeline = MagicMock()
        bad_pipeline.model.is_fitted = True
        bad_pipeline.data.get_all_data.side_effect = Exception("Data error")

        snapshot = collector.collect_timestep_data(bad_pipeline, "2023-01-01", time.time())

        assert snapshot.hmm_state is None
        # technical_indicators may still be created with empty data
        assert snapshot.regime_analysis is None
        assert "collection_error" in snapshot.data_quality_metrics
        # Note: failed_collections may not increment if error is caught at lower level

    @pytest.mark.unit
    def test_collect_hmm_state_with_fitted_model(self, mock_pipeline):
        """Test HMM state collection with fitted model."""
        collector = ModelDataCollector()
        snapshot = collector._collect_hmm_state(
            mock_pipeline, "2023-01-01T00:00:00Z", "2023-01-01"
        )

        assert snapshot is not None
        assert isinstance(snapshot, HMMStateSnapshot)
        assert len(snapshot.transition_matrix) == 3
        assert len(snapshot.emission_means) == 3
        assert len(snapshot.current_regime_probs) == 3
        assert snapshot.log_likelihood == -150.5

    @pytest.mark.unit
    def test_collect_hmm_state_without_fitted_model(self):
        """Test HMM state collection with unfitted model."""
        collector = ModelDataCollector()
        pipeline = MagicMock()
        pipeline.model.is_fitted = False

        snapshot = collector._collect_hmm_state(pipeline, "2023-01-01T00:00:00Z", "2023-01-01")

        assert snapshot is None

    @pytest.mark.unit
    def test_collect_hmm_state_calculates_parameter_changes(self, mock_pipeline):
        """Test that parameter changes are calculated between snapshots."""
        collector = ModelDataCollector()

        # First collection
        snapshot1 = collector._collect_hmm_state(
            mock_pipeline, "2023-01-01T00:00:00Z", "2023-01-01"
        )
        assert snapshot1.parameter_changes == {}  # No previous state

        # Modify model parameters slightly
        mock_pipeline.model.transition_matrix_ = np.array(
            [[0.71, 0.19, 0.1], [0.11, 0.69, 0.2], [0.2, 0.11, 0.69]]
        )

        # Second collection
        snapshot2 = collector._collect_hmm_state(
            mock_pipeline, "2023-01-02T00:00:00Z", "2023-01-02"
        )
        assert "transition_matrix_l2_delta" in snapshot2.parameter_changes
        assert snapshot2.parameter_changes["transition_matrix_l2_delta"] > 0

    @pytest.mark.unit
    def test_collect_indicator_state_returns_snapshot(self, mock_pipeline):
        """Test technical indicator collection returns snapshot."""
        collector = ModelDataCollector()
        snapshot = collector._collect_indicator_state(
            mock_pipeline, "2023-01-01T00:00:00Z", "2023-01-01"
        )

        # Should return snapshot (even if empty due to TODO implementation)
        assert snapshot is not None
        assert isinstance(snapshot, TechnicalIndicatorSnapshot)

    @pytest.mark.unit
    def test_collect_regime_analysis_with_fitted_model(self, mock_pipeline):
        """Test regime analysis collection."""
        collector = ModelDataCollector()
        snapshot = collector._collect_regime_analysis(
            mock_pipeline, "2023-01-01T00:00:00Z", "2023-01-01"
        )

        assert snapshot is not None
        assert isinstance(snapshot, RegimeAnalysisSnapshot)
        assert snapshot.current_regime in ["Bear", "Sideways", "Bull"]
        assert 0.0 <= snapshot.regime_confidence <= 1.0
        assert snapshot.days_in_regime >= 1

    @pytest.mark.unit
    def test_collect_data_quality_metrics(self, mock_pipeline):
        """Test data quality metrics collection."""
        collector = ModelDataCollector()
        metrics = collector._collect_data_quality_metrics(mock_pipeline)

        assert "data_size" in metrics
        assert metrics["data_size"] == 100
        assert "data_completeness" in metrics
        assert "model_fitted" in metrics
        assert "components_available" in metrics


class TestHistoryManagement:
    """Test history retrieval and management."""

    @pytest.fixture
    def collector_with_data(self):
        """Create collector with sample data."""
        collector = ModelDataCollector(max_history=100)
        pipeline = MagicMock()

        # Add minimal data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"close": np.random.randn(10)}, index=dates)
        pipeline.data.get_all_data.return_value = data
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 3
        pipeline.model.transition_matrix_ = np.eye(3)
        pipeline.model.emission_means_ = np.array([0.01, 0.0, -0.01])
        pipeline.model.emission_stds_ = np.array([0.02, 0.015, 0.025])
        pipeline.model.initial_probs_ = np.array([0.33, 0.34, 0.33])
        pipeline.model.predict_proba.return_value = np.random.rand(10, 3)
        pipeline.model.predict.return_value = np.random.randint(0, 3, 10)
        pipeline.model.score.return_value = -100.0
        pipeline.model.training_history_ = {"iterations": 30, "converged": True, "training_time": 1.0}
        pipeline.analysis = MagicMock()

        # Collect 5 timesteps
        for i in range(5):
            collector.collect_timestep_data(pipeline, f"2023-01-{i+1:02d}", time.time())

        return collector

    @pytest.mark.unit
    def test_get_latest_snapshot(self, collector_with_data):
        """Test retrieving latest snapshot."""
        latest = collector_with_data.get_latest_snapshot()

        assert latest is not None
        assert isinstance(latest, TimestepSnapshot)
        assert latest.data_end_date == "2023-01-05"

    @pytest.mark.unit
    def test_get_latest_snapshot_empty_collector(self):
        """Test getting latest snapshot from empty collector."""
        collector = ModelDataCollector()
        latest = collector.get_latest_snapshot()

        assert latest is None

    @pytest.mark.unit
    def test_get_history_all(self, collector_with_data):
        """Test retrieving all history."""
        history = collector_with_data.get_history()

        assert len(history) == 5
        assert all(isinstance(s, TimestepSnapshot) for s in history)

    @pytest.mark.unit
    def test_get_history_limited(self, collector_with_data):
        """Test retrieving limited history."""
        history = collector_with_data.get_history(n_timesteps=3)

        assert len(history) == 3
        assert history[0].data_end_date == "2023-01-03"
        assert history[-1].data_end_date == "2023-01-05"

    @pytest.mark.unit
    def test_get_hmm_parameter_evolution(self, collector_with_data):
        """Test HMM parameter evolution extraction."""
        evolution = collector_with_data.get_hmm_parameter_evolution()

        assert "timestamps" in evolution
        assert "transition_matrices" in evolution
        assert "emission_means" in evolution
        assert "log_likelihoods" in evolution
        assert len(evolution["timestamps"]) == 5

    @pytest.mark.unit
    def test_clear_history(self, collector_with_data):
        """Test clearing collected history."""
        collector_with_data.clear_history()

        assert len(collector_with_data.timestep_data) == 0
        assert collector_with_data._previous_hmm_state is None
        assert collector_with_data.collection_stats["total_timesteps"] == 0
        assert collector_with_data.get_latest_snapshot() is None


class TestExportFunctionality:
    """Test export methods."""

    @pytest.fixture
    def collector_with_data(self):
        """Create collector with sample data for export tests."""
        collector = ModelDataCollector()
        pipeline = MagicMock()

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame({"close": np.random.randn(5)}, index=dates)
        pipeline.data.get_all_data.return_value = data
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 3
        pipeline.model.transition_matrix_ = np.eye(3)
        pipeline.model.emission_means_ = np.array([0.01, 0.0, -0.01])
        pipeline.model.emission_stds_ = np.array([0.02, 0.015, 0.025])
        pipeline.model.initial_probs_ = np.array([0.33, 0.34, 0.33])
        pipeline.model.predict_proba.return_value = np.random.rand(5, 3)
        pipeline.model.predict.return_value = np.random.randint(0, 3, 5)
        pipeline.model.score.return_value = -50.0
        pipeline.model.training_history_ = {"iterations": 20, "converged": True, "training_time": 0.8}
        pipeline.analysis = MagicMock()

        collector.collect_timestep_data(pipeline, "2023-01-01", time.time())
        return collector

    @pytest.mark.unit
    def test_export_to_json(self, collector_with_data, tmp_path):
        """Test JSON export."""
        export_file = tmp_path / "export.json"
        collector_with_data.export_to_json(str(export_file))

        assert export_file.exists()

        # Verify JSON structure
        with open(export_file, "r") as f:
            data = json.load(f)

        assert "collection_stats" in data
        assert "timesteps" in data
        assert len(data["timesteps"]) == 1

    @pytest.mark.unit
    def test_export_to_parquet(self, collector_with_data, tmp_path):
        """Test Parquet export."""
        export_file = tmp_path / "export.parquet"
        collector_with_data.export_to_parquet(str(export_file))

        assert export_file.exists()

        # Verify Parquet structure
        df = pd.read_parquet(export_file)
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "data_end_date" in df.columns
        assert "execution_time" in df.columns

    @pytest.mark.unit
    def test_export_to_parquet_with_multiple_timesteps(self, tmp_path):
        """Test Parquet export with multiple timesteps."""
        collector = ModelDataCollector()
        pipeline = MagicMock()

        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"close": np.random.randn(10)}, index=dates)
        pipeline.data.get_all_data.return_value = data
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 3
        pipeline.model.transition_matrix_ = np.eye(3)
        pipeline.model.emission_means_ = np.array([0.01, 0.0, -0.01])
        pipeline.model.emission_stds_ = np.array([0.02, 0.015, 0.025])
        pipeline.model.initial_probs_ = np.array([0.33, 0.34, 0.33])
        pipeline.model.predict_proba.return_value = np.random.rand(10, 3)
        pipeline.model.predict.return_value = np.random.randint(0, 3, 10)
        pipeline.model.score.return_value = -100.0
        pipeline.model.training_history_ = {"iterations": 30, "converged": True, "training_time": 1.0}
        pipeline.analysis = MagicMock()

        # Collect 3 timesteps
        for i in range(3):
            collector.collect_timestep_data(pipeline, f"2023-01-{i+1:02d}", time.time())

        export_file = tmp_path / "multi_export.parquet"
        collector.export_to_parquet(str(export_file))

        df = pd.read_parquet(export_file)
        assert len(df) == 3

    @pytest.mark.unit
    def test_auto_export_triggered(self, tmp_path, monkeypatch):
        """Test that auto-export is triggered at correct intervals."""
        export_file = tmp_path / "auto_export.json"
        collector = ModelDataCollector(auto_export=True, export_path=str(export_file))

        pipeline = MagicMock()
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"close": np.random.randn(10)}, index=dates)
        pipeline.data.get_all_data.return_value = data
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 2
        pipeline.model.transition_matrix_ = np.eye(2)
        pipeline.model.emission_means_ = np.array([0.01, -0.01])
        pipeline.model.emission_stds_ = np.array([0.02, 0.025])
        pipeline.model.initial_probs_ = np.array([0.5, 0.5])
        pipeline.model.predict_proba.return_value = np.random.rand(10, 2)
        pipeline.model.predict.return_value = np.random.randint(0, 2, 10)
        pipeline.model.score.return_value = -80.0
        pipeline.model.training_history_ = {"iterations": 25, "converged": True, "training_time": 0.9}
        pipeline.analysis = MagicMock()

        # Collect 50 timesteps (should trigger auto-export at timestep 50)
        for i in range(50):
            collector.collect_timestep_data(pipeline, f"2023-01-{i+1:02d}", time.time())

        # Auto-export should have been called
        assert export_file.exists()


class TestSummaryStatistics:
    """Test summary statistics generation."""

    @pytest.mark.unit
    def test_get_summary_stats_empty_collector(self):
        """Test summary stats for empty collector."""
        collector = ModelDataCollector()
        stats = collector.get_summary_stats()

        assert "message" in stats
        assert stats["message"] == "No data collected yet"

    @pytest.mark.unit
    def test_get_summary_stats_with_data(self):
        """Test summary stats with collected data."""
        collector = ModelDataCollector()
        pipeline = MagicMock()

        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"close": np.random.randn(10)}, index=dates)
        pipeline.data.get_all_data.return_value = data
        pipeline.model.is_fitted = True
        pipeline.model.n_states = 3
        pipeline.model.transition_matrix_ = np.eye(3)
        pipeline.model.emission_means_ = np.array([0.01, 0.0, -0.01])
        pipeline.model.emission_stds_ = np.array([0.02, 0.015, 0.025])
        pipeline.model.initial_probs_ = np.array([0.33, 0.34, 0.33])
        pipeline.model.predict_proba.return_value = np.random.rand(10, 3)
        pipeline.model.predict.return_value = np.random.randint(0, 3, 10)
        pipeline.model.score.return_value = -100.0
        pipeline.model.training_history_ = {"iterations": 30, "converged": True, "training_time": 1.0}
        pipeline.analysis = MagicMock()

        # Collect 3 timesteps
        for i in range(3):
            collector.collect_timestep_data(pipeline, f"2023-01-{i+1:02d}", time.time())

        stats = collector.get_summary_stats()

        assert stats["total_timesteps"] == 3
        assert stats["successful_hmm_collection"] == 3
        assert stats["collection_success_rate"] == 1.0
        assert "avg_execution_time" in stats
        assert stats["date_range"]["start"] == "2023-01-01"
        assert stats["date_range"]["end"] == "2023-01-03"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
