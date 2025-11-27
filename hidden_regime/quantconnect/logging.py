"""
Comprehensive logging for QuantConnect regime detection algorithms.

This module provides structured logging for:
- Algorithm initialization and configuration
- Data ingestion (bars received)
- Regime detection pipeline events (retraining, inference)
- Signal generation and interpretation
- Trading decisions and allocations
- Regime transitions and changes
- Performance metrics and statistics
"""

from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class RegimeDetectionLogger:
    """Centralized logging for regime detection algorithm."""

    def __init__(self, algorithm):
        """
        Initialize logger.

        Args:
            algorithm: Reference to QCAlgorithm for Log/Debug calls
        """
        self.algo = algorithm
        self.session_start = datetime.now()
        self.bars_received = {}
        self.regimes_detected = {}

    def log_initialization(self, config: Dict[str, Any]) -> None:
        """Log algorithm initialization."""
        self.algo.Debug("=" * 80)
        self.algo.Debug("REGIME DETECTION ALGORITHM INITIALIZED")
        self.algo.Debug("=" * 80)
        self.algo.Debug(f"Session Start: {self.session_start}")
        self.algo.Debug(f"Configuration:")
        for key, value in config.items():
            self.algo.Debug(f"  {key}: {value}")
        self.algo.Debug("")

    def log_regime_setup(
        self,
        ticker: str,
        n_states: int,
        lookback_days: int,
        retrain_frequency: str,
        allocations: Dict[str, float],
    ) -> None:
        """Log regime detection setup for a ticker."""
        self.algo.Debug(f"SETUP: Regime detection for {ticker}")
        self.algo.Debug(f"  States: {n_states}")
        self.algo.Debug(f"  Lookback: {lookback_days} days")
        self.algo.Debug(f"  Retrain: {retrain_frequency}")
        self.algo.Debug(f"  Allocations: {allocations}")
        self.algo.Debug("")
        self.bars_received[ticker] = 0

    def log_bar_received(self, ticker: str, timestamp: datetime, close: float) -> None:
        """Log when a new bar is received."""
        self.bars_received[ticker] = self.bars_received.get(ticker, 0) + 1
        # DISABLED: Don't log bars to avoid spam in backtests

    def log_regime_readiness(self, ticker: str, bars_available: int, ready: bool) -> None:
        """Log when regime detection becomes ready."""
        if ready:
            self.algo.Debug(f"READY: {ticker} regime detection ready ({bars_available} bars)")
            self.algo.Debug("")

    def log_pipeline_training(
        self, ticker: str, timestamp: datetime, reason: str = "scheduled"
    ) -> None:
        """Log when HMM pipeline is being trained/retrained."""
        self.algo.Debug(f"TRAIN: {ticker} HMM pipeline {reason} @ {timestamp.date()}")

    def log_pipeline_inference(
        self,
        ticker: str,
        timestamp: datetime,
        regime: str,
        state: int,
        confidence: float,
    ) -> None:
        """Log regime inference result for the day."""
        # DISABLED: Don't log daily inferences to reduce spam

    def log_regime_change(
        self,
        ticker: str,
        timestamp: datetime,
        old_regime: str,
        new_regime: str,
        confidence: float,
    ) -> None:
        """Log regime transition."""
        # Log regime changes (important events only)
        self.algo.Debug(
            f"REGIME: {ticker} {old_regime} → {new_regime} "
            f"(confidence: {confidence:.1%})"
        )

        # Track regime changes for analysis
        if ticker not in self.regimes_detected:
            self.regimes_detected[ticker] = []
        self.regimes_detected[ticker].append({
            "timestamp": timestamp,
            "regime": new_regime,
            "confidence": confidence,
        })

    def log_signal_generation(
        self,
        ticker: str,
        timestamp: datetime,
        regime: str,
        allocation: float,
        direction: str,
        strength: str,
        confidence: float,
    ) -> None:
        """Log trading signal generation."""
        # DISABLED: Don't log every daily signal to reduce spam

    def log_position_update(
        self,
        ticker: str,
        timestamp: datetime,
        old_allocation: float,
        new_allocation: float,
        portfolio_value: float,
    ) -> None:
        """Log when position is updated."""
        allocation_change = new_allocation - old_allocation
        self.algo.Log(
            f"TRADE: {ticker} @ {timestamp.date()} "
            f"{old_allocation:.1%} → {new_allocation:.1%} "
            f"(Δ{allocation_change:+.1%}) | "
            f"Portfolio: ${portfolio_value:,.0f}"
        )

    def log_summary(self) -> None:
        """Log end-of-session summary."""
        self.algo.Debug("")
        self.algo.Debug("=" * 80)
        self.algo.Debug("REGIME DETECTION SESSION SUMMARY")
        self.algo.Debug("=" * 80)
        for ticker, count in self.bars_received.items():
            self.algo.Debug(f"{ticker}: {count} bars received")
            if ticker in self.regimes_detected:
                self.algo.Debug(f"  Regime changes: {len(self.regimes_detected[ticker])}")
                for change in self.regimes_detected[ticker]:
                    self.algo.Debug(
                        f"    {change['timestamp'].date()}: {change['regime']} "
                        f"({change['confidence']:.1%})"
                    )
        self.algo.Debug("=" * 80)
        self.algo.Debug("")


def create_csv_export(logger: RegimeDetectionLogger, output_path: str) -> None:
    """
    Export regime detection history to CSV for analysis.

    Args:
        logger: RegimeDetectionLogger instance
        output_path: Path to write CSV file
    """
    rows = []
    for ticker, changes in logger.regimes_detected.items():
        for change in changes:
            rows.append({
                "ticker": ticker,
                "date": change["timestamp"].date(),
                "regime": change["regime"],
                "confidence": change["confidence"],
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Regime detection history exported to {output_path}")
