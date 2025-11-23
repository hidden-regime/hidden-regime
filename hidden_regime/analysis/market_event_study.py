"""
Market Event Study Framework for Regime Analysis.

Provides a reusable API for analyzing regime behavior during market events
(crashes, bubbles, sector rotations, etc.) with minimal code duplication.

Example:
    >>> import hidden_regime as hr
    >>> study = hr.MarketEventStudy(
    ...     ticker='QQQ',
    ...     training_start='2018-01-01',
    ...     training_end='2019-12-31',
    ...     analysis_start='2020-01-01',
    ...     analysis_end='2020-12-31',
    ...     n_states=3,
    ...     key_events={'2020-02-19': 'Market Peak', '2020-03-23': 'Bottom'},
    ...     output_dir='output/covid_study'
    ... )
    >>> results = study.run()
    >>> study.print_summary()
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Lazy imports to avoid circular dependencies
# These will be imported in methods that use them


class MarketEventStudy:
    """
    Reusable framework for analyzing regime behavior during market events.

    This class encapsulates the common pattern of:
    1. Training HMM on pre-event data
    2. Stepping through event period day-by-day
    3. Creating visualizations at key dates
    4. Computing detection and stability metrics
    5. Exporting results for analysis

    Attributes:
        ticker: Stock ticker(s) to analyze
        training_start: Start date for training period
        training_end: End date for training period
        analysis_start: Start date for analysis period
        analysis_end: End date for analysis period (defaults to today)
        n_states: Number of HMM regime states
        key_events: Dictionary of {date: event_name} for snapshot generation
        output_dir: Directory for saving outputs
    """

    def __init__(
        self,
        ticker: Union[str, List[str]],
        training_start: str,
        training_end: str,
        analysis_start: str,
        analysis_end: Optional[str] = None,
        n_states: int = 3,
        key_events: Optional[Dict[str, str]] = None,
        output_dir: str = "output/market_study",
        generate_signals: bool = False,
        signal_strategy: str = 'regime_following',
        **pipeline_kwargs,
    ):
        """
        Initialize market event study.

        Args:
            ticker: Stock ticker symbol(s). Can be string or list of strings.
            training_start: Start date for training period (YYYY-MM-DD)
            training_end: End date for training period (YYYY-MM-DD)
            analysis_start: Start date for analysis period (YYYY-MM-DD)
            analysis_end: End date for analysis period (YYYY-MM-DD), defaults to today
            n_states: Number of regime states for HMM (default: 3)
            key_events: Dict of {date: event_name} for snapshot generation
            output_dir: Directory path for outputs (default: 'output/market_study')
            generate_signals: Whether to generate trading signals (default: False)
            signal_strategy: Signal generation strategy ('regime_following', 'regime_contrarian', 'confidence_weighted')
            **pipeline_kwargs: Additional arguments passed to create_financial_pipeline
        """
        # Handle single ticker vs list
        self.tickers = [ticker] if isinstance(ticker, str) else ticker
        self.ticker = ticker  # Keep original for compatibility

        # Store date ranges
        self.training_start = training_start
        self.training_end = training_end
        self.analysis_start = analysis_start
        self.analysis_end = analysis_end or datetime.now().strftime("%Y-%m-%d")

        # Model configuration
        self.n_states = n_states
        self.pipeline_kwargs = pipeline_kwargs

        # Event configuration
        self.key_events = key_events or {}

        # Signal configuration
        self.generate_signals = generate_signals
        self.signal_strategy = signal_strategy

        # Output configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}
        self.metrics = {}
        self.pipelines = {}

    def run(
        self,
        create_snapshots: bool = True,
        create_animations: bool = False,
        snapshot_window_days: int = 90,
        animation_fps: int = 5,
        testing_mode: bool = False,
        testing_limit_days: int = 30,
    ) -> Dict:
        """
        Run complete market event study workflow.

        Args:
            create_snapshots: Whether to create snapshot PNGs at key events
            create_animations: Whether to create GIF animations
            snapshot_window_days: Number of days to show in snapshot window
            animation_fps: Frames per second for animations
            testing_mode: If True, limit analysis to first N days
            testing_limit_days: Number of days to process in testing mode

        Returns:
            Dictionary with results for each ticker
        """
        from .. import create_financial_pipeline

        self._print_section("Market Event Study")
        print(f"Training Period: {self.training_start} to {self.training_end}")
        print(f"Analysis Period: {self.analysis_start} to {self.analysis_end}")
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Output Directory: {self.output_dir}")

        for ticker in self.tickers:
            self._print_section(f"Analyzing {ticker}")

            # Step 1: Load data
            print(f"\n1. Loading data for {ticker}...")
            training_data, analysis_data, full_data = self._load_data(ticker)

            # Step 2: Train model
            print(f"\n2. Training {ticker} model on {self.training_start} to {self.training_end}...")
            pipeline = self._train_model(ticker, training_data)
            self.pipelines[ticker] = pipeline

            # Step 3: Run temporal analysis
            print(f"\n3. Running temporal analysis for {ticker}...")
            analysis_results = self._run_temporal_analysis(
                ticker,
                pipeline,
                analysis_data,
                full_data,
                testing_mode=testing_mode,
                testing_limit_days=testing_limit_days,
            )
            self.results[ticker] = analysis_results

            # Step 4: Create visualizations
            if create_snapshots and self.key_events:
                print(f"\n4. Creating snapshot PNGs for {ticker}...")
                self._create_snapshot_pngs(
                    analysis_results, ticker, window_days=snapshot_window_days
                )

            if create_animations:
                print(f"\n5. Creating animations for {ticker}...")
                self._create_animations(analysis_results, ticker, fps=animation_fps)

            # Step 5: Compute metrics
            print(f"\n6. Computing metrics for {ticker}...")
            metrics = self._compute_metrics(analysis_results)
            self.metrics[ticker] = metrics

        self._print_section("Analysis Complete")
        return self.results

    def _load_data(self, ticker: str):
        """Load and split data into training and analysis periods."""
        # Lazy imports
        from ..config import FinancialDataConfig
        from ..data import FinancialDataLoader

        config = FinancialDataConfig(
            ticker=ticker, start_date=self.training_start, end_date=self.analysis_end
        )

        loader = FinancialDataLoader(config=config)
        full_data = loader.get_all_data()

        # Remove timezone for temporal controller compatibility
        if full_data.index.tz is not None:
            full_data.index = full_data.index.tz_localize(None)

        # Split into training and analysis
        training_data = full_data[
            (full_data.index >= self.training_start)
            & (full_data.index <= self.training_end)
        ]

        analysis_data = full_data[
            (full_data.index >= self.analysis_start)
            & (full_data.index <= self.analysis_end)
        ]

        print(
            f"  Training: {len(training_data)} days ({self.training_start} to {self.training_end})"
        )
        print(
            f"  Analysis: {len(analysis_data)} days ({self.analysis_start} to {self.analysis_end})"
        )

        return training_data, analysis_data, full_data

    def _train_model(self, ticker: str, training_data: pd.DataFrame):
        """Train HMM model on training period data."""
        from .. import create_financial_pipeline

        pipeline = create_financial_pipeline(
            ticker=ticker,
            n_states=self.n_states,
            start_date=self.training_start,
            end_date=self.training_end,
            **self.pipeline_kwargs,
        )

        # Train on data
        result = pipeline.update()

        # Display trained parameters
        if (
            hasattr(pipeline.model, "model")
            and hasattr(pipeline.model.model, "emission_means_")
        ):
            means = pipeline.model.model.emission_means_
            stds = pipeline.model.model.emission_stds_

            print(f"\n  Trained Regime Parameters:")
            for i in range(self.n_states):
                print(f"    State {i}: μ={means[i]:.4f}, σ={stds[i]:.4f}")

        return pipeline

    def _run_temporal_analysis(
        self,
        ticker: str,
        pipeline,
        analysis_data: pd.DataFrame,
        full_data: pd.DataFrame,
        testing_mode: bool = False,
        testing_limit_days: int = 30,
    ) -> Dict:
        """Step through analysis period day-by-day."""
        # Lazy import
        from ..pipeline import TemporalController

        temporal_controller = TemporalController(pipeline, full_data)

        # Storage for results
        regime_history = []
        regime_data_sequence = []
        evaluation_dates = []
        signal_history = []  # NEW: Store trading signals

        # Initialize signal generator if enabled
        signal_generator = None
        if self.generate_signals:
            from ..signal_generation import FinancialSignalGenerator

            signal_generator = FinancialSignalGenerator(
                strategy_type=self.signal_strategy,
                min_confidence=0.3,
                position_scaling=True  # Continuous signals [-1, 1]
            )
            print(f"  Signal generation enabled: {self.signal_strategy} strategy")

        # Get analysis dates
        analysis_dates = analysis_data.index.tolist()

        # Apply testing mode if enabled
        if testing_mode:
            analysis_dates = analysis_dates[:testing_limit_days]
            print(
                f"  TESTING MODE: Limited to {len(analysis_dates)} days ({analysis_dates[0]} to {analysis_dates[-1]})"
            )
        else:
            print(f"  Processing {len(analysis_dates)} days...")

        # Step through each day with progress bar
        for current_date in tqdm(
            analysis_dates, desc=f"  {ticker}", unit="day", leave=False
        ):
            try:
                # Use temporal controller to update pipeline as-of current date
                date_str = current_date.strftime("%Y-%m-%d")
                temporal_controller.update_as_of(date_str)

                # Extract results from pipeline outputs
                model_output = pipeline.model_output
                data_output = pipeline.data_output
                analysis_output = pipeline.analysis_output

                # Skip if no results
                if len(model_output) == 0 or len(data_output) == 0:
                    continue

                # Get regime for current day (last prediction)
                regime = int(model_output["predicted_state"].iloc[-1])
                confidence = float(model_output["confidence"].iloc[-1])
                regime_name = str(analysis_output["regime_name"].iloc[-1])

                # Get price from data output
                price = float(data_output["close"].iloc[-1])

                # Get state probabilities (for animation)
                state_probs = {}
                for i in range(self.n_states):
                    prob_col = f"state_{i}_prob"
                    if prob_col in model_output.columns:
                        state_probs[prob_col] = float(model_output[prob_col].iloc[-1])
                    else:
                        # Fallback: if prob column missing, use confidence for predicted state
                        state_probs[prob_col] = confidence if i == regime else 0.0

                # Generate trading signal if enabled
                signal_value = None
                signal_discrete = None
                if signal_generator is not None:
                    try:
                        # Generate signal for current timeframe
                        signals = signal_generator.generate_signals(
                            price_data=data_output,
                            additional_data=model_output
                        )
                        signal_value = float(signals.iloc[-1])  # Continuous [-1, 1]
                        signal_discrete = int(np.sign(signal_value))  # Discrete {-1, 0, 1}

                        # Store signal in history
                        signal_history.append({
                            'date': current_date,
                            'signal': signal_value,
                            'signal_discrete': signal_discrete,
                            'regime': regime,
                            'regime_name': regime_name,
                            'confidence': confidence,
                            'price': price,
                        })
                    except Exception as e:
                        warnings.warn(f"Signal generation failed for {current_date}: {e}")
                        signal_value = 0.0
                        signal_discrete = 0

                # Record history
                regime_history.append(
                    {
                        "date": current_date,
                        "regime": regime,
                        "confidence": confidence,
                        "price": price,
                        "regime_name": regime_name,
                        **state_probs,  # Add state probabilities
                    }
                )

                # Build regime data for animation
                current_regime_data = pd.DataFrame(
                    [
                        {
                            "date": r["date"],
                            "predicted_state": r["regime"],
                            "confidence": r["confidence"],
                            "regime_name": r["regime_name"],
                            **{f"state_{i}_prob": r.get(f"state_{i}_prob", 0.0) for i in range(self.n_states)},
                        }
                        for r in regime_history
                    ]
                )
                current_regime_data.set_index("date", inplace=True)

                regime_data_sequence.append(current_regime_data.copy())
                evaluation_dates.append(date_str)

            except ValueError as e:
                # Handle expected errors (insufficient data, etc.)
                if any(
                    keyword in str(e).lower()
                    for keyword in ["insufficient", "not enough", "no data"]
                ):
                    continue
                else:
                    warnings.warn(f"Error on {current_date}: {e}")
                    continue
            except Exception as e:
                warnings.warn(
                    f"Unexpected error on {current_date}: {type(e).__name__}: {e}"
                )
                continue

        print(f"  Completed: {len(regime_history)} days analyzed")

        # Handle empty results
        if len(regime_history) == 0:
            print(f"  ERROR: No data was processed!")
            empty_df = pd.DataFrame(columns=["regime", "confidence", "price"])
            empty_df.index.name = "date"
            return {
                "ticker": ticker,
                "regime_history": empty_df,
                "regime_data_sequence": [],
                "evaluation_dates": [],
                "full_data": full_data,
            }

        # Create DataFrame from history
        regime_df = pd.DataFrame(regime_history)
        regime_df = regime_df.set_index("date")

        print(f"  Regime distribution: {regime_df['regime'].value_counts().to_dict()}")

        # Extract regime_profiles from pipeline for visualization color consistency
        regime_profiles = None
        if hasattr(pipeline, 'analysis') and hasattr(pipeline.analysis, 'get_regime_profiles'):
            regime_profiles = pipeline.analysis.get_regime_profiles()

        # Create signal DataFrame if signals were generated
        signal_df = None
        if len(signal_history) > 0:
            signal_df = pd.DataFrame(signal_history)
            signal_df = signal_df.set_index('date')
            print(f"  Trading signals generated: {len(signal_df)} signals")
            print(f"    Buy signals: {(signal_df['signal_discrete'] > 0).sum()}")
            print(f"    Sell signals: {(signal_df['signal_discrete'] < 0).sum()}")
            print(f"    Hold/Neutral signals: {(signal_df['signal_discrete'] == 0).sum()}")

        return {
            "ticker": ticker,
            "regime_history": regime_df,
            "regime_data_sequence": regime_data_sequence,
            "evaluation_dates": evaluation_dates,
            "full_data": full_data,
            "regime_profiles": regime_profiles,  # For color consistency
            "signal_history": signal_df,  # NEW: Trading signals
        }

    def _create_snapshot_pngs(
        self, analysis_results: Dict, ticker: str, window_days: int = 90
    ):
        """Create static PNG snapshots at key event dates."""
        # Use the API from analysis module for consistent colorblind-safe colors
        # Pass regime_profiles to get colors from RegimeProfile objects (single source of truth)
        from . import get_regime_colors

        regime_profiles = analysis_results.get("regime_profiles")
        regime_colors_map = get_regime_colors(regime_profiles)

        history = analysis_results["regime_history"]
        full_data = analysis_results["full_data"]

        # Compute fixed y-axis limits for consistent scaling
        price_min = full_data["close"].min()
        price_max = full_data["close"].max()
        y_lower = price_min * 0.9
        y_upper = price_max * 1.1

        for date_str, event_name in self.key_events.items():
            snapshot_date = pd.to_datetime(date_str)

            # Check if we have data for this date
            if snapshot_date not in history.index:
                print(f"  Skipping {date_str} (no data)")
                continue

            # Get window
            window_start = snapshot_date - pd.Timedelta(days=window_days)
            window_data = full_data[
                (full_data.index >= window_start) & (full_data.index <= snapshot_date)
            ]
            window_history = history[history.index <= snapshot_date]

            if len(window_data) == 0:
                continue

            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Price with regime overlay
            axes[0].plot(
                window_data.index,
                window_data["close"],
                "k-",
                linewidth=2,
                label="Price",
            )

            # Color background by regime (using regime_name, not state ID)
            for regime_name in window_history["regime_name"].unique():
                regime_mask = window_history["regime_name"] == regime_name
                if regime_mask.sum() > 0:
                    for date in window_history.index[regime_mask]:
                        if date in window_data.index:
                            axes[0].axvspan(
                                date,
                                date + pd.Timedelta(days=1),
                                color=regime_colors_map.get(regime_name, "gray"),
                                alpha=0.2,
                            )

            # Mark current date
            axes[0].axvline(
                snapshot_date, color="red", linestyle="--", linewidth=2, alpha=0.8,
                label="Current Date"
            )

            # Get current regime info
            current_regime_name = history.loc[snapshot_date, "regime_name"]
            current_confidence = history.loc[snapshot_date, "confidence"]

            axes[0].set_ylim(y_lower, y_upper)
            axes[0].set_title(
                f"{ticker} - {event_name} ({date_str})\nRegime: {current_regime_name.capitalize()} (Confidence: {current_confidence:.1%})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0].set_ylabel("Price ($)", fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            # Plot 2: Confidence over window
            axes[1].plot(
                window_history.index,
                window_history["confidence"],
                "purple",
                linewidth=2,
            )
            axes[1].fill_between(
                window_history.index,
                0,
                window_history["confidence"],
                alpha=0.3,
                color="purple",
            )
            axes[1].axvline(
                snapshot_date, color="red", linestyle="--", linewidth=2, alpha=0.8,
                label="Current Date"
            )
            axes[1].set_ylabel("Confidence", fontsize=12)
            axes[1].set_xlabel("Date", fontsize=12)
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()

            # Save
            safe_date = date_str.replace("-", "")
            filename = f"{ticker}_snapshot_{safe_date}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"  Saved: {filename}")

    def _create_animations(self, analysis_results: Dict, ticker: str, fps: int = 5):
        """Generate GIF animations showing regime evolution."""
        try:
            from ..visualization.animations import RegimeAnimator

            animator = RegimeAnimator(color_scheme="colorblind_safe", style="professional")

            # Extract regime_profiles for color consistency
            regime_profiles = analysis_results.get("regime_profiles")

            # Build signal_data_sequence if signals were generated
            signal_data_sequence = None
            if analysis_results.get("signal_history") is not None:
                signal_history = analysis_results["signal_history"]
                evaluation_dates = analysis_results["evaluation_dates"]

                # Create cumulative signal data for each evaluation date
                signal_data_sequence = []
                for date_str in evaluation_dates:
                    date_dt = pd.to_datetime(date_str)
                    # Get signals up to current date
                    cumulative_signals = signal_history[signal_history.index <= date_dt].copy()
                    signal_data_sequence.append(cumulative_signals)

            # Full period animation
            print(f"  Creating animation...")
            animator.create_evolving_regime_animation(
                data=analysis_results["full_data"],
                regime_data_sequence=analysis_results["regime_data_sequence"],
                evaluation_dates=analysis_results["evaluation_dates"],
                window_size=120,
                title=f"{ticker} - Market Event Study",
                save_path=str(self.output_dir / f"{ticker}_full_analysis.gif"),
                fps=fps,
                dpi=120,
                regime_profiles=regime_profiles,
                signal_data_sequence=signal_data_sequence,  # NEW: Pass signal data for visualization
            )
            print(f"    Saved: {ticker}_full_analysis.gif")

        except ImportError:
            warnings.warn("Animation module not available. Skipping animations.")

    def _compute_metrics(self, results: Dict) -> Dict:
        """Compute detection and stability metrics."""
        history = results["regime_history"]

        if len(history) == 0:
            return {
                "detection_lag_days": None,
                "avg_regime_duration": 0,
                "num_transitions": 0,
                "crisis_days": 0,
                "avg_confidence": 0,
            }

        # Find first bearish/crisis regime detection
        analysis_start_dt = pd.to_datetime(self.analysis_start)
        post_start = history[history.index >= analysis_start_dt]

        crisis_mask = post_start["regime_name"].isin(["bearish", "crisis"])
        first_crisis = post_start[crisis_mask]

        if len(first_crisis) > 0:
            detection_lag = (first_crisis.index[0] - analysis_start_dt).days
        else:
            detection_lag = None

        # Compute regime durations
        regime_changes = history["regime"].diff().fillna(0) != 0
        regime_runs = []

        current_regime = history["regime"].iloc[0]
        run_start = history.index[0]
        run_length = 1

        for i in range(1, len(history)):
            if history["regime"].iloc[i] == current_regime:
                run_length += 1
            else:
                regime_runs.append(
                    {
                        "regime": current_regime,
                        "start": run_start,
                        "length": run_length,
                    }
                )
                current_regime = history["regime"].iloc[i]
                run_start = history.index[i]
                run_length = 1

        # Add last run
        regime_runs.append(
            {"regime": current_regime, "start": run_start, "length": run_length}
        )

        avg_regime_duration = np.mean([r["length"] for r in regime_runs])
        num_transitions = len(regime_runs) - 1

        # Crisis duration
        crisis_days = history["regime_name"].isin(["bearish", "crisis"]).sum()

        # Average confidence
        avg_confidence = history["confidence"].mean()

        return {
            "detection_lag_days": detection_lag,
            "avg_regime_duration": avg_regime_duration,
            "num_transitions": num_transitions,
            "crisis_days": crisis_days,
            "avg_confidence": avg_confidence,
            "regime_runs": regime_runs,
        }

    def get_metrics(self, ticker: Optional[str] = None) -> Dict:
        """
        Get computed metrics.

        Args:
            ticker: Specific ticker to get metrics for. If None, returns all.

        Returns:
            Dictionary of metrics
        """
        if ticker:
            return self.metrics.get(ticker, {})
        return self.metrics

    def print_summary(self):
        """Print summary report to console."""
        self._print_section("Market Event Study Summary")

        for ticker, metrics in self.metrics.items():
            print(f"\n{ticker}:")
            print(f"  Detection lag: {metrics['detection_lag_days']} days")
            print(f"  Avg regime duration: {metrics['avg_regime_duration']:.1f} days")
            print(f"  Total transitions: {metrics['num_transitions']}")
            print(f"  Crisis/bear days: {metrics['crisis_days']}")
            print(f"  Avg confidence: {metrics['avg_confidence']:.2%}")

    def export_results(self, format: str = "csv", filename: Optional[str] = None):
        """
        Export results to file.

        Args:
            format: Export format ('csv' or 'json')
            filename: Output filename (defaults to ticker_results.csv/json)

        Returns:
            Path to exported file
        """
        if not filename:
            filename = f"regime_history.{format}"

        filepath = self.output_dir / filename

        if format == "csv":
            # Combine all tickers into single CSV
            combined_histories = {}
            for ticker, results in self.results.items():
                history = results["regime_history"].copy()
                history.columns = [f"{ticker}_{col}" for col in history.columns]
                combined_histories[ticker] = history

            combined = pd.concat(combined_histories.values(), axis=1)
            combined.to_csv(filepath)

        elif format == "json":
            import json

            export_data = {
                ticker: {
                    "regime_history": results["regime_history"].to_dict(
                        orient="index"
                    ),
                    "metrics": self.metrics.get(ticker, {}),
                }
                for ticker, results in self.results.items()
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"  Exported to: {filepath}")
        return str(filepath)

    def add_event(self, date: str, event_name: str):
        """
        Add a key event date for snapshot generation.

        Args:
            date: Event date (YYYY-MM-DD)
            event_name: Description of event
        """
        self.key_events[date] = event_name

    def export_signals_for_quantconnect(self, ticker: Optional[str] = None, filename: Optional[str] = None) -> Optional[str]:
        """
        Export trading signals in QuantConnect-compatible CSV format.

        Args:
            ticker: Specific ticker to export signals for (if None, exports all)
            filename: Custom filename (optional)

        Returns:
            Path to exported CSV file, or None if no signals available

        QuantConnect CSV Format:
            date,signal,confidence,regime,regime_name,price
            2020-02-19,1,0.82,2,bull,300.45
            2020-02-24,-1,0.68,0,bear,285.30
        """
        if not self.generate_signals:
            print("Warning: Signal generation was not enabled. Enable with generate_signals=True")
            return None

        # Determine which tickers to export
        tickers_to_export = [ticker] if ticker else self.tickers

        exported_files = []

        for tick in tickers_to_export:
            if tick not in self.results:
                print(f"Warning: No results found for {tick}")
                continue

            signal_history = self.results[tick].get('signal_history')

            if signal_history is None or len(signal_history) == 0:
                print(f"Warning: No signals generated for {tick}")
                continue

            # Create QuantConnect-compatible DataFrame
            qc_signals = signal_history[['signal_discrete', 'confidence', 'regime', 'regime_name', 'price']].copy()
            qc_signals = qc_signals.rename(columns={'signal_discrete': 'signal'})

            # Generate filename for this ticker (reset filename for each ticker)
            tick_filename = filename if filename else f"{tick}_signals_{self.analysis_start.replace('-', '')}_{self.analysis_end.replace('-', '')}.csv"

            # Export to signals subdirectory
            signals_dir = self.output_dir / "signals"
            signals_dir.mkdir(parents=True, exist_ok=True)
            filepath = signals_dir / tick_filename

            qc_signals.to_csv(filepath)

            print(f"  ✅ Signals exported for {tick}: {filepath}")
            print(f"     Format: QuantConnect-ready CSV")
            print(f"     Signals: {len(qc_signals)} ({(qc_signals['signal'] > 0).sum()} BUY, {(qc_signals['signal'] < 0).sum()} SELL)")

            exported_files.append(str(filepath))

        # Return first file if single ticker, otherwise return list (or None if nothing exported)
        if len(exported_files) == 0:
            return None
        elif len(exported_files) == 1:
            return exported_files[0]
        else:
            return exported_files

    def analyze_signal_consistency(self, ticker: Optional[str] = None) -> Dict:
        """
        Analyze whether generated signal durations are consistent with regime persistence
        patterns observed in the training data.

        This method compares:
        1. Expected regime duration from HMM transition matrix: 1/(1-persistence)
        2. Actual signal duration from generated signals

        Args:
            ticker: Specific ticker to analyze (if None, analyzes all)

        Returns:
            Dictionary with consistency analysis results:
            {
                'ticker': {
                    'expected_durations': {...},  # Expected duration per regime from HMM
                    'actual_durations': {...},     # Actual signal duration statistics
                    'consistency_ratios': {...},   # Ratio of actual/expected
                    'rapid_flips': [...],          # List of rapid signal flips (<3 days)
                    'warnings': [...],             # List of warning messages
                }
            }
        """
        if not self.generate_signals:
            print("Warning: Signal generation was not enabled.")
            return {}

        # Determine which tickers to analyze
        tickers_to_analyze = [ticker] if ticker else self.tickers

        results = {}

        for tick in tickers_to_analyze:
            if tick not in self.results:
                print(f"Warning: No results found for {tick}")
                continue

            if tick not in self.pipelines:
                print(f"Warning: No trained pipeline found for {tick}")
                continue

            signal_history = self.results[tick].get('signal_history')

            if signal_history is None or len(signal_history) == 0:
                print(f"Warning: No signals generated for {tick}")
                continue

            pipeline = self.pipelines[tick]

            # Extract expected regime durations from HMM transition matrix
            expected_durations = {}
            if hasattr(pipeline.model, 'transition_matrix_') and pipeline.model.transition_matrix_ is not None:
                transition_matrix = pipeline.model.transition_matrix_
                n_states = transition_matrix.shape[0]

                # Get regime names from signal_history (more reliable than pipeline output after temporal controller)
                regime_names_map = {}
                for state_id in range(n_states):
                    state_mask = signal_history['regime'] == state_id
                    if state_mask.any():
                        regime_name = signal_history.loc[state_mask, 'regime_name'].iloc[0]
                        regime_names_map[state_id] = regime_name
                    else:
                        # Fallback: use default naming
                        regime_names_map[state_id] = f"regime_{state_id}"

                # Calculate expected duration for each regime
                for i in range(n_states):
                    persistence = transition_matrix[i, i]
                    expected_duration = 1.0 / (1.0 - persistence) if persistence < 1.0 else np.inf
                    regime_name = regime_names_map.get(i, f"regime_{i}")
                    expected_durations[regime_name] = expected_duration
            else:
                print(f"Warning: Could not extract transition matrix for {tick}")
                continue

            # Calculate actual signal durations
            signals = signal_history['signal_discrete'].values
            regimes = signal_history['regime_name'].values
            dates = signal_history.index

            # Find signal runs (consecutive same signal)
            signal_runs = []
            current_signal = signals[0]
            run_start_idx = 0
            run_length = 1

            for i in range(1, len(signals)):
                if signals[i] == current_signal:
                    run_length += 1
                else:
                    # End of run
                    signal_runs.append({
                        'signal': current_signal,
                        'start_date': dates[run_start_idx],
                        'end_date': dates[i-1],
                        'length': run_length,
                        'regime_name': regimes[run_start_idx],  # Regime at start of run
                    })
                    current_signal = signals[i]
                    run_start_idx = i
                    run_length = 1

            # Add last run
            signal_runs.append({
                'signal': current_signal,
                'start_date': dates[run_start_idx],
                'end_date': dates[-1],
                'length': run_length,
                'regime_name': regimes[run_start_idx],
            })

            # Compute statistics by regime
            actual_durations = {}
            for regime_name in expected_durations.keys():
                regime_runs = [r for r in signal_runs if r['regime_name'] == regime_name]
                if len(regime_runs) > 0:
                    lengths = [r['length'] for r in regime_runs]
                    actual_durations[regime_name] = {
                        'mean': np.mean(lengths),
                        'median': np.median(lengths),
                        'std': np.std(lengths),
                        'min': np.min(lengths),
                        'max': np.max(lengths),
                        'count': len(regime_runs),
                    }
                else:
                    actual_durations[regime_name] = {
                        'mean': 0,
                        'median': 0,
                        'std': 0,
                        'min': 0,
                        'max': 0,
                        'count': 0,
                    }

            # Compute consistency ratios
            consistency_ratios = {}
            for regime_name in expected_durations.keys():
                expected = expected_durations[regime_name]
                actual = actual_durations[regime_name]['mean']

                if expected > 0 and not np.isinf(expected):
                    ratio = actual / expected
                    consistency_ratios[regime_name] = ratio
                else:
                    consistency_ratios[regime_name] = None

            # Detect rapid flips (<3 days)
            rapid_flips = [r for r in signal_runs if r['length'] < 3]

            # Generate warnings
            warnings_list = []

            # Check consistency ratios
            for regime_name, ratio in consistency_ratios.items():
                if ratio is None:
                    continue

                if ratio < 0.5:
                    warnings_list.append(
                        f"RED FLAG: {regime_name.capitalize()} signals change {ratio:.1%} as fast as training data suggests "
                        f"(expected: {expected_durations[regime_name]:.1f} days, actual: {actual_durations[regime_name]['mean']:.1f} days)"
                    )
                elif ratio < 0.8:
                    warnings_list.append(
                        f"WARNING: {regime_name.capitalize()} signals change faster than expected "
                        f"(consistency ratio: {ratio:.1%})"
                    )
                elif ratio > 2.0:
                    warnings_list.append(
                        f"RED FLAG: {regime_name.capitalize()} signals persist {ratio:.1%} longer than training data suggests "
                        f"(expected: {expected_durations[regime_name]:.1f} days, actual: {actual_durations[regime_name]['mean']:.1f} days)"
                    )
                elif ratio > 1.2:
                    warnings_list.append(
                        f"WARNING: {regime_name.capitalize()} signals persist longer than expected "
                        f"(consistency ratio: {ratio:.1%})"
                    )

            # Check rapid flips
            if len(rapid_flips) > 0:
                flip_pct = len(rapid_flips) / len(signal_runs) * 100
                if flip_pct > 30:
                    warnings_list.append(
                        f"RED FLAG: {flip_pct:.1f}% of signals flip in <3 days ({len(rapid_flips)}/{len(signal_runs)} runs)"
                    )
                elif flip_pct > 15:
                    warnings_list.append(
                        f"WARNING: {flip_pct:.1f}% of signals flip in <3 days ({len(rapid_flips)}/{len(signal_runs)} runs)"
                    )

            results[tick] = {
                'expected_durations': expected_durations,
                'actual_durations': actual_durations,
                'consistency_ratios': consistency_ratios,
                'rapid_flips': rapid_flips,
                'warnings': warnings_list,
                'signal_runs': signal_runs,  # Full list for detailed analysis
            }

        return results

    def print_signal_consistency_report(self, ticker: Optional[str] = None):
        """
        Print formatted report of signal consistency analysis.

        Args:
            ticker: Specific ticker to report on (if None, reports all)
        """
        consistency_results = self.analyze_signal_consistency(ticker=ticker)

        if not consistency_results:
            print("No signal consistency data available.")
            return

        self._print_section("Signal Consistency Analysis")

        for tick, data in consistency_results.items():
            print(f"\n{tick}:")
            print(f"\n  Expected Regime Durations (from HMM transition matrix):")
            for regime_name, duration in data['expected_durations'].items():
                if np.isinf(duration):
                    print(f"    {regime_name.capitalize()}: ∞ (absorbing state)")
                else:
                    print(f"    {regime_name.capitalize()}: {duration:.1f} days")

            print(f"\n  Actual Signal Durations (from generated signals):")
            for regime_name, stats in data['actual_durations'].items():
                if stats['count'] > 0:
                    print(f"    {regime_name.capitalize()}: {stats['mean']:.1f} days "
                          f"(median: {stats['median']:.1f}, range: {stats['min']}-{stats['max']}, n={stats['count']})")
                else:
                    print(f"    {regime_name.capitalize()}: No signals generated")

            print(f"\n  Consistency Ratios (actual/expected):")
            for regime_name, ratio in data['consistency_ratios'].items():
                if ratio is None:
                    print(f"    {regime_name.capitalize()}: N/A")
                elif 0.8 <= ratio <= 1.2:
                    print(f"    {regime_name.capitalize()}: {ratio:.2f} ✓ (good)")
                elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 2.0:
                    print(f"    {regime_name.capitalize()}: {ratio:.2f} ⚠ (warning)")
                else:
                    print(f"    {regime_name.capitalize()}: {ratio:.2f} ❌ (red flag)")

            print(f"\n  Rapid Flips (<3 days): {len(data['rapid_flips'])}/{len(data['signal_runs'])} "
                  f"({len(data['rapid_flips'])/len(data['signal_runs'])*100:.1f}%)")

            if len(data['warnings']) > 0:
                print(f"\n  ⚠ Warnings:")
                for warning in data['warnings']:
                    print(f"    • {warning}")
            else:
                print(f"\n  ✓ All consistency checks passed")

            print()

    def analyze_regime_paradigm_shift(self, ticker: Optional[str] = None) -> Dict:
        """
        Analyze whether the regime transition dynamics during analysis period differ
        significantly from training period (paradigm shift detection).

        This detects when market structure fundamentally changes - not just new regime
        states, but different rules of regime switching itself.

        Args:
            ticker: Specific ticker to analyze (if None, analyzes all)

        Returns:
            Dictionary with paradigm shift analysis:
            {
                'ticker': {
                    'training_transition_matrix': np.ndarray,  # Expected from HMM
                    'empirical_transition_matrix': np.ndarray, # Observed during analysis
                    'divergence_score': float,                 # Overall shift magnitude
                    'regime_persistence_change': {...},        # Per-regime changes
                    'structural_breaks': [...],                # Detected shift dates
                    'volatility_regime_shift': {...},          # Volatility distribution changes
                }
            }
        """
        # Determine which tickers to analyze
        tickers_to_analyze = [ticker] if ticker else self.tickers

        results = {}

        for tick in tickers_to_analyze:
            if tick not in self.results:
                print(f"Warning: No results found for {tick}")
                continue

            if tick not in self.pipelines:
                print(f"Warning: No trained pipeline found for {tick}")
                continue

            pipeline = self.pipelines[tick]
            regime_history = self.results[tick].get('regime_history')

            if regime_history is None or len(regime_history) == 0:
                print(f"Warning: No regime history for {tick}")
                continue

            # Extract training transition matrix from HMM
            if not hasattr(pipeline.model, 'transition_matrix_') or pipeline.model.transition_matrix_ is None:
                print(f"Warning: Could not extract transition matrix for {tick}")
                continue

            training_transition_matrix = pipeline.model.transition_matrix_
            n_states = training_transition_matrix.shape[0]

            # Compute empirical transition matrix from analysis period
            empirical_transition_matrix = self._compute_empirical_transitions(regime_history, n_states)

            # Calculate divergence score (Frobenius norm)
            divergence_score = np.linalg.norm(training_transition_matrix - empirical_transition_matrix, 'fro')

            # Compute regime persistence changes
            regime_persistence_change = {}
            for i in range(n_states):
                # Get regime name
                state_mask = regime_history['regime'] == i
                if state_mask.any():
                    regime_name = regime_history.loc[state_mask, 'regime_name'].iloc[0]
                else:
                    regime_name = f"regime_{i}"

                expected_persistence = training_transition_matrix[i, i]
                actual_persistence = empirical_transition_matrix[i, i]

                expected_duration = 1.0 / (1.0 - expected_persistence) if expected_persistence < 1.0 else np.inf
                actual_duration = 1.0 / (1.0 - actual_persistence) if actual_persistence < 1.0 else np.inf

                ratio = actual_persistence / expected_persistence if expected_persistence > 0 else None

                regime_persistence_change[regime_name] = {
                    'expected_persistence': expected_persistence,
                    'actual_persistence': actual_persistence,
                    'expected_duration': expected_duration,
                    'actual_duration': actual_duration,
                    'ratio': ratio,
                }

            # Detect structural breaks (rolling window analysis)
            structural_breaks = self._detect_structural_breaks(regime_history, training_transition_matrix)

            # Analyze volatility regime shift (if price data available)
            volatility_regime_shift = None
            if tick in self.results and 'full_data' in self.results[tick]:
                full_data = self.results[tick]['full_data']
                volatility_regime_shift = self._analyze_volatility_regime_shift(
                    full_data,
                    self.training_start,
                    self.training_end,
                    self.analysis_start,
                    self.analysis_end
                )

            results[tick] = {
                'training_transition_matrix': training_transition_matrix,
                'empirical_transition_matrix': empirical_transition_matrix,
                'divergence_score': divergence_score,
                'regime_persistence_change': regime_persistence_change,
                'structural_breaks': structural_breaks,
                'volatility_regime_shift': volatility_regime_shift,
            }

        return results

    def _compute_empirical_transitions(self, regime_history: pd.DataFrame, n_states: int) -> np.ndarray:
        """
        Compute empirical transition matrix from observed regime sequence.

        Args:
            regime_history: DataFrame with regime column
            n_states: Number of regime states

        Returns:
            Empirical transition probability matrix
        """
        # Count transitions
        transition_counts = np.zeros((n_states, n_states))

        regimes = regime_history['regime'].values
        for i in range(len(regimes) - 1):
            from_state = int(regimes[i])
            to_state = int(regimes[i + 1])
            transition_counts[from_state, to_state] += 1

        # Normalize to get probabilities
        empirical_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                empirical_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions from this state, assume staying in same state
                empirical_matrix[i, i] = 1.0

        return empirical_matrix

    def _detect_structural_breaks(self, regime_history: pd.DataFrame, training_matrix: np.ndarray, window_size: int = 20) -> List[Dict]:
        """
        Detect dates when transition dynamics diverged significantly from training.

        Uses rolling window to compute local transition probabilities and compare
        to training expectations.

        Args:
            regime_history: DataFrame with regime sequence
            training_matrix: Expected transition matrix from HMM
            window_size: Size of rolling window for local analysis

        Returns:
            List of structural break events with dates and metrics
        """
        structural_breaks = []

        if len(regime_history) < window_size * 2:
            return structural_breaks

        n_states = training_matrix.shape[0]
        regimes = regime_history['regime'].values
        dates = regime_history.index

        # Slide window through data
        for i in range(window_size, len(regimes) - window_size, 5):  # Step by 5 days
            window_regimes = regimes[i-window_size:i+window_size]

            # Compute local empirical transitions
            local_transitions = np.zeros((n_states, n_states))
            for j in range(len(window_regimes) - 1):
                from_state = int(window_regimes[j])
                to_state = int(window_regimes[j + 1])
                local_transitions[from_state, to_state] += 1

            # Normalize
            local_matrix = np.zeros((n_states, n_states))
            for s in range(n_states):
                row_sum = local_transitions[s, :].sum()
                if row_sum > 0:
                    local_matrix[s, :] = local_transitions[s, :] / row_sum

            # Compute divergence
            local_divergence = np.linalg.norm(training_matrix - local_matrix, 'fro')

            # Flag significant divergences (>0.5 is substantial)
            if local_divergence > 0.5:
                structural_breaks.append({
                    'date': dates[i],
                    'divergence': local_divergence,
                    'window_start': dates[max(0, i-window_size)],
                    'window_end': dates[min(len(dates)-1, i+window_size)],
                })

        return structural_breaks

    def _analyze_volatility_regime_shift(
        self,
        full_data: pd.DataFrame,
        training_start: str,
        training_end: str,
        analysis_start: str,
        analysis_end: str
    ) -> Dict:
        """
        Compare volatility distributions between training and analysis periods.

        Args:
            full_data: Complete price data
            training_start: Training period start
            training_end: Training period end
            analysis_start: Analysis period start
            analysis_end: Analysis period end

        Returns:
            Dict with volatility regime shift metrics
        """
        # Compute returns if not already present
        if 'returns' not in full_data.columns:
            full_data['returns'] = full_data['close'].pct_change()

        training_data = full_data[
            (full_data.index >= training_start) & (full_data.index <= training_end)
        ]
        analysis_data = full_data[
            (full_data.index >= analysis_start) & (full_data.index <= analysis_end)
        ]

        training_returns = training_data['returns'].dropna()
        analysis_returns = analysis_data['returns'].dropna()

        # Compute volatility statistics
        training_vol = training_returns.std() * np.sqrt(252)  # Annualized
        analysis_vol = analysis_returns.std() * np.sqrt(252)

        # Test for statistical significance (Kolmogorov-Smirnov test)
        from scipy import stats
        ks_statistic, ks_pvalue = stats.ks_2samp(training_returns, analysis_returns)

        return {
            'training_volatility': training_vol,
            'analysis_volatility': analysis_vol,
            'volatility_ratio': analysis_vol / training_vol if training_vol > 0 else None,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'is_significant': ks_pvalue < 0.05,  # 5% significance level
        }

    def print_regime_paradigm_report(self, ticker: Optional[str] = None):
        """
        Print formatted report of regime paradigm shift analysis.

        Args:
            ticker: Specific ticker to report on (if None, reports all)
        """
        paradigm_results = self.analyze_regime_paradigm_shift(ticker=ticker)

        if not paradigm_results:
            print("No regime paradigm shift data available.")
            return

        self._print_section("Regime Paradigm Shift Analysis")

        for tick, data in paradigm_results.items():
            print(f"\n{tick}:")

            # Training period dynamics
            print(f"\n  Training Period Dynamics ({self.training_start} to {self.training_end}):")
            print(f"    Transition Matrix:")
            training_matrix = data['training_transition_matrix']
            for i, row in enumerate(training_matrix):
                # Get regime name for this state
                regime_name = list(data['regime_persistence_change'].keys())[i] if i < len(data['regime_persistence_change']) else f"State {i}"
                print(f"      {regime_name:12s}: {' '.join([f'{p:.3f}' for p in row])}")

            print(f"\n    Expected Regime Durations:")
            for regime_name, stats in data['regime_persistence_change'].items():
                persistence = stats['expected_persistence']
                duration = stats['expected_duration']
                if np.isinf(duration):
                    print(f"      {regime_name.capitalize():12s}: ∞ days (persistence: {persistence:.1%})")
                else:
                    print(f"      {regime_name.capitalize():12s}: {duration:.1f} days (persistence: {persistence:.1%})")

            # Analysis period dynamics
            print(f"\n  Analysis Period Dynamics ({self.analysis_start} to {self.analysis_end}):")
            print(f"    Empirical Transition Matrix:")
            empirical_matrix = data['empirical_transition_matrix']
            for i, row in enumerate(empirical_matrix):
                regime_name = list(data['regime_persistence_change'].keys())[i] if i < len(data['regime_persistence_change']) else f"State {i}"
                print(f"      {regime_name:12s}: {' '.join([f'{p:.3f}' for p in row])}")

            print(f"\n    Observed Regime Durations:")
            for regime_name, stats in data['regime_persistence_change'].items():
                persistence = stats['actual_persistence']
                duration = stats['actual_duration']
                if np.isinf(duration):
                    print(f"      {regime_name.capitalize():12s}: ∞ days (persistence: {persistence:.1%})")
                else:
                    print(f"      {regime_name.capitalize():12s}: {duration:.1f} days (persistence: {persistence:.1%})")

            # Paradigm shift metrics
            print(f"\n  Paradigm Shift Metrics:")
            divergence = data['divergence_score']
            if divergence > 0.5:
                print(f"    Divergence Score: {divergence:.3f} ❌ (RED FLAG: >0.5 indicates paradigm shift)")
            elif divergence > 0.3:
                print(f"    Divergence Score: {divergence:.3f} ⚠ (WARNING: >0.3 suggests significant change)")
            else:
                print(f"    Divergence Score: {divergence:.3f} ✓ (Dynamics consistent with training)")

            print(f"\n    Regime Persistence Changes:")
            for regime_name, stats in data['regime_persistence_change'].items():
                ratio = stats['ratio']
                if ratio is None:
                    print(f"      {regime_name.capitalize():12s}: N/A")
                elif 0.8 <= ratio <= 1.2:
                    print(f"      {regime_name.capitalize():12s}: {ratio:.2f}x ✓ (consistent)")
                elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 2.0:
                    change_pct = (ratio - 1.0) * 100
                    print(f"      {regime_name.capitalize():12s}: {ratio:.2f}x ⚠ ({change_pct:+.0f}% change)")
                else:
                    change_pct = (ratio - 1.0) * 100
                    print(f"      {regime_name.capitalize():12s}: {ratio:.2f}x ❌ ({change_pct:+.0f}% change - EXTREME)")

            # Structural breaks
            if data['structural_breaks']:
                print(f"\n    Structural Breaks Detected: {len(data['structural_breaks'])}")
                # Show top 3 most significant breaks
                sorted_breaks = sorted(data['structural_breaks'], key=lambda x: x['divergence'], reverse=True)
                for i, brk in enumerate(sorted_breaks[:3]):
                    date_str = brk['date'].strftime('%Y-%m-%d')
                    # Check if date matches any key events
                    event_label = ""
                    if self.key_events:
                        for event_date, event_name in self.key_events.items():
                            event_dt = pd.to_datetime(event_date)
                            if abs((brk['date'] - event_dt).days) <= 3:  # Within 3 days
                                event_label = f" ({event_name})"
                                break
                    print(f"      {i+1}. {date_str}{event_label}: divergence = {brk['divergence']:.3f}")
            else:
                print(f"\n    Structural Breaks Detected: 0 (stable transition dynamics)")

            # Volatility regime shift
            if data['volatility_regime_shift']:
                vol_data = data['volatility_regime_shift']
                print(f"\n    Volatility Regime Shift:")
                print(f"      Training volatility: {vol_data['training_volatility']:.1%} (annualized)")
                print(f"      Analysis volatility: {vol_data['analysis_volatility']:.1%} (annualized)")
                ratio = vol_data['volatility_ratio']
                if ratio:
                    change = (ratio - 1.0) * 100
                    significance = " (statistically significant)" if vol_data['is_significant'] else ""
                    if ratio > 2.0:
                        print(f"      Volatility ratio: {ratio:.2f}x ❌ ({change:+.0f}% increase - EXTREME){significance}")
                    elif ratio > 1.5:
                        print(f"      Volatility ratio: {ratio:.2f}x ⚠ ({change:+.0f}% increase){significance}")
                    elif ratio < 0.67:
                        print(f"      Volatility ratio: {ratio:.2f}x ⚠ ({change:+.0f}% decrease){significance}")
                    else:
                        print(f"      Volatility ratio: {ratio:.2f}x ✓ ({change:+.0f}% change){significance}")
                print(f"      KS test p-value: {vol_data['ks_pvalue']:.4f}")

            # Generate warnings
            warnings_list = []

            # Check divergence
            if divergence > 0.5:
                warnings_list.append(
                    f"PARADIGM SHIFT DETECTED: Transition dynamics diverged significantly from training (score: {divergence:.3f})"
                )

            # Check regime persistence
            for regime_name, stats in data['regime_persistence_change'].items():
                ratio = stats['ratio']
                if ratio and ratio > 2.0:
                    warnings_list.append(
                        f"{regime_name.capitalize()} regime became {ratio:.1f}x more persistent than training suggests"
                    )
                elif ratio and ratio < 0.5:
                    warnings_list.append(
                        f"{regime_name.capitalize()} regime became {1/ratio:.1f}x less persistent (rapid switching)"
                    )

            # Check volatility
            if data['volatility_regime_shift'] and data['volatility_regime_shift']['is_significant']:
                vol_ratio = data['volatility_regime_shift']['volatility_ratio']
                if vol_ratio > 1.5:
                    warnings_list.append(
                        f"Volatility increased {vol_ratio:.1f}x vs training (different risk regime)"
                    )

            if warnings_list:
                print(f"\n  ⚠ Warnings:")
                for warning in warnings_list:
                    print(f"    • {warning}")
            else:
                print(f"\n  ✓ No significant paradigm shift detected - market dynamics consistent with training")

            print()

    def create_paradigm_shift_visualization(self, ticker: Optional[str] = None, save: bool = True) -> Optional[str]:
        """
        Create comprehensive visualization of regime paradigm shift analysis.

        Generates a 2x3 subplot figure showing:
        1. Training transition matrix heatmap
        2. Empirical transition matrix heatmap
        3. Divergence matrix (difference)
        4. Regime duration comparison bar chart
        5. Regime persistence comparison
        6. Volatility distribution comparison

        Args:
            ticker: Specific ticker to visualize (if None, visualizes all)
            save: Whether to save figure to file (default: True)

        Returns:
            Path to saved figure, or None if not saved
        """
        paradigm_results = self.analyze_regime_paradigm_shift(ticker=ticker)

        if not paradigm_results:
            print("No paradigm shift data available for visualization.")
            return None

        for tick, data in paradigm_results.items():
            # Create figure with 2x3 subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{tick} - Regime Paradigm Shift Analysis\n'
                        f'Training: {self.training_start} to {self.training_end} | '
                        f'Analysis: {self.analysis_start} to {self.analysis_end}',
                        fontsize=16, fontweight='bold')

            # Get regime names
            regime_names = list(data['regime_persistence_change'].keys())
            n_states = len(regime_names)

            # 1. Training transition matrix heatmap (top-left)
            training_matrix = data['training_transition_matrix']
            im1 = axes[0, 0].imshow(training_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[0, 0].set_title('Training Period Transition Matrix\n(Expected Dynamics)', fontweight='bold')
            axes[0, 0].set_xlabel('To Regime')
            axes[0, 0].set_ylabel('From Regime')
            axes[0, 0].set_xticks(range(n_states))
            axes[0, 0].set_yticks(range(n_states))
            axes[0, 0].set_xticklabels([r.capitalize() for r in regime_names], rotation=45, ha='right')
            axes[0, 0].set_yticklabels([r.capitalize() for r in regime_names])

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    text = axes[0, 0].text(j, i, f'{training_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="white" if training_matrix[i, j] > 0.5 else "black",
                                          fontsize=10, fontweight='bold')

            plt.colorbar(im1, ax=axes[0, 0], label='Transition Probability')

            # 2. Empirical transition matrix heatmap (top-middle)
            empirical_matrix = data['empirical_transition_matrix']
            im2 = axes[0, 1].imshow(empirical_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[0, 1].set_title('Analysis Period Transition Matrix\n(Observed Dynamics)', fontweight='bold')
            axes[0, 1].set_xlabel('To Regime')
            axes[0, 1].set_ylabel('From Regime')
            axes[0, 1].set_xticks(range(n_states))
            axes[0, 1].set_yticks(range(n_states))
            axes[0, 1].set_xticklabels([r.capitalize() for r in regime_names], rotation=45, ha='right')
            axes[0, 1].set_yticklabels([r.capitalize() for r in regime_names])

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    text = axes[0, 1].text(j, i, f'{empirical_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="white" if empirical_matrix[i, j] > 0.5 else "black",
                                          fontsize=10, fontweight='bold')

            plt.colorbar(im2, ax=axes[0, 1], label='Transition Probability')

            # 3. Divergence matrix (difference) - top-right
            divergence_matrix = empirical_matrix - training_matrix
            max_abs_div = np.abs(divergence_matrix).max()
            im3 = axes[0, 2].imshow(divergence_matrix, cmap='RdBu_r', aspect='auto',
                                   vmin=-max_abs_div, vmax=max_abs_div)
            axes[0, 2].set_title(f'Transition Matrix Divergence\n(Divergence Score: {data["divergence_score"]:.3f})',
                                fontweight='bold')
            axes[0, 2].set_xlabel('To Regime')
            axes[0, 2].set_ylabel('From Regime')
            axes[0, 2].set_xticks(range(n_states))
            axes[0, 2].set_yticks(range(n_states))
            axes[0, 2].set_xticklabels([r.capitalize() for r in regime_names], rotation=45, ha='right')
            axes[0, 2].set_yticklabels([r.capitalize() for r in regime_names])

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    color = "white" if abs(divergence_matrix[i, j]) > max_abs_div * 0.5 else "black"
                    text = axes[0, 2].text(j, i, f'{divergence_matrix[i, j]:+.2f}',
                                          ha="center", va="center", color=color,
                                          fontsize=10, fontweight='bold')

            plt.colorbar(im3, ax=axes[0, 2], label='Difference (Empirical - Training)')

            # 4. Regime duration comparison (bottom-left)
            expected_durations = [stats['expected_duration'] for stats in data['regime_persistence_change'].values()]
            actual_durations = [stats['actual_duration'] for stats in data['regime_persistence_change'].values()]

            # Filter out infinite durations for plotting
            expected_durations_plot = [min(d, 50) if not np.isinf(d) else 50 for d in expected_durations]
            actual_durations_plot = [min(d, 50) if not np.isinf(d) else 50 for d in actual_durations]

            x = np.arange(len(regime_names))
            width = 0.35

            bars1 = axes[1, 0].bar(x - width/2, expected_durations_plot, width,
                                  label='Training (Expected)', color='#0173B2', alpha=0.8)
            bars2 = axes[1, 0].bar(x + width/2, actual_durations_plot, width,
                                  label='Analysis (Observed)', color='#DE8F05', alpha=0.8)

            axes[1, 0].set_title('Regime Duration Comparison', fontweight='bold')
            axes[1, 0].set_xlabel('Regime')
            axes[1, 0].set_ylabel('Average Duration (days)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([r.capitalize() for r in regime_names])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}',
                                   ha='center', va='bottom', fontsize=9)

            # 5. Regime persistence comparison (bottom-middle)
            expected_persistence = [stats['expected_persistence'] * 100 for stats in data['regime_persistence_change'].values()]
            actual_persistence = [stats['actual_persistence'] * 100 for stats in data['regime_persistence_change'].values()]

            bars1 = axes[1, 1].bar(x - width/2, expected_persistence, width,
                                  label='Training (Expected)', color='#0173B2', alpha=0.8)
            bars2 = axes[1, 1].bar(x + width/2, actual_persistence, width,
                                  label='Analysis (Observed)', color='#DE8F05', alpha=0.8)

            axes[1, 1].set_title('Regime Persistence Comparison', fontweight='bold')
            axes[1, 1].set_xlabel('Regime')
            axes[1, 1].set_ylabel('Persistence (%)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([r.capitalize() for r in regime_names])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].set_ylim(0, 100)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.0f}%',
                                   ha='center', va='bottom', fontsize=9)

            # 6. Volatility distribution comparison (bottom-right)
            if data['volatility_regime_shift']:
                vol_data = data['volatility_regime_shift']

                # Get actual return distributions for histogram
                full_data = self.results[tick]['full_data']
                if 'returns' not in full_data.columns:
                    full_data['returns'] = full_data['close'].pct_change()

                training_data = full_data[
                    (full_data.index >= self.training_start) & (full_data.index <= self.training_end)
                ]
                analysis_data = full_data[
                    (full_data.index >= self.analysis_start) & (full_data.index <= self.analysis_end)
                ]

                training_returns = training_data['returns'].dropna() * 100  # Convert to percentage
                analysis_returns = analysis_data['returns'].dropna() * 100

                # Plot histograms
                axes[1, 2].hist(training_returns, bins=50, alpha=0.6, label='Training',
                               color='#0173B2', density=True, edgecolor='black', linewidth=0.5)
                axes[1, 2].hist(analysis_returns, bins=50, alpha=0.6, label='Analysis',
                               color='#DE8F05', density=True, edgecolor='black', linewidth=0.5)

                axes[1, 2].set_title('Return Distribution Comparison', fontweight='bold')
                axes[1, 2].set_xlabel('Daily Returns (%)')
                axes[1, 2].set_ylabel('Density')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3, axis='y')

                # Add volatility stats as text
                textstr = f"Training Vol: {vol_data['training_volatility']:.1%}\n"
                textstr += f"Analysis Vol: {vol_data['analysis_volatility']:.1%}\n"
                textstr += f"Ratio: {vol_data['volatility_ratio']:.2f}x\n"
                textstr += f"KS p-value: {vol_data['ks_pvalue']:.4f}"

                # Add text box
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                axes[1, 2].text(0.98, 0.97, textstr, transform=axes[1, 2].transAxes,
                               fontsize=10, verticalalignment='top', horizontalalignment='right',
                               bbox=props)
            else:
                axes[1, 2].text(0.5, 0.5, 'Volatility data not available',
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Return Distribution Comparison', fontweight='bold')

            plt.tight_layout()

            # Save figure
            if save:
                filename = f"{tick}_paradigm_shift_analysis.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved paradigm shift visualization: {filename}")
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None

        return None

    def create_full_timeline_visualization(self, save: bool = True) -> Optional[str]:
        """
        Create full-period regime timeline visualization showing all tickers/assets.

        Displays the entire analysis period with colored background regions indicating
        HMM-detected regimes, price line overlay, and vertical event markers.

        Similar to the 2008_timeline.png reference image.

        Args:
            save: Whether to save the figure to disk

        Returns:
            Path to saved PNG file, or None if no tickers to visualize
        """
        if not self.tickers or not self.results:
            print("No results available for timeline visualization")
            return None

        # Prepare figure with subplots (one per ticker)
        n_tickers = len(self.tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(16, 3.5 * n_tickers))

        # Handle single ticker case (axes won't be an array)
        if n_tickers == 1:
            axes = [axes]

        # Color mapping for regime names
        regime_color_map = {
            'bullish': '#4575b4',      # Blue
            'bearish': '#d73027',      # Red
            'sideways': '#fee08b',     # Yellow/Gold
            'crisis': '#a50026',       # Dark Red
        }

        # Track y-axis limits across all subplots
        all_prices = []
        for ticker in self.tickers:
            if ticker in self.results:
                history = self.results[ticker].get('regime_history')
                if history is not None and 'price' in history.columns:
                    all_prices.extend(history['price'].values)

        if all_prices:
            price_min = np.min(all_prices)
            price_max = np.max(all_prices)
            y_lower = price_min * 0.9
            y_upper = price_max * 1.1
        else:
            y_lower, y_upper = None, None

        # Plot each ticker
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.results:
                continue

            analysis_results = self.results[ticker]
            history = analysis_results.get('regime_history')

            if history is None or len(history) == 0:
                continue

            ax = axes[idx]

            # Plot price line
            ax.plot(
                history.index,
                history['price'],
                'k-',
                linewidth=1.5,
                label='Price',
                zorder=3
            )

            # Color background by regime
            current_regime = None
            regime_start = None

            for date in history.index:
                regime_name = history.loc[date, 'regime_name'].lower()

                # Start new regime region if needed
                if regime_name != current_regime:
                    if regime_start is not None and current_regime is not None:
                        # End previous regime region
                        color = regime_color_map.get(current_regime, '#cccccc')
                        ax.axvspan(regime_start, date, color=color, alpha=0.15, zorder=1)

                    current_regime = regime_name
                    regime_start = date

            # Handle last regime region
            if regime_start is not None and current_regime is not None:
                color = regime_color_map.get(current_regime, '#cccccc')
                ax.axvspan(regime_start, history.index[-1] + pd.Timedelta(days=1),
                          color=color, alpha=0.15, zorder=1)

            # Add vertical lines at key events with labels
            for event_date_str, event_name in self.key_events.items():
                event_date = pd.to_datetime(event_date_str)
                if history.index[0] <= event_date <= history.index[-1]:
                    ax.axvline(event_date, color='red', linestyle='--', linewidth=1.5,
                             alpha=0.6, zorder=2)

                    # Add event name label on the line (simple red text, no box)
                    y_pos = ax.get_ylim()[1] * 0.90  # Position near top
                    ax.text(event_date, y_pos, event_name, rotation=90,
                           verticalalignment='top', horizontalalignment='right',
                           color='red', fontsize=8)

            # Format subplot
            if y_lower is not None and y_upper is not None:
                ax.set_ylim(y_lower, y_upper)

            # Add ticker label on the left side
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(history.index[0], y_pos, ticker, fontsize=11,
                   fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='black', alpha=0.8))

            # Set y-axis label with ticker and price info
            ax.set_ylabel(f'{ticker} Price ($)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, zorder=0)

        # Format x-axis (date labels)
        fig.text(0.5, 0.02, 'Date', ha='center', fontsize=12, fontweight='bold')

        # Add legend for regimes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4575b4', alpha=0.15, label='Bullish'),
            Patch(facecolor='#fee08b', alpha=0.15, label='Sideways'),
            Patch(facecolor='#d73027', alpha=0.15, label='Bearish'),
            Patch(facecolor='#a50026', alpha=0.15, label='Crisis'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
                  fontsize=10, title='Regime Types')

        # Add title
        fig.suptitle(
            f'{", ".join(self.tickers)} - HMM-Detected Regime Timeline\n'
            f'Colored regions indicate regimes | Dashed lines mark key events',
            fontsize=14, fontweight='bold', y=0.995
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.99])

        # Save figure
        if save:
            filename = f"full_timeline_visualization.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            print(f"\nSaved full-period timeline visualization: {filename}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def _print_section(self, title: str):
        """Print formatted section header."""
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
