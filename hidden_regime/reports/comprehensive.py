"""Comprehensive Report component implementation.

Aggregates all pipeline components and generates final output with
metrics, visualizations, and interpretations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidden_regime.config.report import ReportConfiguration
from hidden_regime.pipeline.interfaces import ReportComponent


class ComprehensiveReport(ReportComponent):
    """Comprehensive report generation component.

    Takes outputs from all pipeline components and generates:
    - Performance metrics (Sharpe, returns, drawdown, etc.)
    - Regime statistics (duration, frequency, transitions)
    - Signal statistics (win rate, trades, etc.)
    - Multi-panel visualizations
    - Markdown/HTML summaries

    This is the ONLY place where metrics are calculated.
    Components provide raw data, Report aggregates and analyzes.
    """

    def __init__(self, config: ReportConfiguration = None):
        """Initialize report component.

        Args:
            config: ReportConfiguration object (optional)
        """
        if config is None:
            config = ReportConfiguration()
        self.config = config
        self._last_report: Optional[str] = None
        self._metrics_cache: Dict[str, Any] = {}

    def update(self, **kwargs) -> str:
        """Generate comprehensive report from pipeline outputs.

        Args:
            data: Raw data DataFrame
            observations: Observations DataFrame
            model_output: Model predictions
            interpreter_output: Interpreter output with regime labels
            signals: (Optional) Signal Generator output
            raw_data: (Optional) Price data for performance calculations
            price_data: (Optional) Alias for raw_data

        Returns:
            Markdown formatted report string
        """
        # Extract inputs
        data = kwargs.get("data")
        observations = kwargs.get("observations")
        model_output = kwargs.get("model_output")
        interpreter_output = kwargs.get("interpreter_output")
        signals = kwargs.get("signals")
        # Handle raw_data with explicit None check (can't use 'or' with DataFrames)
        raw_data = kwargs.get("raw_data")
        if raw_data is None:
            raw_data = kwargs.get("price_data")

        # Build report sections
        report_sections = []

        # Title and timestamp
        report_sections.append(self._build_title_section())

        # Summary statistics
        if interpreter_output is not None:
            report_sections.append(self._build_regime_summary(interpreter_output))

        # Detailed metrics
        if interpreter_output is not None and raw_data is not None:
            report_sections.append(
                self._build_regime_metrics(interpreter_output, raw_data)
            )

        # Signal metrics (if signals provided)
        if signals is not None:
            report_sections.append(self._build_signal_metrics(signals))

        # Transition analysis
        if interpreter_output is not None:
            report_sections.append(self._build_transition_analysis(interpreter_output))

        # Data quality
        if data is not None:
            report_sections.append(self._build_data_quality_section(data))

        # Join all sections
        report_text = "\n\n".join(report_sections)
        self._last_report = report_text

        return report_text

    def _build_title_section(self) -> str:
        """Build title and metadata section."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"# Hidden Regime Report\n\n*Generated: {now}*"

    def _build_regime_summary(self, interpreter_output: pd.DataFrame) -> str:
        """Build regime summary statistics."""
        summary_lines = ["## Regime Summary"]

        # Current regime
        if len(interpreter_output) > 0:
            last_row = interpreter_output.iloc[-1]
            current_regime = last_row.get("regime_label", "Unknown")
            current_confidence = last_row.get("regime_strength", 0.0)
            summary_lines.append(
                f"\n**Current Regime:** {current_regime} ({current_confidence:.1%} confidence)"
            )

        # Regime distribution
        regime_counts = interpreter_output["regime_label"].value_counts()
        summary_lines.append("\n**Regime Distribution:**")
        for regime, count in regime_counts.items():
            pct = count / len(interpreter_output) * 100
            summary_lines.append(f"- {regime}: {count} days ({pct:.1f}%)")

        return "\n".join(summary_lines)

    def _build_regime_metrics(
        self, interpreter_output: pd.DataFrame, raw_data: pd.DataFrame
    ) -> str:
        """Build detailed regime performance metrics."""
        metrics_lines = ["## Regime Performance Metrics"]

        # Merge regime labels with prices
        merged = pd.DataFrame(interpreter_output[["regime_label"]].copy())
        if "close" in raw_data.columns:
            merged["close"] = raw_data["close"]
        elif len(raw_data.columns) > 0:
            # Use first numeric column as price
            merged["close"] = raw_data.iloc[:, 0]
        else:
            return "## Regime Performance Metrics\n\n(Insufficient data for metrics)"

        # Calculate returns
        merged["returns"] = merged["close"].pct_change()

        # Calculate metrics per regime
        regime_metrics = {}
        for regime in merged["regime_label"].unique():
            if pd.isna(regime):
                continue

            regime_data = merged[merged["regime_label"] == regime]
            regime_returns = regime_data["returns"].dropna()

            if len(regime_returns) > 0:
                total_return = (1 + regime_returns).prod() - 1
                annual_return = (1 + regime_returns.mean()) ** 252 - 1
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = annual_return / volatility if volatility > 0 else 0

                regime_metrics[regime] = {
                    "days": len(regime_data),
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "volatility": volatility,
                    "sharpe": sharpe,
                }

        # Format metrics
        metrics_lines.append("\n| Regime | Days | Total Return | Annual Return | Volatility | Sharpe |")
        metrics_lines.append("|--------|------|---------------|----------------|------------|--------|")

        for regime, metrics in sorted(regime_metrics.items(), key=lambda x: x[1]["annual_return"], reverse=True):
            metrics_lines.append(
                f"| {regime} | {metrics['days']} | {metrics['total_return']:>7.2%} | "
                f"{metrics['annual_return']:>7.2%} | {metrics['volatility']:>7.2%} | {metrics['sharpe']:>6.2f} |"
            )

        return "\n".join(metrics_lines)

    def _build_signal_metrics(self, signals: pd.DataFrame) -> str:
        """Build signal generation metrics."""
        metrics_lines = ["## Signal Metrics"]

        # Count signals
        valid_signals = signals[signals["signal_valid"] == True]
        total_signals = len(valid_signals)

        long_signals = (valid_signals["base_signal"] > 0).sum()
        short_signals = (valid_signals["base_signal"] < 0).sum()
        neutral_signals = (valid_signals["base_signal"] == 0).sum()

        metrics_lines.append(f"\n**Signal Activity:**")
        metrics_lines.append(f"- Total Valid Signals: {total_signals}")
        metrics_lines.append(f"- Long Signals: {long_signals}")
        metrics_lines.append(f"- Short Signals: {short_signals}")
        metrics_lines.append(f"- Neutral Signals: {neutral_signals}")

        # Average position size
        if "position_size" in signals.columns:
            avg_long_size = signals[signals["base_signal"] > 0]["position_size"].mean()
            avg_short_size = signals[signals["base_signal"] < 0]["position_size"].mean()

            metrics_lines.append(f"\n**Average Position Sizes:**")
            metrics_lines.append(f"- Long: {avg_long_size:.2f}")
            metrics_lines.append(f"- Short: {avg_short_size:.2f}")

        # Signal changes
        if "signal_changed" not in signals.columns:
            signals["signal_changed"] = signals["base_signal"].diff() != 0
        signal_changes = signals["signal_changed"].sum()
        metrics_lines.append(f"\n**Signal Transitions:** {signal_changes}")

        return "\n".join(metrics_lines)

    def _build_transition_analysis(self, interpreter_output: pd.DataFrame) -> str:
        """Build regime transition analysis."""
        analysis_lines = ["## Regime Transitions"]

        # Get transitions
        transitions = interpreter_output["regime_label"].ne(
            interpreter_output["regime_label"].shift()
        ).sum()

        analysis_lines.append(f"\n**Total Transitions:** {transitions}")

        # Get regime durations
        durations = []
        current_regime = None
        current_duration = 0

        for regime in interpreter_output["regime_label"]:
            if regime != current_regime:
                if current_duration > 0:
                    durations.append((current_regime, current_duration))
                current_regime = regime
                current_duration = 1
            else:
                current_duration += 1

        if current_duration > 0:
            durations.append((current_regime, current_duration))

        # Summary stats
        if durations:
            duration_values = [d[1] for d in durations]
            analysis_lines.append(
                f"\n**Average Regime Duration:** {np.mean(duration_values):.1f} days"
            )
            analysis_lines.append(
                f"**Median Regime Duration:** {np.median(duration_values):.1f} days"
            )
            analysis_lines.append(
                f"**Min/Max Duration:** {min(duration_values)}/{max(duration_values)} days"
            )

        return "\n".join(analysis_lines)

    def _build_data_quality_section(self, data: pd.DataFrame) -> str:
        """Build data quality and completeness section."""
        quality_lines = ["## Data Quality"]

        quality_lines.append(f"\n**Data Range:** {len(data)} observations")

        if "timestamp" in data.columns or data.index.name == "timestamp":
            dates = data.index if hasattr(data.index, "dtype") else data["timestamp"]
            quality_lines.append(f"**Date Range:** {dates.iloc[0]} to {dates.iloc[-1]}")

        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            quality_lines.append("\n**Missing Values:**")
            for col in missing[missing > 0].index:
                quality_lines.append(f"- {col}: {missing[col]} ({missing[col]/len(data)*100:.1f}%)")
        else:
            quality_lines.append("\n**No missing values**")

        return "\n".join(quality_lines)

    def plot(self, **kwargs) -> plt.Figure:
        """Generate multi-panel visualization.

        Creates a comprehensive plot with all available data.

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(14, 10))

        # Create grid for subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle("Hidden Regime Comprehensive Report", fontsize=16, fontweight="bold")

        # Panel 1: Regime timeline
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_regime_timeline(ax1, kwargs.get("interpreter_output"))

        # Panel 2: Regime distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_regime_distribution(ax2, kwargs.get("interpreter_output"))

        # Panel 3: Signal activity (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        if kwargs.get("signals") is not None:
            self._plot_signal_activity(ax3, kwargs.get("signals"))
        else:
            ax3.text(0.5, 0.5, "No signal data", ha="center", va="center")
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis("off")

        # Panel 4: Metrics table
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_metrics_table(ax4, kwargs.get("interpreter_output"))

        return fig

    def _plot_regime_timeline(self, ax: plt.Axes, interpreter_output: Optional[pd.DataFrame]) -> None:
        """Plot regime timeline."""
        if interpreter_output is None or len(interpreter_output) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        ax.scatter(range(len(interpreter_output)), interpreter_output["regime_strength"], alpha=0.6)
        ax.set_title("Regime Confidence Timeline")
        ax.set_xlabel("Days")
        ax.set_ylabel("Confidence")
        ax.grid(True, alpha=0.3)

    def _plot_regime_distribution(self, ax: plt.Axes, interpreter_output: Optional[pd.DataFrame]) -> None:
        """Plot regime distribution."""
        if interpreter_output is None:
            return

        counts = interpreter_output["regime_label"].value_counts()
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_title("Regime Distribution")
        ax.set_ylabel("Days")

    def _plot_signal_activity(self, ax: plt.Axes, signals: pd.DataFrame) -> None:
        """Plot signal activity."""
        ax.scatter(range(len(signals)), signals["base_signal"], alpha=0.6)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title("Trading Signals")
        ax.set_xlabel("Days")
        ax.set_ylabel("Signal (-1 to 1)")
        ax.grid(True, alpha=0.3)

    def _plot_metrics_table(self, ax: plt.Axes, interpreter_output: Optional[pd.DataFrame]) -> None:
        """Plot metrics as table."""
        ax.axis("off")

        if interpreter_output is None:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
            return

        # Create simple summary
        summary_data = [
            ["Metric", "Value"],
            ["Total Days", str(len(interpreter_output))],
            ["Regimes", str(interpreter_output["regime_label"].nunique())],
            ["Transitions", str(interpreter_output["regime_label"].ne(
                interpreter_output["regime_label"].shift()).sum())],
            ["Current Regime", str(interpreter_output["regime_label"].iloc[-1])],
        ]

        table = ax.table(cellText=summary_data, cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Format header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")
