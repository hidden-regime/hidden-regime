"""
Performance Analysis Tools for Retail Trading

Comprehensive analysis and visualization tools for evaluating regime-based
trading strategies from a retail trader perspective.

Key Features:
- Trading performance metrics and visualizations
- Regime transition analysis and timing
- Risk-adjusted return calculations
- Benchmark comparisons (buy-and-hold, market indices)
- Interactive charts for retail trader insights

Author: Hidden Regime Team
Created: 2025-01-XX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add hidden regime to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hidden_regime.visualization.plotting import (
    setup_financial_plot_style,
    REGIME_COLORS,
    get_regime_colors,
)


class RetailPerformanceAnalyzer:
    """
    Comprehensive performance analysis focused on retail trader needs.

    Provides clear, actionable insights about regime-based trading performance
    with visualizations and metrics that retail traders care about most.
    """

    def __init__(self):
        self.setup_styling()

    def setup_styling(self):
        """Setup consistent styling for all plots."""
        setup_financial_plot_style()
        plt.style.use("seaborn-v0_8-whitegrid")

    def create_performance_dashboard(
        self,
        performance_history: List[Dict],
        regime_history: List[Dict],
        trade_history: List[Dict],
        benchmark_data: pd.DataFrame = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard for retail traders.

        Args:
            performance_history: Portfolio performance over time
            regime_history: Regime predictions over time
            trade_history: Individual trade details
            benchmark_data: Optional benchmark comparison data
            save_path: Optional path to save the dashboard

        Returns:
            Figure object with dashboard
        """
        # Convert to DataFrames
        perf_df = pd.DataFrame(performance_history)
        regime_df = pd.DataFrame(regime_history) if regime_history else pd.DataFrame()

        if len(perf_df) == 0:
            raise ValueError("No performance data available")

        # Create dashboard figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_portfolio_value(ax1, perf_df, benchmark_data)

        # 2. Performance Summary Box
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_performance_summary(ax2, perf_df, benchmark_data)

        # 3. Regime Timeline
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_regime_timeline(ax3, regime_df, perf_df)

        # 4. Daily Returns Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax4, perf_df)

        # 5. Drawdown Analysis
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_drawdown_analysis(ax5, perf_df)

        # 6. Trading Activity
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_trading_activity(ax6, trade_history, regime_df)

        # 7. Regime Performance
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_regime_performance(ax7, perf_df, regime_df)

        # 8. Rolling Metrics
        ax8 = fig.add_subplot(gs[3, 1:])
        self._plot_rolling_metrics(ax8, perf_df)

        # Add title
        fig.suptitle(
            "Hidden Regime Trading Strategy - Performance Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_portfolio_value(
        self, ax, perf_df: pd.DataFrame, benchmark_data: pd.DataFrame = None
    ):
        """Plot portfolio value over time with benchmark comparison."""

        dates = pd.to_datetime(perf_df["timestamp"])
        portfolio_values = perf_df["total_capital"].values
        initial_value = portfolio_values[0]

        # Normalize to starting value for percentage comparison
        portfolio_pct = (portfolio_values / initial_value - 1) * 100

        # Plot strategy performance
        ax.plot(
            dates,
            portfolio_pct,
            linewidth=2.5,
            label="Hidden Regime Strategy",
            color="#2E8B57",
            alpha=0.9,
        )

        # Plot benchmark if available
        if benchmark_data is not None and "price" in benchmark_data.columns:
            benchmark_dates = pd.to_datetime(benchmark_data["date"])
            benchmark_prices = benchmark_data["price"].values

            # Align dates and normalize
            aligned_benchmark = []
            for date in dates:
                closest_idx = np.argmin(np.abs((benchmark_dates - date).days))
                aligned_benchmark.append(benchmark_prices[closest_idx])

            benchmark_pct = (
                np.array(aligned_benchmark) / aligned_benchmark[0] - 1
            ) * 100
            ax.plot(
                dates,
                benchmark_pct,
                linewidth=2,
                label="Buy & Hold",
                color="#CD853F",
                alpha=0.7,
                linestyle="--",
            )

        ax.set_title("Portfolio Performance Comparison", fontweight="bold", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))

    def _plot_performance_summary(
        self, ax, perf_df: pd.DataFrame, benchmark_data: pd.DataFrame = None
    ):
        """Create performance summary text box."""

        # Calculate key metrics
        returns = perf_df["daily_return"].values
        total_return = perf_df["total_return"].iloc[-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = perf_df["drawdown"].max()

        # Format summary text
        summary_text = f"""PERFORMANCE SUMMARY
        
Total Return: {total_return:.1%}
Annualized Return: {(1 + total_return) ** (252 / len(returns)) - 1:.1%}
Volatility: {volatility:.1%}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.1%}

Days Traded: {len(perf_df)}
Final Capital: ${perf_df['total_capital'].iloc[-1]:,.0f}
Peak Capital: ${perf_df['total_capital'].max():,.0f}"""

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def _plot_regime_timeline(self, ax, regime_df: pd.DataFrame, perf_df: pd.DataFrame):
        """Plot regime predictions over time with portfolio performance."""

        if len(regime_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No regime data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        dates = pd.to_datetime(regime_df["timestamp"])
        regimes = regime_df["regime"].values
        confidences = regime_df["confidence"].values

        # Get regime colors
        regime_colors = get_regime_colors(list(set(regimes)))

        # Create regime timeline
        unique_regimes = list(set(regimes))
        regime_to_num = {regime: i for i, regime in enumerate(unique_regimes)}
        regime_nums = [regime_to_num[regime] for regime in regimes]

        # Plot regime blocks with confidence-based alpha
        for i in range(len(dates) - 1):
            regime = regimes[i]
            confidence = confidences[i]
            alpha = 0.3 + 0.5 * confidence  # Alpha based on confidence

            ax.axvspan(
                dates[i],
                dates[i + 1],
                ymin=0.7,
                ymax=1.0,
                color=regime_colors.get(regime, "#808080"),
                alpha=alpha,
                label=regime if i == 0 or regimes[i] != regimes[i - 1] else "",
            )

        # Plot portfolio returns on secondary axis
        ax2 = ax.twinx()
        portfolio_returns = (
            perf_df["daily_return"].values * 100
        )  # Convert to percentage
        dates_perf = pd.to_datetime(perf_df["timestamp"])

        ax2.plot(dates_perf, portfolio_returns, color="black", alpha=0.6, linewidth=1)
        ax2.set_ylabel("Daily Return (%)", fontsize=10)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Format main axis
        ax.set_ylim(0.7, 1.0)
        ax.set_ylabel("Market Regime", fontsize=10)
        ax.set_yticks([0.75, 0.85, 0.95])
        ax.set_yticklabels(["Regime", "Timeline", ""])
        ax.set_title("Regime Detection Timeline with Daily Returns", fontweight="bold")

        # Add legend for regimes
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=regime_colors.get(regime, "#808080"), alpha=0.7
            )
            for regime in unique_regimes
        ]
        ax.legend(handles, unique_regimes, loc="upper left", bbox_to_anchor=(0, 0.65))

    def _plot_returns_distribution(self, ax, perf_df: pd.DataFrame):
        """Plot distribution of daily returns."""

        returns = perf_df["daily_return"].values * 100  # Convert to percentage

        # Create histogram
        ax.hist(
            returns,
            bins=30,
            density=True,
            alpha=0.7,
            color="lightblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add normal distribution overlay
        mu, sigma = np.mean(returns), np.std(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
        ax.plot(
            x, normal_dist, "r-", linewidth=2, alpha=0.8, label="Normal Distribution"
        )

        # Add statistics
        ax.axvline(
            mu, color="blue", linestyle="--", alpha=0.8, label=f"Mean: {mu:.2f}%"
        )
        ax.axvline(
            mu + 2 * sigma,
            color="red",
            linestyle=":",
            alpha=0.8,
            label=f"+2σ: {mu + 2*sigma:.2f}%",
        )
        ax.axvline(
            mu - 2 * sigma,
            color="red",
            linestyle=":",
            alpha=0.8,
            label=f"-2σ: {mu - 2*sigma:.2f}%",
        )

        ax.set_title("Daily Returns Distribution", fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_drawdown_analysis(self, ax, perf_df: pd.DataFrame):
        """Plot drawdown analysis over time."""

        dates = pd.to_datetime(perf_df["timestamp"])
        drawdowns = perf_df["drawdown"].values * 100  # Convert to percentage

        # Plot drawdown
        ax.fill_between(dates, 0, -drawdowns, color="red", alpha=0.3, label="Drawdown")
        ax.plot(dates, -drawdowns, color="red", linewidth=1.5)

        # Highlight maximum drawdown
        max_dd_idx = np.argmax(drawdowns)
        max_dd_date = (
            dates.iloc[max_dd_idx] if hasattr(dates, "iloc") else dates[max_dd_idx]
        )
        max_dd_value = drawdowns[max_dd_idx]

        ax.scatter(
            [max_dd_date],
            [-max_dd_value],
            color="red",
            s=100,
            zorder=5,
            label=f"Max Drawdown: {max_dd_value:.1f}%",
        )

        ax.set_title("Drawdown Analysis", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(-max_dd_value * 1.2, -1), 1)

    def _plot_trading_activity(
        self, ax, trade_history: List[Dict], regime_df: pd.DataFrame
    ):
        """Plot trading activity by regime."""

        if not trade_history:
            ax.text(
                0.5,
                0.5,
                "No trading activity",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        # Count trades by regime
        regime_trades = {}
        for trade in trade_history:
            if trade.get("executed", False):
                regime = trade.get("regime", "Unknown")
                regime_trades[regime] = regime_trades.get(regime, 0) + 1

        if not regime_trades:
            ax.text(
                0.5,
                0.5,
                "No executed trades",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        # Create bar chart
        regimes = list(regime_trades.keys())
        trade_counts = list(regime_trades.values())

        regime_colors = get_regime_colors(regimes)
        colors = [regime_colors.get(regime, "#808080") for regime in regimes]

        bars = ax.bar(regimes, trade_counts, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar, count in zip(bars, trade_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Trading Activity by Regime", fontweight="bold")
        ax.set_xlabel("Market Regime")
        ax.set_ylabel("Number of Trades")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        if len(regimes) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_regime_performance(
        self, ax, perf_df: pd.DataFrame, regime_df: pd.DataFrame
    ):
        """Plot performance by market regime."""

        if len(regime_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No regime data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        # Merge performance and regime data
        perf_df_copy = perf_df.copy()
        regime_df_copy = regime_df.copy()

        perf_df_copy["timestamp"] = pd.to_datetime(perf_df_copy["timestamp"])
        regime_df_copy["timestamp"] = pd.to_datetime(regime_df_copy["timestamp"])

        # Align data (simple approach - match by nearest timestamp)
        aligned_data = []
        for _, perf_row in perf_df_copy.iterrows():
            # Find closest regime prediction
            time_diffs = np.abs(
                (regime_df_copy["timestamp"] - perf_row["timestamp"]).dt.total_seconds()
            )
            closest_regime_idx = np.argmin(time_diffs)

            regime_info = regime_df_copy.iloc[closest_regime_idx]
            aligned_data.append(
                {
                    "regime": regime_info["regime"],
                    "daily_return": perf_row["daily_return"]
                    * 100,  # Convert to percentage
                }
            )

        aligned_df = pd.DataFrame(aligned_data)

        # Calculate average returns by regime
        regime_returns = (
            aligned_df.groupby("regime")["daily_return"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        if len(regime_returns) == 0:
            ax.text(
                0.5,
                0.5,
                "No regime performance data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        # Create bar chart with error bars
        regimes = regime_returns["regime"].values
        mean_returns = regime_returns["mean"].values
        std_returns = regime_returns["std"].values

        regime_colors = get_regime_colors(regimes)
        colors = [regime_colors.get(regime, "#808080") for regime in regimes]

        bars = ax.bar(
            regimes,
            mean_returns,
            color=colors,
            alpha=0.7,
            yerr=std_returns,
            capsize=5,
            edgecolor="black",
        )

        # Add value labels
        for bar, mean_ret in zip(bars, mean_returns):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 if height >= 0 else height - 0.05,
                f"{mean_ret:.2f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontweight="bold",
            )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_title("Average Daily Return by Regime", fontweight="bold")
        ax.set_xlabel("Market Regime")
        ax.set_ylabel("Average Daily Return (%)")
        ax.grid(True, alpha=0.3, axis="y")

        if len(regimes) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_rolling_metrics(self, ax, perf_df: pd.DataFrame):
        """Plot rolling performance metrics."""

        dates = pd.to_datetime(perf_df["timestamp"])
        returns = perf_df["daily_return"].values

        # Calculate rolling metrics (30-day window)
        window = min(30, len(returns) // 3)  # Adaptive window size
        if window < 5:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for rolling metrics",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        rolling_sharpe = []
        rolling_volatility = []

        for i in range(window, len(returns)):
            window_returns = returns[i - window : i]
            sharpe = self._calculate_sharpe_ratio(window_returns)
            vol = np.std(window_returns) * np.sqrt(252) * 100  # Annualized %

            rolling_sharpe.append(sharpe)
            rolling_volatility.append(vol)

        rolling_dates = dates[window:]

        # Plot rolling Sharpe ratio
        ax.plot(
            rolling_dates,
            rolling_sharpe,
            color="blue",
            linewidth=2,
            label=f"{window}-Day Rolling Sharpe Ratio",
        )
        ax.set_ylabel("Sharpe Ratio", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")

        # Plot rolling volatility on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            rolling_dates,
            rolling_volatility,
            color="red",
            linewidth=2,
            label=f"{window}-Day Rolling Volatility",
        )
        ax2.set_ylabel("Volatility (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        ax.set_title(
            f"Rolling Performance Metrics ({window}-Day Window)", fontweight="bold"
        )
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        # Add legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def create_regime_transition_analysis(
        self, regime_history: List[Dict], performance_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze regime transition patterns and their impact on performance.

        Returns:
            Dictionary with transition analysis results
        """
        if not regime_history:
            return {"error": "No regime history available"}

        regime_df = pd.DataFrame(regime_history)
        perf_df = pd.DataFrame(performance_history)

        # Identify regime changes
        transitions = []
        for i in range(1, len(regime_df)):
            if regime_df.iloc[i]["regime"] != regime_df.iloc[i - 1]["regime"]:
                transition = {
                    "timestamp": regime_df.iloc[i]["timestamp"],
                    "from_regime": regime_df.iloc[i - 1]["regime"],
                    "to_regime": regime_df.iloc[i]["regime"],
                    "confidence_before": regime_df.iloc[i - 1]["confidence"],
                    "confidence_after": regime_df.iloc[i]["confidence"],
                }
                transitions.append(transition)

        # Analyze transition performance
        transition_performance = []
        for transition in transitions:
            trans_time = pd.to_datetime(transition["timestamp"])

            # Find corresponding performance data
            perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
            after_transition = perf_df[perf_df["timestamp"] >= trans_time]

            if len(after_transition) >= 5:  # Need at least 5 days after transition
                next_5_days = after_transition.head(5)["daily_return"]
                performance = {
                    "transition": f"{transition['from_regime']} → {transition['to_regime']}",
                    "timestamp": trans_time,
                    "5_day_return": next_5_days.sum(),
                    "confidence_change": transition["confidence_after"]
                    - transition["confidence_before"],
                }
                transition_performance.append(performance)

        return {
            "total_transitions": len(transitions),
            "transition_list": transitions,
            "transition_performance": transition_performance,
            "avg_confidence_change": (
                np.mean([t["confidence_change"] for t in transitions])
                if transitions
                else 0
            ),
        }

    def generate_retail_trader_report(
        self,
        performance_history: List[Dict],
        regime_history: List[Dict],
        trade_history: List[Dict],
        benchmark_data: pd.DataFrame = None,
        asset_name: str = "Asset",
    ) -> str:
        """
        Generate comprehensive report for retail traders.

        Returns:
            HTML report string
        """

        # Calculate key metrics
        perf_df = pd.DataFrame(performance_history)

        if len(perf_df) == 0:
            return "<h1>No Performance Data Available</h1>"

        returns = perf_df["daily_return"].values
        total_return = perf_df["total_return"].iloc[-1]
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = perf_df["drawdown"].max()

        # Regime analysis
        regime_analysis = ""
        if regime_history:
            regime_df = pd.DataFrame(regime_history)
            regime_dist = regime_df["regime"].value_counts(normalize=True)
            regime_analysis = "<h3>Regime Distribution</h3><ul>"
            for regime, pct in regime_dist.items():
                regime_analysis += (
                    f"<li><strong>{regime}</strong>: {pct:.1%} of time</li>"
                )
            regime_analysis += "</ul>"

        # Trading activity
        trading_summary = ""
        if trade_history:
            executed_trades = [t for t in trade_history if t.get("executed", False)]
            trading_summary = (
                f"<p><strong>Total Trades:</strong> {len(executed_trades)}</p>"
            )

        # Create HTML report
        html_report = f"""
        <html>
        <head>
            <title>Hidden Regime Trading Report - {asset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                h1, h2, h3 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>Hidden Regime Trading Report</h1>
            <h2>Asset: {asset_name}</h2>
            
            <h3>Performance Summary</h3>
            <div class="metric">
                <strong>Total Return:</strong> 
                <span class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</span>
            </div>
            <div class="metric">
                <strong>Annualized Volatility:</strong> {volatility:.1%}
            </div>
            <div class="metric">
                <strong>Sharpe Ratio:</strong> {sharpe_ratio:.2f}
            </div>
            <div class="metric">
                <strong>Maximum Drawdown:</strong> 
                <span class="negative">{max_drawdown:.1%}</span>
            </div>
            <div class="metric">
                <strong>Trading Period:</strong> {len(perf_df)} days
            </div>
            
            {regime_analysis}
            
            <h3>Trading Activity</h3>
            {trading_summary}
            
            <h3>Risk Assessment</h3>
            <p>This strategy showed {'strong' if sharpe_ratio > 1.0 else 'moderate' if sharpe_ratio > 0.5 else 'weak'} 
            risk-adjusted performance with a Sharpe ratio of {sharpe_ratio:.2f}.</p>
            
            <h3>Recommendations</h3>
            <ul>
                <li>{'Consider increasing position sizes' if total_return > 0 and max_drawdown < 0.15 else 'Consider reducing position sizes'}</li>
                <li>{'Risk management appears effective' if max_drawdown < 0.2 else 'Review risk management parameters'}</li>
                <li>{'Strategy shows promise for live trading' if sharpe_ratio > 0.8 and total_return > 0 else 'Strategy needs optimization before live trading'}</li>
            </ul>
        </body>
        </html>
        """

        return html_report


# Utility functions for retail analysis
def calculate_benchmark_comparison(
    strategy_returns: np.ndarray, benchmark_returns: np.ndarray
) -> Dict[str, float]:
    """Calculate strategy vs benchmark comparison metrics."""

    # Align arrays if different lengths
    min_length = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_length]
    benchmark_returns = benchmark_returns[:min_length]

    # Calculate metrics
    strategy_total = np.prod(1 + strategy_returns) - 1
    benchmark_total = np.prod(1 + benchmark_returns) - 1

    strategy_vol = np.std(strategy_returns) * np.sqrt(252)
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)

    # Calculate tracking error and information ratio
    active_returns = strategy_returns - benchmark_returns
    tracking_error = np.std(active_returns) * np.sqrt(252)
    information_ratio = (
        np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        if np.std(active_returns) > 0
        else 0
    )

    return {
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "excess_return": strategy_total - benchmark_total,
        "strategy_volatility": strategy_vol,
        "benchmark_volatility": benchmark_vol,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }


def create_retail_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table for retail trader comparison."""

    summary_data = []
    for strategy_name, metrics in results.items():
        if "error" not in metrics:
            summary_data.append(
                {
                    "Strategy": strategy_name,
                    "Total Return": f"{metrics.get('total_return', 0):.1%}",
                    "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                    "Max Drawdown": f"{metrics.get('max_drawdown', 0):.1%}",
                    "Win Rate": f"{metrics.get('win_rate', 0):.1%}",
                    "Trades": metrics.get("num_trades", 0),
                }
            )

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("Hidden Regime Performance Analysis Tools")
    print("=" * 50)
    print("Tools ready for retail trading analysis!")

    # Example usage
    analyzer = RetailPerformanceAnalyzer()
    print("✅ RetailPerformanceAnalyzer initialized")
    print("✅ Ready for performance dashboard creation")
    print("✅ Ready for regime transition analysis")
    print("✅ Ready for retail trader report generation")
