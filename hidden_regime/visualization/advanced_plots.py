"""
Advanced plotting classes and functions for sophisticated regime analysis.

Provides object-oriented plotting interfaces and complex visualizations
for comprehensive regime analysis and performance comparison.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from plotly.subplots import make_subplots

from .plotting import (
    create_regime_legend,
    format_financial_axis,
    get_regime_colors,
    get_regime_names,
    setup_financial_plot_style,
)


class RegimePlotter:
    """
    Advanced plotter for regime analysis with customizable styles and layouts.
    """

    def __init__(self, color_scheme: str = "professional", style: str = "professional"):
        """
        Initialize regime plotter.

        Args:
            color_scheme: Color scheme for regimes
            style: Plot style configuration
        """
        self.color_scheme = color_scheme
        self.style = style
        setup_financial_plot_style(style)

    def plot_regime_evolution(
        self,
        data: pd.DataFrame,
        regime_data: pd.DataFrame,
        price_column: str = "close",
        regime_column: str = "predicted_state",
        confidence_column: str = "confidence",
        window_size: int = 50,
        title: str = "Regime Evolution Analysis",
    ) -> plt.Figure:
        """
        Create evolving view of regime detection with sliding window analysis.

        Args:
            data: Price data
            regime_data: Regime predictions
            price_column: Price column name
            regime_column: Regime column name
            confidence_column: Confidence column name
            window_size: Size of analysis window
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Align data
        aligned_data = data.join(
            regime_data[[regime_column, confidence_column]], how="inner"
        )

        n_states = int(aligned_data[regime_column].max()) + 1
        regime_colors = get_regime_colors(n_states, self.color_scheme)
        regime_names = get_regime_names(n_states)

        # Plot 1: Price with regime backgrounds
        axes[0].plot(
            aligned_data.index,
            aligned_data[price_column],
            color="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Add regime background with varying transparency based on confidence
        for regime in range(n_states):
            regime_mask = aligned_data[regime_column] == regime
            if regime_mask.sum() > 0:
                regime_dates = aligned_data.index[regime_mask]
                confidence_values = aligned_data.loc[regime_mask, confidence_column]

                for date, confidence in zip(regime_dates, confidence_values):
                    alpha = 0.1 + 0.3 * confidence  # Scale alpha with confidence
                    axes[0].axvspan(
                        date,
                        date + pd.Timedelta(days=1),
                        color=regime_colors[regime],
                        alpha=alpha,
                    )

        axes[0].set_title(f"{title} - Price with Regime Confidence")
        axes[0].set_ylabel(f"Price ({price_column})")
        axes[0].grid(True, alpha=0.3)
        create_regime_legend(regime_colors, regime_names, axes[0])

        # Plot 2: Rolling regime distribution
        rolling_regime_dist = {}
        for regime in range(n_states):
            regime_series = (aligned_data[regime_column] == regime).astype(int)
            rolling_regime_dist[regime] = regime_series.rolling(
                window=window_size
            ).mean()

        for regime, rolling_dist in rolling_regime_dist.items():
            axes[1].plot(
                aligned_data.index,
                rolling_dist,
                color=regime_colors[regime],
                linewidth=2,
                label=f"{regime_names[regime]} ({window_size}d avg)",
            )

        axes[1].set_title(f"Rolling Regime Distribution ({window_size}-day window)")
        axes[1].set_ylabel("Regime Probability")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Confidence evolution
        rolling_confidence = (
            aligned_data[confidence_column].rolling(window=window_size).mean()
        )
        axes[2].plot(
            aligned_data.index,
            rolling_confidence,
            color="purple",
            linewidth=2,
            label=f"Rolling Confidence ({window_size}d)",
        )
        axes[2].fill_between(
            aligned_data.index, 0, rolling_confidence, alpha=0.3, color="purple"
        )

        axes[2].set_title("Model Confidence Evolution")
        axes[2].set_ylabel("Average Confidence")
        axes[2].set_xlabel("Date")
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Format x-axis
        format_financial_axis(axes[2])

        plt.tight_layout()
        return fig

    def plot_regime_characteristics_matrix(
        self,
        data: pd.DataFrame,
        regime_data: pd.DataFrame,
        metrics: List[str] = None,
        title: str = "Regime Characteristics Matrix",
    ) -> plt.Figure:
        """
        Create matrix visualization of regime characteristics.

        Args:
            data: Price data
            regime_data: Regime predictions
            metrics: List of metrics to analyze
            title: Plot title

        Returns:
            matplotlib Figure
        """
        if metrics is None:
            metrics = [
                "mean_return",
                "volatility",
                "max_return",
                "min_return",
                "skewness",
            ]

        # Align data and calculate returns
        aligned_data = data.join(regime_data["predicted_state"], how="inner")
        aligned_data["returns"] = aligned_data["close"].pct_change()

        n_states = int(aligned_data["predicted_state"].max()) + 1
        regime_names = get_regime_names(n_states)

        # Calculate metrics for each regime
        regime_metrics = {}
        for regime in range(n_states):
            regime_mask = aligned_data["predicted_state"] == regime
            regime_returns = aligned_data.loc[regime_mask, "returns"].dropna()

            if len(regime_returns) > 0:
                regime_metrics[regime] = {
                    "mean_return": regime_returns.mean() * 252,  # Annualized
                    "volatility": regime_returns.std() * np.sqrt(252),  # Annualized
                    "max_return": regime_returns.max(),
                    "min_return": regime_returns.min(),
                    "skewness": regime_returns.skew(),
                    "kurtosis": regime_returns.kurtosis(),
                    "var_95": regime_returns.quantile(0.05),
                    "var_99": regime_returns.quantile(0.01),
                }

        # Create matrix
        if regime_metrics:
            metrics_matrix = np.zeros((len(metrics), n_states))
            for i, metric in enumerate(metrics):
                for regime in range(n_states):
                    if regime in regime_metrics:
                        metrics_matrix[i, regime] = regime_metrics[regime].get(
                            metric, 0
                        )

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Normalize each metric for better visualization
            normalized_matrix = np.zeros_like(metrics_matrix)
            for i in range(len(metrics)):
                row = metrics_matrix[i, :]
                if row.max() != row.min():
                    normalized_matrix[i, :] = (row - row.min()) / (
                        row.max() - row.min()
                    )

            im = ax.imshow(normalized_matrix, cmap="RdYlGn", aspect="auto")

            # Set ticks and labels
            ax.set_xticks(range(n_states))
            ax.set_xticklabels(regime_names)
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)

            # Add text annotations with actual values
            for i in range(len(metrics)):
                for j in range(n_states):
                    if j < len(regime_metrics):
                        value = metrics_matrix[i, j]
                        text = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color="black" if normalized_matrix[i, j] > 0.5 else "white",
                        )

            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="Normalized Value")
            plt.tight_layout()
            return fig

        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No regime data available", ha="center", va="center")
            return fig


class PerformancePlotter:
    """
    Specialized plotter for performance analysis and comparison.
    """

    def __init__(self, color_scheme: str = "professional"):
        """Initialize performance plotter."""
        self.color_scheme = color_scheme

    def plot_strategy_comparison(
        self,
        strategies: Dict[str, pd.Series],
        benchmark: pd.Series = None,
        title: str = "Strategy Performance Comparison",
    ) -> plt.Figure:
        """
        Compare multiple strategies with comprehensive metrics.

        Args:
            strategies: Dictionary of strategy name -> returns series
            benchmark: Benchmark returns series
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Calculate cumulative returns
        cumulative_returns = {}
        for name, returns in strategies.items():
            cumulative_returns[name] = (1 + returns).cumprod()

        if benchmark is not None:
            cumulative_returns["Benchmark"] = (1 + benchmark).cumprod()

        # Plot 1: Cumulative returns
        for name, cum_ret in cumulative_returns.items():
            axes[0].plot(cum_ret.index, cum_ret.values, linewidth=2, label=name)

        axes[0].set_title("Cumulative Returns")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        format_financial_axis(axes[0])

        # Plot 2: Rolling Sharpe ratio
        window = 252  # 1 year
        for name, returns in strategies.items():
            if len(returns) >= window:
                rolling_sharpe = (
                    returns.rolling(window).mean()
                    / returns.rolling(window).std()
                    * np.sqrt(252)
                )
                axes[1].plot(returns.index, rolling_sharpe, linewidth=2, label=name)

        axes[1].set_title(f"Rolling Sharpe Ratio ({window} days)")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        format_financial_axis(axes[1])

        # Plot 3: Return distribution
        all_returns = []
        labels = []
        for name, returns in strategies.items():
            all_returns.append(returns.dropna().values)
            labels.append(name)

        axes[2].hist(all_returns, bins=50, alpha=0.7, label=labels, density=True)
        axes[2].set_title("Return Distribution")
        axes[2].set_xlabel("Daily Return")
        axes[2].set_ylabel("Density")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Risk-return scatter
        risk_return_data = {}
        for name, returns in strategies.items():
            clean_returns = returns.dropna()
            if len(clean_returns) > 0:
                annual_return = clean_returns.mean() * 252
                annual_vol = clean_returns.std() * np.sqrt(252)
                risk_return_data[name] = (annual_vol, annual_return)

        if risk_return_data:
            vols = [data[0] for data in risk_return_data.values()]
            rets = [data[1] for data in risk_return_data.values()]
            names = list(risk_return_data.keys())

            scatter = axes[3].scatter(vols, rets, s=100, alpha=0.7)

            for i, name in enumerate(names):
                axes[3].annotate(
                    name, (vols[i], rets[i]), xytext=(5, 5), textcoords="offset points"
                )

            axes[3].set_title("Risk-Return Profile")
            axes[3].set_xlabel("Annual Volatility")
            axes[3].set_ylabel("Annual Return")
            axes[3].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_drawdown_analysis(
        self, returns: pd.Series, title: str = "Drawdown Analysis"
    ) -> plt.Figure:
        """
        Detailed drawdown analysis plot.

        Args:
            returns: Returns series
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Calculate cumulative returns and drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Plot 1: Cumulative returns with drawdown periods
        axes[0].plot(
            cumulative.index,
            cumulative.values,
            linewidth=2,
            color="blue",
            label="Cumulative Return",
        )
        axes[0].plot(
            running_max.index,
            running_max.values,
            linewidth=1,
            color="green",
            alpha=0.7,
            label="High Water Mark",
        )

        # Shade drawdown periods
        axes[0].fill_between(
            cumulative.index,
            cumulative.values,
            running_max.values,
            where=(cumulative < running_max),
            alpha=0.3,
            color="red",
            label="Drawdown",
        )

        axes[0].set_title("Cumulative Returns with Drawdowns")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Drawdown series
        axes[1].fill_between(drawdown.index, 0, drawdown.values, alpha=0.7, color="red")
        axes[1].plot(drawdown.index, drawdown.values, linewidth=1, color="darkred")

        axes[1].set_title("Drawdown Over Time")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(drawdown.min() * 1.1, 0.01)

        # Plot 3: Drawdown distribution
        drawdown_values = drawdown.values[drawdown.values < 0]
        if len(drawdown_values) > 0:
            axes[2].hist(
                drawdown_values, bins=30, alpha=0.7, color="red", edgecolor="black"
            )
            axes[2].axvline(
                drawdown.min(),
                color="darkred",
                linestyle="--",
                label=f"Max DD: {drawdown.min():.2%}",
            )
            axes[2].axvline(
                np.percentile(drawdown_values, 5),
                color="orange",
                linestyle="--",
                label=f"5th percentile: {np.percentile(drawdown_values, 5):.2%}",
            )

        axes[2].set_title("Drawdown Distribution")
        axes[2].set_xlabel("Drawdown")
        axes[2].set_ylabel("Frequency")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Format x-axis
        format_financial_axis(axes[1])
        format_financial_axis(axes[2])

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig


class ComparisonPlotter:
    """
    Plotter for comparing different models and strategies.
    """

    def __init__(self):
        """Initialize comparison plotter."""
        pass

    def plot_model_comparison(
        self,
        models_data: Dict[str, Dict[str, pd.DataFrame]],
        title: str = "Model Comparison Analysis",
    ) -> plt.Figure:
        """
        Compare multiple models across different metrics.

        Args:
            models_data: Dictionary of model_name -> {data, regime_data, performance}
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        model_names = list(models_data.keys())
        n_models = len(model_names)

        if n_models == 0:
            axes[0].text(0.5, 0.5, "No model data provided", ha="center", va="center")
            return fig

        # Colors for different models
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        # Plot 1: Regime detection comparison (example with regime counts)
        regime_distributions = {}
        for model_name, model_data in models_data.items():
            if "regime_data" in model_data:
                regime_data = model_data["regime_data"]
                if "predicted_state" in regime_data.columns:
                    regime_counts = (
                        regime_data["predicted_state"].value_counts().sort_index()
                    )
                    regime_distributions[model_name] = regime_counts

        if regime_distributions:
            width = 0.8 / n_models
            x_positions = np.arange(
                max(len(dist) for dist in regime_distributions.values())
            )

            for i, (model_name, distribution) in enumerate(
                regime_distributions.items()
            ):
                x_pos = x_positions[: len(distribution)] + i * width
                axes[0].bar(
                    x_pos,
                    distribution.values,
                    width,
                    alpha=0.7,
                    color=colors[i],
                    label=model_name,
                )

            axes[0].set_title("Regime Distribution Comparison")
            axes[0].set_xlabel("Regime")
            axes[0].set_ylabel("Number of Days")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Performance metrics comparison
        if all("performance" in model_data for model_data in models_data.values()):
            metrics = [
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
            ]
            x_pos = np.arange(len(metrics))
            width = 0.8 / n_models

            for i, (model_name, model_data) in enumerate(models_data.items()):
                performance = model_data.get("performance", {})
                values = [performance.get(metric, 0) for metric in metrics]

                axes[1].bar(
                    x_pos + i * width,
                    values,
                    width,
                    alpha=0.7,
                    color=colors[i],
                    label=model_name,
                )

            axes[1].set_title("Performance Metrics Comparison")
            axes[1].set_xlabel("Metric")
            axes[1].set_ylabel("Value")
            axes[1].set_xticks(x_pos + width * (n_models - 1) / 2)
            axes[1].set_xticklabels(metrics, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Cumulative performance comparison
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if "performance_series" in model_data:
                cumulative = (1 + model_data["performance_series"]).cumprod()
                axes[2].plot(
                    cumulative.index,
                    cumulative.values,
                    linewidth=2,
                    color=colors[i],
                    label=model_name,
                )

        axes[2].set_title("Cumulative Performance Comparison")
        axes[2].set_ylabel("Cumulative Return")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        format_financial_axis(axes[2])

        # Plot 4: Model confidence comparison
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if (
                "regime_data" in model_data
                and "confidence" in model_data["regime_data"].columns
            ):
                confidence = model_data["regime_data"]["confidence"]
                axes[3].hist(
                    confidence.values,
                    bins=20,
                    alpha=0.5,
                    color=colors[i],
                    label=f"{model_name} (μ={confidence.mean():.2f})",
                )

        axes[3].set_title("Model Confidence Comparison")
        axes[3].set_xlabel("Confidence Score")
        axes[3].set_ylabel("Frequency")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig


class InteractivePlotter:
    """
    Interactive plotting using Plotly for web-based visualizations.
    """

    def __init__(self):
        """Initialize interactive plotter."""
        pass

    def create_interactive_regime_plot(
        self,
        data: pd.DataFrame,
        regime_data: pd.DataFrame,
        price_column: str = "close",
        regime_column: str = "predicted_state",
        confidence_column: str = "confidence",
    ) -> go.Figure:
        """
        Create interactive plot with price and regime data.

        Args:
            data: Price data
            regime_data: Regime predictions
            price_column: Price column name
            regime_column: Regime column name
            confidence_column: Confidence column name

        Returns:
            Plotly Figure
        """
        # Align data
        aligned_data = data.join(
            regime_data[[regime_column, confidence_column]], how="inner"
        )

        n_states = int(aligned_data[regime_column].max()) + 1
        regime_colors_dict = {0: "red", 1: "orange", 2: "green", 3: "blue", 4: "purple"}
        regime_names = get_regime_names(n_states)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxis=True,
            subplot_titles=("Price with Regimes", "Regime Timeline", "Confidence"),
            vertical_spacing=0.05,
        )

        # Plot 1: Price line
        fig.add_trace(
            go.Scatter(
                x=aligned_data.index,
                y=aligned_data[price_column],
                mode="lines",
                name="Price",
                line=dict(color="black", width=2),
            ),
            row=1,
            col=1,
        )

        # Plot 2: Regime timeline
        for regime in range(n_states):
            regime_mask = aligned_data[regime_column] == regime
            if regime_mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=aligned_data.index[regime_mask],
                        y=[regime] * regime_mask.sum(),
                        mode="markers",
                        name=regime_names[regime],
                        marker=dict(
                            color=regime_colors_dict.get(regime, "gray"),
                            size=8,
                            opacity=0.8,
                        ),
                    ),
                    row=2,
                    col=1,
                )

        # Plot 3: Confidence
        fig.add_trace(
            go.Scatter(
                x=aligned_data.index,
                y=aligned_data[confidence_column],
                mode="lines",
                name="Confidence",
                line=dict(color="purple", width=2),
                fill="tonexty",
            ),
            row=3,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title="Interactive Regime Analysis", showlegend=True, height=800
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig


def create_regime_timeline_plot(
    regime_data: pd.DataFrame,
    regime_column: str = "predicted_state",
    confidence_column: str = "confidence",
    title: str = "Regime Timeline with Transitions",
    color_scheme: str = "professional",
) -> plt.Figure:
    """
    Create detailed timeline plot showing regime transitions and durations.

    Args:
        regime_data: DataFrame with regime predictions
        regime_column: Column name for regime predictions
        confidence_column: Column name for confidence scores
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    n_states = int(regime_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Plot 1: Regime timeline with transitions
    current_regime = regime_data[regime_column].iloc[0]
    start_date = regime_data.index[0]

    for i in range(1, len(regime_data)):
        if (
            regime_data[regime_column].iloc[i] != current_regime
            or i == len(regime_data) - 1
        ):
            end_date = (
                regime_data.index[i - 1]
                if i < len(regime_data) - 1
                else regime_data.index[i]
            )

            # Draw regime period
            duration = (end_date - start_date).days
            axes[0].barh(
                current_regime,
                duration,
                left=start_date,
                height=0.8,
                color=regime_colors[current_regime],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Add duration text
            mid_date = start_date + (end_date - start_date) / 2
            axes[0].text(
                mid_date,
                current_regime,
                f"{duration}d",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

            # Update for next period
            if i < len(regime_data):
                current_regime = regime_data[regime_column].iloc[i]
                start_date = regime_data.index[i]

    axes[0].set_yticks(range(n_states))
    axes[0].set_yticklabels(regime_names)
    axes[0].set_title(f"{title} - Regime Periods and Durations")
    axes[0].set_ylabel("Market Regime")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Plot 2: Confidence evolution
    axes[1].plot(
        regime_data.index,
        regime_data[confidence_column],
        color="purple",
        linewidth=2,
        alpha=0.8,
    )
    axes[1].fill_between(
        regime_data.index, 0, regime_data[confidence_column], alpha=0.3, color="purple"
    )

    # Add regime change markers
    regime_changes = regime_data[regime_column].diff() != 0
    change_dates = regime_data.index[regime_changes]
    change_confidences = regime_data.loc[regime_changes, confidence_column]

    axes[1].scatter(
        change_dates,
        change_confidences,
        color="red",
        s=50,
        alpha=0.8,
        label="Regime Changes",
        zorder=5,
    )

    axes[1].set_title("Model Confidence with Regime Change Points")
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Format x-axis
    format_financial_axis(axes[1])

    plt.tight_layout()
    return fig


def create_multi_asset_regime_comparison(
    multi_asset_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    title: str = "Multi-Asset Regime Comparison",
) -> plt.Figure:
    """
    Compare regime detection across multiple assets.

    Args:
        multi_asset_data: Dictionary of asset_name -> (price_data, regime_data)
        title: Plot title

    Returns:
        matplotlib Figure
    """
    n_assets = len(multi_asset_data)
    fig, axes = plt.subplots(n_assets, 1, figsize=(15, 4 * n_assets), sharex=True)

    if n_assets == 1:
        axes = [axes]

    asset_names = list(multi_asset_data.keys())

    for i, (asset_name, (price_data, regime_data)) in enumerate(
        multi_asset_data.items()
    ):
        ax = axes[i]

        # Align data
        aligned_data = price_data.join(regime_data["predicted_state"], how="inner")

        # Plot price
        ax.plot(
            aligned_data.index,
            aligned_data["close"],
            color="black",
            linewidth=1.5,
            alpha=0.8,
            label="Price",
        )

        # Add regime backgrounds
        n_states = int(aligned_data["predicted_state"].max()) + 1
        regime_colors = get_regime_colors(n_states, self.color_scheme)
        regime_names = get_regime_names(n_states)

        for regime in range(n_states):
            regime_mask = aligned_data["predicted_state"] == regime
            if regime_mask.sum() > 0:
                regime_dates = aligned_data.index[regime_mask]

                for date in regime_dates:
                    ax.axvspan(
                        date,
                        date + pd.Timedelta(days=1),
                        color=regime_colors[regime],
                        alpha=0.2,
                    )

        ax.set_title(f"{asset_name} - Regime Detection")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)

        # Add regime legend only for the first subplot
        if i == 0:
            create_regime_legend(regime_colors, regime_names, ax)

    # Format x-axis for bottom plot
    format_financial_axis(axes[-1])
    axes[-1].set_xlabel("Date")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig
