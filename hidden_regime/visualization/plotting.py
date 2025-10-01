"""
Core plotting functions for hidden-regime visualizations.

Provides consistent plotting functionality and styling across all components
including data loaders, HMM models, state standardizers, and financial analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Define regime color schemes
REGIME_COLORS = {
    "classic": ["#d32f2f", "#f57c00", "#388e3c"],  # Red, Orange, Green
    "professional": ["#c62828", "#ff8f00", "#2e7d32"],  # Darker variants
    "academic": ["#b71c1c", "#e65100", "#1b5e20"],  # Even darker for papers
    "pastel": ["#ffcdd2", "#ffe0b2", "#c8e6c9"],  # Light variants for backgrounds
    # Colorblind-friendly alternatives (deuteranopia/protanopia safe)
    "colorblind_safe": ["#d73027", "#fee08b", "#4575b4"],  # Red, Yellow, Blue
    "high_contrast": ["#c51b7d", "#fde68a", "#2166ac"],  # Magenta, Light Yellow, Blue
    "blue_red": ["#b2182b", "#f7f7f7", "#2166ac"],  # Red, Gray, Blue
    "viridis": [
        "#440154",
        "#21908c",
        "#fde725",
    ],  # Purple, Teal, Yellow (viridis-inspired)
    "okabe_ito": [
        "#e69f00",
        "#56b4e9",
        "#009e73",
    ],  # Orange, Sky Blue, Green (Okabe-Ito safe)
}

REGIME_NAMES_MAP = {
    2: ["Bear", "Bull"],
    3: ["Bear", "Sideways", "Bull"],
    4: ["Crisis", "Bear", "Sideways", "Bull"],
    5: ["Crisis", "Bear", "Sideways", "Bull", "Strong Bull"],
    6: ["Deep Crisis", "Crisis", "Bear", "Sideways", "Bull", "Strong Bull"],
}


def setup_financial_plot_style(style: str = "professional") -> None:
    """
    Set up consistent plotting style for financial charts.

    Args:
        style: Style name ('professional', 'academic', 'presentation')
    """
    plt.style.use("default")

    if style == "professional":
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "grid.alpha": 0.3,
                "axes.grid": True,
                "figure.facecolor": "white",
            }
        )
    elif style == "academic":
        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "grid.alpha": 0.2,
                "axes.grid": True,
                "figure.facecolor": "white",
            }
        )
    elif style == "presentation":
        plt.rcParams.update(
            {
                "figure.figsize": (14, 10),
                "font.size": 14,
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 14,
                "grid.alpha": 0.4,
                "axes.grid": True,
                "figure.facecolor": "white",
            }
        )


def get_regime_colors(
    n_states: int,
    color_scheme: str = "professional",
    regime_profiles: Optional[Dict[int, Any]] = None,
) -> List[str]:
    """
    Get appropriate colors for regime visualization.

    Args:
        n_states: Number of regime states
        color_scheme: Color scheme name
        regime_profiles: Optional RegimeProfile objects from financial characterization

    Returns:
        List of color strings for each regime
    """
    # Use financial characterization if available
    if regime_profiles:
        from ..utils.regime_mapping import get_regime_mapping_for_visualization

        _, colors = get_regime_mapping_for_visualization(regime_profiles, n_states)
        return colors

    # Fallback to original color scheme logic
    base_colors = REGIME_COLORS.get(color_scheme, REGIME_COLORS["professional"])

    if n_states <= len(base_colors):
        return base_colors[:n_states]

    # Generate additional colors if needed
    additional_colors = []
    for i in range(n_states - len(base_colors)):
        hue = (i * 360 / (n_states - len(base_colors))) / 360
        color = plt.cm.tab10(hue)
        additional_colors.append(color)

    return base_colors + additional_colors


def get_regime_names(
    n_states: int,
    custom_names: Optional[List[str]] = None,
    regime_profiles: Optional[Dict[int, Any]] = None,
) -> List[str]:
    """
    Get appropriate regime names based on number of states and financial characterization.

    Args:
        n_states: Number of regime states
        custom_names: Optional custom regime names
        regime_profiles: Optional RegimeProfile objects from financial characterization

    Returns:
        List of regime names
    """
    if custom_names:
        if len(custom_names) != n_states:
            raise ValueError(
                f"Custom names must have {n_states} elements, got {len(custom_names)}"
            )
        return custom_names

    # Use financial characterization if available
    if regime_profiles:
        from ..utils.regime_mapping import get_regime_mapping_for_visualization

        labels, _ = get_regime_mapping_for_visualization(regime_profiles, n_states)
        return labels

    # Fallback to generic mapping
    if n_states in REGIME_NAMES_MAP:
        return REGIME_NAMES_MAP[n_states]

    # Generate generic names for unusual numbers of states
    return [f"Regime {i+1}" for i in range(n_states)]


def format_financial_axis(ax: plt.Axes, date_format: str = "%Y-%m") -> None:
    """
    Format axes for financial time series plots.

    Args:
        ax: Matplotlib axes object
        date_format: Date format string for x-axis
    """
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Format y-axis for currency/percentages
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:.2f}" if x > 1 else f"{x:.1%}")
    )

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def create_regime_legend(
    regime_colors: List[str],
    regime_names: List[str],
    ax: plt.Axes,
    location: str = "upper right",
) -> None:
    """
    Create consistent regime legend for plots.

    Args:
        regime_colors: List of colors for each regime
        regime_names: List of names for each regime
        ax: Matplotlib axes object
        location: Legend location
    """
    handles = []
    for color, name in zip(regime_colors, regime_names):
        handles.append(Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=name))

    ax.legend(handles=handles, loc=location, framealpha=0.9)


def plot_returns_with_regimes(
    data: pd.DataFrame,
    regime_data: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    price_column: str = "close",
    regime_column: str = "predicted_state",
    confidence_column: str = "confidence",
    title: str = "Price Chart with Market Regimes",
    color_scheme: str = "professional",
    regime_profiles: Optional[Dict[int, Any]] = None,
) -> plt.Figure:
    """
    Plot price series with regime background coloring.

    Args:
        data: DataFrame with price data
        regime_data: DataFrame with regime predictions
        ax: Optional matplotlib axes
        price_column: Column name for price data
        regime_column: Column name for regime predictions
        confidence_column: Column name for confidence scores
        title: Plot title
        color_scheme: Color scheme for regimes
        regime_profiles: Optional RegimeProfile objects for financial characterization

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Align data on dates
    aligned_data = data.join(
        regime_data[[regime_column, confidence_column]], how="inner"
    )

    # Get regime info
    n_states = int(aligned_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme, regime_profiles)
    regime_names = get_regime_names(n_states, regime_profiles=regime_profiles)

    # Plot price line
    ax.plot(
        aligned_data.index,
        aligned_data[price_column],
        color="black",
        linewidth=1.5,
        alpha=0.8,
        label="Price",
    )

    # Add regime background coloring
    for regime in range(n_states):
        regime_mask = aligned_data[regime_column] == regime
        if regime_mask.sum() > 0:
            regime_dates = aligned_data.index[regime_mask]

            # Create background spans for continuous periods
            if len(regime_dates) > 0:
                current_start = regime_dates[0]
                for i in range(1, len(regime_dates)):
                    # Check if there's a gap
                    if (regime_dates[i] - regime_dates[i - 1]).days > 1:
                        # End current span
                        ax.axvspan(
                            current_start,
                            regime_dates[i - 1],
                            color=regime_colors[regime],
                            alpha=0.2,
                        )
                        current_start = regime_dates[i]

                # Add final span
                ax.axvspan(
                    current_start,
                    regime_dates[-1],
                    color=regime_colors[regime],
                    alpha=0.2,
                )

    # Format axes
    format_financial_axis(ax)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(f"Price ({price_column})")

    # Add regime legend
    create_regime_legend(regime_colors, regime_names, ax)

    plt.tight_layout()
    return fig


def plot_regime_heatmap(
    regime_data: pd.DataFrame,
    regime_column: str = "predicted_state",
    confidence_column: str = "confidence",
    ax: Optional[plt.Axes] = None,
    title: str = "Regime Detection Heatmap",
    color_scheme: str = "professional",
    regime_profiles: Optional[Dict[int, Any]] = None,
) -> plt.Figure:
    """
    Create heatmap showing regime probabilities over time.

    Args:
        regime_data: DataFrame with regime predictions and confidence
        regime_column: Column name for regime predictions
        confidence_column: Column name for confidence scores
        ax: Optional matplotlib axes
        title: Plot title
        regime_profiles: Optional RegimeProfile objects for financial characterization

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    # Create heatmap data
    n_states = int(regime_data[regime_column].max()) + 1
    regime_names = get_regime_names(n_states, regime_profiles=regime_profiles)

    # Create a matrix for heatmap (time x states)
    heatmap_data = np.zeros((len(regime_data), n_states))

    for i, (idx, row) in enumerate(regime_data.iterrows()):
        regime = int(row[regime_column])
        confidence = row[confidence_column] if confidence_column in row else 1.0
        heatmap_data[i, regime] = confidence

    # Create heatmap with colorblind-friendly colormap
    im = ax.imshow(
        heatmap_data.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Format axes
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(regime_names)
    ax.set_xlabel("Time")
    ax.set_ylabel("Market Regime")
    ax.set_title(title, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Confidence")

    # Format x-axis with dates
    if len(regime_data) > 0:
        date_ticks = np.linspace(0, len(regime_data) - 1, 5, dtype=int)
        ax.set_xticks(date_ticks)
        ax.set_xticklabels(
            [regime_data.index[i].strftime("%Y-%m-%d") for i in date_ticks], rotation=45
        )

    plt.tight_layout()
    return fig


def plot_regime_statistics(
    regime_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    regime_column: str = "predicted_state",
    price_column: str = "close",
    ax: Optional[plt.Axes] = None,
    title: str = "Regime Performance Statistics",
) -> plt.Figure:
    """
    Plot regime performance statistics (returns, volatility, duration).

    Args:
        regime_data: DataFrame with regime predictions
        raw_data: DataFrame with price data
        regime_column: Column name for regime predictions
        price_column: Column name for price data
        ax: Optional matplotlib axes
        title: Plot title

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig = ax.figure
        axes = [ax]

    # Align data
    aligned_data = raw_data.join(regime_data[regime_column], how="inner")

    # Calculate returns
    aligned_data["returns"] = aligned_data[price_column].pct_change()

    n_states = int(aligned_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Statistics by regime
    regime_stats = {}
    for regime in range(n_states):
        mask = aligned_data[regime_column] == regime
        if mask.sum() > 1:
            regime_returns = aligned_data.loc[mask, "returns"].dropna()
            regime_stats[regime] = {
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "count": mask.sum(),
                "total_return": (1 + regime_returns).prod() - 1,
            }

    if len(axes) >= 4:
        # Plot 1: Mean returns by regime
        regimes = list(regime_stats.keys())
        mean_returns = [
            regime_stats[r]["mean_return"] * 252 for r in regimes
        ]  # Annualized
        bars1 = axes[0].bar(
            [regime_names[r] for r in regimes],
            mean_returns,
            color=[regime_colors[r] for r in regimes],
            alpha=0.7,
        )
        axes[0].set_title("Annualized Returns by Regime")
        axes[0].set_ylabel("Return (%)")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Plot 2: Volatility by regime
        volatilities = [
            regime_stats[r]["volatility"] * np.sqrt(252) for r in regimes
        ]  # Annualized
        bars2 = axes[1].bar(
            [regime_names[r] for r in regimes],
            volatilities,
            color=[regime_colors[r] for r in regimes],
            alpha=0.7,
        )
        axes[1].set_title("Annualized Volatility by Regime")
        axes[1].set_ylabel("Volatility (%)")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Plot 3: Days in each regime
        days_in_regime = [regime_stats[r]["count"] for r in regimes]
        bars3 = axes[2].bar(
            [regime_names[r] for r in regimes],
            days_in_regime,
            color=[regime_colors[r] for r in regimes],
            alpha=0.7,
        )
        axes[2].set_title("Days in Each Regime")
        axes[2].set_ylabel("Number of Days")

        # Plot 4: Sharpe ratio by regime
        sharpe_ratios = []
        for r in regimes:
            if regime_stats[r]["volatility"] > 0:
                sharpe = (
                    regime_stats[r]["mean_return"]
                    / regime_stats[r]["volatility"]
                    * np.sqrt(252)
                )
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        bars4 = axes[3].bar(
            [regime_names[r] for r in regimes],
            sharpe_ratios,
            color=[regime_colors[r] for r in regimes],
            alpha=0.7,
        )
        axes[3].set_title("Sharpe Ratio by Regime")
        axes[3].set_ylabel("Sharpe Ratio")
        axes[3].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Format all subplots
        for ax in axes:
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(title, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def plot_regime_transitions(
    regime_data: pd.DataFrame,
    regime_column: str = "predicted_state",
    ax: Optional[plt.Axes] = None,
    title: str = "Regime Transition Timeline",
    color_scheme: str = "professional",
) -> plt.Figure:
    """
    Plot regime transitions over time as a timeline.

    Args:
        regime_data: DataFrame with regime predictions
        regime_column: Column name for regime predictions
        ax: Optional matplotlib axes
        title: Plot title

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    n_states = int(regime_data[regime_column].max()) + 1
    regime_colors = get_regime_colors(n_states, color_scheme)
    regime_names = get_regime_names(n_states)

    # Plot regime timeline
    y_pos = 0.5
    for i, (date, regime) in enumerate(regime_data[regime_column].items()):
        color = regime_colors[int(regime)]
        ax.scatter(
            date, y_pos, c=color, s=50, alpha=0.8, edgecolors="black", linewidth=0.5
        )

    # Customize plot
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title(title, fontweight="bold")

    # Add regime legend
    create_regime_legend(regime_colors, regime_names, ax)

    # Format x-axis
    format_financial_axis(ax)

    plt.tight_layout()
    return fig


def create_multi_panel_regime_plot(
    data: pd.DataFrame,
    regime_data: pd.DataFrame,
    price_column: str = "close",
    regime_column: str = "predicted_state",
    confidence_column: str = "confidence",
    title: str = "Comprehensive Regime Analysis",
    color_scheme: str = "professional",
    regime_profiles: Optional[Dict[int, Any]] = None,
) -> plt.Figure:
    """
    Create comprehensive multi-panel plot with price, regimes, and statistics.

    Args:
        data: DataFrame with price data
        regime_data: DataFrame with regime predictions
        price_column: Column name for price data
        regime_column: Column name for regime predictions
        confidence_column: Column name for confidence scores
        title: Overall plot title
        color_scheme: Color scheme for regimes
        regime_profiles: Optional RegimeProfile objects for financial characterization

    Returns:
        matplotlib Figure with multiple subplots
    """
    fig = plt.figure(figsize=(16, 12))

    # Create subplot layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Main price chart with regimes
    ax1 = fig.add_subplot(gs[0, :])
    plot_returns_with_regimes(
        data,
        regime_data,
        ax=ax1,
        price_column=price_column,
        regime_column=regime_column,
        confidence_column=confidence_column,
        title="Price with Market Regimes",
        color_scheme=color_scheme,
        regime_profiles=regime_profiles,
    )

    # Regime heatmap
    ax2 = fig.add_subplot(gs[1, :])
    plot_regime_heatmap(
        regime_data,
        regime_column=regime_column,
        confidence_column=confidence_column,
        ax=ax2,
        title="Regime Confidence Heatmap",
        color_scheme=color_scheme,
        regime_profiles=regime_profiles,
    )

    # Regime statistics
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # Calculate regime distribution
    regime_counts = regime_data[regime_column].value_counts().sort_index()
    n_states = len(regime_counts)
    regime_colors = get_regime_colors(n_states, color_scheme, regime_profiles)
    regime_names = get_regime_names(n_states, regime_profiles=regime_profiles)

    # Pie chart of regime distribution
    ax3.pie(
        regime_counts.values,
        labels=[regime_names[i] for i in regime_counts.index],
        colors=[regime_colors[i] for i in regime_counts.index],
        autopct="%1.1f%%",
    )
    ax3.set_title("Regime Distribution")

    # Confidence distribution
    if confidence_column in regime_data.columns:
        ax4.hist(regime_data[confidence_column], bins=20, alpha=0.7, color="steelblue")
        ax4.set_xlabel("Confidence Score")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Confidence Distribution")
        ax4.axvline(
            regime_data[confidence_column].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {regime_data[confidence_column].mean():.2f}",
        )
        ax4.legend()

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig
