"""
Animation and GIF generation for evolving regime analysis.

Provides functions to create animated visualizations showing how regime
detection evolves over time during case studies.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from PIL import Image

from ..utils.formatting import format_strategy_names_dict
from .plotting import (
    create_regime_legend,
    format_financial_axis,
    get_regime_colors,
    get_regime_names,
    setup_financial_plot_style,
)


class RegimeAnimator:
    """
    Creates animated visualizations for regime evolution analysis.
    """

    def __init__(self, color_scheme: str = "colorblind_safe", style: str = "professional"):
        """
        Initialize regime animator.

        Args:
            color_scheme: Color scheme for regimes
            style: Plot style configuration
        """
        self.color_scheme = color_scheme
        self.style = style
        setup_financial_plot_style(style)

    def create_evolving_regime_animation(
        self,
        data: pd.DataFrame,
        regime_data_sequence: List[pd.DataFrame],
        evaluation_dates: List[str],
        price_column: str = "close",
        regime_column: str = "predicted_state",
        confidence_column: str = "confidence",
        window_size: int = 100,
        title: str = "Evolving Regime Analysis",
        save_path: Optional[str] = None,
        fps: int = 2,
        dpi: int = 100,
        regime_profiles: Optional[Dict[int, Any]] = None,
        signal_data_sequence: Optional[List[pd.DataFrame]] = None,
    ) -> animation.FuncAnimation:
        """
        Create animation showing regime detection evolving over time.

        Args:
            data: Complete price data
            regime_data_sequence: List of regime DataFrames for each evaluation date
            evaluation_dates: List of evaluation dates
            price_column: Price column name
            regime_column: Regime column name
            confidence_column: Confidence column name
            window_size: Size of sliding window to display
            title: Animation title
            save_path: Path to save GIF (optional)
            fps: Frames per second
            dpi: DPI for output
            regime_profiles: Optional regime profiles for color consistency
            signal_data_sequence: Optional list of signal DataFrames for trading signal visualization

        Returns:
            matplotlib FuncAnimation object
        """
        # Determine number of panels based on whether signals are provided
        n_panels = 4 if signal_data_sequence is not None else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 10 if n_panels == 3 else 13))

        # Determine regime info from first regime data
        if regime_data_sequence:
            n_states = int(regime_data_sequence[0][regime_column].max()) + 1
            regime_colors = get_regime_colors(
                n_states, self.color_scheme, regime_profiles
            )

            # Use regime names from data if available, otherwise generate generic names
            if 'regime_name' in regime_data_sequence[0].columns:
                # Extract regime names from data by mapping state index to name
                # Concatenate all regime data to get complete state-to-name mapping
                all_regime_data = pd.concat(regime_data_sequence)
                state_to_name = all_regime_data.groupby(regime_column)['regime_name'].first().to_dict()
                regime_names = [state_to_name.get(i, f"Regime {i}") for i in range(n_states)]
            else:
                regime_names = get_regime_names(n_states, regime_profiles=regime_profiles)
        else:
            n_states = 3
            regime_colors = get_regime_colors(3, self.color_scheme, regime_profiles)
            regime_names = get_regime_names(3, regime_profiles=regime_profiles)

        # Calculate fixed price limits for smooth animation (0.9x min to 1.1x max)
        price_min = data[price_column].min()
        price_max = data[price_column].max()
        price_y_lower = price_min * 0.9
        price_y_upper = price_max * 1.1


        def animate(frame):
            try:
                # Clear all axes
                for ax in axes:
                    ax.clear()

                # Safety check for frame bounds
                if frame >= len(evaluation_dates) or frame >= len(regime_data_sequence):
                    print(f"Frame {frame} out of bounds")
                    return axes

                # Get current date and regime data
                current_date = evaluation_dates[frame]
                current_regime_data = regime_data_sequence[frame]

            except Exception as e:
                print(f"Animation frame {frame} error: {e}")
                return axes

            # Get window of data around current date
            current_date_dt = pd.to_datetime(current_date)

            # Handle timezone compatibility
            if data.index.tz is not None:
                # If data index is timezone-aware, make current_date_dt compatible
                if current_date_dt.tz is None:
                    current_date_dt = current_date_dt.tz_localize(data.index.tz)
            else:
                # If data index is timezone-naive, ensure current_date_dt is also naive
                if current_date_dt.tz is not None:
                    current_date_dt = current_date_dt.tz_localize(None)

            window_start = current_date_dt - pd.Timedelta(days=window_size)

            # Filter data to window
            window_data = data[
                (data.index >= window_start) & (data.index <= current_date_dt)
            ]
            window_regime_data = current_regime_data[
                current_regime_data.index <= current_date_dt
            ]

            if len(window_data) == 0:
                return axes

            # Plot 1: Price chart with regime backgrounds
            axes[0].plot(
                window_data.index,
                window_data[price_column],
                color="black",
                linewidth=2,
                alpha=0.8,
            )

            # Add regime backgrounds - color by regime_name for consistency with snapshots
            if len(window_regime_data) > 0:
                # Import colorblind-safe regime colors (name-based)
                # Use regime_profiles for color consistency (single source of truth)
                from ..analysis import get_regime_colors as get_name_based_colors
                regime_colors_map = get_name_based_colors(regime_profiles)

                # Color by regime_name if available, otherwise fall back to state ID
                if 'regime_name' in window_regime_data.columns:
                    for regime_name in window_regime_data['regime_name'].unique():
                        regime_mask = window_regime_data['regime_name'] == regime_name
                        if regime_mask.sum() > 0:
                            for date in window_regime_data.index[regime_mask]:
                                if date in window_data.index:
                                    axes[0].axvspan(
                                        date,
                                        date + pd.Timedelta(days=1),
                                        color=regime_colors_map.get(regime_name, "gray"),
                                        alpha=0.3,
                                    )
                else:
                    # Fallback to state ID coloring
                    for regime in range(n_states):
                        regime_mask = window_regime_data[regime_column] == regime
                        if regime_mask.sum() > 0:
                            regime_dates = window_regime_data.index[regime_mask]

                            for date in regime_dates:
                                if date in window_data.index:
                                    axes[0].axvspan(
                                        date,
                                        date + pd.Timedelta(days=1),
                                        color=regime_colors[regime],
                                        alpha=0.3,
                                    )

            # Highlight current date
            axes[0].axvline(
                x=current_date_dt,
                color="red",
                linewidth=3,
                alpha=0.8,
                linestyle="--",
                label=f"Current: {current_date}",
            )

            axes[0].set_title(f"{title} - {current_date}")
            axes[0].set_ylabel(f"Price ({price_column})")
            axes[0].set_ylim(price_y_lower, price_y_upper)  # Fixed y-axis for smooth animation
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Regime probability timeline (show ALL state probabilities)
            if len(window_regime_data) > 0:
                # Plot all state probabilities (not just predicted regime)
                for state_id in range(n_states):
                    prob_column = f"state_{state_id}_prob"
                    if prob_column in window_regime_data.columns:
                        # Get ALL probabilities for this state (convert to percentage)
                        state_probs = window_regime_data[prob_column] * 100

                        # Determine regime name and color for this state ID
                        if 'regime_name' in window_regime_data.columns:
                            # Map state_id to regime name
                            state_to_name_map = window_regime_data.groupby(regime_column)['regime_name'].first().to_dict()
                            regime_name = state_to_name_map.get(state_id, f"Regime {state_id}")
                            # Use name-based color mapping for consistency
                            from ..analysis import get_regime_colors as get_name_based_colors
                            regime_colors_map = get_name_based_colors(regime_profiles)
                            color = regime_colors_map.get(regime_name.lower(), regime_colors[state_id])
                        else:
                            regime_name = regime_names[state_id]
                            color = regime_colors[state_id]

                        axes[1].scatter(
                            window_regime_data.index,
                            state_probs,
                            c=color,
                            s=20,  # Smaller dots since we have 3x more
                            alpha=0.6,  # More transparent for overlapping
                            label=regime_name,
                        )

            axes[1].axvline(
                x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
            )
            axes[1].set_title("Regime Probability Timeline")
            axes[1].set_ylabel("Probability (%)")
            axes[1].set_ylim(0, 100)  # Y-axis is now 0-100%
            axes[1].legend(loc='upper left', fontsize=8, ncol=1)
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Model confidence
            if (
                len(window_regime_data) > 0
                and confidence_column in window_regime_data.columns
            ):
                # Convert confidence to percentage (0-100%)
                confidence_pct = window_regime_data[confidence_column] * 100

                axes[2].plot(
                    window_regime_data.index,
                    confidence_pct,
                    color="purple",
                    linewidth=2,
                    alpha=0.8,
                )
                axes[2].fill_between(
                    window_regime_data.index,
                    0,
                    confidence_pct,
                    alpha=0.3,
                    color="purple",
                )

                # Show current confidence
                current_confidence_pct = (
                    confidence_pct.iloc[-1]
                    if len(window_regime_data) > 0
                    else 0
                )
                axes[2].axhline(
                    y=current_confidence_pct,
                    color="red",
                    linewidth=2,
                    alpha=0.8,
                    linestyle=":",
                    label=f"Current: {current_confidence_pct:.1f}%",
                )

            axes[2].axvline(
                x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
            )

            # Get current regime name for title
            current_regime_name = "Unknown"
            if len(window_regime_data) > 0:
                if 'regime_name' in window_regime_data.columns:
                    current_regime_name = window_regime_data['regime_name'].iloc[-1].capitalize()
                else:
                    current_regime_state = int(window_regime_data[regime_column].iloc[-1])
                    current_regime_name = regime_names[current_regime_state]

            axes[2].set_title(f"Confidence in Predicted Regime: {current_regime_name}")
            axes[2].set_ylabel("Confidence (%)")
            axes[2].set_xlabel("Date")
            axes[2].set_ylim(0, 100)  # Y-axis is now 0-100%
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Trading Signals (if available)
            if signal_data_sequence is not None and frame < len(signal_data_sequence):
                current_signal_data = signal_data_sequence[frame]
                window_signal_data = current_signal_data[current_signal_data.index <= current_date_dt]

                if len(window_signal_data) > 0 and 'signal' in window_signal_data.columns:
                    # Define colorblind-safe colors: BLUE for buy, ORANGE for sell
                    buy_color = '#0173B2'   # Blue (colorblind-safe)
                    sell_color = '#DE8F05'  # Orange (colorblind-safe)

                    # Plot signal line
                    axes[3].plot(
                        window_signal_data.index,
                        window_signal_data['signal'],
                        color='black',
                        linewidth=2,
                        alpha=0.8,
                        label='Signal Value'
                    )

                    # Color background regions based on signal value
                    for i in range(len(window_signal_data) - 1):
                        date_start = window_signal_data.index[i]
                        date_end = window_signal_data.index[i + 1]
                        signal_val = window_signal_data['signal'].iloc[i]

                        if signal_val > 0:  # BUY signal
                            axes[3].axvspan(
                                date_start, date_end,
                                color=buy_color,
                                alpha=0.2
                            )
                        elif signal_val < 0:  # SELL signal
                            axes[3].axvspan(
                                date_start, date_end,
                                color=sell_color,
                                alpha=0.2
                            )

                    # Add reference lines
                    axes[3].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    axes[3].axhline(y=0.0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
                    axes[3].axhline(y=-1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                    # Highlight current date
                    axes[3].axvline(
                        x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
                    )

                    # Get current signal value
                    current_signal = window_signal_data['signal'].iloc[-1]
                    signal_label = "BUY" if current_signal > 0 else "SELL" if current_signal < 0 else "HOLD"

                    axes[3].set_title(f"Trading Signals: {signal_label} ({current_signal:+.2f})")
                    axes[3].set_ylabel("Signal Strength")
                    axes[3].set_xlabel("Date")
                    axes[3].set_ylim(-1.1, 1.1)
                    axes[3].grid(True, alpha=0.3)

                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=buy_color, alpha=0.4, label='BUY (>0)'),
                        Patch(facecolor=sell_color, alpha=0.4, label='SELL (<0)')
                    ]
                    axes[3].legend(handles=legend_elements, loc='upper left', fontsize=8)

            # Format x-axes - only apply financial formatting to price (axes[0])
            # Other axes are confidence/signal plots with percentage/ratio y-axis
            for i, ax in enumerate(axes):
                if len(window_data) > 0:
                    ax.set_xlim(window_data.index.min(), window_data.index.max())
                    if i == 0:  # Only price chart needs financial formatting ($ on y-axis)
                        format_financial_axis(ax, date_format="%m/%d")
                    else:  # Confidence/signal plots - just format x-axis dates
                        import matplotlib.dates as mdates
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            return axes

        # Create animation
        try:
            frames_count = min(len(evaluation_dates), len(regime_data_sequence))
            if frames_count == 0:
                print("Warning: No frames to animate")
                return None

            anim = animation.FuncAnimation(
                fig,
                animate,
                frames=frames_count,
                interval=1000 // fps,
                blit=False,
                repeat=True,
            )

            # Save as GIF if path provided
            if save_path:
                import os

                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)

            return anim

        except Exception as e:
            print(f"Animation creation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def create_performance_evolution_animation(
        self,
        performance_data_sequence: List[Dict[str, Any]],
        evaluation_dates: List[str],
        title: str = "Performance Evolution",
        save_path: Optional[str] = None,
        fps: int = 2,
    ) -> animation.FuncAnimation:
        """
        Create animation showing performance metrics evolving over time.

        Args:
            performance_data_sequence: List of performance dictionaries for each date
            evaluation_dates: List of evaluation dates
            title: Animation title
            save_path: Path to save GIF
            fps: Frames per second

        Returns:
            matplotlib FuncAnimation object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        def animate(frame):
            # Clear all axes
            for ax in axes:
                ax.clear()

            current_date = evaluation_dates[frame]
            current_performance = performance_data_sequence[frame]

            # Collect data up to current frame
            cumulative_data = {
                "dates": evaluation_dates[: frame + 1],
                "returns": [],
                "sharpe": [],
                "drawdown": [],
                "regime_distribution": current_performance.get(
                    "regime_distribution", {}
                ),
            }

            for i in range(frame + 1):
                perf = performance_data_sequence[i]
                cumulative_data["returns"].append(perf.get("cumulative_return", 0))
                cumulative_data["sharpe"].append(perf.get("sharpe_ratio", 0))
                cumulative_data["drawdown"].append(perf.get("max_drawdown", 0))

            # Plot 1: Cumulative returns evolution
            dates = [pd.to_datetime(d) for d in cumulative_data["dates"]]
            axes[0].plot(dates, cumulative_data["returns"], "b-", linewidth=2)
            axes[0].set_title(f"Cumulative Return Evolution - {current_date}")
            axes[0].set_ylabel("Cumulative Return")
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Plot 2: Sharpe ratio evolution
            axes[1].plot(dates, cumulative_data["sharpe"], "g-", linewidth=2)
            axes[1].set_title("Sharpe Ratio Evolution")
            axes[1].set_ylabel("Sharpe Ratio")
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Plot 3: Drawdown evolution
            axes[2].plot(dates, cumulative_data["drawdown"], "r-", linewidth=2)
            axes[2].fill_between(
                dates, 0, cumulative_data["drawdown"], alpha=0.3, color="red"
            )
            axes[2].set_title("Maximum Drawdown Evolution")
            axes[2].set_ylabel("Max Drawdown")
            axes[2].set_xlabel("Date")
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Current regime distribution
            regime_dist = cumulative_data["regime_distribution"]
            if regime_dist:
                regimes = list(regime_dist.keys())
                counts = list(regime_dist.values())
                regime_names = get_regime_names(len(regimes))
                colors = get_regime_colors(len(regimes), self.color_scheme)

                axes[3].bar(
                    regime_names[: len(regimes)], counts, color=colors, alpha=0.7
                )
                axes[3].set_title("Current Regime Distribution")
                axes[3].set_ylabel("Days in Regime")
                plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

            # Format x-axes for time series
            for i in range(3):
                format_financial_axis(axes[i], date_format="%m/%d")

            plt.tight_layout()
            return axes

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(evaluation_dates),
            interval=1000 // fps,
            blit=False,
            repeat=True,
        )

        # Save as GIF if path provided
        if save_path:
            anim.save(save_path, writer="pillow", fps=fps)

        return anim


def create_regime_comparison_gif(
    strategies_data: Dict[str, List[Dict[str, Any]]],
    evaluation_dates: List[str],
    title: str = "Strategy Comparison Evolution",
    save_path: str = "regime_comparison.gif",
    fps: int = 2,
    dpi: int = 100,
) -> None:
    """
    Create GIF comparing multiple strategies over time.

    Args:
        strategies_data: Dictionary of strategy_name -> list of performance data
        evaluation_dates: List of evaluation dates
        title: GIF title
        save_path: Path to save GIF
        fps: Frames per second
        dpi: DPI for output
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Format strategy names consistently
    formatted_strategies_data = format_strategy_names_dict(strategies_data)
    strategy_names = list(formatted_strategies_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))

    def animate(frame):
        for ax in axes:
            ax.clear()

        current_date = evaluation_dates[frame]

        # Collect cumulative data for each strategy
        strategy_cumulative = {}
        for strategy_name in strategy_names:
            cumulative_returns = []
            sharpe_ratios = []
            drawdowns = []

            for i in range(frame + 1):
                perf = formatted_strategies_data[strategy_name][i]
                cumulative_returns.append(perf.get("cumulative_return", 0))
                sharpe_ratios.append(perf.get("sharpe_ratio", 0))
                drawdowns.append(perf.get("max_drawdown", 0))

            strategy_cumulative[strategy_name] = {
                "returns": cumulative_returns,
                "sharpe": sharpe_ratios,
                "drawdowns": drawdowns,
            }

        dates = [pd.to_datetime(d) for d in evaluation_dates[: frame + 1]]

        # Plot 1: Cumulative returns comparison
        for i, (strategy_name, data) in enumerate(strategy_cumulative.items()):
            axes[0].plot(
                dates,
                data["returns"],
                color=colors[i],
                linewidth=2,
                label=strategy_name,
                alpha=0.8,
            )

        axes[0].set_title(f"Cumulative Returns Comparison - {current_date}")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # Plot 2: Sharpe ratio comparison
        for i, (strategy_name, data) in enumerate(strategy_cumulative.items()):
            axes[1].plot(
                dates,
                data["sharpe"],
                color=colors[i],
                linewidth=2,
                label=strategy_name,
                alpha=0.8,
            )

        axes[1].set_title("Sharpe Ratio Comparison")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # Plot 3: Current performance bar chart
        current_returns = [
            formatted_strategies_data[name][frame].get("cumulative_return", 0)
            for name in strategy_names
        ]
        bars = axes[2].bar(strategy_names, current_returns, color=colors, alpha=0.7)
        axes[2].set_title(f"Current Performance - {current_date}")
        axes[2].set_ylabel("Cumulative Return")
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Risk-return scatter (current position)
        current_sharpe = [
            formatted_strategies_data[name][frame].get("sharpe_ratio", 0)
            for name in strategy_names
        ]
        current_vol = [
            formatted_strategies_data[name][frame].get("volatility", 0.1)
            for name in strategy_names
        ]

        for i, name in enumerate(strategy_names):
            axes[3].scatter(
                current_vol[i],
                current_returns[i],
                color=colors[i],
                s=100,
                alpha=0.8,
                label=name,
            )

        axes[3].set_title("Risk-Return Profile")
        axes[3].set_xlabel("Volatility")
        axes[3].set_ylabel("Return")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Format time axes
        for i in range(2):
            format_financial_axis(axes[i], date_format="%m/%d")

        plt.tight_layout()
        return axes

    # Create and save animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(evaluation_dates),
        interval=1000 // fps,
        blit=False,
        repeat=True,
    )

    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def save_individual_frames(
    data: pd.DataFrame,
    regime_data_sequence: List[pd.DataFrame],
    evaluation_dates: List[str],
    output_directory: str,
    frame_prefix: str = "frame",
    regime_profiles: Optional[Dict[int, Any]] = None,
    signal_data_sequence: Optional[List[pd.DataFrame]] = None,
    **kwargs,
) -> List[str]:
    """
    Save individual frames for manual GIF creation or inspection.

    Args:
        data: Price data
        regime_data_sequence: List of regime DataFrames
        evaluation_dates: List of evaluation dates
        output_directory: Directory to save frames
        frame_prefix: Prefix for frame filenames
        regime_profiles: Optional regime profiles for color consistency
        signal_data_sequence: Optional list of signal DataFrames
        **kwargs: Additional arguments for plotting

    Returns:
        List of saved frame paths
    """
    os.makedirs(output_directory, exist_ok=True)
    saved_paths = []

    # Create animator
    animator = RegimeAnimator(
        color_scheme=kwargs.get("color_scheme", "colorblind_safe"),
        style=kwargs.get("style", "professional"),
    )

    # Determine number of panels
    n_panels = 4 if signal_data_sequence is not None else 3

    for i, (date, regime_data) in enumerate(
        zip(evaluation_dates, regime_data_sequence)
    ):
        # Create single frame plot
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 10 if n_panels == 3 else 13))

        # Plot current state (similar to animation frame)
        window_size = kwargs.get("window_size", 100)
        current_date_dt = pd.to_datetime(date)
        window_start = current_date_dt - pd.Timedelta(days=window_size)

        window_data = data[
            (data.index >= window_start) & (data.index <= current_date_dt)
        ]
        window_regime_data = regime_data[regime_data.index <= current_date_dt]

        if len(window_data) > 0:
            # Price chart
            axes[0].plot(
                window_data.index,
                window_data["close"],
                color="black",
                linewidth=2,
                alpha=0.8,
            )

            # Add regime backgrounds - color by regime_name for consistency with snapshots
            if len(window_regime_data) > 0:
                # Import colorblind-safe regime colors (name-based)
                # Use regime_profiles for color consistency (single source of truth)
                from ..analysis import get_regime_colors as get_name_based_colors
                regime_colors_map = get_name_based_colors(regime_profiles)

                n_states = int(window_regime_data["predicted_state"].max()) + 1
                regime_colors = get_regime_colors(
                    n_states, kwargs.get("color_scheme", "colorblind_safe"), regime_profiles
                )

                # Color by regime_name if available, otherwise fall back to state ID
                if 'regime_name' in window_regime_data.columns:
                    for regime_name in window_regime_data['regime_name'].unique():
                        regime_mask = window_regime_data['regime_name'] == regime_name
                        if regime_mask.sum() > 0:
                            for d in window_regime_data.index[regime_mask]:
                                if d in window_data.index:
                                    axes[0].axvspan(
                                        d,
                                        d + pd.Timedelta(days=1),
                                        color=regime_colors_map.get(regime_name, "gray"),
                                        alpha=0.3,
                                    )
                else:
                    # Fallback to state ID coloring
                    for regime in range(n_states):
                        regime_mask = window_regime_data["predicted_state"] == regime
                        if regime_mask.sum() > 0:
                            regime_dates = window_regime_data.index[regime_mask]
                            for d in regime_dates:
                                if d in window_data.index:
                                    axes[0].axvspan(
                                        d,
                                        d + pd.Timedelta(days=1),
                                        color=regime_colors[regime],
                                        alpha=0.3,
                                    )

            axes[0].axvline(
                x=current_date_dt,
                color="red",
                linewidth=3,
                alpha=0.8,
                linestyle="--",
                label=f"Current: {date}",
            )
            axes[0].set_title(f"Price Analysis - {date}")
            axes[0].set_ylabel("Price")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Regime probability timeline (show ALL state probabilities)
            if len(window_regime_data) > 0:
                regime_names_list = get_regime_names(n_states)
                # Plot all state probabilities (not just predicted regime)
                for state_id in range(n_states):
                    prob_column = f"state_{state_id}_prob"
                    if prob_column in window_regime_data.columns:
                        # Get ALL probabilities for this state (convert to percentage)
                        state_probs = window_regime_data[prob_column] * 100

                        # Determine regime name and color for this state ID
                        if 'regime_name' in window_regime_data.columns:
                            # Map state_id to regime name
                            state_to_name_map = window_regime_data.groupby("predicted_state")['regime_name'].first().to_dict()
                            regime_name = state_to_name_map.get(state_id, f"Regime {state_id}")
                            # Use name-based color mapping for consistency
                            from ..analysis import get_regime_colors as get_name_based_colors
                            regime_colors_map = get_name_based_colors(regime_profiles)
                            color = regime_colors_map.get(regime_name.lower(), regime_colors[state_id])
                        else:
                            regime_name = regime_names_list[state_id]
                            color = regime_colors[state_id]

                        axes[1].scatter(
                            window_regime_data.index,
                            state_probs,
                            c=color,
                            s=20,  # Smaller dots since we have 3x more
                            alpha=0.6,  # More transparent for overlapping
                            label=regime_name,
                        )

            axes[1].axvline(
                x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
            )
            axes[1].set_title("Regime Probability Timeline")
            axes[1].set_ylabel("Probability (%)")
            axes[1].set_ylim(0, 100)  # Y-axis is now 0-100%
            axes[1].legend(loc='upper left', fontsize=8, ncol=1)
            axes[1].grid(True, alpha=0.3)

            # Confidence
            if (
                len(window_regime_data) > 0
                and "confidence" in window_regime_data.columns
            ):
                # Convert confidence to percentage (0-100%)
                confidence_pct = window_regime_data["confidence"] * 100

                axes[2].plot(
                    window_regime_data.index,
                    confidence_pct,
                    color="purple",
                    linewidth=2,
                    alpha=0.8,
                )
                axes[2].fill_between(
                    window_regime_data.index,
                    0,
                    confidence_pct,
                    alpha=0.3,
                    color="purple",
                )

            axes[2].axvline(
                x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
            )

            # Get current regime name for title
            current_regime_name = "Unknown"
            if len(window_regime_data) > 0:
                if 'regime_name' in window_regime_data.columns:
                    current_regime_name = window_regime_data['regime_name'].iloc[-1].capitalize()
                else:
                    current_regime_state = int(window_regime_data["predicted_state"].iloc[-1])
                    current_regime_name = regime_names[current_regime_state]

            axes[2].set_title(f"Confidence in Predicted Regime: {current_regime_name}")
            axes[2].set_ylabel("Confidence (%)")
            axes[2].set_xlabel("Date")
            axes[2].set_ylim(0, 100)  # Y-axis is now 0-100%
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Trading Signals (if available)
            if signal_data_sequence is not None and i < len(signal_data_sequence):
                signal_data = signal_data_sequence[i]
                window_signal_data = signal_data[signal_data.index <= current_date_dt]

                if len(window_signal_data) > 0 and 'signal' in window_signal_data.columns:
                    # Define colorblind-safe colors: BLUE for buy, ORANGE for sell
                    buy_color = '#0173B2'   # Blue (colorblind-safe)
                    sell_color = '#DE8F05'  # Orange (colorblind-safe)

                    # Plot signal line
                    axes[3].plot(
                        window_signal_data.index,
                        window_signal_data['signal'],
                        color='black',
                        linewidth=2,
                        alpha=0.8,
                        label='Signal Value'
                    )

                    # Color background regions based on signal value
                    for j in range(len(window_signal_data) - 1):
                        date_start = window_signal_data.index[j]
                        date_end = window_signal_data.index[j + 1]
                        signal_val = window_signal_data['signal'].iloc[j]

                        if signal_val > 0:  # BUY signal
                            axes[3].axvspan(
                                date_start, date_end,
                                color=buy_color,
                                alpha=0.2
                            )
                        elif signal_val < 0:  # SELL signal
                            axes[3].axvspan(
                                date_start, date_end,
                                color=sell_color,
                                alpha=0.2
                            )

                    # Add reference lines
                    axes[3].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    axes[3].axhline(y=0.0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
                    axes[3].axhline(y=-1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                    # Highlight current date
                    axes[3].axvline(
                        x=current_date_dt, color="red", linewidth=3, alpha=0.8, linestyle="--"
                    )

                    # Get current signal value
                    current_signal = window_signal_data['signal'].iloc[-1]
                    signal_label = "BUY" if current_signal > 0 else "SELL" if current_signal < 0 else "HOLD"

                    axes[3].set_title(f"Trading Signals: {signal_label} ({current_signal:+.2f})")
                    axes[3].set_ylabel("Signal Strength")
                    axes[3].set_xlabel("Date")
                    axes[3].set_ylim(-1.1, 1.1)
                    axes[3].grid(True, alpha=0.3)

                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=buy_color, alpha=0.4, label='BUY (>0)'),
                        Patch(facecolor=sell_color, alpha=0.4, label='SELL (<0)')
                    ]
                    axes[3].legend(handles=legend_elements, loc='upper left', fontsize=8)

            # Format axes - only apply financial formatting to price (axes[0])
            # Other axes are confidence/signal plots with percentage/ratio y-axis
            for j, ax in enumerate(axes):
                if j == 0:  # Only price chart needs financial formatting ($ on y-axis)
                    format_financial_axis(ax, date_format="%m/%d")
                else:  # Confidence/signal plots - just format x-axis dates
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(
            output_directory, f"{frame_prefix}_{i:04d}_{date}.png"
        )
        fig.savefig(frame_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(frame_path)

    return saved_paths


def create_gif_from_frames(
    frame_paths: List[str], output_path: str, duration: int = 500, loop: int = 0
) -> None:
    """
    Create GIF from saved frame images.

    Args:
        frame_paths: List of paths to frame images
        output_path: Path for output GIF
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")

    # Load images
    images = []
    for path in frame_paths:
        if os.path.exists(path):
            images.append(Image.open(path))

    if images:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
    else:
        raise ValueError("No valid images found in frame paths")
