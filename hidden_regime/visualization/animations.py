"""
Animation and GIF generation for evolving regime analysis.

Provides functions to create animated visualizations showing how regime
detection evolves over time during case studies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Optional, Dict, List, Any, Tuple, Callable
from datetime import datetime, timedelta
import os
import io
from PIL import Image

from .plotting import (
    get_regime_colors, get_regime_names, format_financial_axis,
    create_regime_legend, setup_financial_plot_style
)
from ..utils.formatting import format_strategy_names_dict


class RegimeAnimator:
    """
    Creates animated visualizations for regime evolution analysis.
    """

    def __init__(self, color_scheme: str = "professional", style: str = "professional"):
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
        price_column: str = 'close',
        regime_column: str = 'predicted_state',
        confidence_column: str = 'confidence',
        window_size: int = 100,
        title: str = "Evolving Regime Analysis",
        save_path: Optional[str] = None,
        fps: int = 2,
        dpi: int = 100,
        regime_profiles: Optional[Dict[int, Any]] = None
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

        Returns:
            matplotlib FuncAnimation object
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Determine regime info from first regime data
        if regime_data_sequence:
            n_states = int(regime_data_sequence[0][regime_column].max()) + 1
            regime_colors = get_regime_colors(n_states, self.color_scheme, regime_profiles)
            regime_names = get_regime_names(n_states, regime_profiles=regime_profiles)
        else:
            n_states = 3
            regime_colors = get_regime_colors(3, self.color_scheme, regime_profiles)
            regime_names = get_regime_names(3, regime_profiles=regime_profiles)

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
            window_data = data[(data.index >= window_start) & (data.index <= current_date_dt)]
            window_regime_data = current_regime_data[current_regime_data.index <= current_date_dt]

            if len(window_data) == 0:
                return axes

            # Plot 1: Price chart with regime backgrounds
            axes[0].plot(window_data.index, window_data[price_column],
                        color='black', linewidth=2, alpha=0.8)

            # Add regime backgrounds
            if len(window_regime_data) > 0:
                for regime in range(n_states):
                    regime_mask = window_regime_data[regime_column] == regime
                    if regime_mask.sum() > 0:
                        regime_dates = window_regime_data.index[regime_mask]

                        for date in regime_dates:
                            if date in window_data.index:
                                axes[0].axvspan(date, date + pd.Timedelta(days=1),
                                              color=regime_colors[regime], alpha=0.3)

            # Highlight current date
            axes[0].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8,
                           linestyle='--', label=f'Current: {current_date}')

            axes[0].set_title(f'{title} - {current_date}')
            axes[0].set_ylabel(f'Price ({price_column})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Regime timeline
            if len(window_regime_data) > 0:
                for regime in range(n_states):
                    regime_mask = window_regime_data[regime_column] == regime
                    if regime_mask.sum() > 0:
                        y_values = [regime] * regime_mask.sum()
                        axes[1].scatter(window_regime_data.index[regime_mask], y_values,
                                       c=regime_colors[regime], s=30, alpha=0.8,
                                       label=regime_names[regime])

            axes[1].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8, linestyle='--')
            axes[1].set_title('Regime Timeline')
            axes[1].set_ylabel('Regime')
            axes[1].set_yticks(range(n_states))
            axes[1].set_yticklabels(regime_names)
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Model confidence
            if len(window_regime_data) > 0 and confidence_column in window_regime_data.columns:
                axes[2].plot(window_regime_data.index, window_regime_data[confidence_column],
                            color='purple', linewidth=2, alpha=0.8)
                axes[2].fill_between(window_regime_data.index, 0, window_regime_data[confidence_column],
                                    alpha=0.3, color='purple')

                # Show current confidence
                current_confidence = window_regime_data[confidence_column].iloc[-1] if len(window_regime_data) > 0 else 0
                axes[2].axhline(y=current_confidence, color='red', linewidth=2, alpha=0.8,
                               linestyle=':', label=f'Current: {current_confidence:.2f}')

            axes[2].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8, linestyle='--')
            axes[2].set_title('Model Confidence')
            axes[2].set_ylabel('Confidence')
            axes[2].set_xlabel('Date')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Format x-axes
            for ax in axes:
                if len(window_data) > 0:
                    ax.set_xlim(window_data.index.min(), window_data.index.max())
                    format_financial_axis(ax, date_format="%m/%d")

            plt.tight_layout()
            return axes

        # Create animation
        try:
            frames_count = min(len(evaluation_dates), len(regime_data_sequence))
            if frames_count == 0:
                print("Warning: No frames to animate")
                return None

            anim = animation.FuncAnimation(
                fig, animate, frames=frames_count,
                interval=1000//fps, blit=False, repeat=True
            )

            # Save as GIF if path provided
            if save_path:
                import os
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)

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
        fps: int = 2
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
                'dates': evaluation_dates[:frame+1],
                'returns': [],
                'sharpe': [],
                'drawdown': [],
                'regime_distribution': current_performance.get('regime_distribution', {})
            }

            for i in range(frame + 1):
                perf = performance_data_sequence[i]
                cumulative_data['returns'].append(perf.get('cumulative_return', 0))
                cumulative_data['sharpe'].append(perf.get('sharpe_ratio', 0))
                cumulative_data['drawdown'].append(perf.get('max_drawdown', 0))

            # Plot 1: Cumulative returns evolution
            dates = [pd.to_datetime(d) for d in cumulative_data['dates']]
            axes[0].plot(dates, cumulative_data['returns'], 'b-', linewidth=2)
            axes[0].set_title(f'Cumulative Return Evolution - {current_date}')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Plot 2: Sharpe ratio evolution
            axes[1].plot(dates, cumulative_data['sharpe'], 'g-', linewidth=2)
            axes[1].set_title('Sharpe Ratio Evolution')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Plot 3: Drawdown evolution
            axes[2].plot(dates, cumulative_data['drawdown'], 'r-', linewidth=2)
            axes[2].fill_between(dates, 0, cumulative_data['drawdown'], alpha=0.3, color='red')
            axes[2].set_title('Maximum Drawdown Evolution')
            axes[2].set_ylabel('Max Drawdown')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)

            # Plot 4: Current regime distribution
            regime_dist = cumulative_data['regime_distribution']
            if regime_dist:
                regimes = list(regime_dist.keys())
                counts = list(regime_dist.values())
                regime_names = get_regime_names(len(regimes))
                colors = get_regime_colors(len(regimes), self.color_scheme)

                axes[3].bar(regime_names[:len(regimes)], counts, color=colors, alpha=0.7)
                axes[3].set_title('Current Regime Distribution')
                axes[3].set_ylabel('Days in Regime')
                plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

            # Format x-axes for time series
            for i in range(3):
                format_financial_axis(axes[i], date_format="%m/%d")

            plt.tight_layout()
            return axes

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(evaluation_dates),
            interval=1000//fps, blit=False, repeat=True
        )

        # Save as GIF if path provided
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)

        return anim


def create_regime_comparison_gif(
    strategies_data: Dict[str, List[Dict[str, Any]]],
    evaluation_dates: List[str],
    title: str = "Strategy Comparison Evolution",
    save_path: str = "regime_comparison.gif",
    fps: int = 2,
    dpi: int = 100
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
                cumulative_returns.append(perf.get('cumulative_return', 0))
                sharpe_ratios.append(perf.get('sharpe_ratio', 0))
                drawdowns.append(perf.get('max_drawdown', 0))

            strategy_cumulative[strategy_name] = {
                'returns': cumulative_returns,
                'sharpe': sharpe_ratios,
                'drawdowns': drawdowns
            }

        dates = [pd.to_datetime(d) for d in evaluation_dates[:frame+1]]

        # Plot 1: Cumulative returns comparison
        for i, (strategy_name, data) in enumerate(strategy_cumulative.items()):
            axes[0].plot(dates, data['returns'], color=colors[i], linewidth=2,
                        label=strategy_name, alpha=0.8)

        axes[0].set_title(f'Cumulative Returns Comparison - {current_date}')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Plot 2: Sharpe ratio comparison
        for i, (strategy_name, data) in enumerate(strategy_cumulative.items()):
            axes[1].plot(dates, data['sharpe'], color=colors[i], linewidth=2,
                        label=strategy_name, alpha=0.8)

        axes[1].set_title('Sharpe Ratio Comparison')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Plot 3: Current performance bar chart
        current_returns = [formatted_strategies_data[name][frame].get('cumulative_return', 0)
                          for name in strategy_names]
        bars = axes[2].bar(strategy_names, current_returns, color=colors, alpha=0.7)
        axes[2].set_title(f'Current Performance - {current_date}')
        axes[2].set_ylabel('Cumulative Return')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Risk-return scatter (current position)
        current_sharpe = [formatted_strategies_data[name][frame].get('sharpe_ratio', 0)
                         for name in strategy_names]
        current_vol = [formatted_strategies_data[name][frame].get('volatility', 0.1)
                      for name in strategy_names]

        for i, name in enumerate(strategy_names):
            axes[3].scatter(current_vol[i], current_returns[i], color=colors[i],
                           s=100, alpha=0.8, label=name)

        axes[3].set_title('Risk-Return Profile')
        axes[3].set_xlabel('Volatility')
        axes[3].set_ylabel('Return')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Format time axes
        for i in range(2):
            format_financial_axis(axes[i], date_format="%m/%d")

        plt.tight_layout()
        return axes

    # Create and save animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(evaluation_dates),
        interval=1000//fps, blit=False, repeat=True
    )

    anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)


def save_individual_frames(
    data: pd.DataFrame,
    regime_data_sequence: List[pd.DataFrame],
    evaluation_dates: List[str],
    output_directory: str,
    frame_prefix: str = "frame",
    **kwargs
) -> List[str]:
    """
    Save individual frames for manual GIF creation or inspection.

    Args:
        data: Price data
        regime_data_sequence: List of regime DataFrames
        evaluation_dates: List of evaluation dates
        output_directory: Directory to save frames
        frame_prefix: Prefix for frame filenames
        **kwargs: Additional arguments for plotting

    Returns:
        List of saved frame paths
    """
    os.makedirs(output_directory, exist_ok=True)
    saved_paths = []

    # Create animator
    animator = RegimeAnimator(
        color_scheme=kwargs.get('color_scheme', 'professional'),
        style=kwargs.get('style', 'professional')
    )

    for i, (date, regime_data) in enumerate(zip(evaluation_dates, regime_data_sequence)):
        # Create single frame plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot current state (similar to animation frame)
        window_size = kwargs.get('window_size', 100)
        current_date_dt = pd.to_datetime(date)
        window_start = current_date_dt - pd.Timedelta(days=window_size)

        window_data = data[(data.index >= window_start) & (data.index <= current_date_dt)]
        window_regime_data = regime_data[regime_data.index <= current_date_dt]

        if len(window_data) > 0:
            # Price chart
            axes[0].plot(window_data.index, window_data['close'],
                        color='black', linewidth=2, alpha=0.8)

            # Add regime backgrounds
            if len(window_regime_data) > 0:
                n_states = int(window_regime_data['predicted_state'].max()) + 1
                regime_colors = get_regime_colors(n_states, kwargs.get('color_scheme', 'professional'))

                for regime in range(n_states):
                    regime_mask = window_regime_data['predicted_state'] == regime
                    if regime_mask.sum() > 0:
                        regime_dates = window_regime_data.index[regime_mask]
                        for d in regime_dates:
                            if d in window_data.index:
                                axes[0].axvspan(d, d + pd.Timedelta(days=1),
                                              color=regime_colors[regime], alpha=0.3)

            axes[0].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8,
                           linestyle='--', label=f'Current: {date}')
            axes[0].set_title(f'Price Analysis - {date}')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Regime timeline
            if len(window_regime_data) > 0:
                regime_names = get_regime_names(n_states)
                for regime in range(n_states):
                    regime_mask = window_regime_data['predicted_state'] == regime
                    if regime_mask.sum() > 0:
                        y_values = [regime] * regime_mask.sum()
                        axes[1].scatter(window_regime_data.index[regime_mask], y_values,
                                       c=regime_colors[regime], s=30, alpha=0.8,
                                       label=regime_names[regime])

                axes[1].set_yticks(range(n_states))
                axes[1].set_yticklabels(regime_names)

            axes[1].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8, linestyle='--')
            axes[1].set_title('Regime Timeline')
            axes[1].set_ylabel('Regime')
            axes[1].grid(True, alpha=0.3)

            # Confidence
            if len(window_regime_data) > 0 and 'confidence' in window_regime_data.columns:
                axes[2].plot(window_regime_data.index, window_regime_data['confidence'],
                            color='purple', linewidth=2, alpha=0.8)
                axes[2].fill_between(window_regime_data.index, 0, window_regime_data['confidence'],
                                    alpha=0.3, color='purple')

            axes[2].axvline(x=current_date_dt, color='red', linewidth=3, alpha=0.8, linestyle='--')
            axes[2].set_title('Model Confidence')
            axes[2].set_ylabel('Confidence')
            axes[2].set_xlabel('Date')
            axes[2].set_ylim(0, 1)
            axes[2].grid(True, alpha=0.3)

            # Format axes
            for ax in axes:
                format_financial_axis(ax, date_format="%m/%d")

        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(output_directory, f"{frame_prefix}_{i:04d}_{date}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(frame_path)

    return saved_paths


def create_gif_from_frames(
    frame_paths: List[str],
    output_path: str,
    duration: int = 500,
    loop: int = 0
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
            loop=loop
        )
    else:
        raise ValueError("No valid images found in frame paths")