"""
Core plotting utilities for hidden-regime package.

Provides shared visualization functions, consistent styling, and helper utilities
for creating clear, informative financial plots with proper date handling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib and/or seaborn not available. Install with: pip install matplotlib seaborn")

# Global style settings
FINANCIAL_STYLE_CONFIG = {
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
}

# Colorblind-friendly regime color palette
# Based on Okabe-Ito colorblind-safe research palette with user preferences
REGIME_COLORS = {
    'Bear': '#E69F00',      # Dark Orange (colorblind-safe alternative to red)
    'Crisis': '#CC79A7',    # Pink/Magenta (distinctive, alarming alternative)
    'Sideways': '#F0E442',  # Yellow (neutral, user-requested)
    'Bull': '#0072B2',      # Blue (positive, user-requested)
    'Euphoric': '#9467BD',  # Purple (euphoric, user-requested)
    'Unknown': '#7f7f7f',   # Gray (unchanged)
    # Fallback for numbered states
    0: '#E69F00',           # Dark Orange (typically bear)
    1: '#F0E442',           # Yellow (typically sideways)
    2: '#0072B2',           # Blue (typically bull) 
    3: '#CC79A7',           # Pink/Magenta (typically crisis)
    4: '#9467BD'            # Purple (typically euphoric)
}

# Colorblind-friendly seaborn color palette for regime analysis
REGIME_PALETTE = ['#E69F00', '#F0E442', '#0072B2', '#CC79A7', '#9467BD']

# Line styles for additional accessibility (for colorblind users)
REGIME_LINE_STYLES = {
    'Bear': '--',           # Dashed (negative/declining)
    'Crisis': ':',          # Dotted (volatile/uncertain)
    'Sideways': '-',        # Solid (stable/neutral)
    'Bull': '-',            # Solid (positive/rising)
    'Euphoric': '-.',       # Dash-dot (excited/extreme)
    'Unknown': '-'          # Solid (default)
}

# Marker styles for scatter plots (additional accessibility)
REGIME_MARKERS = {
    'Bear': 'v',            # Triangle down (declining)
    'Crisis': 'X',          # X marker (crisis/extreme)
    'Sideways': 's',        # Square (stable)
    'Bull': '^',            # Triangle up (rising)
    'Euphoric': '*',        # Star (euphoric/extreme)
    'Unknown': 'o'          # Circle (default)
}


def check_plotting_available():
    """Check if plotting libraries are available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting functionality requires matplotlib and seaborn. "
            "Install with: pip install matplotlib seaborn"
        )


def setup_financial_plot_style():
    """Set up consistent styling for financial plots."""
    check_plotting_available()
    
    # Set style
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update(FINANCIAL_STYLE_CONFIG)
    
    # Set seaborn style
    sns.set_palette(REGIME_PALETTE)
    sns.set_context("notebook", font_scale=1.0)


def get_regime_colors(regime_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get consistent color mapping for regimes.
    
    Args:
        regime_names: List of regime names. If None, returns full color dict.
        
    Returns:
        Dictionary mapping regime names to colors
    """
    if regime_names is None:
        return REGIME_COLORS.copy()
    
    color_map = {}
    for name in regime_names:
        if name in REGIME_COLORS:
            color_map[name] = REGIME_COLORS[name]
        else:
            # Fallback to default colors by index
            idx = len(color_map) % len(REGIME_PALETTE)
            color_map[name] = REGIME_PALETTE[idx]
    
    return color_map


def get_regime_line_styles(regime_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get consistent line style mapping for regimes (accessibility feature).
    
    Args:
        regime_names: List of regime names. If None, returns full line style dict.
        
    Returns:
        Dictionary mapping regime names to line styles
    """
    if regime_names is None:
        return REGIME_LINE_STYLES.copy()
    
    line_style_map = {}
    line_styles = ['-', '--', '-.', ':', '-']  # Cycle through available styles
    for i, name in enumerate(regime_names):
        if name in REGIME_LINE_STYLES:
            line_style_map[name] = REGIME_LINE_STYLES[name]
        else:
            line_style_map[name] = line_styles[i % len(line_styles)]
    
    return line_style_map


def get_regime_markers(regime_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get consistent marker mapping for regimes (accessibility feature).
    
    Args:
        regime_names: List of regime names. If None, returns full marker dict.
        
    Returns:
        Dictionary mapping regime names to marker styles
    """
    if regime_names is None:
        return REGIME_MARKERS.copy()
    
    marker_map = {}
    markers = ['o', 's', '^', 'v', '*', 'D', 'p']  # Cycle through available markers
    for i, name in enumerate(regime_names):
        if name in REGIME_MARKERS:
            marker_map[name] = REGIME_MARKERS[name]
        else:
            marker_map[name] = markers[i % len(markers)]
    
    return marker_map


def format_financial_axis(ax, dates: Optional[np.ndarray] = None, 
                         date_format: str = 'auto') -> None:
    """
    Format axis for financial time series plots.
    
    Args:
        ax: Matplotlib axis to format
        dates: Optional date array for x-axis
        date_format: Date format string or 'auto' for intelligent formatting
    """
    check_plotting_available()
    
    if dates is not None:
        # Convert to datetime if needed
        first_date = dates.iloc[0] if hasattr(dates, 'iloc') else dates[0]
        if not isinstance(first_date, (pd.Timestamp, np.datetime64)):
            dates = pd.to_datetime(dates)
        
        # Set x-axis - handle both arrays and Series
        if hasattr(dates, 'iloc'):
            # pandas Series
            start_date, end_date = dates.iloc[0], dates.iloc[-1]
        else:
            # numpy array or list
            start_date, end_date = dates[0], dates[-1]
        
        ax.set_xlim(start_date, end_date)
        
        # Intelligent date formatting based on data range
        if date_format == 'auto':
            date_range = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            
            if date_range <= 90:  # 3 months or less
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            elif date_range <= 730:  # 2 years or less
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            else:  # More than 2 years
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def create_regime_legend(regime_names: List[str], colors: Dict[str, str]) -> List:
    """
    Create consistent legend entries for regime plots.
    
    Args:
        regime_names: List of regime names
        colors: Color mapping dictionary
        
    Returns:
        List of legend handles
    """
    check_plotting_available()
    
    from matplotlib.patches import Patch
    
    legend_elements = []
    for name in regime_names:
        color = colors.get(name, REGIME_COLORS.get('Unknown'))
        legend_elements.append(Patch(facecolor=color, label=name))
    
    return legend_elements


def plot_returns_with_regimes(returns: np.ndarray, 
                             regime_states: np.ndarray,
                             dates: Optional[np.ndarray] = None,
                             regime_names: Optional[Dict[int, str]] = None,
                             ax: Optional[plt.Axes] = None,
                             title: str = "Returns Colored by Regime",
                             use_accessibility_features: bool = True) -> plt.Axes:
    """
    Plot returns time series colored by regime states with accessibility features.
    
    Args:
        returns: Array of returns
        regime_states: Array of regime state assignments
        dates: Optional date array
        regime_names: Optional mapping of state indices to names
        ax: Optional matplotlib axis
        title: Plot title
        use_accessibility_features: Use different markers for colorblind accessibility
        
    Returns:
        Matplotlib axis with the plot
    """
    check_plotting_available()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare regime names and styling
    unique_states = np.unique(regime_states)
    if regime_names is None:
        regime_names = {i: f"State {i}" for i in unique_states}
    
    colors = get_regime_colors(list(regime_names.values()))
    markers = get_regime_markers(list(regime_names.values())) if use_accessibility_features else {}
    
    # Plot returns colored by regime
    if dates is not None:
        x_values = dates
    else:
        x_values = np.arange(len(returns))
    
    for state in unique_states:
        mask = regime_states == state
        regime_name = regime_names.get(state, f"State {state}")
        color = colors.get(regime_name, REGIME_COLORS.get(state, '#7f7f7f'))
        marker = markers.get(regime_name, 'o') if use_accessibility_features else 'o'
        
        ax.scatter(x_values[mask], returns[mask], 
                  c=color, marker=marker, label=regime_name, alpha=0.7, s=20,
                  edgecolors='black', linewidth=0.5)  # Add edge for better visibility
    
    # Format plot
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Returns', fontsize=10)
    ax.set_xlabel('Date' if dates is not None else 'Time', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates if provided
    if dates is not None:
        format_financial_axis(ax, dates)
    
    return ax


def plot_regime_heatmap(data: Union[np.ndarray, pd.DataFrame],
                       regime_labels: Optional[List[str]] = None,
                       title: str = "Regime Analysis Heatmap",
                       ax: Optional[plt.Axes] = None,
                       cmap: str = 'viridis',
                       annot: bool = True,
                       fmt: str = '.3f') -> plt.Axes:
    """
    Create a heatmap for regime-related analysis.
    
    Args:
        data: Data matrix for heatmap
        regime_labels: Optional labels for regimes
        title: Plot title
        ax: Optional matplotlib axis
        cmap: Colormap name
        annot: Whether to annotate cells
        fmt: Format for annotations
        
    Returns:
        Matplotlib axis with the heatmap
    """
    check_plotting_available()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    if isinstance(data, pd.DataFrame):
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                   cbar_kws={'label': 'Value'})
    else:
        # Convert to DataFrame for better labeling
        if regime_labels is not None:
            if data.ndim == 2 and data.shape[0] == data.shape[1]:
                # Square matrix (e.g., transition matrix)
                df = pd.DataFrame(data, index=regime_labels, columns=regime_labels)
            else:
                df = pd.DataFrame(data, columns=regime_labels)
        else:
            df = pd.DataFrame(data)
        
        sns.heatmap(df, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                   cbar_kws={'label': 'Value'})
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return ax


def plot_regime_statistics(regime_stats: Dict[str, Dict[str, float]],
                          metrics: List[str] = ['mean_return', 'std_return', 'frequency'],
                          ax: Optional[plt.Axes] = None,
                          plot_type: str = 'box') -> plt.Axes:
    """
    Plot regime statistics in various formats.
    
    Args:
        regime_stats: Dictionary of regime statistics
        metrics: List of metrics to plot
        ax: Optional matplotlib axis
        plot_type: Type of plot ('box', 'bar', 'violin')
        
    Returns:
        Matplotlib axis with the plot
    """
    check_plotting_available()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    regimes = list(regime_stats.keys())
    colors = get_regime_colors(regimes)
    
    if plot_type == 'bar':
        # Bar plot for multiple metrics
        x = np.arange(len(regimes))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = [regime_stats[regime].get(metric, 0) for regime in regimes]
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=metric.replace('_', ' ').title(),
                         alpha=0.8)
            
            # Color bars by regime
            for bar, regime in zip(bars, regimes):
                bar.set_color(colors.get(regime, '#7f7f7f'))
        
        ax.set_xlabel('Regime')
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend()
        
    else:
        # For single metric visualization
        if len(metrics) == 1:
            metric = metrics[0]
            values = [regime_stats[regime].get(metric, 0) for regime in regimes]
            
            if plot_type == 'box':
                # Box plot (simulated since we have summary stats)
                bp = ax.boxplot([values], patch_artist=True)
                bp['boxes'][0].set_facecolor(colors.get(regimes[0], '#7f7f7f'))
            else:
                # Simple bar plot
                bars = ax.bar(regimes, values)
                for bar, regime in zip(bars, regimes):
                    bar.set_color(colors.get(regime, '#7f7f7f'))
            
            ax.set_ylabel(metric.replace('_', ' ').title())
        else:
            # Multiple metrics - use normalized comparison
            data_matrix = np.array([[regime_stats[regime].get(metric, 0) 
                                   for metric in metrics] for regime in regimes])
            
            # Normalize for comparison
            data_matrix_norm = (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0)
            
            im = ax.imshow(data_matrix_norm.T, cmap='RdBu_r', aspect='auto')
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels(regimes)
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Normalized Value')
    
    ax.set_title('Regime Statistics Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 150, 
              bbox_inches: str = 'tight') -> None:
    """
    Save plot with consistent settings.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save file
        dpi: Resolution for saving
        bbox_inches: Bounding box setting
    """
    check_plotting_available()
    
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {filepath}")
    except Exception as e:
        warnings.warn(f"Failed to save plot: {e}")


def create_subplot_grid(n_plots: int, max_cols: int = 3) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a grid of subplots for multiple regime visualizations.
    
    Args:
        n_plots: Number of plots needed
        max_cols: Maximum columns in grid
        
    Returns:
        Tuple of (figure, list of axes)
    """
    check_plotting_available()
    
    if n_plots <= max_cols:
        rows, cols = 1, n_plots
    else:
        rows = (n_plots + max_cols - 1) // max_cols
        cols = max_cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes[:n_plots]