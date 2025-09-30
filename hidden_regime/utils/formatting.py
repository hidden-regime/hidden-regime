"""
Formatting utilities for consistent display across the hidden regime system.

Provides standardized formatting for strategy names, regime types, and other
display elements to ensure consistency between reports, visualizations, and outputs.
"""


def format_strategy_name(strategy_name: str) -> str:
    """
    Format strategy name for consistent display across reports and visualizations.

    Args:
        strategy_name: Raw strategy name (e.g., 'ta_rsi', 'hmm_regime_following')

    Returns:
        Formatted strategy name for display (e.g., 'Ta Rsi', 'Hmm Regime Following')
    """
    # Handle technical indicators
    if strategy_name.startswith("ta_"):
        indicator_name = strategy_name[3:]  # Remove 'ta_' prefix
        return f"Ta {indicator_name.replace('_', ' ').title()}"

    # Handle HMM strategies
    if strategy_name.startswith("hmm_"):
        strategy_type = strategy_name[4:]  # Remove 'hmm_' prefix
        return f"Hmm {strategy_type.replace('_', ' ').title()}"

    # Handle other strategies
    return strategy_name.replace("_", " ").title()


def format_strategy_names_dict(strategies_dict: dict) -> dict:
    """
    Format all strategy names in a dictionary while preserving the structure.

    Args:
        strategies_dict: Dictionary with strategy names as keys

    Returns:
        Dictionary with formatted strategy names as keys
    """
    return {
        format_strategy_name(strategy_name): strategy_data
        for strategy_name, strategy_data in strategies_dict.items()
    }


def format_regime_type(regime_type: str) -> str:
    """
    Format regime type for display.

    Args:
        regime_type: Raw regime type (e.g., 'BULLISH', 'bearish')

    Returns:
        Formatted regime type (e.g., 'Bullish', 'Bearish')
    """
    return regime_type.title()


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage.

    Args:
        value: Decimal value (e.g., 0.0523 for 5.23%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., '5.23%')
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format a value as currency.

    Args:
        value: Monetary value
        decimals: Number of decimal places

    Returns:
        Formatted currency string (e.g., '$123,456')
    """
    return f"${value:,.{decimals}f}"
