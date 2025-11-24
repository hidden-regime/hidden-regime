"""
MCP tools for Hidden Regime regime detection.

Implements three core tools:
1. detect_regime - Current regime detection
2. get_regime_statistics - Detailed regime analysis
3. get_transition_probabilities - Transition matrix and forecasting
"""

import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

import pandas as pd
from fastmcp.exceptions import ToolError

from hidden_regime import create_financial_pipeline
from hidden_regime.data import FinancialDataLoader
from hidden_regime_mcp.cache import get_cache
from hidden_regime_mcp.errors import (
    ValidationError,
    DataError,
    NetworkError,
    ModelError,
    MCPToolError,
    ErrorCode,
)
from hidden_regime_mcp.retry import (
    get_retry_manager,
    RetryConfig,
    get_yfinance_circuit_breaker,
)

logger = logging.getLogger(__name__)


def handle_pipeline_error(ticker: str, error: Exception, operation: str) -> None:
    """
    Standardized error handling for pipeline operations.

    Args:
        ticker: Stock symbol that failed
        error: The exception that occurred
        operation: Operation name (e.g., "Regime detection", "Regime statistics")

    Raises:
        ToolError: With standardized user-friendly message and error code
    """
    error_msg = str(error).lower()

    # Log the error for debugging
    logger.warning(f"{operation} failed for {ticker}: {error}", exc_info=True)

    # Classify error and create appropriate exception
    if "no data" in error_msg or "could not load" in error_msg:
        mcp_error = DataError(
            message=f"Unable to load data for {ticker}",
            details=str(error),
            retriable=True,
        )
    elif "insufficient" in error_msg:
        mcp_error = DataError(
            message=f"Insufficient data for {ticker}. Need at least 100 observations.",
            details=str(error),
            retriable=False,
        )
    elif "timeout" in error_msg or "connection" in error_msg:
        mcp_error = NetworkError(
            message=f"Network error while analyzing {ticker}",
            details=str(error),
            retriable=True,
        )
    elif "training" in error_msg or "hmm" in error_msg:
        mcp_error = ModelError(
            message=f"Model training failed for {ticker}",
            details=str(error),
            retriable=False,
        )
    else:
        mcp_error = MCPToolError(
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"{operation} failed for {ticker}",
            details=str(error),
            retriable=False,
        )

    # Convert to ToolError with additional context
    error_info = mcp_error.to_error_info()
    error_response = f"{error_info.message}"
    if error_info.suggestion:
        error_response += f". {error_info.suggestion}"

    raise ToolError(error_response)


def validate_ticker(ticker: str) -> None:
    """
    Validate ticker symbol format.

    Args:
        ticker: Stock symbol to validate

    Raises:
        ValidationError: If ticker is invalid
    """
    if not ticker:
        raise ValidationError("Ticker symbol is required")

    if not ticker.replace(".", "").replace("-", "").isalnum():
        raise ValidationError(
            message=f"Invalid ticker symbol: {ticker}",
            details="Tickers must be alphanumeric with optional . or - (e.g., AAPL, BRK.A, SPY)",
        )

    if len(ticker) > 10:
        raise ValidationError(
            message=f"Ticker symbol too long: {ticker}",
            details="Ticker symbols must be 10 characters or less",
        )


def validate_n_states(n_states: int) -> None:
    """
    Validate number of states.

    Args:
        n_states: Number of HMM states

    Raises:
        ValidationError: If n_states is out of range
    """
    if n_states < 2 or n_states > 5:
        raise ValidationError(
            message=f"Invalid number of regimes: {n_states}",
            details="Number of regimes (states) must be between 2 (minimal) and 5 (maximum)",
        )


def validate_date(date_str: Optional[str], param_name: str) -> None:
    """
    Validate date format and check for future dates.

    Args:
        date_str: Date string (YYYY-MM-DD) or None
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If date format is invalid or date is in the future
    """
    if date_str is None:
        return

    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValidationError(
            message=f"Invalid date format for {param_name}",
            details=f"Expected YYYY-MM-DD format, got '{date_str}'",
        )

    # Check if date is in the future
    if parsed_date.date() > datetime.now().date():
        raise ValidationError(
            message=f"{param_name} cannot be in the future",
            details=f"Date must be on or before today ({datetime.now().date()})",
        )


def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> None:
    """
    Validate that start_date is before end_date.

    Args:
        start_date: Start date string (YYYY-MM-DD) or None
        end_date: End date string (YYYY-MM-DD) or None

    Raises:
        ValidationError: If start_date is after end_date
    """
    if start_date is None or end_date is None:
        return

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if start > end:
        raise ValidationError(
            message="Invalid date range",
            details=f"start_date ({start_date}) must be before or equal to end_date ({end_date})",
        )


def calculate_price_performance(data_output: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate price performance metrics (YTD, 30d, 7d returns).

    Args:
        data_output: Pipeline data output DataFrame with price data

    Returns:
        Dictionary with performance metrics
    """

    # Guard against empty DataFrame
    if len(data_output) == 0:
        raise ValueError("data_output DataFrame is empty - no price data available")

    # Get current price and date
    current_price = float(data_output["close"].iloc[-1])
    current_date = data_output.index[-1]

    # Calculate YTD return
    year_start = pd.Timestamp(f"{current_date.year}-01-01", tz=current_date.tz)
    ytd_data = data_output[data_output.index >= year_start]
    if len(ytd_data) > 1:
        ytd_start_price = float(ytd_data["close"].iloc[0])
        ytd_return = (current_price - ytd_start_price) / ytd_start_price if ytd_start_price != 0 else 0.0
    else:
        ytd_return = 0.0

    # Calculate 30-day return
    date_30d_ago = current_date - timedelta(days=30)
    data_30d = data_output[data_output.index >= date_30d_ago]
    if len(data_30d) > 1:
        price_30d_ago = float(data_30d["close"].iloc[0])
        return_30d = (current_price - price_30d_ago) / price_30d_ago if price_30d_ago != 0 else 0.0
    else:
        return_30d = 0.0

    # Calculate 7-day return
    date_7d_ago = current_date - timedelta(days=7)
    data_7d = data_output[data_output.index >= date_7d_ago]
    if len(data_7d) > 1:
        price_7d_ago = float(data_7d["close"].iloc[0])
        return_7d = (current_price - price_7d_ago) / price_7d_ago if price_7d_ago != 0 else 0.0
    else:
        return_7d = 0.0

    # Determine price trend
    if return_7d > 0.02:
        trend = "up"
    elif return_7d < -0.02:
        trend = "down"
    else:
        trend = "consolidating"

    return {
        "ytd_return": round(ytd_return, 4),
        "ytd_return_pct": f"{ytd_return*100:+.1f}%",
        "30d_return": round(return_30d, 4),
        "30d_return_pct": f"{return_30d*100:+.1f}%",
        "7d_return": round(return_7d, 4),
        "7d_return_pct": f"{return_7d*100:+.1f}%",
        "current_price": round(current_price, 2),
        "price_trend": trend,
    }


def determine_regime_status(days_in_regime: float, expected_duration: float) -> str:
    """
    Determine regime status based on completion percentage.

    Args:
        days_in_regime: Current days in regime
        expected_duration: Expected total duration

    Returns:
        Status string: "early", "mid", "mature", or "overdue"
    """
    if expected_duration <= 0:
        return "unknown"

    percent_complete = (days_in_regime / expected_duration) * 100

    if percent_complete < 25:
        return "early"
    elif percent_complete < 60:
        return "mid"
    elif percent_complete < 100:
        return "mature"
    else:
        return "overdue"


def analyze_regime_stability(analysis_output: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze regime stability and transition history.

    Args:
        analysis_output: Pipeline analysis output DataFrame

    Returns:
        Dictionary with stability metrics
    """
    # Guard against empty DataFrame
    if len(analysis_output) == 0:
        raise ValueError("analysis_output DataFrame is empty - no regime data available")

    # Get last 30 days of data
    last_30_days = analysis_output.tail(min(30, len(analysis_output)))

    # Count unique regimes in last 30 days
    # Support both old (regime_name) and new (regime_label) column names
    regime_column = "regime_label" if "regime_label" in last_30_days.columns else "regime_name"

    unique_regimes = last_30_days[regime_column].unique()
    regime_changes = len(unique_regimes) - 1

    # Determine stability
    if regime_changes == 0:
        stability = "stable"
    elif regime_changes <= 2:
        stability = "moderate"
    else:
        stability = "volatile"

    # Get previous regime and transition date
    current_regime = analysis_output[regime_column].iloc[-1]
    current_episode = analysis_output["regime_episode"].iloc[-1]

    # Find where current episode started
    episode_data = analysis_output[analysis_output["regime_episode"] == current_episode]
    transition_date = episode_data.index[0].strftime("%Y-%m-%d")

    # Find previous regime
    if len(analysis_output) > len(episode_data):
        previous_data = analysis_output[analysis_output["regime_episode"] < current_episode]
        if len(previous_data) > 0:
            previous_regime = previous_data[regime_column].iloc[-1]
        else:
            previous_regime = "unknown"
    else:
        previous_regime = "none"

    return {
        "regime_stability": stability,
        "recent_transitions": regime_changes,
        "previous_regime": previous_regime,
        "last_transition_date": transition_date,
    }


def generate_regime_interpretation(
    regime_name: str, mean_return: float, volatility: float, price_perf: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate human-readable interpretation of regime.

    Args:
        regime_name: Detected regime name
        mean_return: Mean return in regime
        volatility: Volatility in regime
        price_perf: Price performance metrics

    Returns:
        Dictionary with interpretation and explanation
    """
    # Determine volatility level
    if volatility > 0.05:
        vol_desc = "high volatility"
    elif volatility > 0.02:
        vol_desc = "moderate volatility"
    else:
        vol_desc = "low volatility"

    # Determine return level
    if mean_return > 0.002:
        return_desc = "strong positive returns"
    elif mean_return > 0:
        return_desc = "modest positive returns"
    elif mean_return > -0.002:
        return_desc = "modest negative returns"
    else:
        return_desc = "significant negative returns"

    # Create interpretation
    interpretation = f"{vol_desc.capitalize()} phase with {return_desc}"

    # Create explanation reconciling regime with price performance
    ytd_return = price_perf["ytd_return"]
    recent_30d = price_perf["30d_return"]

    if regime_name in ["bearish", "bear"] and ytd_return > 0.10:
        explanation = (
            f"While the stock is up {price_perf['ytd_return_pct']} YTD, recent behavior "
            f"shows increased volatility ({volatility*100:.1f}% daily std dev) and "
            f"{abs(recent_30d)*100:.1f}% {'decline' if recent_30d < 0 else 'gain'} over 30 days. "
            f"The model detects this as a {regime_name} regime, indicating transition from "
            f"the strong uptrend that characterized earlier performance. This is typical "
            f"consolidation or correction after a major rally."
        )
    elif regime_name in ["bullish", "bull"] and ytd_return < -0.10:
        explanation = (
            f"Despite being down {price_perf['ytd_return_pct']} YTD, recent behavior "
            f"shows improving conditions with {vol_desc} and {return_desc}. "
            f"The model detects this as a {regime_name} regime, indicating potential "
            f"recovery or bottom formation."
        )
    else:
        explanation = (
            f"The {regime_name} regime is characterized by {vol_desc} "
            f"({volatility*100:.1f}% daily) and {return_desc} "
            f"({mean_return*100:.2f}% daily avg). "
            f"YTD performance is {price_perf['ytd_return_pct']}, with recent 30-day "
            f"returns at {price_perf['30d_return_pct']}."
        )

    return {"interpretation": interpretation, "explanation": explanation}


async def detect_regime(
    ticker: str,
    n_states: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect market regime for a stock using HMM with temporal context and interpretation.

    Args:
        ticker: Stock symbol (e.g., 'SPY', 'AAPL', 'NVDA')
        n_states: Number of regimes (2-5, default: 3)
        start_date: Start date for analysis (YYYY-MM-DD, optional)
        end_date: End date for analysis (YYYY-MM-DD, optional)

    Returns:
        Dictionary containing:

        Basic regime information:
        - ticker: Stock symbol
        - current_regime: Current regime name (e.g., 'bull', 'bear', 'sideways')
        - confidence: Confidence in current regime (0-1)
        - mean_return: Mean return for current regime
        - volatility: Volatility for current regime
        - last_updated: Date of last data point
        - n_states: Number of states used
        - analysis_period: Dict with start and end dates

        Temporal context:
        - days_in_regime: Number of days in current regime
        - expected_duration: Expected duration of this regime (days)
        - percent_complete: Percentage of expected duration completed
        - regime_status: "early", "mid", "mature", or "overdue"
        - days_until_expected_transition: Days until expected regime change

        Price context:
        - price_performance: Dict with ytd/30d/7d returns and percentages
        - current_price: Latest closing price
        - price_trend: "up", "down", or "flat"

        Stability metrics:
        - regime_stability: "stable", "moderate", or "volatile"
        - recent_transitions: Number of regime changes in last 30 days
        - previous_regime: Previous regime name
        - last_transition_date: Date of transition to current regime

        Interpretation:
        - interpretation: Brief human-readable description
        - explanation: Detailed narrative reconciling regime with price action

    Example:
        {
            "ticker": "NVDA",
            "current_regime": "bearish",
            "confidence": 0.72,
            "mean_return": -0.001234,
            "volatility": 0.034,
            "last_updated": "2025-10-31",
            "n_states": 3,
            "analysis_period": {"start": "2024-01-01", "end": "2025-10-31"},
            "days_in_regime": 12,
            "expected_duration": 15.3,
            "percent_complete": 78.4,
            "regime_status": "mature",
            "days_until_expected_transition": 3.3,
            "price_performance": {
                "ytd_return": 0.187,
                "ytd_return_pct": "+18.7%",
                "30d_return": -0.034,
                "30d_return_pct": "-3.4%",
                ...
            },
            "current_price": 142.50,
            "price_trend": "down",
            "regime_stability": "stable",
            "recent_transitions": 1,
            "previous_regime": "bullish",
            "last_transition_date": "2025-10-19",
            "interpretation": "High volatility phase with modest negative returns",
            "explanation": "While the stock is up +18.7% YTD, recent behavior shows..."
        }

    Raises:
        ToolError: If parameters are invalid or regime detection fails
    """
    # Validate inputs
    try:
        validate_ticker(ticker)
        validate_n_states(n_states)
        validate_date(start_date, "start_date")
        validate_date(end_date, "end_date")
        validate_date_range(start_date, end_date)
    except ValidationError as e:
        error_info = e.to_error_info()
        msg = f"{error_info.message}"
        if error_info.details:
            msg += f": {error_info.details}"
        raise ToolError(msg)

    # Check cache first
    cache = get_cache()
    cached_result = cache.get(ticker, n_states, start_date, end_date, tool_name="detect_regime")
    if cached_result is not None:
        logger.info(f"Cache hit for {ticker} (regime detection) - returning cached result")
        return cached_result

    logger.info(f"Cache miss for {ticker} - running fresh regime analysis")

    try:
        # Create pipeline and run regime detection
        logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
        pipeline = create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            start_date=start_date,
            end_date=end_date,
            include_report=False,  # Don't generate report, return result object
        )

        logger.info(f"Training {n_states}-state HMM model for {ticker}...")
        _ = pipeline.update()  # Returns a string, but we'll use pipeline internals

        # Validate pipeline outputs are not empty
        if len(pipeline.interpreter_output) == 0:
            raise ToolError(
                f"No data available for {ticker} in the specified date range. "
                "This may be due to weekends, holidays, or insufficient trading data."
            )

        # Log data loading success
        n_observations = len(pipeline.data_output)
        date_range = f"{pipeline.data_output.index[0].strftime('%Y-%m-%d')} to {pipeline.data_output.index[-1].strftime('%Y-%m-%d')}"
        logger.info(f"Downloaded {n_observations} observations for {ticker} ({date_range})")
        logger.info(f"Model training completed for {ticker}")

        # Extract current regime information from pipeline interpreter_output (v2.0.0)
        latest = pipeline.interpreter_output.iloc[-1]

        # Support both old (regime_name) and new (regime_label) column names
        regime_column = "regime_label" if "regime_label" in pipeline.interpreter_output.columns else "regime_name"

        # Normalize regime name to lowercase and map to expected format
        regime_name = str(latest[regime_column]).lower()
        # Map new FinancialInterpreter labels to old format for backward compatibility
        regime_mapping = {
            "bull": "bullish",
            "bear": "bearish",
            "sideways": "sideways",
            "crisis": "crisis",
            "mixed": "mixed"
        }
        regime_name = regime_mapping.get(regime_name, regime_name)

        confidence = float(latest["confidence"])
        expected_return = float(latest["expected_return"])
        expected_volatility = float(latest["expected_volatility"])
        days_in_regime = int(latest["days_in_regime"])
        expected_duration = float(latest["expected_duration"])

        # Determine actual date range used
        actual_start = pipeline.data_output.index[0].strftime("%Y-%m-%d")
        actual_end = pipeline.data_output.index[-1].strftime("%Y-%m-%d")

        # Calculate temporal context
        percent_complete = (days_in_regime / expected_duration) * 100 if expected_duration > 0 else 0
        days_until_transition = max(0, expected_duration - days_in_regime)
        regime_status = determine_regime_status(days_in_regime, expected_duration)

        # Calculate price performance context
        price_perf = calculate_price_performance(pipeline.data_output)

        # Analyze regime stability
        stability_metrics = analyze_regime_stability(pipeline.interpreter_output)

        # Generate interpretation
        interpretation = generate_regime_interpretation(
            regime_name, expected_return, expected_volatility, price_perf
        )

        response = {
            # Basic regime information
            "ticker": ticker.upper(),
            "current_regime": regime_name,
            "confidence": round(confidence, 4),
            "mean_return": round(expected_return, 6),
            "volatility": round(expected_volatility, 6),
            "last_updated": actual_end,
            "n_states": n_states,
            "analysis_period": {"start": actual_start, "end": actual_end},
            # Temporal context
            "days_in_regime": days_in_regime,
            "expected_duration": round(expected_duration, 2),
            "percent_complete": round(percent_complete, 1),
            "regime_status": regime_status,
            "days_until_expected_transition": round(days_until_transition, 1),
            # Price context
            "price_performance": price_perf,
            "current_price": price_perf["current_price"],
            "price_trend": price_perf["price_trend"],
            # Stability metrics
            "regime_stability": stability_metrics["regime_stability"],
            "recent_transitions": stability_metrics["recent_transitions"],
            "previous_regime": stability_metrics["previous_regime"],
            "last_transition_date": stability_metrics["last_transition_date"],
            # Interpretation
            "interpretation": interpretation["interpretation"],
            "explanation": interpretation["explanation"],
        }

        # Cache the result
        cache.set(ticker, n_states, response, start_date, end_date, tool_name="detect_regime")
        logger.info(f"Regime detection completed for {ticker} - result cached")

        return response

    except Exception as e:
        handle_pipeline_error(ticker, e, "Regime detection")


async def get_regime_statistics(
    ticker: str,
    n_states: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get statistical analysis of detected regimes.

    Args:
        ticker: Stock symbol
        n_states: Number of regimes (2-5, default: 3)
        start_date: Start date for analysis (YYYY-MM-DD, optional)
        end_date: End date for analysis (YYYY-MM-DD, optional)

    Returns:
        Dictionary containing:
        - ticker: Stock symbol
        - regimes: Dict mapping regime names to statistics
        - analysis_period: Dict with start and end dates
        - n_states: Number of states used

        Each regime includes:
        - mean_return: Average return in this regime
        - volatility: Standard deviation of returns
        - duration_days: Average duration in days
        - win_rate: Percentage of positive return days
        - observations: Number of days in this regime

    Example:
        {
            "ticker": "SPY",
            "regimes": {
                "bull": {
                    "mean_return": 0.03,
                    "volatility": 0.10,
                    "duration_days": 45.2,
                    "win_rate": 0.68,
                    "observations": 234
                },
                ...
            },
            "analysis_period": {...},
            "n_states": 3
        }

    Raises:
        ToolError: If parameters are invalid or analysis fails
    """
    # Validate inputs
    try:
        validate_ticker(ticker)
        validate_n_states(n_states)
        validate_date(start_date, "start_date")
        validate_date(end_date, "end_date")
        validate_date_range(start_date, end_date)
    except ValidationError as e:
        error_info = e.to_error_info()
        msg = f"{error_info.message}"
        if error_info.details:
            msg += f": {error_info.details}"
        raise ToolError(msg)

    # Check cache first
    cache = get_cache()
    cached_result = cache.get(ticker, n_states, start_date, end_date, tool_name="get_regime_statistics")
    if cached_result is not None:
        logger.info(f"Cache hit for {ticker} (regime statistics) - returning cached result")
        return cached_result

    logger.info(f"Cache miss for {ticker} - computing regime statistics")

    try:
        # Create pipeline and run regime detection
        logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
        pipeline = create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            start_date=start_date,
            end_date=end_date,
            include_report=False,  # Don't generate report, return result object
        )

        logger.info(f"Training {n_states}-state HMM model for {ticker}...")
        _ = pipeline.update()  # Returns a string, but we'll use pipeline internals

        # Validate pipeline outputs are not empty
        if len(pipeline.interpreter_output) == 0:
            raise ToolError(
                f"No data available for {ticker} in the specified date range. "
                "This may be due to weekends, holidays, or insufficient trading data."
            )

        # Log data loading success
        n_observations = len(pipeline.data_output)
        date_range = f"{pipeline.data_output.index[0].strftime('%Y-%m-%d')} to {pipeline.data_output.index[-1].strftime('%Y-%m-%d')}"
        logger.info(f"Downloaded {n_observations} observations for {ticker} ({date_range})")
        logger.info(f"Analyzing regime statistics for {ticker}...")

        # Build regime statistics from interpreter_output (v2.0.0)
        analysis = pipeline.interpreter_output
        regimes = {}

        # Group by regime_label (or regime_name for backward compatibility) and calculate statistics
        regime_column = "regime_label" if "regime_label" in analysis.columns else "regime_name"
        for regime_name in analysis[regime_column].unique():
            regime_data = analysis[analysis[regime_column] == regime_name]

            regimes[regime_name] = {
                "mean_return": round(float(regime_data["expected_return"].mean()), 6),
                "volatility": round(float(regime_data["expected_volatility"].mean()), 6),
                "duration_days": round(float(regime_data["expected_duration"].mean()), 2),
                "win_rate": round(float(regime_data["win_rate"].mean()), 4),
                "observations": int(len(regime_data)),
            }

        # Determine actual date range
        actual_start = pipeline.data_output.index[0].strftime("%Y-%m-%d")
        actual_end = pipeline.data_output.index[-1].strftime("%Y-%m-%d")
        total_days = len(pipeline.data_output)

        response = {
            "ticker": ticker.upper(),
            "regimes": regimes,
            "analysis_period": {
                "start": actual_start,
                "end": actual_end,
                "total_days": total_days,
            },
            "n_states": n_states,
        }

        # Cache the result
        cache.set(ticker, n_states, response, start_date, end_date, tool_name="get_regime_statistics")
        logger.info(f"Regime statistics computed for {ticker} - result cached")

        return response

    except Exception as e:
        handle_pipeline_error(ticker, e, "Regime statistics")


async def get_transition_probabilities(
    ticker: str, n_states: int = 3
) -> Dict[str, Any]:
    """
    Get regime transition probabilities and expected durations.

    Args:
        ticker: Stock symbol
        n_states: Number of regimes (2-5, default: 3)

    Returns:
        Dictionary containing:
        - ticker: Stock symbol
        - transition_matrix: Dict mapping from_regime -> to_regime -> probability
        - expected_durations: Dict mapping regime -> expected days
        - steady_state: Dict mapping regime -> long-term probability
        - n_states: Number of states used

    Example:
        {
            "ticker": "SPY",
            "transition_matrix": {
                "bull": {"bull": 0.85, "bear": 0.05, "sideways": 0.10},
                "bear": {"bull": 0.10, "bear": 0.80, "sideways": 0.10},
                "sideways": {"bull": 0.20, "bear": 0.15, "sideways": 0.65}
            },
            "expected_durations": {
                "bull": 20.0,
                "bear": 10.0,
                "sideways": 6.7
            },
            "steady_state": {
                "bull": 0.40,
                "bear": 0.25,
                "sideways": 0.35
            },
            "n_states": 3
        }

    Raises:
        ToolError: If parameters are invalid or analysis fails
    """
    # Validate inputs
    try:
        validate_ticker(ticker)
        validate_n_states(n_states)
    except ValidationError as e:
        error_info = e.to_error_info()
        msg = f"{error_info.message}"
        if error_info.details:
            msg += f": {error_info.details}"
        raise ToolError(msg)

    # Check cache first
    cache = get_cache()
    cached_result = cache.get(ticker, n_states, tool_name="get_transition_probabilities")
    if cached_result is not None:
        logger.info(f"Cache hit for {ticker} (transition probabilities) - returning cached result")
        return cached_result

    logger.info(f"Cache miss for {ticker} - computing transition probabilities")

    try:
        # Create pipeline and run regime detection
        logger.info(f"Fetching data for {ticker} from Yahoo Finance...")
        pipeline = create_financial_pipeline(
            ticker=ticker, n_states=n_states, include_report=False
        )

        logger.info(f"Training {n_states}-state HMM model for {ticker}...")
        _ = pipeline.update()  # Returns a string, but we'll use pipeline internals

        # Validate pipeline outputs are not empty
        if len(pipeline.interpreter_output) == 0:
            raise ToolError(
                f"No data available for {ticker}. "
                "This may be due to insufficient trading data or invalid ticker symbol."
            )

        # Log data loading success
        n_observations = len(pipeline.data_output)
        logger.info(f"Downloaded {n_observations} observations for {ticker}")
        logger.info(f"Computing transition probabilities for {ticker}...")

        # Get transition matrix from model
        trans_matrix = pipeline.model.transition_matrix_

        # Get regime labels from interpreter_output (v2.0.0)
        analysis = pipeline.interpreter_output

        # Support both old (regime_name) and new (regime_label) column names
        regime_column = "regime_label" if "regime_label" in analysis.columns else "regime_name"

        regime_names = analysis[regime_column].unique()
        state_to_regime = {int(analysis.iloc[0]["predicted_state"]): regime_names[0]}

        # Build mapping from state number to regime name
        for i, regime_name in enumerate(regime_names):
            regime_rows = analysis[analysis[regime_column] == regime_name]
            if len(regime_rows) > 0:
                state_num = int(regime_rows.iloc[0]["predicted_state"])
                state_to_regime[state_num] = regime_name

        # Build transition matrix in readable format with regime names
        transition_matrix = {}
        for i in range(n_states):
            from_regime = state_to_regime.get(i, f"state_{i}")
            transition_matrix[from_regime] = {}
            for j in range(n_states):
                to_regime = state_to_regime.get(j, f"state_{j}")
                prob = float(trans_matrix[i, j])
                transition_matrix[from_regime][to_regime] = round(prob, 4)

        # Calculate expected durations (1 / (1 - self-transition-prob))
        expected_durations = {}
        for i in range(n_states):
            regime = state_to_regime.get(i, f"state_{i}")
            self_prob = float(trans_matrix[i, i])
            if self_prob < 1.0:
                duration = 1.0 / (1.0 - self_prob)
                expected_durations[regime] = round(duration, 2)
            else:
                expected_durations[regime] = float("inf")

        # Calculate steady state probabilities using empirical frequencies
        predicted_states = analysis["predicted_state"].values
        steady_state = {}
        for i in range(n_states):
            regime = state_to_regime.get(i, f"state_{i}")
            count = (predicted_states == i).sum()
            steady_state[regime] = round(count / len(predicted_states), 4)

        response = {
            "ticker": ticker.upper(),
            "transition_matrix": transition_matrix,
            "expected_durations": expected_durations,
            "steady_state": steady_state,
            "n_states": n_states,
        }

        # Cache the result
        cache.set(ticker, n_states, response, tool_name="get_transition_probabilities")
        logger.info(f"Transition probabilities computed for {ticker} - result cached")

        return response

    except Exception as e:
        handle_pipeline_error(ticker, e, "Transition probability analysis")
