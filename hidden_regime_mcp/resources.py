"""
MCP resources for Hidden Regime regime detection.

Resources provide URI-based access to regime data:
1. regime://{ticker}/current - Current regime status
2. regime://{ticker}/transitions - Transition probabilities
"""

import json
from typing import Any

from fastmcp.exceptions import ToolError

from hidden_regime_mcp.tools import detect_regime, get_transition_probabilities


async def get_current_regime_resource(ticker: str) -> str:
    """
    Get current regime for ticker as a resource.

    URI: regime://{ticker}/current

    Args:
        ticker: Stock symbol

    Returns:
        JSON string with current regime information or error

    Example:
        regime://SPY/current returns:
        {
            "ticker": "SPY",
            "current_regime": "bull",
            "confidence": 0.85,
            "mean_return": 0.02,
            "volatility": 0.15,
            "last_updated": "2025-10-31"
        }
    """
    try:
        result = await detect_regime(ticker=ticker.upper(), n_states=3)
        return json.dumps(result, indent=2)
    except ToolError as e:
        return json.dumps({"error": str(e), "ticker": ticker.upper()}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}", "ticker": ticker.upper()}, indent=2)


async def get_transitions_resource(ticker: str) -> str:
    """
    Get transition probabilities for ticker as a resource.

    URI: regime://{ticker}/transitions

    Args:
        ticker: Stock symbol

    Returns:
        JSON string with transition probability matrix or error

    Example:
        regime://SPY/transitions returns:
        {
            "ticker": "SPY",
            "transition_matrix": {
                "bull": {"bull": 0.85, "bear": 0.05, "sideways": 0.10},
                ...
            },
            "expected_durations": {...},
            "steady_state": {...}
        }
    """
    try:
        result = await get_transition_probabilities(ticker=ticker.upper(), n_states=3)
        return json.dumps(result, indent=2)
    except ToolError as e:
        return json.dumps({"error": str(e), "ticker": ticker.upper()}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}", "ticker": ticker.upper()}, indent=2)
