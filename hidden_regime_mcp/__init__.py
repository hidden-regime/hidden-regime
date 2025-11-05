"""
Hidden Regime MCP Server

Model Context Protocol server exposing Hidden Regime's HMM regime detection
capabilities to AI assistants like Claude Desktop.

This package provides:
- MCP tools for regime detection and analysis
- MCP resources for current regime data
- Local STDIO transport (runs on user's machine)

Example usage in Claude:
    "What's SPY's current market regime?"
    "Analyze NVDA's regime statistics for 2024"
    "What's the probability QQQ transitions to a bear regime?"
"""

__version__ = "0.1.0"

from hidden_regime_mcp.server import mcp, main

__all__ = ["mcp", "main", "__version__"]
