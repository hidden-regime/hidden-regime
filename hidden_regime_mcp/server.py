"""
Hidden Regime MCP Server

Main FastMCP server exposing Hidden Regime's HMM regime detection capabilities
to AI assistants via the Model Context Protocol.

This server runs locally using STDIO transport - no hosting required.
"""

from fastmcp import FastMCP

from hidden_regime_mcp.tools import (
    detect_regime,
    get_regime_statistics,
    get_transition_probabilities,
)
from hidden_regime_mcp.resources import (
    get_current_regime_resource,
    get_transitions_resource,
)

# Create FastMCP server
mcp = FastMCP(
    name="Hidden Regime",
    version="0.1.0",
    instructions="HMM-based market regime detection for financial analysis",
)

# Register tools
mcp.tool(detect_regime)
mcp.tool(get_regime_statistics)
mcp.tool(get_transition_probabilities)

# Register resources with URI templates
mcp.resource("regime://{ticker}/current")(get_current_regime_resource)
mcp.resource("regime://{ticker}/transitions")(get_transitions_resource)


def main():
    """
    Main entry point for the MCP server.

    Starts the server in STDIO mode for local Claude Desktop integration.
    """
    # Run server with STDIO transport (local execution)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
