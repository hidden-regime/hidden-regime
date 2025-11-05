"""
Tests for Hidden Regime MCP server.

Tests server initialization and tool/resource registration.
"""

import pytest
from hidden_regime_mcp.server import mcp, main


class TestServer:
    """Test MCP server initialization."""

    @pytest.mark.unit
    def test_server_exists(self):
        """Test that server is created."""
        assert mcp is not None
        assert mcp.name == "Hidden Regime"
        assert mcp.version == "0.1.0"

    @pytest.mark.integration
    def test_server_has_correct_name(self):
        """Test that server has correct configuration."""
        assert mcp.name == "Hidden Regime"
        assert mcp.version == "0.1.0"

    @pytest.mark.integration
    def test_main_function_exists(self):
        """Test that main entry point exists."""
        assert callable(main)
