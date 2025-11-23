"""
Tests for MCP prompts.

Tests verify that all prompts:
1. Generate valid prompt text
2. Handle arguments correctly (required and optional)
3. Reference appropriate MCP tools
4. Include proper instructions and formatting
"""

import pytest

from hidden_regime_mcp.prompts import (
    regime_quick_check,
    regime_deep_dive,
    regime_strategy_advisor,
    regime_multi_asset_comparison,
    regime_risk_assessment,
    regime_historical_analogs,
    regime_portfolio_review,
)


class TestPromptGeneration:
    """Test that all prompts generate valid text"""

    @pytest.mark.asyncio
    async def test_regime_quick_check_basic(self):
        """Test quick check prompt generation"""
        prompt = await regime_quick_check("SPY")

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Substantial prompt
        assert "SPY" in prompt
        assert "detect_regime" in prompt.lower()
        assert "3-4 sentence" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_deep_dive_basic(self):
        """Test deep dive prompt generation"""
        prompt = await regime_deep_dive("AAPL")

        assert isinstance(prompt, str)
        assert len(prompt) > 500  # Comprehensive prompt
        assert "AAPL" in prompt
        assert "detect_regime" in prompt.lower()
        assert "get_regime_statistics" in prompt.lower()
        assert "get_transition_probabilities" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_strategy_advisor_basic(self):
        """Test strategy advisor prompt generation"""
        prompt = await regime_strategy_advisor("NVDA")

        assert isinstance(prompt, str)
        assert len(prompt) > 400
        assert "NVDA" in prompt
        assert "detect_regime" in prompt.lower()
        assert "strategy" in prompt.lower() or "strategies" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_multi_asset_comparison_basic(self):
        """Test multi-asset comparison prompt generation"""
        prompt = await regime_multi_asset_comparison("SPY,QQQ,IWM")

        assert isinstance(prompt, str)
        assert len(prompt) > 400
        assert "SPY,QQQ,IWM" in prompt
        assert "detect_regime" in prompt.lower()
        assert "comparison" in prompt.lower() or "compare" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_risk_assessment_basic(self):
        """Test risk assessment prompt generation"""
        prompt = await regime_risk_assessment("TSLA")

        assert isinstance(prompt, str)
        assert len(prompt) > 500
        assert "TSLA" in prompt
        assert "detect_regime" in prompt.lower()
        assert "get_transition_probabilities" in prompt.lower()
        assert "risk" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_historical_analogs_basic(self):
        """Test historical analogs prompt generation"""
        prompt = await regime_historical_analogs("GLD")

        assert isinstance(prompt, str)
        assert len(prompt) > 400
        assert "GLD" in prompt
        assert "detect_regime" in prompt.lower()
        assert "get_regime_statistics" in prompt.lower()
        assert "historical" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_portfolio_review_basic(self):
        """Test portfolio review prompt generation"""
        portfolio_json = '{"SPY": 0.5, "QQQ": 0.3, "GLD": 0.2}'
        prompt = await regime_portfolio_review(portfolio_json)

        assert isinstance(prompt, str)
        assert len(prompt) > 500
        assert portfolio_json in prompt
        assert "detect_regime" in prompt.lower()
        assert "portfolio" in prompt.lower()


class TestPromptArguments:
    """Test argument validation and defaults"""

    @pytest.mark.asyncio
    async def test_regime_quick_check_ticker_normalization(self):
        """Test ticker normalization (uppercase, strip)"""
        prompt_lower = await regime_quick_check("spy")
        prompt_spaces = await regime_quick_check("  SPY  ")

        assert "SPY" in prompt_lower
        assert "spy" not in prompt_lower
        assert "SPY" in prompt_spaces

    @pytest.mark.asyncio
    async def test_regime_deep_dive_default_period(self):
        """Test default lookback period"""
        prompt = await regime_deep_dive("AAPL")
        assert "2y" in prompt  # Default value

    @pytest.mark.asyncio
    async def test_regime_deep_dive_custom_period(self):
        """Test custom lookback period"""
        prompt_1y = await regime_deep_dive("AAPL", lookback_period="1y")
        prompt_5y = await regime_deep_dive("AAPL", lookback_period="5y")

        assert "1y" in prompt_1y
        assert "5y" in prompt_5y

    @pytest.mark.asyncio
    async def test_regime_strategy_advisor_default_risk(self):
        """Test default risk tolerance"""
        prompt = await regime_strategy_advisor("SPY")
        assert "medium" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_strategy_advisor_risk_levels(self):
        """Test all risk tolerance levels"""
        for risk in ["low", "medium", "high"]:
            prompt = await regime_strategy_advisor("SPY", risk_tolerance=risk)
            assert risk in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_strategy_advisor_invalid_risk(self):
        """Test invalid risk tolerance defaults to medium"""
        prompt = await regime_strategy_advisor("SPY", risk_tolerance="invalid")
        assert "medium" in prompt.lower()

    @pytest.mark.asyncio
    async def test_regime_historical_analogs_default_period(self):
        """Test default lookback period for historical analogs"""
        prompt = await regime_historical_analogs("SPY")
        assert "5y" in prompt  # Default value

    @pytest.mark.asyncio
    async def test_regime_historical_analogs_custom_period(self):
        """Test custom lookback period for historical analogs"""
        prompt = await regime_historical_analogs("SPY", lookback_period="10y")
        assert "10y" in prompt


class TestPromptContent:
    """Test prompt content quality and completeness"""

    @pytest.mark.asyncio
    async def test_quick_check_includes_key_sections(self):
        """Verify quick check includes all required analysis points"""
        prompt = await regime_quick_check("SPY")

        # Key analysis points
        assert "confidence" in prompt.lower()
        assert "regime" in prompt.lower()
        assert "performance" in prompt.lower()

    @pytest.mark.asyncio
    async def test_deep_dive_includes_all_tools(self):
        """Verify deep dive references all three tools"""
        prompt = await regime_deep_dive("SPY")

        assert "detect_regime" in prompt.lower()
        assert "get_regime_statistics" in prompt.lower()
        assert "get_transition_probabilities" in prompt.lower()

    @pytest.mark.asyncio
    async def test_deep_dive_includes_sections(self):
        """Verify deep dive includes all report sections"""
        prompt = await regime_deep_dive("SPY")

        # Required sections
        assert "current regime state" in prompt.lower()
        assert "historical context" in prompt.lower()
        assert "transition" in prompt.lower()
        assert "statistical summary" in prompt.lower()
        assert "trading implications" in prompt.lower()

    @pytest.mark.asyncio
    async def test_strategy_advisor_includes_regime_types(self):
        """Verify strategy advisor covers all regime types"""
        prompt = await regime_strategy_advisor("SPY")

        assert "bull" in prompt.lower()
        assert "bear" in prompt.lower()
        assert "sideways" in prompt.lower()

    @pytest.mark.asyncio
    async def test_strategy_advisor_includes_guidance_sections(self):
        """Verify strategy advisor includes all guidance sections"""
        prompt = await regime_strategy_advisor("SPY")

        assert "regime assessment" in prompt.lower()
        assert "recommended strategies" in prompt.lower() or "strategy" in prompt.lower()
        assert "entry" in prompt.lower() and "exit" in prompt.lower()
        assert "risk management" in prompt.lower()
        assert "monitoring" in prompt.lower()

    @pytest.mark.asyncio
    async def test_multi_asset_includes_comparison_sections(self):
        """Verify multi-asset includes all comparison sections"""
        prompt = await regime_multi_asset_comparison("SPY,QQQ,IWM")

        assert "regime overview" in prompt.lower()
        assert "distribution" in prompt.lower()
        assert "diversification" in prompt.lower()
        assert "correlation" in prompt.lower()
        assert "portfolio implications" in prompt.lower()

    @pytest.mark.asyncio
    async def test_risk_assessment_includes_risk_components(self):
        """Verify risk assessment includes all risk components"""
        prompt = await regime_risk_assessment("SPY")

        assert "risk score" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "transition" in prompt.lower()
        assert "stability" in prompt.lower()
        assert "mitigation" in prompt.lower()

    @pytest.mark.asyncio
    async def test_historical_analogs_includes_sections(self):
        """Verify historical analogs includes all sections"""
        prompt = await regime_historical_analogs("SPY")

        assert "current regime pattern" in prompt.lower() or "current pattern" in prompt.lower()
        assert "historical" in prompt.lower()
        assert "transition outcomes" in prompt.lower()
        assert "timeline" in prompt.lower()

    @pytest.mark.asyncio
    async def test_portfolio_review_includes_sections(self):
        """Verify portfolio review includes all sections"""
        portfolio_json = '{"SPY": 0.5, "QQQ": 0.5}'
        prompt = await regime_portfolio_review(portfolio_json)

        assert "portfolio holdings" in prompt.lower()
        assert "regime breakdown" in prompt.lower()
        assert "weighted metrics" in prompt.lower()
        assert "concentration" in prompt.lower()
        assert "diversification" in prompt.lower()
        assert "rebalancing" in prompt.lower()


class TestPromptInstructions:
    """Test that prompts provide clear instructions"""

    @pytest.mark.asyncio
    async def test_prompts_have_numbered_steps(self):
        """Verify prompts use numbered steps for clarity"""
        quick = await regime_quick_check("SPY")
        deep = await regime_deep_dive("SPY")
        strategy = await regime_strategy_advisor("SPY")

        # All should have numbered steps (1., 2., 3., etc.)
        for prompt in [quick, deep, strategy]:
            assert "1." in prompt
            assert "2." in prompt

    @pytest.mark.asyncio
    async def test_prompts_specify_output_format(self):
        """Verify prompts specify desired output format"""
        quick = await regime_quick_check("SPY")
        deep = await regime_deep_dive("SPY")

        # Quick check specifies sentence structure
        assert "sentence" in quick.lower()

        # Deep dive specifies sections
        assert "##" in deep  # Markdown sections

    @pytest.mark.asyncio
    async def test_prompts_include_tool_call_syntax(self):
        """Verify prompts show proper tool call syntax"""
        deep = await regime_deep_dive("SPY")
        risk = await regime_risk_assessment("SPY")

        # Should show tool call format like "detect_regime(ticker)"
        assert "detect_regime" in deep
        assert "get_regime_statistics" in deep

        assert "detect_regime" in risk
        assert "get_transition_probabilities" in risk


class TestPromptEdgeCases:
    """Test edge cases and special scenarios"""

    @pytest.mark.asyncio
    async def test_ticker_with_special_characters(self):
        """Test tickers with dots and dashes"""
        prompt_dot = await regime_quick_check("BRK.B")
        prompt_dash = await regime_quick_check("ABC-D")

        assert "BRK.B" in prompt_dot
        assert "ABC-D" in prompt_dash

    @pytest.mark.asyncio
    async def test_multi_asset_single_ticker(self):
        """Test multi-asset comparison with single ticker"""
        prompt = await regime_multi_asset_comparison("SPY")
        assert "SPY" in prompt
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_multi_asset_many_tickers(self):
        """Test multi-asset comparison with many tickers"""
        tickers = "SPY,QQQ,IWM,DIA,GLD,SLV,TLT,HYG"
        prompt = await regime_multi_asset_comparison(tickers)
        assert tickers in prompt

    @pytest.mark.asyncio
    async def test_portfolio_empty_weights(self):
        """Test portfolio review with empty portfolio"""
        portfolio_json = '{}'
        prompt = await regime_portfolio_review(portfolio_json)
        assert isinstance(prompt, str)
        assert portfolio_json in prompt

    @pytest.mark.asyncio
    async def test_portfolio_complex_json(self):
        """Test portfolio review with complex portfolio"""
        portfolio_json = '{"SPY": 0.3, "QQQ": 0.2, "IWM": 0.15, "GLD": 0.15, "TLT": 0.1, "VNQ": 0.1}'
        prompt = await regime_portfolio_review(portfolio_json)
        assert portfolio_json in prompt


class TestPromptDocumentation:
    """Test that prompts have proper documentation"""

    def test_all_prompts_have_docstrings(self):
        """Verify all prompt functions have docstrings"""
        prompts = [
            regime_quick_check,
            regime_deep_dive,
            regime_strategy_advisor,
            regime_multi_asset_comparison,
            regime_risk_assessment,
            regime_historical_analogs,
            regime_portfolio_review,
        ]

        for prompt_func in prompts:
            assert prompt_func.__doc__ is not None
            assert len(prompt_func.__doc__.strip()) > 50  # Substantial docstring

    def test_docstrings_include_args_and_returns(self):
        """Verify docstrings document Args and Returns"""
        prompts = [
            regime_quick_check,
            regime_deep_dive,
            regime_strategy_advisor,
        ]

        for prompt_func in prompts:
            docstring = prompt_func.__doc__
            assert "Args:" in docstring
            assert "Returns:" in docstring

    def test_docstrings_include_examples(self):
        """Verify docstrings include usage examples"""
        prompts = [
            regime_quick_check,
            regime_deep_dive,
            regime_strategy_advisor,
        ]

        for prompt_func in prompts:
            docstring = prompt_func.__doc__
            # Should have example section
            assert "Example:" in docstring or ">>>" in docstring


class TestPromptIntegration:
    """Integration-style tests (require understanding of actual usage)"""

    @pytest.mark.asyncio
    async def test_all_prompts_callable(self):
        """Verify all prompts can be called without errors"""
        # All prompts should be callable with basic args
        await regime_quick_check("SPY")
        await regime_deep_dive("SPY")
        await regime_strategy_advisor("SPY")
        await regime_multi_asset_comparison("SPY,QQQ")
        await regime_risk_assessment("SPY")
        await regime_historical_analogs("SPY")
        await regime_portfolio_review('{"SPY": 1.0}')

    @pytest.mark.asyncio
    async def test_prompts_are_async(self):
        """Verify prompts are properly async functions"""
        import inspect

        prompts = [
            regime_quick_check,
            regime_deep_dive,
            regime_strategy_advisor,
            regime_multi_asset_comparison,
            regime_risk_assessment,
            regime_historical_analogs,
            regime_portfolio_review,
        ]

        for prompt_func in prompts:
            assert inspect.iscoroutinefunction(prompt_func)

    @pytest.mark.asyncio
    async def test_prompt_return_types(self):
        """Verify all prompts return strings"""
        results = [
            await regime_quick_check("SPY"),
            await regime_deep_dive("SPY"),
            await regime_strategy_advisor("SPY"),
            await regime_multi_asset_comparison("SPY,QQQ"),
            await regime_risk_assessment("SPY"),
            await regime_historical_analogs("SPY"),
            await regime_portfolio_review('{"SPY": 1.0}'),
        ]

        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
