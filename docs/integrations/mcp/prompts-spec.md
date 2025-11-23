# MCP Prompts Interface - Technical Specification

**Version:** 1.0
**Date:** 2025-11-18
**Author:** Hidden Regime Development Team
**Status:** Implementation Ready

---

## 1. Executive Summary

This specification defines the implementation of the **MCP Prompts Interface** for the Hidden Regime MCP server. Prompts provide pre-configured, expert-level analysis templates that users can invoke with simple commands, democratizing access to sophisticated regime detection analysis.

**Objectives:**
- Implement 7 curated prompt templates covering common regime analysis workflows
- Enable one-command access to complex multi-step analysis
- Improve user discoverability of Hidden Regime capabilities
- Provide consistent, expert-level analytical frameworks

**Expected Outcomes:**
- Reduced barrier to entry for regime analysis
- Increased user engagement with advanced features
- Standardized analysis methodology
- Educational value for learning regime detection concepts

---

## 2. MCP Prompts Interface Overview

### 2.1 What are MCP Prompts?

MCP Prompts are **pre-configured prompt templates** exposed by an MCP server that AI assistants can present to users. When a user selects a prompt:

1. The MCP server returns a structured prompt template with argument placeholders
2. Claude Desktop UI shows available prompts in a discoverable list
3. User selects a prompt and fills in required arguments (e.g., ticker symbol)
4. Claude receives the completed prompt and executes the analysis workflow
5. Claude uses available tools (detect_regime, get_regime_statistics, etc.) to fulfill the prompt

### 2.2 Benefits Over Plain Tools

| Aspect | Tools Only | Tools + Prompts |
|--------|-----------|-----------------|
| **User Experience** | Users must know what to ask | Users browse curated options |
| **Complexity** | Users craft multi-step queries | Single command → full workflow |
| **Consistency** | Variable quality of analysis | Standardized expert-level analysis |
| **Discovery** | Users may not know capabilities | Prompts reveal possibilities |
| **Learning** | Trial and error | Templates teach best practices |

### 2.3 FastMCP Prompts API

FastMCP provides a simple decorator-based API for registering prompts:

```python
from fastmcp import FastMCP

mcp = FastMCP(name="Hidden Regime")

@mcp.prompt()
async def my_prompt(arg1: str, arg2: int = 10) -> str:
    """
    Prompt description shown to users.

    Args:
        arg1: Description of required argument
        arg2: Description of optional argument with default

    Returns:
        The prompt text to send to Claude
    """
    return f"Analyze {arg1} using parameter {arg2}..."
```

**Key features:**
- Type-annotated arguments with validation
- Optional arguments with defaults
- Async support for dynamic prompt generation
- Automatic argument schema generation
- Docstring-based descriptions

---

## 3. Architecture & Design Decisions

### 3.1 Module Structure

```
hidden_regime_mcp/
├── server.py          # Main MCP server (registers prompts)
├── tools.py           # Existing tools (detect_regime, etc.)
├── resources.py       # Existing resources
├── prompts.py         # NEW: Prompt implementations
└── tests/
    └── test_prompts.py # NEW: Prompt tests
```

### 3.2 Design Principles

**1. Separation of Concerns**
- Prompts are **instructional** (tell Claude what to do)
- Tools are **operational** (execute regime detection)
- Resources are **informational** (expose static data)

**2. Prompt Complexity Ladder**
- **Quick** prompts: 1-2 tool calls, < 5 second response
- **Standard** prompts: 2-4 tool calls, 5-15 second response
- **Deep** prompts: 4+ tool calls, 15-30 second response

**3. Argument Design**
- **Required arguments:** Only ticker symbols (core to all analysis)
- **Optional arguments:** Everything else has sensible defaults
- **Validation:** Use existing validators from tools.py

**4. Prompt Text Structure**
Each prompt follows this structure:
```
[CONTEXT] - What is the user trying to accomplish?
[STEPS] - Numbered steps Claude should execute
[OUTPUT] - Format/structure of final response to user
[CONSTRAINTS] - Important caveats or limitations
```

### 3.3 Error Handling Strategy

Prompts should be **robust** and **graceful**:
- Don't fail on individual tool errors - work with available data
- Provide context when data is missing
- Suggest alternatives when analysis isn't possible
- Always give the user something actionable

---

## 4. Detailed Prompt Specifications

### 4.1 Prompt 1: regime-quick-check

**Purpose:** Fast market health check for day-to-day decision making

**Arguments:**
- `ticker` (required, str): Stock symbol to analyze

**Workflow:**
1. Call `detect_regime(ticker)` for current regime
2. Interpret results focusing on:
   - Current regime + confidence
   - Days in regime vs expected duration
   - Recent price performance vs regime characteristics
   - Immediate risks or opportunities

**Output Format:**
- 3-4 sentence executive summary
- Key metric: regime name, confidence, status (early/mid/mature)
- Actionable insight (what this means for traders)

**Example User Flow:**
```
User: [Selects regime-quick-check prompt]
Claude: What ticker would you like to check?
User: SPY
Claude: [Calls detect_regime("SPY")]
Claude: SPY is in a bearish regime (89% confidence) that started 12 days ago...
```

**Implementation:**
```python
async def regime_quick_check(ticker: str) -> str:
    """Quick market health check for a ticker"""
    return f"""Perform a quick market health check for {ticker}:

1. Call detect_regime for {ticker}
2. Focus your analysis on these key points:
   - Current regime and confidence level
   - Regime maturity (early/mid/mature/overdue based on days_in_regime vs expected_duration)
   - Recent price performance context (YTD, 30d trends)
   - Any immediate risks (approaching transition, low confidence, etc.)

3. Provide a concise 3-4 sentence summary formatted as:
   - Sentence 1: Current regime, confidence, and how long it's been active
   - Sentence 2: Price performance context
   - Sentence 3: Key implication for traders/investors

Keep it brief and actionable - this is for quick decision-making.
"""
```

**Success Criteria:**
- Response time < 5 seconds
- Clear go/no-go signal for traders
- Non-technical language accessible to beginners

---

### 4.2 Prompt 2: regime-deep-dive

**Purpose:** Comprehensive regime analysis with historical context

**Arguments:**
- `ticker` (required, str): Stock symbol to analyze
- `lookback_period` (optional, str): Time period, default "2y"

**Workflow:**
1. Parse lookback_period into start_date (e.g., "2y" → 2 years ago)
2. Call `detect_regime(ticker, start_date=...)` for current state
3. Call `get_regime_statistics(ticker, start_date=...)` for historical context
4. Call `get_transition_probabilities(ticker)` for forward-looking analysis
5. Synthesize comprehensive regime profile

**Output Format:**
- **Current State** section (regime, confidence, temporal context)
- **Historical Context** section (regime statistics, how often each regime occurs)
- **Transition Analysis** section (transition matrix, expected durations)
- **Statistical Summary** section (performance metrics by regime)
- **Interpretation** section (what this means, trading implications)

**Implementation:**
```python
async def regime_deep_dive(ticker: str, lookback_period: str = "2y") -> str:
    """Comprehensive regime analysis with historical context"""
    return f"""Perform a comprehensive regime analysis for {ticker} over the past {lookback_period}:

ANALYSIS STEPS:

1. Calculate the start_date based on lookback_period:
   - "1y" = 1 year ago from today
   - "2y" = 2 years ago from today
   - "3y" = 3 years ago from today
   - If specific date format (YYYY-MM-DD), use as-is

2. Call detect_regime({ticker}, start_date=calculated_date) for current regime

3. Call get_regime_statistics({ticker}, start_date=calculated_date) for historical context

4. Call get_transition_probabilities({ticker}) for transition analysis

5. Synthesize results into a comprehensive report with these sections:

   ## Current Regime State
   - Regime name, confidence, last updated
   - Days in regime vs expected duration
   - Regime status (early/mid/mature/overdue)
   - Current price and trend

   ## Historical Context ({lookback_period})
   - Regime distribution (% time in each regime)
   - Performance by regime (mean return, volatility, win rate)
   - Typical regime durations

   ## Transition Dynamics
   - Transition probabilities from current regime
   - Expected duration remaining
   - Most likely next regime

   ## Statistical Summary
   - Best performing regime historically
   - Most stable vs most volatile regimes
   - Long-term steady state probabilities

   ## Trading Implications
   - What does current regime suggest for strategy?
   - Key risks to monitor
   - Potential catalysts for regime change

Format the report with clear markdown sections and tables where appropriate.
"""
```

**Success Criteria:**
- Comprehensive yet readable analysis
- Historical context helps interpret current regime
- Clear action items for portfolio management

---

### 4.3 Prompt 3: regime-strategy-advisor

**Purpose:** Trading strategy recommendations based on current regime

**Arguments:**
- `ticker` (required, str): Stock symbol to analyze
- `risk_tolerance` (optional, str): "low", "medium", or "high", default "medium"

**Workflow:**
1. Call `detect_regime(ticker)` for current regime and stability
2. Map regime + risk tolerance → strategy recommendations
3. Provide specific entry/exit signals based on regime maturity
4. Include risk warnings specific to regime characteristics

**Strategy Mapping:**
```
Bull Regime:
  Low Risk: Long-only, trailing stops
  Medium Risk: Trend following, momentum strategies
  High Risk: Leveraged long, breakout trading

Bear Regime:
  Low Risk: Cash, defensive sectors
  Medium Risk: Short positions, inverse ETFs
  High Risk: Volatility trading, aggressive shorts

Sideways Regime:
  Low Risk: Range-bound trading, covered calls
  Medium Risk: Mean reversion, pairs trading
  High Risk: Iron condors, straddle selling
```

**Output Format:**
- **Current Regime Assessment** (regime, confidence, stability)
- **Recommended Strategies** (2-3 strategies suited to regime + risk)
- **Entry/Exit Signals** (specific price levels or conditions)
- **Risk Management** (stop losses, position sizing, warnings)
- **Regime Change Monitoring** (what to watch for transition)

**Implementation:**
```python
async def regime_strategy_advisor(ticker: str, risk_tolerance: str = "medium") -> str:
    """Trading strategy recommendation based on current regime"""
    # Validate risk_tolerance
    valid_risk = ["low", "medium", "high"]
    if risk_tolerance.lower() not in valid_risk:
        risk_tolerance = "medium"

    return f"""Provide trading strategy recommendations for {ticker} based on current regime:

ANALYSIS STEPS:

1. Call detect_regime({ticker}) to get:
   - Current regime (bull/bear/sideways)
   - Confidence and stability
   - Regime maturity (days_in_regime vs expected_duration)
   - Price trend and performance

2. Based on regime and risk_tolerance="{risk_tolerance}", recommend strategies:

   For BULL regimes:
   - Low risk: Long-only positions, trailing stops, dollar-cost averaging
   - Medium risk: Trend-following, momentum strategies, breakout trading
   - High risk: Leveraged long positions, aggressive growth allocation

   For BEAR regimes:
   - Low risk: Cash positions, defensive sectors, quality bonds
   - Medium risk: Short positions, inverse ETFs, hedging strategies
   - High risk: Aggressive shorts, put options, volatility trading

   For SIDEWAYS regimes:
   - Low risk: Range-bound trading, covered calls, cash-secured puts
   - Medium risk: Mean reversion, pairs trading, sector rotation
   - High risk: Iron condors, straddles, high-frequency range trading

3. Provide specific guidance in these sections:

   ## Current Regime Assessment
   - Regime type and confidence
   - Stability (stable/moderate/volatile)
   - Regime maturity status

   ## Recommended Strategies (for {risk_tolerance} risk tolerance)
   - List 2-3 specific strategies suited to current regime
   - Explain why each strategy fits the regime characteristics
   - Note which strategy is most appropriate given regime maturity

   ## Entry/Exit Signals
   - Specific conditions for entering positions
   - Price levels or technical signals to monitor
   - Exit criteria (profit targets and stop losses)

   ## Risk Management
   - Position sizing recommendations
   - Stop loss placement
   - Warning signs that strategy may need adjustment

   ## Regime Transition Monitoring
   - What signals to watch for regime change
   - How to adjust strategy if regime shifts
   - Days until expected transition

Keep recommendations practical and specific to {ticker}'s current regime characteristics.
"""
```

**Success Criteria:**
- Actionable strategy recommendations
- Risk-appropriate for user's tolerance
- Clear entry/exit criteria
- Warns of regime-specific risks

---

### 4.4 Prompt 4: regime-multi-asset-comparison

**Purpose:** Compare regimes across multiple assets for portfolio analysis

**Arguments:**
- `tickers` (required, str): Comma-separated list of tickers (e.g., "SPY,QQQ,IWM")

**Workflow:**
1. Parse comma-separated tickers
2. Call `detect_regime(ticker)` for each ticker in parallel
3. Compare regime states, confidence, maturity
4. Identify regime correlations and diversification opportunities
5. Highlight concentration risks

**Output Format:**
- **Regime Overview Table** (ticker, regime, confidence, status)
- **Regime Distribution** (how many assets in each regime)
- **Correlation Analysis** (are all assets in same regime? diversified?)
- **Portfolio Implications** (concentration risk, rebalancing suggestions)

**Implementation:**
```python
async def regime_multi_asset_comparison(tickers: str) -> str:
    """Compare regimes across multiple assets"""
    return f"""Perform a multi-asset regime comparison for: {tickers}

ANALYSIS STEPS:

1. Parse the comma-separated ticker list: {tickers}
   - Remove whitespace and convert to uppercase
   - Validate each ticker is reasonable (1-10 characters)

2. For each ticker, call detect_regime(ticker) to get:
   - Current regime
   - Confidence level
   - Regime status (early/mid/mature)
   - Recent performance (YTD, 30d)

3. Create a comparison analysis with these sections:

   ## Regime Overview
   Create a comparison table with columns:
   | Ticker | Regime | Confidence | Status | YTD Return | 30d Return |

   ## Regime Distribution
   - Count assets in each regime (bull/bear/sideways)
   - Calculate percentage distribution
   - Note if portfolio is concentrated in one regime

   ## Diversification Analysis
   - Are assets showing regime diversity or all moving together?
   - Which regime combinations provide best diversification?
   - Identify any regime-based concentration risk

   ## Correlation Insights
   - Are regime transitions synchronized across assets?
   - Which assets provide regime diversification?
   - Note any unusual regime patterns (e.g., one asset bearish while others bullish)

   ## Portfolio Implications
   - Is the portfolio diversified across regimes?
   - Concentration risk assessment
   - Rebalancing suggestions based on regime distribution
   - Assets that may need increased/decreased allocation

   ## Risk Assessment
   - What percentage of portfolio is in mature/overdue regimes?
   - Transition risk (assets close to regime changes)
   - Overall portfolio regime health score

Use tables and clear formatting for easy comparison across assets.
"""
```

**Success Criteria:**
- Easy visual comparison across assets
- Identifies diversification gaps
- Actionable rebalancing recommendations

---

### 4.5 Prompt 5: regime-risk-assessment

**Purpose:** Risk analysis based on regime stability and transition probabilities

**Arguments:**
- `ticker` (required, str): Stock symbol to analyze

**Workflow:**
1. Call `detect_regime(ticker)` for stability metrics
2. Call `get_transition_probabilities(ticker)` for transition risk
3. Calculate risk scores based on:
   - Regime confidence (low confidence = high risk)
   - Regime maturity (overdue = high transition risk)
   - Stability (volatile = high risk)
   - Transition probabilities to unfavorable regimes

**Output Format:**
- **Risk Score** (0-100, higher = more risk)
- **Risk Breakdown** (component scores: confidence, stability, transition)
- **Key Risk Factors** (what's driving the risk score)
- **Risk Mitigation** (specific actions to reduce risk)
- **Monitoring Plan** (what to watch, when to act)

**Implementation:**
```python
async def regime_risk_assessment(ticker: str) -> str:
    """Risk analysis based on regime stability and transitions"""
    return f"""Perform a comprehensive risk assessment for {ticker} based on regime analysis:

ANALYSIS STEPS:

1. Call detect_regime({ticker}) to get:
   - Current regime and confidence
   - Regime stability (stable/moderate/volatile)
   - Days in regime vs expected duration
   - Recent transitions count

2. Call get_transition_probabilities({ticker}) to get:
   - Transition probabilities from current regime
   - Expected duration in current regime
   - Likelihood of moving to each other regime

3. Calculate and present a risk assessment with these components:

   ## Overall Risk Score (0-100)
   Calculate based on:
   - Confidence risk: (1 - confidence) * 30 points
     * >80% confidence = low risk (0-6 pts)
     * 50-80% confidence = medium risk (6-15 pts)
     * <50% confidence = high risk (15-30 pts)

   - Transition risk: (days_in_regime / expected_duration) * 30 points
     * Early regime (<25% complete) = low risk (0-7 pts)
     * Mid regime (25-60% complete) = low risk (7-18 pts)
     * Mature regime (60-100% complete) = medium risk (18-30 pts)
     * Overdue regime (>100% complete) = high risk (30+ pts)

   - Stability risk: based on recent_transitions * 20 points
     * Stable (0 transitions) = low risk (0 pts)
     * Moderate (1-2 transitions) = medium risk (10 pts)
     * Volatile (3+ transitions) = high risk (20 pts)

   - Adverse transition risk: probability of transitioning to unfavorable regime * 20 points
     * If in bull: risk = P(bear transition) * 20
     * If in bear: risk = P(staying in bear) * 20
     * If in sideways: risk = P(bear transition) * 20

   Total score and interpretation:
   * 0-30: Low risk
   * 31-60: Medium risk
   * 61-100: High risk

   ## Risk Component Breakdown
   Show score for each component with explanation

   ## Key Risk Factors
   List top 3 risk drivers in priority order

   ## Risk Mitigation Strategies
   For each key risk factor, provide specific mitigation:
   - Low confidence → Reduce position size, wait for confirmation
   - High transition risk → Tighten stops, consider partial profit-taking
   - High volatility → Reduce exposure, hedge with options
   - Adverse transition risk → Prepare exit strategy, set alerts

   ## Monitoring Plan
   - What specific metrics to watch daily/weekly
   - Alert thresholds (e.g., "if confidence drops below X%")
   - Regime transition signals to monitor
   - Recommended review frequency

Provide specific, actionable risk management guidance based on the current regime state.
"""
```

**Success Criteria:**
- Quantitative risk score for objective comparison
- Clear breakdown of risk components
- Specific mitigation actions
- Proactive monitoring plan

---

### 4.6 Prompt 6: regime-historical-analogs

**Purpose:** Find similar historical regime patterns to predict outcomes

**Arguments:**
- `ticker` (required, str): Stock symbol to analyze
- `lookback_period` (optional, str): How far back to search, default "5y"

**Workflow:**
1. Call `detect_regime(ticker)` for current regime characteristics
2. Call `get_regime_statistics(ticker, start_date=...)` with extended history
3. Identify historical episodes with similar characteristics:
   - Same regime type
   - Similar confidence levels
   - Similar days in regime
4. Analyze outcomes: what happened next?
5. Calculate expected time to next transition

**Output Format:**
- **Current Pattern** (regime, confidence, duration)
- **Historical Matches** (3-5 similar past episodes)
- **Outcome Analysis** (what happened after each match)
- **Expected Timeline** (time to next transition based on analogs)
- **Probability Distribution** (likelihood of different outcomes)

**Implementation:**
```python
async def regime_historical_analogs(ticker: str, lookback_period: str = "5y") -> str:
    """Find similar historical regime patterns"""
    return f"""Find historical regime patterns similar to {ticker}'s current state:

ANALYSIS STEPS:

1. Calculate start_date from lookback_period: {lookback_period}

2. Call detect_regime({ticker}) to get current regime:
   - Current regime type
   - Confidence level
   - Days in regime
   - Expected duration
   - Regime maturity status

3. Call get_regime_statistics({ticker}, start_date=calculated_date) to get historical context

4. Analyze and present findings:

   ## Current Regime Pattern
   - Regime type and characteristics
   - Days active vs expected duration (X% complete)
   - Confidence and stability
   - Current price trend

   ## Pattern Matching Analysis
   Based on the regime statistics, identify how often {ticker} has been in similar situations:
   - How many times has this regime type occurred in the past {lookback_period}?
   - What was the average duration historically?
   - What was the typical confidence level?
   - How does current episode compare to historical average?

   ## Historical Regime Episodes
   Describe the characteristics of past episodes of this regime:
   - Average duration: X days
   - Shortest duration: Y days
   - Longest duration: Z days
   - Current episode duration context (shorter/longer than average)

   ## Transition Outcomes
   When this regime has ended in the past, what happened next?
   - Use transition probabilities to show likelihood of each outcome
   - Expected duration until next transition
   - Most likely next regime
   - Probability distribution across possible outcomes

   ## Timeline Projection
   Based on historical patterns:
   - Current regime is X% complete (days_in_regime / expected_duration)
   - Expected days until transition: Y days
   - Confidence interval: typically transitions between Y-Z and Y+Z days
   - Regime status: early/mid/mature/overdue

   ## Pattern Insights
   - Is current episode behaving like historical patterns?
   - Any unusual characteristics compared to past episodes?
   - What external factors might affect regime duration this time?
   - Risk of early vs late transition

Use the historical data to provide probabilistic guidance on what to expect next.
"""
```

**Success Criteria:**
- Identifies meaningful historical comparisons
- Probabilistic (not deterministic) outcome predictions
- Clear timeline expectations with confidence intervals

---

### 4.7 Prompt 7: regime-portfolio-review

**Purpose:** Portfolio-wide regime analysis with allocation recommendations

**Arguments:**
- `portfolio` (required, str): JSON string with format: `{"ticker": weight, ...}` (e.g., `{"SPY": 0.4, "QQQ": 0.3, "IWM": 0.3}`)

**Workflow:**
1. Parse portfolio JSON (tickers and weights)
2. Call `detect_regime(ticker)` for each holding
3. Calculate portfolio-level regime exposure
4. Identify concentration risks and diversification gaps
5. Generate rebalancing recommendations

**Output Format:**
- **Portfolio Regime Breakdown** (% in bull/bear/sideways)
- **Weighted Risk Score** (portfolio-level regime risk)
- **Concentration Analysis** (over/under-exposed regimes)
- **Diversification Assessment** (how well diversified across regimes)
- **Rebalancing Recommendations** (specific allocation changes)

**Implementation:**
```python
async def regime_portfolio_review(portfolio: str) -> str:
    """Portfolio-wide regime analysis"""
    return f"""Perform a comprehensive regime analysis of your portfolio:

Portfolio: {portfolio}

ANALYSIS STEPS:

1. Parse the portfolio JSON:
   - Extract ticker symbols and weights
   - Validate that weights sum to ~1.0 (or 100%)
   - Note: weights should be decimal (0.4 = 40%) or percentage (40 = 40%)

2. For each ticker in the portfolio, call detect_regime(ticker) to get:
   - Current regime
   - Confidence
   - Regime status (early/mid/mature)
   - Volatility
   - Recent performance

3. Create a portfolio-level regime analysis:

   ## Portfolio Holdings
   Table showing each position:
   | Ticker | Weight | Regime | Confidence | Status | YTD Return |

   ## Portfolio Regime Breakdown
   Calculate weighted regime exposure:
   - % in Bull regimes (sum of weights for all bull-regime holdings)
   - % in Bear regimes
   - % in Sideways regimes

   Show as both percentages and visual representation

   ## Weighted Metrics
   Portfolio-level weighted averages:
   - Average regime confidence (weighted by position size)
   - Average volatility (weighted)
   - Average YTD return (weighted)
   - Percentage in mature/overdue regimes (transition risk)

   ## Concentration Risk Analysis
   Identify risks:
   - Regime concentration: Is >60% in one regime type?
   - Maturity concentration: Are multiple large positions in overdue regimes?
   - Volatility concentration: Are high-volatility regimes over-weighted?
   - Correlation risk: Are all positions in synchronized regimes?

   ## Diversification Assessment
   Score diversification (0-100):
   - Regime diversity: better if spread across bull/bear/sideways
   - Maturity diversity: better if spread across early/mid/mature
   - Confidence diversity: better if mix of high/medium confidence

   Overall diversification score and interpretation

   ## Rebalancing Recommendations
   Specific suggestions to improve regime diversification:
   - If over-exposed to bull regimes → Consider adding defensive positions
   - If over-exposed to mature regimes → Reduce positions approaching transition
   - If low diversification → Suggest specific tickers to add
   - If high concentration risk → Suggest reducing positions

   Provide 3-5 specific action items with reasoning

   ## Risk Summary
   - Overall portfolio regime health score
   - Top 3 portfolio risks based on regime analysis
   - Recommended monitoring frequency
   - Key signals that would warrant immediate rebalancing

Format with clear sections and tables for easy portfolio review.
"""
```

**Success Criteria:**
- Clear portfolio-level regime exposure
- Identifies concentration risks
- Actionable rebalancing recommendations
- Considers position sizing in analysis

---

## 5. Implementation Plan

### 5.1 File Structure

```python
# hidden_regime_mcp/prompts.py

"""
MCP prompts for Hidden Regime regime detection.

Provides 7 curated prompt templates for sophisticated regime analysis:
1. regime_quick_check - Fast market health check
2. regime_deep_dive - Comprehensive regime analysis
3. regime_strategy_advisor - Trading strategy recommendations
4. regime_multi_asset_comparison - Multi-asset regime comparison
5. regime_risk_assessment - Risk analysis and mitigation
6. regime_historical_analogs - Historical pattern matching
7. regime_portfolio_review - Portfolio regime analysis
"""

from typing import Any

# Individual prompt functions (shown in sections 4.1-4.7)
async def regime_quick_check(ticker: str) -> str:
    """Quick market health check for a ticker"""
    # Implementation from section 4.1
    pass

# ... (all 7 prompts)
```

### 5.2 Server Registration

Update `hidden_regime_mcp/server.py`:

```python
from hidden_regime_mcp.prompts import (
    regime_quick_check,
    regime_deep_dive,
    regime_strategy_advisor,
    regime_multi_asset_comparison,
    regime_risk_assessment,
    regime_historical_analogs,
    regime_portfolio_review,
)

# Register prompts
mcp.prompt()(regime_quick_check)
mcp.prompt()(regime_deep_dive)
mcp.prompt()(regime_strategy_advisor)
mcp.prompt()(regime_multi_asset_comparison)
mcp.prompt()(regime_risk_assessment)
mcp.prompt()(regime_historical_analogs)
mcp.prompt()(regime_portfolio_review)
```

### 5.3 Testing Strategy

Create `tests/test_mcp/test_prompts.py`:

```python
import pytest
from hidden_regime_mcp.prompts import (
    regime_quick_check,
    regime_deep_dive,
    # ... all prompts
)


class TestPromptGeneration:
    """Test that prompts generate valid text"""

    @pytest.mark.asyncio
    async def test_regime_quick_check_basic(self):
        """Test quick check prompt generation"""
        prompt = await regime_quick_check("SPY")
        assert "detect_regime" in prompt.lower()
        assert "SPY" in prompt
        assert len(prompt) > 100  # Substantial prompt

    # ... tests for each prompt


class TestPromptArguments:
    """Test argument validation and defaults"""

    @pytest.mark.asyncio
    async def test_regime_deep_dive_default_period(self):
        """Test default lookback period"""
        prompt = await regime_deep_dive("AAPL")
        assert "2y" in prompt  # Default value

    @pytest.mark.asyncio
    async def test_regime_deep_dive_custom_period(self):
        """Test custom lookback period"""
        prompt = await regime_deep_dive("AAPL", lookback_period="5y")
        assert "5y" in prompt


class TestPromptContent:
    """Test prompt content quality"""

    @pytest.mark.asyncio
    async def test_prompts_include_tool_calls(self):
        """Verify prompts reference actual MCP tools"""
        quick = await regime_quick_check("SPY")
        deep = await regime_deep_dive("SPY")

        assert "detect_regime" in quick
        assert all(tool in deep for tool in [
            "detect_regime",
            "get_regime_statistics",
            "get_transition_probabilities"
        ])

    @pytest.mark.asyncio
    async def test_strategy_advisor_risk_levels(self):
        """Test all risk tolerance levels"""
        for risk in ["low", "medium", "high"]:
            prompt = await regime_strategy_advisor("SPY", risk_tolerance=risk)
            assert risk in prompt.lower()
```

### 5.4 Documentation Updates

Update `README_MCP.md` with new section:

```markdown
## Available Prompts

Prompts provide pre-configured analysis workflows for sophisticated regime detection.

### 1. regime-quick-check

**Quick market health check**

**Arguments:**
- `ticker` (required): Stock symbol

**What it does:**
1. Analyzes current regime and confidence
2. Checks regime maturity and stability
3. Provides 3-4 sentence actionable summary

**Example:**
```
Use the regime-quick-check prompt for SPY
```

### 2. regime-deep-dive
...

## Using Prompts in Claude Desktop

1. Look for prompt suggestions in the prompt library
2. Select the desired prompt
3. Fill in required arguments (e.g., ticker symbol)
4. Claude automatically executes the full analysis workflow

Prompts are faster than manual queries because they provide expert-level analytical frameworks automatically.
```

---

## 6. Testing & Validation

### 6.1 Unit Tests

**Coverage targets:**
- Prompt generation: 100% (all prompts can be called)
- Argument handling: 100% (defaults, validation)
- Content validation: 90% (prompts reference correct tools)

**Test cases:**
- ✅ Each prompt generates non-empty text
- ✅ Required arguments are enforced
- ✅ Optional arguments use correct defaults
- ✅ Prompts reference appropriate tools
- ✅ Argument values appear in generated prompts

### 6.2 Integration Tests

**Manual testing with Claude Desktop:**
1. Install updated MCP server
2. Restart Claude Desktop
3. Verify prompts appear in prompt library
4. Test each prompt end-to-end
5. Verify Claude uses appropriate tools
6. Check output format matches specification

**Success criteria:**
- All 7 prompts visible in Claude Desktop
- Arguments render correctly in UI
- Claude executes correct tool calls
- Output format matches specification
- Response time acceptable (<30s for deep prompts)

### 6.3 User Acceptance Testing

**Test scenarios:**
1. Beginner user explores prompts → Should discover capabilities easily
2. Experienced user needs quick analysis → Quick check provides instant insight
3. Portfolio manager reviews holdings → Portfolio prompt actionable
4. Risk manager assesses exposure → Risk assessment quantitative and clear

---

## 7. Success Metrics

### 7.1 Technical Metrics

- ✅ All 7 prompts implemented and tested
- ✅ 100% prompt generation test coverage
- ✅ Zero errors in Claude Desktop prompt library
- ✅ All prompts execute successfully with valid arguments

### 7.2 User Experience Metrics

- ⭐ Prompt discovery: Users find prompts without documentation
- ⭐ Prompt adoption: >30% of users try at least one prompt
- ⭐ Prompt preference: Prompts used more than manual tool calls
- ⭐ User satisfaction: Positive feedback on prompt quality

### 7.3 Quality Metrics

- 📊 Response quality: Claude output matches specification
- 📊 Consistency: Same prompt → similar quality analysis
- 📊 Completeness: All specified sections included in output
- 📊 Actionability: Users can act on recommendations

---

## 8. Future Enhancements

### 8.1 Phase 2 Prompts (Post-Launch)

Based on user feedback, consider adding:

**Advanced Analysis:**
- `regime-sector-rotation` - Sector-specific regime analysis
- `regime-options-strategy` - Options strategies based on regime + volatility
- `regime-backtest` - Historical performance of regime-based strategies

**Educational:**
- `regime-explainer` - Deep dive into how HMM regime detection works
- `regime-case-study` - Analysis of famous market events through regime lens

**Automation:**
- `regime-watchlist` - Monitor multiple tickers for regime changes
- `regime-alert-config` - Set up regime transition alerts

### 8.2 Prompt Composition

Allow combining multiple prompts:
```python
async def regime_full_report(ticker: str) -> str:
    """Combines quick-check + deep-dive + risk-assessment"""
    # Orchestrate multiple prompts into comprehensive report
```

### 8.3 Dynamic Prompts

Generate prompts dynamically based on context:
- Detect portfolio holdings from user message → Auto-suggest portfolio-review
- Detect multiple tickers → Auto-suggest multi-asset-comparison
- Detect risk keywords → Auto-suggest risk-assessment

---

## 9. Risk Assessment & Mitigation

### 9.1 Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Prompts generate incorrect tool calls | High | Low | Extensive testing, clear tool references |
| Users confused by prompt arguments | Medium | Medium | Clear descriptions, sensible defaults |
| Prompts too verbose for practical use | Medium | Low | Keep prompts concise, focus on essentials |
| Claude ignores prompt instructions | High | Low | Test with actual Claude Desktop, iterate |
| Prompts don't match user mental models | Medium | Medium | User testing, gather feedback |

### 9.2 Mitigation Strategies

**Testing:**
- Manual testing with Claude Desktop before release
- Unit tests verify prompt generation
- Integration tests verify end-to-end workflows

**Documentation:**
- Clear prompt descriptions in docstrings
- Examples in README_MCP.md
- User guide with screenshots

**Iteration:**
- Start with 3 core prompts, validate, then expand
- Gather user feedback early
- Iterate on prompt quality based on usage

---

## 10. Timeline & Milestones

### Day 1: Core Implementation
- ✅ Create `prompts.py` module
- ✅ Implement prompts 1-3 (quick-check, deep-dive, strategy-advisor)
- ✅ Update `server.py` to register prompts
- ✅ Basic unit tests

### Day 2: Advanced Prompts
- ✅ Implement prompts 4-7 (multi-asset, risk, historical, portfolio)
- ✅ Comprehensive unit tests
- ✅ Integration testing with Claude Desktop

### Day 3: Documentation & Testing
- ✅ Update README_MCP.md with prompts section
- ✅ Add usage examples
- ✅ Manual testing and refinement
- ✅ Commit and document

---

## 11. Appendix

### 11.1 MCP Prompts Specification Reference

From the Model Context Protocol specification:

```typescript
interface Prompt {
  name: string;              // Unique prompt identifier
  description?: string;      // Human-readable description
  arguments?: [              // Optional prompt arguments
    {
      name: string;          // Argument name
      description?: string;  // Argument description
      required?: boolean;    // Whether argument is required
    }
  ];
}

interface GetPromptResult {
  description?: string;      // Optional description
  messages: [                // Prompt messages
    {
      role: "user" | "assistant";
      content: {
        type: "text";
        text: string;        // The actual prompt text
      }
    }
  ];
}
```

### 11.2 FastMCP Implementation Pattern

```python
from fastmcp import FastMCP

mcp = FastMCP(name="My Server")

@mcp.prompt()
async def my_prompt_name(
    required_arg: str,
    optional_arg: str = "default"
) -> str:
    """
    Prompt description shown to users.

    Args:
        required_arg: Description of required argument
        optional_arg: Description of optional argument

    Returns:
        The prompt text to send to Claude
    """
    return f"Prompt text with {required_arg} and {optional_arg}"
```

FastMCP automatically:
- Converts function name to prompt name (snake_case → kebab-case)
- Extracts description from docstring
- Generates argument schema from type hints
- Handles argument validation
- Returns prompt text in MCP format

---

## 12. Conclusion

This specification provides a complete blueprint for implementing the MCP Prompts interface for Hidden Regime. The 7 curated prompts cover the most common and valuable regime analysis workflows, from quick health checks to comprehensive portfolio reviews.

**Key Benefits:**
- **Accessibility:** Non-experts get expert-level analysis
- **Efficiency:** One command replaces complex multi-step queries
- **Consistency:** Standardized analytical frameworks
- **Discovery:** Users learn what's possible with regime detection

**Implementation Approach:**
- Start with core prompts (1-3) for validation
- Expand to advanced prompts (4-7) based on success
- Iterate based on user feedback
- Future enhancements in Phase 2

This implementation will significantly enhance the user experience of the Hidden Regime MCP server and demonstrate the power of the MCP Prompts interface.

---

**End of Technical Specification**
