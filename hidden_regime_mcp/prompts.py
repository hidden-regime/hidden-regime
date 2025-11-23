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

Each prompt returns a structured instruction that guides Claude through
sophisticated regime analysis workflows using the available MCP tools.
"""


async def regime_quick_check(ticker: str) -> str:
    """
    Quick market health check for a ticker.

    Provides a fast, actionable summary of current regime status suitable for
    day-to-day trading decisions. Returns a 3-4 sentence executive summary
    focusing on regime, confidence, maturity, and immediate implications.

    Args:
        ticker: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'NVDA')

    Returns:
        Prompt text instructing Claude to perform quick regime check

    Example:
        >>> prompt = await regime_quick_check("SPY")
        >>> # Claude will call detect_regime and provide concise summary
    """
    ticker = ticker.upper().strip()

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


async def regime_deep_dive(ticker: str, lookback_period: str = "2y") -> str:
    """
    Comprehensive regime analysis with historical context.

    Performs a thorough multi-dimensional regime analysis including current state,
    historical patterns, transition dynamics, and statistical summary. Ideal for
    portfolio reviews, research, and strategic planning.

    Args:
        ticker: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'NVDA')
        lookback_period: Time period for historical analysis (default: "2y")
            Examples: "1y", "2y", "3y", "5y", or specific date "2020-01-01"

    Returns:
        Prompt text instructing Claude to perform comprehensive regime analysis

    Example:
        >>> prompt = await regime_deep_dive("NVDA", lookback_period="3y")
        >>> # Claude will call multiple tools and provide detailed report
    """
    ticker = ticker.upper().strip()

    return f"""Perform a comprehensive regime analysis for {ticker} over the past {lookback_period}:

ANALYSIS STEPS:

1. Calculate the start_date based on lookback_period:
   - "1y" = 1 year ago from today
   - "2y" = 2 years ago from today
   - "3y" = 3 years ago from today
   - "5y" = 5 years ago from today
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


async def regime_strategy_advisor(
    ticker: str, risk_tolerance: str = "medium"
) -> str:
    """
    Trading strategy recommendation based on current regime.

    Provides specific trading strategy recommendations tailored to the current
    regime and user's risk tolerance. Includes entry/exit signals, risk management
    guidelines, and regime transition monitoring.

    Args:
        ticker: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'NVDA')
        risk_tolerance: Risk level - "low", "medium", or "high" (default: "medium")

    Returns:
        Prompt text instructing Claude to provide strategy recommendations

    Example:
        >>> prompt = await regime_strategy_advisor("QQQ", risk_tolerance="high")
        >>> # Claude will recommend aggressive strategies suitable for bull regimes
    """
    ticker = ticker.upper().strip()

    # Validate risk_tolerance
    valid_risk = ["low", "medium", "high"]
    if risk_tolerance.lower() not in valid_risk:
        risk_tolerance = "medium"
    else:
        risk_tolerance = risk_tolerance.lower()

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


async def regime_multi_asset_comparison(tickers: str) -> str:
    """
    Compare regimes across multiple assets.

    Analyzes regime states across a portfolio of assets to identify diversification
    opportunities, concentration risks, and correlation patterns. Ideal for
    portfolio construction and risk management.

    Args:
        tickers: Comma-separated list of ticker symbols (e.g., "SPY,QQQ,IWM")

    Returns:
        Prompt text instructing Claude to perform multi-asset regime comparison

    Example:
        >>> prompt = await regime_multi_asset_comparison("SPY,QQQ,IWM,DIA")
        >>> # Claude will compare regimes and provide diversification insights
    """
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


async def regime_risk_assessment(ticker: str) -> str:
    """
    Risk analysis based on regime stability and transitions.

    Provides a quantitative risk assessment based on regime characteristics,
    including confidence, stability, transition probabilities, and maturity.
    Calculates an overall risk score (0-100) with component breakdown and
    specific mitigation strategies.

    Args:
        ticker: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'NVDA')

    Returns:
        Prompt text instructing Claude to perform risk assessment

    Example:
        >>> prompt = await regime_risk_assessment("NVDA")
        >>> # Claude will calculate risk score and provide mitigation strategies
    """
    ticker = ticker.upper().strip()

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


async def regime_historical_analogs(
    ticker: str, lookback_period: str = "5y"
) -> str:
    """
    Find similar historical regime patterns.

    Identifies historical regime episodes similar to the current pattern and
    analyzes their outcomes. Provides probabilistic guidance on expected
    transition timing and likely next regimes based on historical precedent.

    Args:
        ticker: Stock symbol to analyze (e.g., 'SPY', 'AAPL', 'NVDA')
        lookback_period: How far back to search for patterns (default: "5y")
            Examples: "3y", "5y", "10y", or specific date "2015-01-01"

    Returns:
        Prompt text instructing Claude to find and analyze historical patterns

    Example:
        >>> prompt = await regime_historical_analogs("SPY", lookback_period="10y")
        >>> # Claude will find similar past regime episodes and analyze outcomes
    """
    ticker = ticker.upper().strip()

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


async def regime_portfolio_review(portfolio: str) -> str:
    """
    Portfolio-wide regime analysis.

    Analyzes regime exposure across an entire portfolio, calculating weighted
    metrics, identifying concentration risks, and providing rebalancing
    recommendations to improve regime diversification.

    Args:
        portfolio: JSON string with format: {"ticker": weight, ...}
            Example: '{"SPY": 0.4, "QQQ": 0.3, "IWM": 0.3}'
            Weights should sum to 1.0 (or 100 for percentages)

    Returns:
        Prompt text instructing Claude to perform portfolio regime analysis

    Example:
        >>> portfolio_json = '{"SPY": 0.5, "QQQ": 0.3, "GLD": 0.2}'
        >>> prompt = await regime_portfolio_review(portfolio_json)
        >>> # Claude will analyze portfolio-wide regime exposure
    """
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
