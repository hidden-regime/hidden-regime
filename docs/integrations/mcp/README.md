# Hidden Regime MCP Server

**Model Context Protocol server for HMM-based market regime detection**

Enable AI assistants like Claude to detect and analyze market regimes using Hidden Markov Models directly in natural language conversations.

---

## What is This?

The Hidden Regime MCP server exposes regime detection capabilities to AI assistants via the [Model Context Protocol](https://modelcontextprotocol.io). Once configured, you can ask Claude questions like:

- "What's SPY's current market regime?"
- "Analyze NVDA's regime statistics for 2024"
- "What's the probability QQQ transitions to a bear regime?"

Claude will automatically call the Hidden Regime tools and provide analysis based on HMM regime detection.

---

## Quick Start

### 1. Install

```bash
# Activate your virtual environment
source ~/hidden-regime-pyenv/bin/activate

# Install with MCP extras
pip install hidden-regime[mcp]
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS:**
```bash
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Linux:**
```bash
nano ~/.config/Claude/claude_desktop_config.json
```

**Windows:**
```bash
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**Configuration:**
```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/home/aoaustin/hidden-regime-pyenv/bin/python",
      "args": [
        "-m",
        "fastmcp",
        "run",
        "/path/to/hidden-regime/hidden_regime_mcp/server.py"
      ]
    }
  }
}
```

**Note:** Replace `/path/to/hidden-regime` with your actual installation path. You can find it with:
```bash
python -c "import hidden_regime_mcp; print(hidden_regime_mcp.__file__.replace('/__init__.py', ''))"
```

### 3. Restart Claude Desktop

Completely quit Claude Desktop (Cmd+Q on Mac, not just close window) and restart.

### 4. Test

Ask Claude:
```
What's SPY's current market regime?
```

Claude should automatically detect the regime and provide analysis!

---

## Available Tools

### 1. detect_regime

**Detect current market regime for a stock with rich temporal context and interpretation**

**Parameters:**
- `ticker` (required): Stock symbol (e.g., 'SPY', 'AAPL', 'NVDA')
- `n_states` (optional): Number of regimes (2-5, default: 3)
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Example Query:**
```
"What's NVDA's current market regime?"
"Detect the regime for QQQ using 4 states"
"What was SPY's regime on March 15, 2024?"
```

**Response Structure:**

The tool returns a comprehensive analysis with four categories of information:

**Basic Regime Information:**
- `ticker`: Stock symbol
- `current_regime`: Current regime name (e.g., 'bull', 'bear', 'sideways')
- `confidence`: Confidence in current regime (0-1)
- `mean_return`: Mean daily return for current regime
- `volatility`: Daily volatility (standard deviation)
- `last_updated`: Date of last data point
- `n_states`: Number of states used in analysis
- `analysis_period`: Dict with start and end dates

**Temporal Context:**
- `days_in_regime`: Number of days in current regime
- `expected_duration`: Expected duration of this regime (days)
- `percent_complete`: Percentage of expected duration completed
- `regime_status`: "early", "mid", "mature", or "overdue"
- `days_until_expected_transition`: Days until expected regime change

**Price Context:**
- `price_performance`: YTD, 30-day, and 7-day returns (both decimal and formatted %)
- `current_price`: Latest closing price
- `price_trend`: "up", "down", "consolidating", or "flat"

**Stability & Interpretation:**
- `regime_stability`: "stable", "moderate", or "volatile"
- `recent_transitions`: Number of regime changes in last 30 days
- `previous_regime`: Previous regime name
- `last_transition_date`: Date of transition to current regime
- `interpretation`: Brief human-readable description
- `explanation`: Detailed narrative reconciling regime with price action

**Example Response:**
```json
{
  "ticker": "NVDA",
  "current_regime": "bearish",
  "confidence": 0.89,
  "mean_return": -0.000751,
  "volatility": 0.214,
  "last_updated": "2025-10-31",
  "n_states": 3,
  "analysis_period": {"start": "2023-11-03", "end": "2025-10-31"},
  "days_in_regime": 2,
  "expected_duration": 9.21,
  "percent_complete": 21.7,
  "regime_status": "early",
  "days_until_expected_transition": 7.2,
  "price_performance": {
    "ytd_return": 0.464,
    "ytd_return_pct": "+46.4%",
    "30d_return": 0.081,
    "30d_return_pct": "+8.1%",
    "7d_return": 0.087,
    "7d_return_pct": "+8.7%",
    "current_price": 202.49,
    "price_trend": "up"
  },
  "current_price": 202.49,
  "price_trend": "up",
  "regime_stability": "moderate",
  "recent_transitions": 1,
  "previous_regime": "bullish",
  "last_transition_date": "2025-10-30",
  "interpretation": "High volatility phase with modest negative returns",
  "explanation": "While the stock is up +46.4% YTD, recent behavior shows increased volatility (21.4% daily std dev) and 8.1% gain over 30 days. The model detects this as a bearish regime, indicating transition from the strong uptrend that characterized earlier performance. This is typical consolidation or correction after a major rally."
}
```

**Key Features:**
- **Temporal Awareness**: Understand how long the regime has persisted and when it might change
- **Price Context**: See how the regime relates to actual price performance (YTD, 30d, 7d)
- **Smart Interpretation**: Explanations that reconcile seemingly contradictory signals (e.g., "bearish" regime despite YTD gains)
- **Stability Analysis**: Know if regimes are stable or frequently changing

**Error Response Example:**
```json
{
  "error": "Unable to load data for INVALID. Please check the ticker symbol and try again."
}
```

Or:
```json
{
  "error": "Insufficient data for NEWIPO. Need at least 100 observations for reliable regime detection."
}
```

Or:
```json
{
  "error": "No data available for SPY in the specified date range. This may be due to weekends, holidays, or insufficient trading data."
}
```

### 2. get_regime_statistics

**Get detailed statistics for all detected regimes**

**Parameters:**
- `ticker` (required): Stock symbol
- `n_states` (optional): Number of regimes (default: 3)
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Example Query:**
```
"Show me regime statistics for NVDA in 2024"
"What are the characteristics of each regime for SPY?"
```

**Response:**
```json
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
    "bear": {...},
    "sideways": {...}
  },
  "analysis_period": {
    "start": "2024-01-01",
    "end": "2025-10-31",
    "total_days": 669
  }
}
```

### 3. get_transition_probabilities

**Get regime transition matrix and forecasts**

**Parameters:**
- `ticker` (required): Stock symbol
- `n_states` (optional): Number of regimes (default: 3)

**Example Query:**
```
"What's the probability SPY transitions to a bear regime?"
"Show me the transition matrix for QQQ"
"How long do bull regimes typically last for AAPL?"
```

**Response:**
```json
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
  }
}
```

---

## Available Resources

Resources provide URI-based access to regime data.

### 1. regime://{ticker}/current

Get current regime for any ticker.

**Example:** `regime://SPY/current`

### 2. regime://{ticker}/transitions

Get transition probabilities for any ticker.

**Example:** `regime://SPY/transitions`

---

## Available Prompts

Prompts provide pre-configured analysis workflows that guide Claude through sophisticated regime detection tasks. Instead of manually crafting complex queries, you can select a prompt and let it orchestrate the entire analysis.

### Why Use Prompts?

- **Expert-Level Analysis:** Get sophisticated multi-step analysis with a single command
- **Consistency:** Standardized analytical frameworks ensure thorough coverage
- **Discoverability:** Browse available prompts to learn what's possible
- **Efficiency:** One prompt replaces 5+ manual tool calls

### How to Use Prompts

In Claude Desktop, prompts appear in the prompt library. Select a prompt, fill in the required arguments (like ticker symbol), and Claude automatically executes the complete analysis workflow.

---

### 1. regime-quick-check

**Quick market health check for day-to-day decisions**

**Arguments:**
- `ticker` (required): Stock symbol (e.g., 'SPY', 'AAPL', 'NVDA')

**What it does:**
1. Analyzes current regime and confidence
2. Checks regime maturity and stability
3. Reviews recent price performance
4. Provides 3-4 sentence actionable summary

**Best for:** Quick market checks, daily decision-making

**Example:**
```
Use regime-quick-check for SPY
```

**Sample Output:**
```
SPY is in a bearish regime (89% confidence) that started 12 days ago
and is in early phase (13% of expected duration complete). Despite being
up +17.7% YTD, recent volatility has increased to 8.4% daily, indicating
a consolidation phase. The regime is stable with no recent transitions,
suggesting traders should wait for clearer signals before aggressive positioning.
```

---

### 2. regime-deep-dive

**Comprehensive regime analysis with historical context**

**Arguments:**
- `ticker` (required): Stock symbol
- `lookback_period` (optional): Time period, default "2y" (e.g., "1y", "3y", "5y")

**What it does:**
1. Calls `detect_regime` for current state
2. Calls `get_regime_statistics` for historical patterns
3. Calls `get_transition_probabilities` for forward-looking analysis
4. Synthesizes comprehensive multi-section report

**Report Sections:**
- Current Regime State (regime, confidence, temporal context)
- Historical Context (regime distribution, performance by regime)
- Transition Dynamics (transition matrix, expected durations)
- Statistical Summary (best/worst regimes, steady state)
- Trading Implications (strategies, risks, catalysts)

**Best for:** Portfolio reviews, research, strategic planning

**Example:**
```
Use regime-deep-dive for NVDA with lookback_period="3y"
```

---

### 3. regime-strategy-advisor

**Trading strategy recommendations based on current regime**

**Arguments:**
- `ticker` (required): Stock symbol
- `risk_tolerance` (optional): "low", "medium", or "high", default "medium"

**What it does:**
1. Analyzes current regime characteristics
2. Maps regime + risk tolerance → specific strategies
3. Provides entry/exit signals based on regime maturity
4. Includes risk management guidelines

**Strategy Recommendations:**

**Bull Regimes:**
- Low risk: Long-only, trailing stops, DCA
- Medium risk: Trend-following, momentum strategies
- High risk: Leveraged long, aggressive growth

**Bear Regimes:**
- Low risk: Cash, defensive sectors
- Medium risk: Short positions, inverse ETFs
- High risk: Aggressive shorts, put options

**Sideways Regimes:**
- Low risk: Range-bound trading, covered calls
- Medium risk: Mean reversion, pairs trading
- High risk: Iron condors, straddles

**Best for:** Strategy selection, risk-appropriate position sizing

**Example:**
```
Use regime-strategy-advisor for QQQ with risk_tolerance="high"
```

---

### 4. regime-multi-asset-comparison

**Compare regimes across multiple assets**

**Arguments:**
- `tickers` (required): Comma-separated list (e.g., "SPY,QQQ,IWM,DIA")

**What it does:**
1. Analyzes regime for each ticker
2. Creates comparison table (regime, confidence, status, performance)
3. Calculates regime distribution across portfolio
4. Identifies concentration risks and diversification opportunities

**Analysis Sections:**
- Regime Overview Table
- Regime Distribution (% in bull/bear/sideways)
- Diversification Analysis
- Correlation Insights
- Portfolio Implications
- Risk Assessment

**Best for:** Portfolio construction, diversification analysis, multi-asset monitoring

**Example:**
```
Use regime-multi-asset-comparison for "SPY,QQQ,IWM,DIA,GLD,TLT"
```

**Sample Output:**
```
| Ticker | Regime   | Confidence | Status  | YTD    | 30d    |
|--------|----------|------------|---------|--------|--------|
| SPY    | bearish  | 89%        | early   | +17.7% | +2.0%  |
| QQQ    | bullish  | 76%        | mature  | +24.3% | +5.1%  |
| IWM    | sideways | 64%        | mid     | +8.2%  | -1.3%  |
| DIA    | bearish  | 82%        | early   | +14.1% | +1.8%  |
| GLD    | bullish  | 91%        | mid     | +22.5% | +3.7%  |
| TLT    | sideways | 58%        | mature  | -2.1%  | -0.5%  |

Portfolio shows 33% bull, 33% bear, 33% sideways - well diversified!
```

---

### 5. regime-risk-assessment

**Quantitative risk analysis based on regime stability**

**Arguments:**
- `ticker` (required): Stock symbol

**What it does:**
1. Analyzes regime confidence, stability, and maturity
2. Calculates transition probabilities to unfavorable regimes
3. Computes overall risk score (0-100) with component breakdown
4. Provides specific mitigation strategies

**Risk Score Components:**
- **Confidence Risk (30 pts):** Lower confidence = higher risk
- **Transition Risk (30 pts):** Mature/overdue regimes = higher risk
- **Stability Risk (20 pts):** Frequent transitions = higher risk
- **Adverse Transition Risk (20 pts):** High probability of bad outcome

**Risk Score Interpretation:**
- 0-30: Low risk - stable regime, high confidence
- 31-60: Medium risk - some caution warranted
- 61-100: High risk - significant risk factors present

**Best for:** Risk management, position sizing, portfolio monitoring

**Example:**
```
Use regime-risk-assessment for TSLA
```

**Sample Output:**
```
Overall Risk Score: 67/100 (High Risk)

Component Breakdown:
- Confidence Risk: 18/30 (62% confidence)
- Transition Risk: 28/30 (regime 95% complete, overdue)
- Stability Risk: 15/20 (2 transitions in last 30 days)
- Adverse Transition Risk: 6/20 (30% probability of bear transition)

Top Risk Factors:
1. Overdue regime (95% of expected duration complete)
2. Moderate confidence (62%) suggests uncertainty
3. Recent volatility (2 transitions in 30 days)

Mitigation Strategies:
- Tighten stop losses to 5% below current price
- Reduce position size by 30-50%
- Set alert for regime transition
- Consider partial profit-taking
```

---

### 6. regime-historical-analogs

**Find similar historical regime patterns**

**Arguments:**
- `ticker` (required): Stock symbol
- `lookback_period` (optional): How far back to search, default "5y"

**What it does:**
1. Identifies current regime characteristics
2. Finds similar historical episodes
3. Analyzes what happened after past episodes
4. Projects timeline for next transition

**Analysis Sections:**
- Current Regime Pattern
- Pattern Matching Analysis
- Historical Regime Episodes
- Transition Outcomes
- Timeline Projection
- Pattern Insights

**Best for:** Outcome prediction, timeline expectations, pattern recognition

**Example:**
```
Use regime-historical-analogs for SPY with lookback_period="10y"
```

**Sample Output:**
```
Current Pattern: Bull regime, 23 days active (45% complete)

Historical Analysis (past 10 years):
- This regime type occurred 8 times previously
- Average duration: 51 days (current: 23 days)
- Shortest: 12 days, Longest: 89 days
- Current episode is tracking close to historical average

Transition Outcomes:
- 62% transitioned to sideways regime (5 of 8 episodes)
- 25% transitioned to bear regime (2 of 8 episodes)
- 13% extended beyond expected duration (1 of 8 episodes)

Timeline Projection:
- Expected transition in ~28 days (±10 days)
- Current regime is 45% complete
- Regime status: mid-phase
```

---

### 7. regime-portfolio-review

**Portfolio-wide regime analysis and rebalancing**

**Arguments:**
- `portfolio` (required): JSON with format `{"ticker": weight, ...}`
  - Example: `{"SPY": 0.4, "QQQ": 0.3, "GLD": 0.2, "TLT": 0.1}`

**What it does:**
1. Analyzes regime for each portfolio holding
2. Calculates weighted portfolio metrics
3. Identifies concentration risks
4. Provides specific rebalancing recommendations

**Analysis Sections:**
- Portfolio Holdings Table
- Portfolio Regime Breakdown (weighted %)
- Weighted Metrics (confidence, volatility, returns)
- Concentration Risk Analysis
- Diversification Assessment (0-100 score)
- Rebalancing Recommendations
- Risk Summary

**Best for:** Portfolio management, rebalancing decisions, risk monitoring

**Example:**
```
Use regime-portfolio-review with portfolio='{"SPY": 0.5, "QQQ": 0.3, "GLD": 0.2}'
```

**Sample Output:**
```
Portfolio Regime Breakdown:
- 80% in Bull regimes (SPY: 50%, QQQ: 30%)
- 20% in Sideways regimes (GLD: 20%)
- 0% in Bear regimes

⚠️ Concentration Risk: HIGH
- Over-exposed to bull regimes (>60% threshold)
- 50% of portfolio in mature regimes (approaching transition)
- Low regime diversification score: 35/100

Rebalancing Recommendations:
1. Reduce QQQ allocation from 30% → 20% (mature bull regime, high transition risk)
2. Add defensive position (TLT or cash): 10% allocation
3. Increase GLD from 20% → 25% (stable sideways regime)
4. Consider adding IWM: 5% allocation (different regime pattern than SPY/QQQ)
5. Target regime distribution: 50% bull, 25% sideways, 25% defensive

This would improve diversification score from 35 → 68/100
```

---

## Prompts vs Tools: When to Use Each

**Use Prompts When:**
- You want expert-level analysis frameworks
- You need multi-step analysis (multiple tool calls)
- You're learning what's possible with regime detection
- You want consistent, thorough analysis
- You prefer guided workflows

**Use Tools Directly When:**
- You need specific data points only
- You're building custom analysis
- You want programmatic access
- You're experienced with regime analysis
- You need granular control

**Best Practice:** Start with prompts to learn, then use tools directly for custom workflows.

---

## Logging & Transparency

The MCP server provides structured logging to give you visibility into operations. Claude Desktop displays these logs so you can see what's happening during regime analysis.

### What Gets Logged

**Cache Operations:**
- Cache hits (returning cached results instantly)
- Cache misses (running fresh analysis)

**Data Loading:**
- When data is being fetched from Yahoo Finance
- Number of observations downloaded
- Date range of downloaded data

**Model Training:**
- When HMM model training starts
- Number of states being trained
- When training completes

**Result Caching:**
- When results are cached for future use

### Example Log Output

```
[INFO] Cache miss for SPY - running fresh regime analysis
[INFO] Fetching data for SPY from Yahoo Finance...
[INFO] Training 3-state HMM model for SPY...
[INFO] Downloaded 500 observations for SPY (2023-01-01 to 2024-12-31)
[INFO] Model training completed for SPY
[INFO] Regime detection completed for SPY - result cached
```

### Why Logging is Useful

**Transparency:**
- Understand what's happening during slow operations
- See when cached results are being used
- Know when data is being fetched vs using cache

**Performance Insights:**
- Identify cache efficiency
- Understand which operations take time
- Optimize query patterns

**Debugging:**
- Diagnose issues with data loading
- Verify model training occurred
- Confirm caching behavior

### Log Levels

- **INFO**: Normal operations (cache, data loading, training)
- **WARNING**: Non-critical issues (retries, degraded performance)
- **ERROR**: Failures requiring attention

### Viewing Logs in Claude Desktop

Logs appear automatically in Claude Desktop during tool execution. You'll see them in real-time as the MCP server processes your request.

For persistent logging, check Claude Desktop logs:
- **macOS**: `~/Library/Logs/Claude/`
- **Linux**: `~/.config/Claude/logs/`
- **Windows**: `%APPDATA%\Claude\logs\`

---

## Example Usage Scenarios

### Scenario 1: Quick Market Check with Context

**You:**
```
What's the current market regime for SPY and QQQ?
```

**Claude:**
```
Let me check the current regimes for both tickers.

[Calls detect_regime for SPY and QQQ]

SPY: Bearish regime (94% confidence)
- Has been in this regime for 63 days (89.6% of expected duration)
- Regime status: mature
- YTD performance: +17.7%
- 30-day performance: +2.0%
- Stability: stable (0 transitions in last 30 days)
- Interpretation: "High volatility phase with modest positive returns"

While SPY is up significantly YTD, the current regime reflects increased
volatility (8.4% daily) indicating consolidation after the strong rally.
This is a mature regime approaching expected transition.

QQQ: Similar pattern with bull regime characteristics and positive YTD returns.

Both indices show stable regimes but approaching potential transition points.
Monitor for regime changes as signals for strategy adjustments.
```

### Scenario 2: Historical Analysis

**You:**
```
Analyze NVDA's regime behavior during 2024. Were there significant regime changes?
```

**Claude:**
```
Let me analyze NVDA's regimes throughout 2024.

[Calls get_regime_statistics with start_date="2024-01-01", end_date="2024-12-31"]

NVDA showed 3 distinct regimes in 2024:

Bull Regime (68% of year):
- Mean return: +3.2%
- Average duration: 45 days
- Win rate: 72%

Sideways Regime (22%):
- Mean return: -0.3%
- Average duration: 12 days

Bear Regime (10%):
- Mean return: -2.8%
- Average duration: 8 days

Notable: NVDA spent most of 2024 in bull regime, with brief
sideways consolidations. Bear regime appearances were short-lived.
```

### Scenario 3: Risk Assessment

**You:**
```
I'm holding SPY in a bull regime. What's the risk it transitions to bear?
```

**Claude:**
```
[Calls get_transition_probabilities for SPY]

Based on SPY's historical transition patterns:

Current: Bull regime

Transition probabilities:
- Stay in bull: 85%
- Move to sideways: 10%
- Move to bear: 5%

Expected bull regime duration: ~20 days

Analysis: Direct bull → bear transitions are rare (5%). More
common path is bull → sideways → bear. Current risk is low,
but watch for sideways regime as early warning signal.
```

### Scenario 4: Understanding Regime vs. Price Action

**You:**
```
NVDA is at a YTD high but you're saying it's in a bearish regime? That doesn't make sense.
```

**Claude:**
```
[Calls detect_regime for NVDA]

Great question! Let me explain what's happening with NVDA:

Current Status:
- Regime: Bearish (89% confidence)
- YTD Return: +46.4%
- 30-day Return: +8.1%
- Current Price: $202.49 (up trend)

Explanation:
The bearish regime doesn't mean NVDA is down - it's actually up significantly!
The regime reflects the *statistical characteristics* of recent price behavior:

- High volatility: 21.4% daily (very elevated)
- Just transitioned from bullish regime (2 days ago)
- This is typical consolidation after a major rally

Think of it as: "The strong uptrend has paused, and we're now in a high-volatility
consolidation phase." The HMM detects the change in behavior patterns, not just
price direction.

This is actually useful for risk management - the regime tells you that despite
the YTD gains, recent volatility is elevated and behavior is different from the
strong trending phase that drove earlier returns.
```

### Scenario 5: Strategy Selection

**You:**
```
I'm trading QQQ today. Should I use trend-following or mean-reversion?
```

**Claude:**
```
[Calls detect_regime for QQQ]

QQQ is currently in a bull regime (78% confidence):
- Mean return: +2.8%
- Volatility: 14%
- Regime status: mid (45% complete)
- Days in regime: 23 of expected 51 days
- Stability: stable

Recommendation: Trend-following

Bull regimes favor directional strategies:
- Moving average crossovers
- Breakout systems
- Trailing stops

This regime is still in mid-phase with good stability (no recent transitions),
suggesting the trend has room to run. Avoid mean-reversion in bull regimes -
prices tend to trend rather than revert.

Consider re-evaluating when regime approaches "mature" status or shows signs
of transition to sideways.
```

---

## Troubleshooting

### Server Not Appearing

**Symptoms:** Claude doesn't show Hidden Regime tools

**Solutions:**
1. Check JSON syntax in config file (use JSONLint)
2. Verify Python path: `which python` in terminal
3. Verify server.py path exists
4. Check Claude Desktop logs (Help → View Logs)
5. Completely quit Claude (Cmd+Q), don't just close window

### Tools Not Working

**Symptoms:** Tools appear but return errors

**Solutions:**
1. Check hidden-regime package is installed: `pip show hidden-regime`
2. Install MCP extras: `pip install hidden-regime[mcp]`
3. Test server manually: `fastmcp run /path/to/server.py`
4. Check for firewall/permission issues

### Slow Responses

**Symptoms:** Tools take >10 seconds to respond

**Solutions:**
1. First query typically takes 2-5 seconds to fetch data and train HMM model
2. Check internet connection (yfinance needs data access)
3. Try shorter date ranges
4. Use fewer states (n_states=2 is faster than 5)

### Import Errors

**Symptoms:** "ModuleNotFoundError: No module named 'fastmcp'"

**Solutions:**
1. Install MCP extras: `pip install hidden-regime[mcp]`
2. Verify correct virtual environment is activated
3. Check pip installation: `pip list | grep fastmcp`

### Data Loading Errors

**Symptoms:** "Unable to load data for TICKER"

**Solutions:**
1. Verify ticker symbol is correct (use Yahoo Finance symbol)
2. Check internet connection
3. Try different ticker (some have limited data)
4. Avoid weekends/holidays (markets closed)

### Common Error Messages

**Error:** `"Unable to load data for TICKER. Please check the ticker symbol and try again."`
- **Cause:** Invalid ticker symbol or no data available from Yahoo Finance
- **Solution:** Verify ticker exists on Yahoo Finance, check spelling, try a different ticker

**Error:** `"Insufficient data for TICKER. Need at least 100 observations for reliable regime detection."`
- **Cause:** Ticker has too few trading days (e.g., recently IPO'd stocks, delisted stocks)
- **Solution:** Use a longer date range or choose a ticker with more history

**Error:** `"No data available for TICKER in the specified date range. This may be due to weekends, holidays, or insufficient trading data."`
- **Cause:** Date range contains no trading days (weekends/holidays) or empty DataFrame
- **Solution:** Use a date range that includes trading days, avoid single-day queries on weekends

**Error:** `"start_date must be before or equal to end_date"`
- **Cause:** Invalid date range (start date is after end date)
- **Solution:** Check date parameters, ensure start_date < end_date

**Error:** `"end_date cannot be in the future"`
- **Cause:** Trying to analyze future dates
- **Solution:** Use current or past dates only

**Error:** `"n_states must be between 2 and 5"`
- **Cause:** Invalid number of regimes specified
- **Solution:** Use n_states value between 2 and 5 (3 is recommended default)

**Error:** `"Ticker symbol is required"`
- **Cause:** Empty or missing ticker parameter
- **Solution:** Provide a valid ticker symbol

**Error:** `"Ticker symbol too long: TICKER. Max 10 characters"`
- **Cause:** Ticker exceeds maximum length
- **Solution:** Use standard ticker symbols (typically 1-5 characters)

---

## Performance Tips

### Query Performance

Each query requires fetching data from Yahoo Finance and training an HMM model, which takes 2-5 seconds.

**Factors affecting speed:**
- **Ticker popularity**: Major indices (SPY, QQQ) have reliable data and load quickly
- **Date range**: Longer date ranges require more data processing
- **Number of states**: More states (n_states=4 or 5) require more complex model training

### Optimal Usage

**Faster queries:**
```
"What's SPY's current regime?"
- Default parameters (n_states=3)
- Major ticker with reliable data
- Recent date range (default)
```

**Slower queries:**
```
"Analyze regime for OBSCURETICKER with 5 states from 2000-2024"
- Obscure ticker (may have missing data)
- 24-year date range
- 5 states (more complex model)
```

**Recommendation:** Use default parameters (n_states=3) for best balance of speed and insight.

---

## Development & Testing

### Test Server Locally

```bash
# Activate virtual environment
source ~/hidden-regime-pyenv/bin/activate

# Install with dev extras
pip install -e ".[mcp,dev]"

# Run server manually
fastmcp run hidden_regime_mcp/server.py

# Test with MCP inspector
npx @modelcontextprotocol/inspector fastmcp run hidden_regime_mcp/server.py
```

### Run Tests

```bash
pytest tests/test_mcp/ -v
```

### View Logs

**Claude Desktop logs:**
- macOS: `~/Library/Logs/Claude/`
- Linux: `~/.config/Claude/logs/`
- Windows: `%APPDATA%\Claude\logs\`

---

## Advanced Configuration

### Custom Virtual Environment

```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/custom/path/to/venv/bin/python",
      "args": ["-m", "fastmcp", "run", "/path/to/server.py"],
      "env": {
        "PYTHONPATH": "/custom/path"
      }
    }
  }
}
```

### Multiple Configurations

```json
{
  "mcpServers": {
    "hidden-regime-3state": {
      "command": "python",
      "args": ["-m", "fastmcp", "run", "/path/to/server.py"],
      "env": {
        "DEFAULT_N_STATES": "3"
      }
    },
    "hidden-regime-4state": {
      "command": "python",
      "args": ["-m", "fastmcp", "run", "/path/to/server.py"],
      "env": {
        "DEFAULT_N_STATES": "4"
      }
    }
  }
}
```

### Debug Mode

```json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "python",
      "args": ["-m", "fastmcp", "run", "/path/to/server.py", "--log-level", "DEBUG"]
    }
  }
}
```

---

## Limitations & Constraints

### Data Requirements

**Minimum Observations:**
- Regime detection requires at least **100 trading days** of data
- Fewer observations will result in "Insufficient data" error
- Recommendation: Use tickers with at least 6 months of trading history

**Data Source:**
- All data comes from Yahoo Finance via the `yfinance` library
- Data quality and availability depend on Yahoo Finance
- Some tickers may have gaps, missing data, or be delisted

### Temporal Constraints

**Market Hours:**
- No data available on weekends, holidays, or non-trading days
- Queries on weekends/holidays will return "No data available" error
- Use date ranges that include actual trading days

**Data Frequency:**
- Only **daily** data is supported (close prices or OHLC average)
- No intraday regime detection (1-min, 5-min, hourly, etc.)
- Historical data only - cannot analyze real-time tick data

**Date Ranges:**
- Future dates are not allowed
- `start_date` must be before or equal to `end_date`
- Very long date ranges (>10 years) may be slow

### Model Constraints

**Number of Regimes:**
- Must specify between 2 and 5 states (`n_states`)
- Default is 3 (typically: bull, bear, sideways)
- More states require more data for reliable estimation

**Model Assumptions:**
- Assumes Gaussian emission distributions (may underestimate tail risk)
- Point estimates only - no uncertainty quantification
- Batch processing - no online/incremental learning

### Performance Constraints

**Query Speed:**
- Each query takes 2-5 seconds (data fetch + HMM training)
- No caching - every query recomputes from scratch
- Longer date ranges and more states are slower

**Rate Limits:**
- Subject to Yahoo Finance rate limits
- Too many rapid queries may trigger temporary blocking
- Recommendation: Space out queries if analyzing many tickers

### System Constraints

**Local Execution:**
- MCP server runs on your local machine (not cloud-hosted)
- Requires active internet connection for Yahoo Finance data
- Claude Desktop must be running for MCP to work

**Platform Support:**
- Tested on macOS, Linux, and Windows (via WSL)
- Requires Python 3.9+ environment
- Requires `hidden-regime` package with MCP extras installed

---

## FAQ

### Q: Does this send my data to a server?

**A:** No. The MCP server runs entirely on your local machine. Your queries and regime detection results never leave your computer.

### Q: Do I need an API key?

**A:** No. The server uses yfinance which is free and doesn't require API keys.

### Q: Can I use this with other AI assistants?

**A:** Yes, any MCP-compatible client works (Claude Desktop, VS Code with MCP extension, custom applications).

### Q: How accurate is the regime detection?

**A:** Regime detection is probabilistic. Confidence scores indicate certainty. High confidence (>80%) suggests strong regime signal. Low confidence (<50%) suggests uncertain/transitional period.

### Q: Can I analyze multiple tickers at once?

**A:** Yes, ask Claude to analyze multiple tickers and it will call the tool multiple times in parallel.

### Q: What data source does it use?

**A:** Yahoo Finance via the yfinance library. Same data source as the core Hidden Regime package.

### Q: How far back can I analyze?

**A:** Depends on ticker. Most major indices (SPY, QQQ) have data back to 1990s. Individual stocks vary.

### Q: Can I analyze intraday regimes?

**A:** No, currently only daily data is supported.

---

## Support

**Issues:** https://github.com/hidden-regime/hidden-regime/issues
**Documentation:** https://docs.hiddenregime.com
**Website:** https://hiddenregime.com

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io) - Anthropic's MCP specification
- [Hidden Regime](https://github.com/hidden-regime/hidden-regime) - Core HMM library

---

**Version:** 0.1.0
**Last Updated:** October 31, 2025
