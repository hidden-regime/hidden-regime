# Hidden Regime MCP - Error Handling Guide

Comprehensive guide to understanding, diagnosing, and resolving errors in Hidden Regime MCP.

## Table of Contents
1. [Error Categories](#error-categories)
2. [Error Code Reference](#error-code-reference)
3. [Common Errors and Solutions](#common-errors-and-solutions)
4. [Retry Logic](#retry-logic)
5. [Troubleshooting Guide](#troubleshooting-guide)

---

## Error Categories

Hidden Regime organizes errors into five main categories:

### 1. Validation Errors (1xxx)
Input validation failures - parameter issues that prevent tool execution.

**Characteristics:**
- User-provided incorrect input
- Invalid format or out-of-range values
- **Not retriable** - same input will fail again

**Examples:**
- Invalid ticker symbol format
- Date in wrong format
- Start date after end date

### 2. Data Errors (2xxx)
Data loading and processing failures.

**Characteristics:**
- Issue with market data availability or quality
- May be **retriable** (temporary data source issue)
- May be **non-retriable** (insufficient historical data)

**Examples:**
- No data found for ticker
- Insufficient historical data (< 100 days)
- Data processing failed

### 3. Model Errors (3xxx)
HMM training and inference failures.

**Characteristics:**
- Issue with statistical model operations
- Usually **non-retriable** - indicates model constraints
- May require different parameters

**Examples:**
- Model training failed
- Invalid model state
- Parameter estimation failed

### 4. Network Errors (4xxx)
Network connectivity and data source failures.

**Characteristics:**
- **Retriable** - temporary network issues
- Exponential backoff recommended
- May indicate data provider outage

**Examples:**
- Network timeout
- Connection failed
- Rate limit exceeded

### 5. Resource Errors (5xxx)
System resource constraints.

**Characteristics:**
- **Retriable** - may succeed after resource is freed
- May indicate system overload

**Examples:**
- Out of memory
- Request timeout
- Resource exhausted

---

## Error Code Reference

### Validation Errors (1xxx)

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 1001 | Invalid Ticker | Ticker format wrong | Use standard format (e.g., SPY, AAPL, BRK.A) |
| 1002 | Invalid N States | States not 2-5 | Use between 2 (minimal) and 5 (max) |
| 1003 | Invalid Date Format | Date not YYYY-MM-DD | Use format like 2025-01-15 |
| 1004 | Invalid Date Range | Start > End | Ensure start_date < end_date |
| 1005 | Invalid Parameter | Other parameter issue | Check parameter values |

### Data Errors (2xxx)

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 2001 | Data Not Found | No data for ticker | Check ticker symbol, try different date range |
| 2002 | Insufficient Data | < 100 observations | Use longer date range or established ticker |
| 2003 | Data Processing Failed | Data transformation error | Try with different parameters |
| 2004 | Missing Required Field | Data incomplete | Check data source |
| 2005 | Data Validation Failed | Data quality issue | Try different ticker or date range |

### Model Errors (3xxx)

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 3001 | Model Training Failed | HMM training issue | Try n_states=2 or longer date range |
| 3002 | Model Inference Failed | Viterbi algorithm error | Insufficient training data |
| 3003 | Invalid Model State | Model not properly trained | Retrain with valid parameters |
| 3004 | Parameter Estimation Failed | EM algorithm issue | Different date range might help |

### Network Errors (4xxx)

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 4001 | Network Timeout | Request took too long | Check internet, try again |
| 4002 | Connection Failed | Cannot reach server | Verify internet connectivity |
| 4003 | Data Source Unavailable | yfinance down | Wait and retry, check status |
| 4004 | yfinance Error | Yahoo Finance API issue | Wait a moment and retry |
| 4005 | Rate Limit Exceeded | Too many requests | Wait before retrying |

### Resource Errors (5xxx)

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| 5001 | Out of Memory | Insufficient RAM | Reduce date range, restart server |
| 5002 | Timeout | Operation too slow | Try simpler query, restart server |
| 5003 | Resource Exhausted | System overloaded | Wait and retry, increase resources |

---

## Common Errors and Solutions

### Error: "Invalid ticker symbol: ABC123"

**Problem:** Ticker format is invalid.

**Root Cause:**
- Contains non-alphanumeric characters (except . and -)
- More than 10 characters
- Empty or null value

**Solutions:**
```
✓ Valid:    SPY, AAPL, BRK.A, MSFT
✗ Invalid:  ABC123!, SPY-OLD, VeryLongTickerName
```

**Action:**
- Use standard stock ticker symbols
- Check for typos
- Verify symbol on Yahoo Finance or your broker

---

### Error: "Insufficient data for TICKER. Need at least 100 observations"

**Problem:** Not enough historical data available.

**Root Cause:**
- Ticker has limited trading history
- Date range too narrow (< 100 trading days)
- Ticker recently IPO'd or delisted

**Solutions:**

**Option 1: Use longer date range**
```
Instead of:   start_date=2025-09-01, end_date=2025-11-09
Try:          start_date=2025-01-01, end_date=2025-11-09
              (Provides ~10 months = ~240 trading days)
```

**Option 2: Use established ticker**
```
✓ Works:      SPY, QQQ, AAPL, MSFT (extensive history)
✗ May fail:   New IPOs, penny stocks, micro-caps
```

**Option 3: Check ticker validity**
```python
# Test if ticker has enough data
import yfinance as yf
data = yf.download('YOUR_TICKER', period='2y')
print(f"Available: {len(data)} trading days")
```

---

### Error: "Unable to load data for TICKER. Network timeout after 30s"

**Problem:** Network request to yfinance timed out.

**Root Cause:**
- Internet connectivity issue
- yfinance server slow or unavailable
- Rate limiting (too many requests)

**Solutions:**

**Option 1: Check internet connection**
```bash
# Test connectivity
ping www.google.com

# Test yfinance specifically
python -c "import yfinance as yf; print(yf.download('SPY', period='1d'))"
```

**Option 2: Wait and retry**
- Automatic retry with exponential backoff (3 attempts)
- Manual wait: 10-30 seconds before retrying

**Option 3: Check yfinance status**
- Visit: https://yfinance.io
- Check: https://github.com/ranaroussi/yfinance/issues

**Option 4: Use different ticker**
```
# If SPY times out, try:
- QQQ, IWM, EFA (other major ETFs)
- Large cap stocks: AAPL, MSFT, GOOGL
```

---

### Error: "Model training failed for TICKER: singular matrix"

**Problem:** HMM model training hit mathematical issue.

**Root Cause:**
- Data has no variance (all same price)
- Number of states too close to data points
- Data quality issue

**Solutions:**

**Option 1: Reduce number of states**
```
Instead of:   n_states=5
Try:          n_states=3 (or 2)
```

**Option 2: Use different date range**
```
Instead of:   A period with constant prices
Try:          A period with price movement
```

**Option 3: Check data quality**
```python
import yfinance as yf
data = yf.download('TICKER')
print(f"Price range: {data['Close'].min()} - {data['Close'].max()}")
print(f"Unique prices: {data['Close'].nunique()}")
# If only 1-2 unique prices, data is problematic
```

---

### Error: "Request timeout after 60 seconds"

**Problem:** Operation took too long to complete.

**Root Cause:**
- Very large date range (10+ years)
- System resource constraints (low RAM)
- Server overloaded

**Solutions:**

**Option 1: Use shorter date range**
```
Instead of:   start_date=2000-01-01 (25 years)
Try:          start_date=2020-01-01 (5 years)
```

**Option 2: Restart MCP server**
```bash
# Close Claude Desktop
# Kill any running Python processes
pkill -f "hidden_regime_mcp"

# Wait 5 seconds
sleep 5

# Reopen Claude Desktop
```

**Option 3: Reduce n_states**
```
Instead of:   n_states=5
Try:          n_states=2 or 3
```

---

## Retry Logic

### Automatic Retry Behavior

Hidden Regime automatically retries **retriable errors** with exponential backoff:

**Configuration:**
- **Max attempts**: 3
- **Initial delay**: 1 second
- **Backoff multiplier**: 2x each attempt
- **Max delay**: 30 seconds

**Retry Timeline:**
```
Attempt 1: Fails immediately
           Wait 1 second + jitter
Attempt 2: Fails after 2 seconds
           Wait 2 seconds + jitter
Attempt 3: Fails after 4 seconds
           Give up
```

### Retriable vs Non-Retriable Errors

**Automatically Retried (Network/Resource Errors):**
- 4001: Network Timeout
- 4002: Connection Failed
- 4003: Data Source Unavailable
- 4005: Rate Limit Exceeded
- 5001: Out of Memory
- 5002: Timeout
- 5003: Resource Exhausted

**Not Retried (Validation/Model Errors):**
- 1xxx: Validation errors (bad input)
- 3xxx: Model errors (need different parameters)

### Manual Retry Strategy

For non-retriable errors, manual action needed:

```
User Input Error → Fix input and try again
Model Error → Change n_states or date range and try again
Data Error → Use different ticker or date range and try again
```

---

## Troubleshooting Guide

### Step 1: Identify Error Category

Check the error message for keywords:

| Keyword | Category | Action |
|---------|----------|--------|
| "Invalid", "format" | Validation (1xxx) | Fix input |
| "No data", "insufficient" | Data (2xxx) | Try different ticker/dates |
| "Model", "training" | Model (3xxx) | Change n_states |
| "Timeout", "connection" | Network (4xxx) | Retry or wait |
| "Memory", "resource" | Resource (5xxx) | Restart server |

### Step 2: Check Error Code

Match the error code (1001-5003) to the reference table above.

### Step 3: Apply Solution

Use the troubleshooting steps in [Common Errors and Solutions](#common-errors-and-solutions).

### Step 4: Verify Fix

Test with a simple query:
```
Ask Claude: "What's the current regime for SPY?"
```

If it works, try your original query again.

### Step 5: Get Help

If still failing:

1. **Collect information:**
   ```
   - Error code
   - Error message (full text)
   - Ticker you used
   - Date range you used
   - Your n_states value
   ```

2. **Check logs:**
   ```
   For Claude Desktop:
   - macOS: ~/Library/Logs/Claude
   - Windows: %APPDATA%\Claude\Logs
   - Linux: ~/.config/Claude/logs
   ```

3. **Report issue:**
   - [GitHub Issues](https://github.com/hidden-regime/issues)
   - Include: error code, ticker, dates, n_states
   - Attach: verification script output

---

## Error Recovery

### Circuit Breaker Pattern

Hidden Regime uses circuit breaker to prevent cascading failures:

**States:**
- **CLOSED**: Normal operation
- **OPEN**: Too many failures, requests blocked
- **HALF_OPEN**: Testing if service recovered

**Example:**
```
5 consecutive failures → Circuit OPENS
Wait 60 seconds → Circuit tries HALF_OPEN
If next request succeeds → Circuit CLOSES
If next request fails → Circuit stays OPEN
```

### Graceful Degradation

When cache is available:
```
Live query fails → Return cached result (if available)
                 → Provides stale but valid data
```

---

## Error Message Anatomy

Hidden Regime error messages have this structure:

```
{short_message}: {details}. {suggestion}
```

**Example:**
```
"Invalid number of regimes: 7. Number of regimes (states) must be
between 2 (minimal) and 5 (maximum). Use n_states between 2 and 5."
```

**Components:**
- **Short message**: What went wrong
- **Details**: Why it went wrong
- **Suggestion**: How to fix it

---

## For Developers

### Adding Retry to a Function

```python
from hidden_regime_mcp.retry import async_retry, RetryConfig

@async_retry(RetryConfig(max_attempts=3, initial_delay_seconds=2))
async def my_function():
    # This will automatically retry on network errors
    pass
```

### Custom Error Handling

```python
from hidden_regime_mcp.errors import (
    ValidationError,
    DataError,
    NetworkError,
)

try:
    # Your code
except ValidationError as e:
    error_info = e.to_error_info()
    # Handle validation error
except NetworkError as e:
    # Handle network error (retriable)
```

---

**Last Updated**: November 2025
**Version**: 1.1.0+
**Status**: Production Ready
