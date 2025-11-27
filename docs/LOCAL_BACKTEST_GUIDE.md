# Local Backtesting with QuantConnect API

This guide explains how to run regime-based backtests locally using your QuantConnect PRO account, without needing Docker or the LEAN engine.

## Architecture

```
Your Machine
├── QuantConnect API Client
│   ├── Pulls historical data from QC via REST API
│   ├── Caches data locally (~/.cache/hidden-regime/qc-data/)
│   └── Handles authentication
├── Regime Detection (HMM)
│   ├── Trains Hidden Markov Models locally
│   ├── No cloud computation needed
│   └── Full control over parameters
└── Backtester
    ├── Simulates portfolio rebalancing
    ├── Tracks performance metrics
    └── Exports results (CSV, JSON)
```

## Prerequisites

1. **QuantConnect Account**
   - Sign up at https://www.quantconnect.com (free tier available)
   - Upgrade to PRO account for historical data access
   - Get API credentials from https://www.quantconnect.com/account

2. **Python Environment**
   ```bash
   source /home/aoaustin/hidden-regime-pyenv/bin/activate
   pip install -e .[dev]
   ```

## Setup: Store Your QC Credentials

### Option 1: Environment Variables (Recommended for CI/CD)

```bash
export QC_API_KEY="your-api-key-here"
export QC_API_SECRET="your-api-secret-here"
python examples/local_backtest_with_qc_api.py
```

### Option 2: Config File (Recommended for Development)

Create `~/.qc-credentials.json` with restricted permissions:

```bash
cat > ~/.qc-credentials.json << 'EOF'
{
  "api_key": "your-api-key-here",
  "api_secret": "your-api-secret-here"
}
EOF

chmod 600 ~/.qc-credentials.json
```

Then use directly:

```python
from hidden_regime.local import LocalBacktester

backtester = LocalBacktester(ticker="SPY")
```

### Option 3: Programmatic (For Custom Scripts)

```python
from hidden_regime.local import LocalBacktester

backtester = LocalBacktester(
    ticker="SPY",
    api_key="your-key",
    api_secret="your-secret"
)
```

## Quick Start: Run Your First Local Backtest

```bash
source /home/aoaustin/hidden-regime-pyenv/bin/activate
python examples/local_backtest_with_qc_api.py
```

This will:
1. ✓ Verify your QuantConnect credentials
2. ✓ Pull SPY historical data (Jan 2020 - Dec 2023)
3. ✓ Run regime detection with 3-state HMM
4. ✓ Simulate regime-based portfolio rebalancing
5. ✓ Export results to `./backtest_results/SPY_YYYYMMDD_HHMMSS/`

## Example 1: Basic Backtest

```python
from hidden_regime.local import LocalBacktester

# Create backtester
backtester = LocalBacktester(ticker="SPY", initial_cash=100000)

# Run backtest
results = backtester.run(
    start_date="2020-01-01",
    end_date="2023-12-31",
    n_states=3,
    lookback_days=252,
    allocations={
        "Bull": 1.0,      # 100% long
        "Bear": 0.0,      # Cash
        "Sideways": 0.5,  # 50% long
    }
)

# Print summary
print(results["metrics"])

# Export results
backtester.export_results()
```

## Example 2: Parameter Optimization

Test different regime allocations:

```python
from hidden_regime.local import LocalBacktester
import pandas as pd

results_summary = []

allocations_to_test = [
    {"Bull": 1.0, "Bear": 0.0, "Sideways": 0.5},    # Conservative
    {"Bull": 1.2, "Bear": -0.2, "Sideways": 0.5},   # Moderate
    {"Bull": 1.5, "Bear": -0.5, "Sideways": 0.5},   # Aggressive
]

for i, allocations in enumerate(allocations_to_test):
    print(f"\nTesting allocation set {i+1}: {allocations}")

    backtester = LocalBacktester(ticker="SPY")
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
        allocations=allocations
    )

    results_summary.append({
        "allocation": str(allocations),
        "total_return": results["metrics"]["total_return"],
        "annual_return": results["metrics"]["annual_return"],
        "sharpe_ratio": results["metrics"]["sharpe_ratio"],
        "max_drawdown": results["metrics"]["max_drawdown"],
    })

# Compare results
summary_df = pd.DataFrame(results_summary)
print("\n" + "="*70)
print(summary_df.to_string())
print("="*70)
```

## Example 3: Multi-Ticker Analysis

```python
from hidden_regime.local import LocalBacktester
import pandas as pd

tickers = ["SPY", "QQQ", "IWM"]
results_by_ticker = {}

for ticker in tickers:
    print(f"\nBacktesting {ticker}...")

    backtester = LocalBacktester(ticker=ticker)
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
    )

    results_by_ticker[ticker] = results["metrics"]
    print(f"{ticker}: Sharpe={results['metrics']['sharpe_ratio']:.2f}, "
          f"Return={results['metrics']['total_return']:.1%}")

# Compare across tickers
comparison = pd.DataFrame(results_by_ticker).T
print("\n" + comparison.to_string())
```

## Data Caching

Data is cached locally to avoid repeated API calls:

```bash
# Cache location
~/.cache/hidden-regime/qc-data/

# Check cache stats
python -c "
from hidden_regime.local import LocalBacktester
bt = LocalBacktester()
stats = bt.api_client.get_cache_stats()
print(f\"Cached files: {stats['total_files']}\")
print(f\"Cache size: {stats['total_size_mb']:.1f} MB\")
"

# Clear cache if needed
python -c "
from hidden_regime.local import LocalBacktester
bt = LocalBacktester()
bt.api_client.clear_cache()
print('Cache cleared')
"
```

## Understanding Results

### Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| `total_return` | Final return over entire period | >0% |
| `annual_return` | Annualized return | >10% |
| `sharpe_ratio` | Risk-adjusted return | >1.0 |
| `max_drawdown` | Largest peak-to-trough decline | <-20% |
| `win_rate` | % of profitable trades | >60% |

### Output Files

After running a backtest, you get:

```
backtest_results/SPY_20250127_143025/
├── results.csv          # Daily portfolio values and regimes
├── trades.csv           # Executed trades with dates and prices
├── metrics.json         # Performance statistics
└── summary.txt          # Human-readable summary
```

### Exported CSV Columns

**results.csv:**
- `date`: Trading date
- `close`: SPY closing price
- `portfolio_value`: Daily portfolio value
- `regime`: Detected regime (Bull/Bear/Sideways)
- `confidence`: Confidence in regime detection (0.0-1.0)

**trades.csv:**
- `date`: Trade execution date
- `regime`: Regime at time of trade
- `shares_traded`: Positive (buy) or negative (sell)
- `price`: Execution price
- `value`: Trade value in dollars

## Workflow: Local Development → Cloud Deployment

### Phase 1: Develop Locally (SPY)

```bash
# Test parameters locally with SPY
python examples/local_backtest_with_qc_api.py

# Optimize parameters
python scripts/optimize_parameters.py

# Analyze results
jupyter notebook examples/analyze_local_backtest.ipynb
```

### Phase 2: Deploy to QC Cloud

1. Upload your strategy to https://www.quantconnect.com/ide
2. Run with different tickers (QQQ, IWM, TLT, GLD, etc.)
3. Use QuantConnect's built-in backtester for faster processing
4. A/B test strategies in cloud vs. local

### Phase 3: Paper/Live Trading

After validation:
1. Use the strategy on QC Cloud's paper trading
2. Monitor live performance
3. Iterate based on real-world results

## Troubleshooting

### Error: "Credentials not found"

```
ValueError: QuantConnect credentials not found
```

**Solution:** Set up credentials using one of the three methods above:
- Environment variables: `QC_API_KEY`, `QC_API_SECRET`
- Config file: `~/.qc-credentials.json`
- Programmatic: Pass to `LocalBacktester(api_key=..., api_secret=...)`

### Error: "No data returned for ticker"

```
ValueError: No data returned for SPY from 2020-01-01 to 2023-12-31
```

**Possible causes:**
1. Ticker not available on your QC plan
2. Date range is invalid (weekends, holidays, future dates)
3. API is temporarily unavailable

**Solution:**
- Check ticker is valid on QC (most major indices work on free tier)
- Verify date range (avoid future dates)
- Try with SPY first (most reliable)

### Error: "Access denied - PRO subscription required"

```
ValueError: Access denied. You may need a PRO subscription for this ticker.
```

**Solution:**
- Free tier: Limited to major indices (SPY, QQQ, IWM, TLT, GLD)
- For individual stocks: Upgrade to PRO account
- Or use the Docker local LEAN engine for SPY-only backtests

### Slow API Responses

**Solution:**
- Responses are cached in `~/.cache/hidden-regime/qc-data/`
- First run takes longer, subsequent runs use cache
- Clear cache only if you need fresh data: `backtester.api_client.clear_cache()`

## Performance Tips

### Speed Up Backtests

1. **Use cached data:**
   ```python
   backtester = LocalBacktester(cache_enabled=True)  # Default
   ```

2. **Shorter date ranges for testing:**
   ```python
   results = backtester.run(
       start_date="2023-01-01",  # 1 year instead of 10 years
       end_date="2023-12-31"
   )
   ```

3. **Reduce lookback window for faster HMM training:**
   ```python
   results = backtester.run(
       lookback_days=126,  # 6 months instead of 1 year
   )
   ```

## API Rate Limits

QuantConnect API has rate limits:
- Free tier: ~10-20 requests/minute
- PRO tier: Higher limits

**Impact:** First backtest takes time, but caching prevents repeated calls.

## Advanced: Custom Regime Allocations

The allocations dictionary maps regime names to position sizes:

```python
# Standard 3-state HMM
allocations = {
    "Bull": 1.0,      # 100% long (full position)
    "Bear": 0.0,      # Cash (no position)
    "Sideways": 0.5,  # 50% long (half position)
}

# Aggressive (with shorts)
allocations = {
    "Bull": 1.5,      # Leveraged long (150%)
    "Bear": -0.5,     # Short (50%)
    "Sideways": 0.5,
}

# Very conservative
allocations = {
    "Bull": 0.5,      # 50% long
    "Bear": 0.0,      # Cash
    "Sideways": 0.25, # 25% long
}
```

Regime names depend on your HMM configuration. Run a backtest and check `results["results_df"]["regime"]` to see what regimes were detected.

## Next Steps

1. ✓ Install hidden-regime: `pip install -e .[dev]`
2. ✓ Set up QC credentials
3. ✓ Run first backtest: `python examples/local_backtest_with_qc_api.py`
4. → Modify parameters and test locally
5. → Deploy optimized strategy to QC Cloud
6. → Monitor live performance

## More Information

- **Main Documentation:** `/working/README.md`
- **API Reference:** `/working/docs/reference/`
- **QuantConnect Docs:** https://www.quantconnect.com/docs
- **Discord:** https://discord.gg/quantconnect
