# BasicRegimeSwitching Algorithm Parameters

This document describes all available parameters for the `BasicRegimeSwitching` algorithm. Parameters can be modified without editing code or rebuilding the container.

## How to Use Parameters

### QuantConnect Cloud/IDE (Recommended for Multiple Tickers)
1. Create an account at https://www.quantconnect.com
2. Create a new project and upload `basic_regime_switching.py`
3. Open the algorithm settings
4. Add parameters with the names and values listed below
5. Click "Backtest" to run with your parameters
6. View results, charts, and statistics instantly

**Advantages:**
- Access to 2M+ securities (stocks, crypto, forex, futures, options)
- Fast backtesting on cloud infrastructure
- Beautiful web-based analysis and charting
- Live trading support
- No data limitations

### Command Line (Local LEAN - Limited Data)
```bash
./lean backtest "quantconnect_templates/basic_regime_switching.py" \
  --parameter ticker=SPY \
  --parameter start_year=2019 \
  --parameter bull_allocation=0.8
```

**Note:** Local LEAN Docker only includes SPY data. Use `ticker=SPY` for local testing.

### Docker (Local - SPY Only)
```bash
docker run -v $(pwd):/app lean-hidden-regime:latest \
  --parameter ticker=SPY \
  --parameter n_states=4 \
  --parameter lookback_days=126
```

## Available Parameters

### Asset Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | string | `SPY` | Stock ticker symbol (e.g., QQQ, TLT, GLD, AAPL) |

### Date Range

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_year` | int | `2020` | Backtest start year (4-digit, e.g., 2020) |
| `start_month` | int | `1` | Backtest start month (1-12, e.g., 1 for January) |
| `start_day` | int | `1` | Backtest start day (1-31) |
| `end_year` | int | `2022` | Backtest end year (4-digit, e.g., 2022) |
| `end_month` | int | `1` | Backtest end month (1-12) |
| `end_day` | int | `1` | Backtest end day (1-31) |

### Portfolio Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cash` | float | `100000` | Initial portfolio cash in USD |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | `3` | Number of HMM states (2-5 recommended) |
| `lookback_days` | int | `252` | Historical window for HMM training (trading days) |
| `min_confidence` | float | `0.6` | Minimum confidence threshold for regime signals (0.0-1.0) |

### Regime Allocations

These parameters control position sizing for each regime:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bull_allocation` | float | `1.0` | Allocation in bull regime (1.0 = 100% long, 0.0 = cash) |
| `bear_allocation` | float | `0.0` | Allocation in bear regime (0.0 = cash, -1.0 = fully short) |
| `sideways_allocation` | float | `0.5` | Allocation in sideways regime (0.5 = 50% long) |

## Example Parameter Sets

### Conservative (Lower Volatility)
```
ticker=TLT
n_states=3
bull_allocation=0.8
bear_allocation=0.0
sideways_allocation=0.3
lookback_days=252
```

### Aggressive (Maximum Returns)
```
ticker=QQQ
n_states=3
bull_allocation=1.5
bear_allocation=-0.5
sideways_allocation=0.5
lookback_days=126
```

### Tech-Focused
```
ticker=AAPL
n_states=4
bull_allocation=1.0
bear_allocation=0.0
sideways_allocation=0.5
lookback_days=252
```

### Multi-Asset (Run separately)
- SPY (broad market)
- QQQ (technology)
- IWM (small cap)
- TLT (bonds)
- GLD (gold)

## Tips for Parameter Tuning

1. **Date Range**: Use 2+ years of data for stable HMM training
2. **Lookback Days**: 252 = 1 year of trading days (standard), 126 = 6 months
3. **Allocations**: Values between -1.0 and 1.5 are typical
   - Negative values = short positions
   - Values > 1.0 = leveraged long
4. **n_states**: 3 is a good default (Bull/Sideways/Bear)
5. **min_confidence**: Increase from 0.6 to 0.8+ for more selective trading

## Regime Definitions

- **Bull**: Strong uptrend, positive returns expected
- **Sideways**: Range-bound, neutral outlook
- **Bear**: Downtrend, negative returns expected

The allocation for each regime is applied automatically when the regime is detected.

## No Code Changes Required

All parameters are read at runtime via `GetParameter()`. You can change:
- Different tickers
- Different date ranges
- Different position sizes
- Different model configurations

**Without rebuilding the Docker container or editing any code.**

---

## Data Availability

### Local Docker (Included in lean-hidden-regime image)
- **SPY** ✅ (free tier, all dates)
- Most other tickers ❌ (requires QuantConnect subscription or cloud)

The Docker image inherits from QuantConnect's official `quantconnect/lean:17389` which includes SPY data with the free tier. Additional tickers require either:
1. QuantConnect Cloud (recommended)
2. Manually downloading data files to `/Lean/Data/` in the Docker container

**Note:** The backtest script doesn't mount a data volume, so local data additions require rebuilding the Docker image.

### QuantConnect Cloud (Recommended for Multiple Tickers)
- **2M+ securities** including:
  - 10,000+ US stocks
  - 50+ cryptocurrencies
  - Forex pairs
  - Futures contracts
  - Options data
- All going back 10+ years for most assets
- Real-time data for live trading
- **Free tier:** Access to major indices (SPY, IWM, QQQ, TLT, etc.)
- **Premium tiers:** Unlocks additional data

### Workflow Recommendation
1. **Develop locally:** Use `ticker=SPY` to test your strategy logic
2. **Cloud testing:** Push to https://www.quantconnect.com to test with other tickers
   - Free account includes major US indices
   - No data download/setup needed on cloud
3. **Parameter optimization:** Run sweeps on cloud (supports parameter iterations)
4. **Deployment:** Go live with funded accounts or backtested strategy

---

See `basic_regime_switching.py` lines 59-76 for the complete parameter list.
