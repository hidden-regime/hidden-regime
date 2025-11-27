# Deploying to QuantConnect Cloud

This guide walks you through uploading and running the `BasicRegimeSwitching` strategy on QuantConnect's cloud platform.

## Prerequisites

- QuantConnect account (free at https://www.quantconnect.com)
- `basic_regime_switching.py` strategy file
- Internet connection

## Step-by-Step Deployment

### 1. Create QuantConnect Account (if needed)

1. Go to https://www.quantconnect.com
2. Click "Sign Up" (or sign in if you have an account)
3. Complete registration (free account is sufficient)
4. Verify email address

### 2. Create a New Project

1. Log in to QuantConnect
2. Click **"Create Algorithm"** or go to Dashboard → New Project
3. Choose **Python** as the language
4. Name your project (e.g., "Hidden Regime Switching")
5. Click **Create**

You'll see the QuantConnect IDE with a blank strategy template.

### 3. Upload the Strategy

**Option A: Copy-Paste (Easiest)**
1. Open `basic_regime_switching.py` locally
2. Copy all the code
3. In QuantConnect IDE, select all text (Ctrl+A)
4. Paste your strategy code (Ctrl+V)
5. Click **Save**

**Option B: Upload File**
1. In QuantConnect IDE, click the **⋮** menu (top right)
2. Look for "Upload" or "Import" option
3. Select `basic_regime_switching.py` from your computer
4. Click **Save**

### 4. Configure Backtest Parameters

Now you can test with different parameters without editing code:

1. Click **"Backtest"** button
2. A settings panel will appear on the left
3. Under **"Parameters"**, add your settings:

```
Parameter Name          Value           Description
─────────────────────────────────────────────────
ticker                  QQQ             Stock ticker
start_year              2020            Start year
start_month             1               Start month
start_day               1               Start day
end_year                2022            End year
end_month               1               End month
end_day                 1               End day
cash                    100000          Initial cash
n_states                3               HMM states
lookback_days           252             Training window
min_confidence          0.6             Confidence threshold
bull_allocation         1.0             Bull position %
bear_allocation         0.0             Bear position %
sideways_allocation     0.5             Sideways position %
```

### 5. Run the Backtest

1. Review your parameter settings
2. Click **"Backtest"** (or **"Live Trade"** if using live account)
3. Wait for results (usually 10-30 seconds depending on date range)

### 6. View Results

After the backtest completes, you'll see:

- **Summary Statistics:**
  - Total Return
  - Annual Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - And 20+ other metrics

- **Charts:**
  - Equity curve (portfolio value over time)
  - Drawdown chart
  - Returns distribution
  - Rolling Sharpe ratio

- **Logs:**
  - Algorithm logs (regime changes, trades)
  - Debug messages
  - Errors (if any)

### 7. Test Different Tickers

To test another ticker (e.g., TLT for bonds):

1. Click **"Backtest"** again
2. Change `ticker` parameter to **TLT**
3. Click **"Backtest"**

**Available Free Tier Tickers:**
- SPY (S&P 500)
- QQQ (Nasdaq 100)
- IWM (Russell 2000)
- TLT (Long-term Treasury Bond ETF)
- GLD (Gold ETF)
- And many others

### 8. Optimize Parameters (Optional)

To run multiple backtests with different parameter combinations:

1. Click **"Optimize"** instead of "Backtest"
2. Define parameter ranges:
   ```
   Parameter          Min    Max    Step
   ─────────────────────────────────────
   n_states           2      4      1
   lookback_days      126    252    126
   bull_allocation    0.8    1.0    0.1
   ```
3. Click **"Optimize"**
4. View results sorted by Sharpe ratio

### 9. Save & Share

**Save Your Work:**
- Click **"Save"** (Ctrl+S)
- Changes auto-save every 30 seconds

**Share with Others:**
1. Click the **Share** button
2. Generate a shareable link
3. Others can view (read-only) or fork your strategy

## Common Parameter Combinations

### Conservative (Lower Risk)
```
n_states: 3
lookback_days: 252
bull_allocation: 0.8
bear_allocation: 0.0
sideways_allocation: 0.3
min_confidence: 0.7
```

### Aggressive (Higher Returns)
```
n_states: 3
lookback_days: 126
bull_allocation: 1.0
bear_allocation: -0.5
sideways_allocation: 0.5
min_confidence: 0.6
```

### Tech-Focused (QQQ)
```
ticker: QQQ
n_states: 4
lookback_days: 252
bull_allocation: 1.0
bear_allocation: 0.0
sideways_allocation: 0.5
```

### Bond-Focused (TLT)
```
ticker: TLT
n_states: 3
lookback_days: 252
bull_allocation: 0.9
bear_allocation: 0.0
sideways_allocation: 0.4
```

## Troubleshooting

### "Strategy compilation failed"
- Check for Python syntax errors
- Ensure all imports are correct
- Click "Logs" to see the error message

### "No data for ticker"
- Verify ticker symbol is correct (e.g., "SPY" not "spy")
- Check date range is valid
- Use one of the free tier tickers listed above

### "Algorithm took too long to run"
- Reduce date range
- Decrease lookback_days (train on less historical data)
- Increase start_year to begin more recently

### Results look incorrect
- Check that regime changes are happening (view logs)
- Verify parameters are being read correctly
- Try with SPY first to verify it works

## Next Steps

### After Successful Backtest
1. **Compare strategies:** Run backtests with different parameters
2. **Analyze regimes:** Look at logs to see when regimes change
3. **Forward test:** Run on more recent data
4. **Live trade:** (Advanced) Upgrade account and trade live if desired

### Learning Resources
- QuantConnect Documentation: https://www.quantconnect.com/docs
- API Reference: https://www.quantconnect.com/docs/v2/python-api
- Tutorial Videos: https://www.quantconnect.com/learning

## Tips & Best Practices

1. **Start with SPY:** Test your strategy on the broad market first
2. **Document your changes:** Add comments to track what you tried
3. **Compare results:** Keep track of Sharpe ratios for different parameters
4. **Use reasonable dates:** Avoid overfitting to tiny date ranges
5. **Check the logs:** Always read algorithm logs to verify regime detection is working
6. **Test multiple assets:** Good strategies work across different asset classes

## What Happens Next

Once you deploy to QuantConnect Cloud:

- Your strategy has access to 2M+ securities
- No data limitations (free tier includes major indices)
- Results are instant (cloud infrastructure)
- You can optimize parameters with parameter sweeps
- Beautiful charting and analysis tools
- Option to go live with funded accounts

---

Questions? Check the [QuantConnect community](https://www.quantconnect.com/forum) or this repo's documentation.
