"""
Minimal 10-Line Example - Hidden Regime Detection

The absolute simplest way to use hidden-regime: detect market regimes in 10 lines.
Perfect for getting started or quick prototyping.
"""

import hidden_regime as hr

# Create and run a financial pipeline in one line
pipeline = hr.create_financial_pipeline(ticker="SPY", start_date="2020-01-01", end_date="2024-01-01", n_states=3)

# Get results with regime detection
result = pipeline.update()

# Print summary
print(f"Detected {result['regime'].nunique()} regimes across {len(result)} days")
print(f"Current regime: {result['regime'].iloc[-1]}")
print(f"Regime distribution:\n{result['regime'].value_counts()}")
