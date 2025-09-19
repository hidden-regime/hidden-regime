#!/usr/bin/env python3
"""Debug script to understand FinancialDataLoader behavior."""

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Create sample data exactly like the test
dates = pd.date_range('2023-01-01', periods=15, freq='D')
sample_data = pd.DataFrame({
    'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
    'Volume': [1000000] * 15,
    'Adj Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
}, index=dates)

print("Sample data shape:", sample_data.shape)
print("Sample data columns:", sample_data.columns.tolist())
print("Sample data head:")
print(sample_data.head())

# Try to simulate the _process_raw_data method
print("\n--- Processing simulation ---")

# Create empty result DataFrame
result = pd.DataFrame()

# Check if we have the date column
if isinstance(sample_data.index, pd.DatetimeIndex):
    data = sample_data.reset_index()
else:
    data = sample_data.copy()

print("After reset_index, data columns:", data.columns.tolist())
print("Data shape:", data.shape)

# Set index for result
if "Date" in data.columns:
    result.index = pd.to_datetime(data["Date"])
    print("Set index from Date column")
elif "index" in data.columns:
    result.index = pd.to_datetime(data["index"])
    print("Set index from index column")
else:
    result.index = sample_data.index
    print("Set index from original index")

print("Result index:", result.index[:3])

# Add OHLCV columns
if "Open" in data.columns:
    result["open"] = data["Open"]
    print("Added open column")
if "High" in data.columns:
    result["high"] = data["High"]
    print("Added high column")
if "Low" in data.columns:
    result["low"] = data["Low"]
    print("Added low column")
if "Close" in data.columns:
    result["close"] = data["Close"]
    print("Added close column")
if "Volume" in data.columns:
    result["volume"] = data["Volume"]
    print("Added volume column")

print("Result columns after OHLCV:", result.columns.tolist())
print("Result shape after OHLCV:", result.shape)

# Calculate price
if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
    result["price"] = data["Close"]
    print("Added price column from Close")
else:
    result["price"] = data["Close"]
    print("Added price column (fallback)")

print("Result columns after price:", result.columns.tolist())
print("Result shape after price:", result.shape)

# Check for NaN values
print("Price column has NaN:", result["price"].isna().any())
if result["price"].isna().any():
    print("Number of NaN values:", result["price"].isna().sum())

# Apply dropna
print("\n--- Before dropna ---")
print("Shape:", result.shape)
result_after_dropna = result.dropna(subset=["price"]).reset_index(drop=False)
print("\n--- After dropna ---")
print("Shape:", result_after_dropna.shape)
print("Columns:", result_after_dropna.columns.tolist())

if len(result_after_dropna) == 0:
    print("ERROR: Result is empty after dropna!")
else:
    print("SUCCESS: Result has data")
    print(result_after_dropna.head())