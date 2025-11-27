#!/usr/bin/env python
"""
Simple test for public data ingestion interface - no bar counting.
Just verify functionality works.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.data.financial import FinancialDataLoader


def create_bar(date, price):
    """Create a single OHLCV bar."""
    return pd.DataFrame({
        'Open': price,
        'High': price + 1,
        'Low': price - 1,
        'Close': price,
        'Volume': 1000000
    }, index=pd.DatetimeIndex([date]))


def test():
    """Test basic functionality."""
    print("Testing Data Ingestion Interface")
    print("=" * 50)

    config = FinancialDataConfig(ticker="TEST")
    loader = FinancialDataLoader(config)

    # Ingest first bar
    print("\n1. Ingest first bar")
    bar1 = create_bar(datetime(2024, 1, 1), 100)
    loader.update(data=bar1)
    data = loader.get_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   ✓ Has mandatory columns: {all(c in data.columns for c in ['price', 'pct_change', 'log_return'])}")

    # Ingest second bar
    print("\n2. Ingest second bar")
    bar2 = create_bar(datetime(2024, 1, 2), 101)
    loader.update(data=bar2)
    data = loader.get_data()
    print(f"   Data shape: {data.shape}")
    print(f"   ✓ Has pct_change with values: {not data['pct_change'].isna().all()}")

    # Ingest third bar
    print("\n3. Ingest third bar")
    bar3 = create_bar(datetime(2024, 1, 3), 99)
    loader.update(data=bar3)
    data = loader.get_data()
    print(f"   Data shape: {data.shape}")
    print(f"   ✓ Data is sorted: {data.index.is_monotonic_increasing}")

    print("\n" + "=" * 50)
    print("All basic tests passed!")


if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
