"""
Test fixtures for QuantConnect integration tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Simulate realistic price movements with trends
    returns = np.random.normal(0.0005, 0.02, len(dates))
    returns[:500] = np.random.normal(0.001, 0.015, 500)  # Bull regime
    returns[500:750] = np.random.normal(-0.001, 0.025, 250)  # Bear regime
    returns[750:] = np.random.normal(0.0, 0.01, len(dates) - 750)  # Sideways

    prices = 100 * np.exp(np.cumsum(returns))

    # Add OHLCV data
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, len(dates))
    }, index=dates)

    return df


@pytest.fixture
def sample_multi_asset_data():
    """Generate multi-asset price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    assets = {}
    for i, ticker in enumerate(['SPY', 'QQQ', 'TLT', 'GLD']):
        returns = np.random.normal(0.0005 + i * 0.0001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        assets[ticker] = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'Close': prices,
            'Volume': np.random.uniform(1e6, 5e6, len(dates))
        }, index=dates)

    return assets


@pytest.fixture
def mock_tradebar_data():
    """Create mock TradeBar-like objects for testing."""
    class MockTradeBar:
        def __init__(self, time, open_price, high, low, close, volume):
            self.Time = time
            self.Open = open_price
            self.High = high
            self.Low = low
            self.Close = close
            self.Volume = volume

    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)

    bars = []
    price = 100.0
    for date in dates:
        price = price * (1 + np.random.normal(0.001, 0.02))
        bars.append(MockTradeBar(
            time=date,
            open_price=price * (1 + np.random.uniform(-0.01, 0.01)),
            high=price * (1 + np.random.uniform(0, 0.02)),
            low=price * (1 - np.random.uniform(0, 0.02)),
            close=price,
            volume=np.random.uniform(1e6, 5e6)
        ))

    return bars


@pytest.fixture
def sample_regime_states():
    """Sample regime states for testing."""
    return {
        'Bull': {'volatility': 0.15, 'trend': 0.05, 'label': 'Bull'},
        'Bear': {'volatility': 0.25, 'trend': -0.03, 'label': 'Bear'},
        'Sideways': {'volatility': 0.10, 'trend': 0.00, 'label': 'Sideways'},
        'Crisis': {'volatility': 0.40, 'trend': -0.10, 'label': 'Crisis'}
    }


@pytest.fixture
def sample_regime_allocations():
    """Sample regime allocations for testing."""
    return {
        'Bull': 1.0,
        'Bear': 0.0,
        'Sideways': 0.5,
        'Crisis': -0.3
    }


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    class MockPipeline:
        def __init__(self):
            self.current_regime = 'Bull'
            self.regime_probabilities = np.array([0.8, 0.1, 0.1])
            self.regime_labels = ['Bull', 'Bear', 'Sideways']

        def run(self, data=None):
            """Mock run method."""
            return {
                'regime': self.current_regime,
                'probabilities': self.regime_probabilities,
                'confidence': 0.8
            }

        def analyze(self):
            """Mock analyze method."""
            return {
                'current_regime': self.current_regime,
                'regime_probabilities': self.regime_probabilities
            }

    return MockPipeline()


@pytest.fixture
def qc_config():
    """Sample QuantConnect configuration for testing."""
    from hidden_regime.quantconnect.config import QuantConnectConfig

    return QuantConnectConfig(
        lookback_days=252,
        retrain_frequency='weekly',
        warmup_days=252,
        use_cache=True,
        cache_size=100,
        batch_updates=True,
        max_workers=4,
        enable_profiling=False
    )


@pytest.fixture
def performance_test_data():
    """Generate larger dataset for performance testing."""
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, len(dates))
    }, index=dates)

    return df


@pytest.fixture
def mock_qc_algorithm():
    """Create a mock QCAlgorithm for testing."""
    class MockQCAlgorithm:
        def __init__(self):
            self.portfolio = {}
            self.securities = {}
            self.holdings = {}
            self.transactions = []
            self.logs = []
            self.time = datetime(2020, 1, 1)

        def SetHoldings(self, symbol, allocation):
            """Mock SetHoldings."""
            self.holdings[symbol] = allocation
            self.transactions.append({
                'time': self.time,
                'symbol': symbol,
                'allocation': allocation
            })

        def Liquidate(self, symbol=None):
            """Mock Liquidate."""
            if symbol:
                self.holdings[symbol] = 0
                self.transactions.append({
                    'time': self.time,
                    'symbol': symbol,
                    'allocation': 0
                })
            else:
                self.holdings = {}

        def Log(self, message):
            """Mock Log."""
            self.logs.append({'time': self.time, 'message': message})

        def Debug(self, message):
            """Mock Debug."""
            self.logs.append({'time': self.time, 'message': f"DEBUG: {message}"})

    return MockQCAlgorithm()
