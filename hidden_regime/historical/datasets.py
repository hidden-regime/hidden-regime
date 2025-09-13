"""
Historical Market Datasets

Curated datasets for major market events with known regime characteristics.
Enables validation of HMM regime detection against well-documented market periods.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..data import DataLoader, DataConfig


# Major historical market events with expected regime characteristics
MAJOR_MARKET_EVENTS = {
    "2008_financial_crisis": {
        "name": "2008 Financial Crisis",
        "start_date": "2008-09-01",
        "end_date": "2009-03-31",
        "expected_regime": "bear",
        "description": "Lehman Brothers collapse and subsequent financial crisis",
        "key_tickers": ["SPY", "XLF", "VTI", "QQQ"],
        "characteristics": {
            "mean_return": -0.025,  # ~-2.5% daily average
            "volatility": 0.035,    # ~3.5% daily volatility
            "duration_days": 212,
            "total_decline": -0.57  # ~57% peak-to-trough decline
        }
    },
    
    "covid_crash_2020": {
        "name": "COVID-19 Market Crash and Recovery", 
        "start_date": "2020-02-15",
        "end_date": "2020-06-15",
        "expected_regime": "crisis_to_bull",
        "description": "Rapid crash in March 2020 followed by aggressive recovery",
        "key_tickers": ["SPY", "QQQ", "VTI", "TSLA", "ZOOM"],
        "characteristics": {
            "crash_phase": {
                "start": "2020-02-15",
                "end": "2020-03-23",
                "mean_return": -0.045,  # Extreme negative returns
                "volatility": 0.055     # Extreme volatility
            },
            "recovery_phase": {
                "start": "2020-03-24", 
                "end": "2020-06-15",
                "mean_return": 0.025,   # Strong positive returns
                "volatility": 0.030     # High but decreasing volatility
            }
        }
    },
    
    "dotcom_bubble_burst": {
        "name": "Dot-com Bubble Burst",
        "start_date": "2000-03-01", 
        "end_date": "2002-10-31",
        "expected_regime": "bear",
        "description": "Technology sector bubble burst and prolonged bear market",
        "key_tickers": ["QQQ", "SPY", "MSFT", "AAPL", "AMZN"],
        "characteristics": {
            "mean_return": -0.008,  # Persistent negative returns
            "volatility": 0.025,    # Moderate volatility
            "duration_days": 975,   # Very long bear market
            "total_decline": -0.78  # ~78% decline in NASDAQ
        }
    },
    
    "great_bull_run_2016_2018": {
        "name": "Great Bull Run 2016-2018",
        "start_date": "2016-11-01",
        "end_date": "2018-01-31", 
        "expected_regime": "bull",
        "description": "Post-election bull market with low volatility",
        "key_tickers": ["SPY", "QQQ", "VTI", "DIA"],
        "characteristics": {
            "mean_return": 0.008,   # Consistent positive returns
            "volatility": 0.012,    # Very low volatility  
            "duration_days": 457,
            "total_gain": 0.35      # ~35% total gain
        }
    },

    "volmageddon_2018": {
        "name": "Volmageddon 2018",
        "start_date": "2018-01-29",
        "end_date": "2018-02-26",
        "expected_regime": "crisis",
        "description": "VIX spike and volatility crisis in February 2018",
        "key_tickers": ["SPY", "QQQ", "VXX", "UVXY"],
        "characteristics": {
            "mean_return": -0.015,  # Sharp negative returns
            "volatility": 0.025,    # Sudden volatility spike
            "duration_days": 28,    # Short but intense
            "key_feature": "volatility_explosion"
        }
    }
}


def load_historical_period(
    event_name: str,
    tickers: Optional[List[str]] = None,
    data_config: Optional[DataConfig] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load historical data for a specific market event.
    
    Args:
        event_name: Name of the event from MAJOR_MARKET_EVENTS
        tickers: List of tickers to load (uses event defaults if None)
        data_config: DataLoader configuration (uses defaults if None)
        
    Returns:
        Tuple of (data_dict, event_metadata)
        
    Raises:
        ValueError: If event_name not found in MAJOR_MARKET_EVENTS
        
    Example:
        >>> data, metadata = load_historical_period('2008_financial_crisis')
        >>> spy_data = data['SPY']
        >>> print(f"Loaded {len(spy_data)} days of {metadata['name']}")
    """
    if event_name not in MAJOR_MARKET_EVENTS:
        available = list(MAJOR_MARKET_EVENTS.keys())
        raise ValueError(f"Unknown event '{event_name}'. Available: {available}")
    
    event_info = MAJOR_MARKET_EVENTS[event_name]
    
    # Use event's key tickers if none specified
    if tickers is None:
        tickers = event_info["key_tickers"]
    
    # Configure data loader for reliable historical data
    if data_config is None:
        data_config = DataConfig(
            use_ohlc_average=True,
            include_volume=True,
            cache_enabled=True,
            max_missing_data_pct=0.05,  # Strict for historical validation
        )
    
    loader = DataLoader(data_config)
    
    print(f"Loading historical data for: {event_info['name']}")
    print(f"Period: {event_info['start_date']} to {event_info['end_date']}")
    print(f"Tickers: {tickers}")
    
    try:
        data_dict = loader.load_multiple_stocks(
            tickers, 
            event_info["start_date"],
            event_info["end_date"]
        )
        
        print(f"✓ Successfully loaded {len(data_dict)} out of {len(tickers)} tickers")
        
        # Add event metadata to each dataset
        for ticker, data in data_dict.items():
            data.attrs['event_name'] = event_name
            data.attrs['event_info'] = event_info
            data.attrs['expected_regime'] = event_info['expected_regime']
            
        return data_dict, event_info
        
    except Exception as e:
        print(f"✗ Failed to load historical data: {e}")
        raise


def load_crisis_2008(tickers: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load 2008 Financial Crisis data.
    
    Convenience function for the 2008 crisis period from Lehman collapse
    through market bottom in March 2009.
    
    Args:
        tickers: List of tickers (defaults to ['SPY', 'XLF', 'VTI', 'QQQ'])
        
    Returns:
        Tuple of (data_dict, event_metadata)
    """
    return load_historical_period("2008_financial_crisis", tickers)


def load_covid_crash_2020(tickers: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load COVID-19 crash and recovery data.
    
    Covers the rapid March 2020 crash and subsequent recovery through June.
    Demonstrates both crisis and bull regime transitions.
    
    Args:
        tickers: List of tickers (defaults to ['SPY', 'QQQ', 'VTI', 'TSLA', 'ZOOM'])
        
    Returns:
        Tuple of (data_dict, event_metadata)  
    """
    return load_historical_period("covid_crash_2020", tickers)


def load_dotcom_bubble(tickers: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load dot-com bubble burst data.
    
    Covers the prolonged bear market from March 2000 through October 2002.
    Demonstrates sustained bear regime characteristics.
    
    Args:
        tickers: List of tickers (defaults to ['QQQ', 'SPY', 'MSFT', 'AAPL', 'AMZN'])
        
    Returns:
        Tuple of (data_dict, event_metadata)
    """
    return load_historical_period("dotcom_bubble_burst", tickers)


def load_bull_run_2016_2018(tickers: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load 2016-2018 bull market data.
    
    Post-election bull run with consistently low volatility and positive returns.
    Perfect example of sustained bull regime.
    
    Args:
        tickers: List of tickers (defaults to ['SPY', 'QQQ', 'VTI', 'DIA'])
        
    Returns:
        Tuple of (data_dict, event_metadata)
    """
    return load_historical_period("great_bull_run_2016_2018", tickers)


def load_volmageddon_2018(tickers: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load Volmageddon 2018 data.
    
    Short but intense volatility crisis in February 2018.
    Demonstrates sudden regime transitions and crisis detection.
    
    Args:
        tickers: List of tickers (defaults to ['SPY', 'QQQ', 'VXX', 'UVXY'])
        
    Returns:
        Tuple of (data_dict, event_metadata)
    """
    return load_historical_period("volmageddon_2018", tickers)


def get_event_summary() -> pd.DataFrame:
    """
    Get a summary DataFrame of all available historical events.
    
    Returns:
        DataFrame with event information for easy reference
    """
    summary_data = []
    
    for event_key, event_info in MAJOR_MARKET_EVENTS.items():
        row = {
            'event_key': event_key,
            'name': event_info['name'],
            'start_date': event_info['start_date'],
            'end_date': event_info['end_date'],
            'expected_regime': event_info['expected_regime'],
            'description': event_info['description'],
            'key_tickers': ', '.join(event_info['key_tickers']),
        }
        
        # Add characteristics if available
        if 'characteristics' in event_info:
            chars = event_info['characteristics']
            if 'mean_return' in chars:
                row['mean_return'] = chars['mean_return']
            if 'volatility' in chars:
                row['volatility'] = chars['volatility']
            if 'duration_days' in chars:
                row['duration_days'] = chars['duration_days']
                
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def validate_event_data(data_dict: Dict[str, pd.DataFrame], event_info: Dict) -> Dict:
    """
    Validate loaded historical data against expected characteristics.
    
    Args:
        data_dict: Dictionary of loaded stock data
        event_info: Event metadata from MAJOR_MARKET_EVENTS
        
    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {
        'event_name': event_info['name'],
        'tickers_loaded': list(data_dict.keys()),
        'validation_passed': True,
        'issues': [],
        'statistics': {}
    }
    
    for ticker, data in data_dict.items():
        if data.empty:
            validation_results['issues'].append(f"{ticker}: No data loaded")
            validation_results['validation_passed'] = False
            continue
            
        # Calculate actual statistics
        returns = data['log_return'].dropna()
        actual_stats = {
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'days_of_data': len(data),
            'min_return': returns.min(),
            'max_return': returns.max(),
        }
        
        validation_results['statistics'][ticker] = actual_stats
        
        # Check against expected characteristics if available
        if 'characteristics' in event_info:
            expected = event_info['characteristics']
            
            # Check if we have simple characteristics (not nested like COVID)
            if 'mean_return' in expected and isinstance(expected['mean_return'], (int, float)):
                actual_mean = actual_stats['mean_return']
                expected_mean = expected['mean_return']
                
                # Allow 50% tolerance for historical validation
                tolerance = abs(expected_mean * 0.5)
                if abs(actual_mean - expected_mean) > tolerance:
                    validation_results['issues'].append(
                        f"{ticker}: Mean return {actual_mean:.4f} differs significantly "
                        f"from expected {expected_mean:.4f}"
                    )
    
    return validation_results