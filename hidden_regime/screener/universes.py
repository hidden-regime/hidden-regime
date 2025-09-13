"""
Stock Universe Definitions

Predefined stock universes for systematic screening including major indices
and sector classifications. Provides easy access to common screening targets.
"""

import warnings
from typing import Dict, List, Optional

import pandas as pd

# Major stock universes with representative tickers
# Note: In production, these would be loaded from external data sources
STOCK_UNIVERSES = {
    'sp500_sample': [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ORCL', 'ADBE',
        # Financials  
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CME',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'MRK', 'AMGN',
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG',
        # Industrial
        'BA', 'CAT', 'HON', 'UNP', 'LMT', 'RTX', 'DE', 'UPS', 'GE', 'MMM',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'IFF',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ES',
        # Communication Services
        'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'
    ],
    
    'tech_giants': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ORCL', 'ADBE',
        'CRM', 'INTC', 'AMD', 'QCOM', 'UBER', 'SPOT', 'ZM', 'DOCU', 'SNOW', 'PLTR'
    ],
    
    'financial_sector': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CME',
        'ICE', 'MCO', 'SCHW', 'USB', 'TFC', 'PNC', 'COF', 'AON', 'MMC', 'AJG'
    ],
    
    'energy_sector': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
        'KMI', 'WMB', 'EPD', 'ET', 'MPLX', 'BKR', 'DVN', 'FANG', 'MRO', 'APA'
    ],
    
    'healthcare_biotech': [
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'MRK', 'AMGN',
        'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'DHR', 'SYK', 'BSX', 'MDT', 'EW'
    ],
    
    'retail_consumer': [
        'AMZN', 'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'NKE', 'SBUX', 'MCD',
        'DIS', 'BKNG', 'ABNB', 'ETSY', 'EBAY', 'LULU', 'ROST', 'DG', 'DLTR', 'BBY'
    ],
    
    'industrials': [
        'BA', 'CAT', 'HON', 'UNP', 'LMT', 'RTX', 'DE', 'UPS', 'GE', 'MMM',
        'CSX', 'NSC', 'FDX', 'ITW', 'EMR', 'ETN', 'PH', 'ROK', 'DOV', 'XYL'
    ],
    
    'etf_major': [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG', 'TLT',
        'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU'
    ],
    
    'growth_stocks': [
        'TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM', 'DOCU', 'SNOW', 'PLTR', 'ARKK', 'SQ',
        'ROKU', 'TWLO', 'OKTA', 'CRWD', 'ZS', 'DDOG', 'NET', 'FSLY', 'ESTC', 'MDB'
    ],
    
    'value_stocks': [
        'BRK.B', 'JPM', 'JNJ', 'PG', 'XOM', 'CVX', 'WMT', 'KO', 'PEP', 'MRK',
        'VZ', 'T', 'IBM', 'GE', 'F', 'GM', 'C', 'BAC', 'WFC', 'USB'
    ],
    
    'dividend_aristocrats': [
        'MMM', 'ABT', 'ABBV', 'ADP', 'ALB', 'APD', 'AFL', 'A', 'AOS', 'LNT',
        'AMCR', 'ABC', 'AME', 'AMGN', 'APH', 'ATO', 'T', 'AZO', 'BDX', 'BF.B'
    ],
    
    'small_cap_sample': [
        'ROKU', 'PENN', 'PLUG', 'FCEL', 'SPCE', 'NKLA', 'RIDE', 'WKHS', 'HYLN', 'BLNK',
        'CHPT', 'LAZR', 'VLDR', 'LIDR', 'MVIS', 'GMHI', 'ACTC', 'CCIV', 'PSTH', 'IPOE'
    ],
    
    'crypto_exposed': [
        'COIN', 'MSTR', 'TSLA', 'SQ', 'PYPL', 'HOOD', 'RIOT', 'MARA', 'CAN', 'BITF',
        'HUT', 'ARBK', 'BTBT', 'EBON', 'SOS', 'NCTY', 'HVBT', 'CLSK', 'DPW', 'NILE'
    ]
}

# Sector mappings for classification
SECTOR_MAPPINGS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ORCL', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CME'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'MRK', 'AMGN'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG'],
    'Industrials': ['BA', 'CAT', 'HON', 'UNP', 'LMT', 'RTX', 'DE', 'UPS', 'GE', 'MMM'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'IFF'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ES'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'],
    'Communication Services': ['META', 'NFLX', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI'],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WY', 'DLR', 'BXP', 'O', 'WELL']
}


def get_sp500_universe(sample_size: Optional[int] = None) -> List[str]:
    """
    Get S&P 500 universe of stocks.
    
    Args:
        sample_size: Optional number of stocks to sample (returns all if None)
        
    Returns:
        List of S&P 500 stock tickers
    """
    universe = STOCK_UNIVERSES['sp500_sample'].copy()
    
    if sample_size is not None and sample_size < len(universe):
        import random
        random.seed(42)  # For reproducibility
        universe = random.sample(universe, sample_size)
    
    return universe


def get_russell2000_universe(sample_size: Optional[int] = None) -> List[str]:
    """
    Get Russell 2000 universe (small cap stocks).
    
    Args:
        sample_size: Optional number of stocks to sample
        
    Returns:
        List of Russell 2000 stock tickers
    """
    # Note: This is a sample representation, not the complete Russell 2000
    universe = STOCK_UNIVERSES['small_cap_sample'].copy()
    
    if sample_size is not None and sample_size < len(universe):
        import random
        random.seed(42)
        universe = random.sample(universe, sample_size)
    
    return universe


def get_sector_universe(sector: str, sample_size: Optional[int] = None) -> List[str]:
    """
    Get stocks from a specific sector.
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Financials')
        sample_size: Optional number of stocks to sample
        
    Returns:
        List of stock tickers in the sector
    """
    if sector not in SECTOR_MAPPINGS:
        available_sectors = list(SECTOR_MAPPINGS.keys())
        raise ValueError(f"Unknown sector '{sector}'. Available: {available_sectors}")
    
    universe = SECTOR_MAPPINGS[sector].copy()
    
    if sample_size is not None and sample_size < len(universe):
        import random
        random.seed(42)
        universe = random.sample(universe, sample_size)
    
    return universe


def get_custom_universe(universe_name: str, sample_size: Optional[int] = None) -> List[str]:
    """
    Get a predefined custom universe of stocks.
    
    Args:
        universe_name: Name of the universe (e.g., 'tech_giants', 'energy_sector')
        sample_size: Optional number of stocks to sample
        
    Returns:
        List of stock tickers in the universe
    """
    if universe_name not in STOCK_UNIVERSES:
        available_universes = list(STOCK_UNIVERSES.keys())
        raise ValueError(f"Unknown universe '{universe_name}'. Available: {available_universes}")
    
    universe = STOCK_UNIVERSES[universe_name].copy()
    
    if sample_size is not None and sample_size < len(universe):
        import random
        random.seed(42)
        universe = random.sample(universe, sample_size)
    
    return universe


def create_custom_universe(tickers: List[str], name: str) -> List[str]:
    """
    Create and register a custom universe of stocks.
    
    Args:
        tickers: List of stock tickers
        name: Name for the custom universe
        
    Returns:
        The ticker list (for chaining)
    """
    # Validate tickers
    if not tickers:
        raise ValueError("Ticker list cannot be empty")
    
    # Remove duplicates and sort
    clean_tickers = sorted(list(set(tickers)))
    
    # Register the universe
    STOCK_UNIVERSES[name] = clean_tickers
    
    return clean_tickers


def get_universe_info(universe_name: str) -> Dict:
    """
    Get information about a stock universe.
    
    Args:
        universe_name: Name of the universe
        
    Returns:
        Dictionary with universe information
    """
    if universe_name not in STOCK_UNIVERSES:
        available_universes = list(STOCK_UNIVERSES.keys())
        return {
            'exists': False,
            'available_universes': available_universes
        }
    
    tickers = STOCK_UNIVERSES[universe_name]
    
    # Analyze sector distribution if possible
    sector_distribution = {}
    for sector, sector_tickers in SECTOR_MAPPINGS.items():
        overlap = set(tickers) & set(sector_tickers)
        if overlap:
            sector_distribution[sector] = len(overlap)
    
    return {
        'exists': True,
        'name': universe_name,
        'size': len(tickers),
        'tickers': tickers,
        'sector_distribution': sector_distribution,
        'sample_tickers': tickers[:10]  # First 10 for preview
    }


def list_available_universes() -> List[str]:
    """
    List all available stock universes.
    
    Returns:
        List of universe names
    """
    return list(STOCK_UNIVERSES.keys())


def get_market_cap_universe(market_cap: str = 'large') -> List[str]:
    """
    Get universe by market capitalization category.
    
    Args:
        market_cap: Market cap category ('large', 'small', 'growth', 'value')
        
    Returns:
        List of stock tickers for the market cap category
    """
    market_cap_mappings = {
        'large': 'sp500_sample',
        'small': 'small_cap_sample', 
        'growth': 'growth_stocks',
        'value': 'value_stocks'
    }
    
    if market_cap not in market_cap_mappings:
        available_caps = list(market_cap_mappings.keys())
        raise ValueError(f"Unknown market cap '{market_cap}'. Available: {available_caps}")
    
    universe_name = market_cap_mappings[market_cap]
    return STOCK_UNIVERSES[universe_name].copy()


def get_thematic_universe(theme: str) -> List[str]:
    """
    Get universe by investment theme.
    
    Args:
        theme: Investment theme ('crypto', 'dividend', 'etf')
        
    Returns:
        List of stock tickers for the theme
    """
    theme_mappings = {
        'crypto': 'crypto_exposed',
        'dividend': 'dividend_aristocrats',
        'etf': 'etf_major'
    }
    
    if theme not in theme_mappings:
        available_themes = list(theme_mappings.keys())
        raise ValueError(f"Unknown theme '{theme}'. Available: {available_themes}")
    
    universe_name = theme_mappings[theme]
    return STOCK_UNIVERSES[universe_name].copy()


def load_universe_from_file(filepath: str, 
                           ticker_column: str = 'ticker',
                           name: Optional[str] = None) -> List[str]:
    """
    Load stock universe from CSV file.
    
    Args:
        filepath: Path to CSV file containing tickers
        ticker_column: Name of column containing ticker symbols
        name: Optional name to register the universe
        
    Returns:
        List of stock tickers
    """
    try:
        df = pd.read_csv(filepath)
        
        if ticker_column not in df.columns:
            raise ValueError(f"Column '{ticker_column}' not found in file. Available: {list(df.columns)}")
        
        tickers = df[ticker_column].dropna().unique().tolist()
        tickers = [str(ticker).upper().strip() for ticker in tickers]  # Clean and standardize
        
        if name:
            create_custom_universe(tickers, name)
        
        return tickers
        
    except Exception as e:
        raise ValueError(f"Failed to load universe from file: {str(e)}")


def combine_universes(*universe_names: str, name: Optional[str] = None) -> List[str]:
    """
    Combine multiple universes into one.
    
    Args:
        *universe_names: Names of universes to combine
        name: Optional name to register the combined universe
        
    Returns:
        Combined list of unique tickers
    """
    combined_tickers = []
    
    for universe_name in universe_names:
        if universe_name in STOCK_UNIVERSES:
            combined_tickers.extend(STOCK_UNIVERSES[universe_name])
        else:
            warnings.warn(f"Universe '{universe_name}' not found, skipping")
    
    # Remove duplicates and sort
    unique_tickers = sorted(list(set(combined_tickers)))
    
    if name:
        create_custom_universe(unique_tickers, name)
    
    return unique_tickers


def filter_universe(universe_name: str, 
                   exclude_tickers: Optional[List[str]] = None,
                   include_only: Optional[List[str]] = None) -> List[str]:
    """
    Filter an existing universe by inclusion/exclusion criteria.
    
    Args:
        universe_name: Name of universe to filter
        exclude_tickers: Tickers to exclude
        include_only: If provided, only include these tickers
        
    Returns:
        Filtered list of tickers
    """
    if universe_name not in STOCK_UNIVERSES:
        raise ValueError(f"Universe '{universe_name}' not found")
    
    tickers = STOCK_UNIVERSES[universe_name].copy()
    
    if include_only is not None:
        tickers = [t for t in tickers if t in include_only]
    
    if exclude_tickers is not None:
        tickers = [t for t in tickers if t not in exclude_tickers]
    
    return tickers