"""
QuantConnect REST API client for pulling historical market data locally.

This module provides a clean interface to the QuantConnect REST API for:
- Pulling historical OHLCV data
- Caching results to avoid repeated API calls
- Handling authentication with PRO API keys
- Managing rate limits and retries

Usage:
    from hidden_regime.quantconnect.api_client import QCApiClient

    client = QCApiClient(api_key="your-key", api_secret="your-secret")
    data = client.get_historical_data(
        ticker="SPY",
        start_date="2020-01-01",
        end_date="2023-12-31",
        resolution="daily"
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class QCApiClient:
    """QuantConnect REST API client for historical data retrieval."""

    BASE_URL = "https://www.quantconnect.com/api/v2"
    CACHE_DIR = Path.home() / ".cache" / "hidden-regime" / "qc-data"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        timeout: int = 30,
    ):
        """
        Initialize QuantConnect API client.

        Args:
            api_key: QuantConnect API key (from account settings)
            api_secret: QuantConnect API secret (from account settings)
            cache_enabled: Enable local caching of API responses (default: True)
            cache_dir: Directory for caching (default: ~/.cache/hidden-regime/qc-data)
            timeout: Request timeout in seconds (default: 30)

        Note:
            Get API credentials at https://www.quantconnect.com/account
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.timeout = timeout

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_cache_path(self, ticker: str, start_date: str, end_date: str, resolution: str) -> Path:
        """Get cache file path for given parameters."""
        cache_key = f"{ticker}_{start_date}_{end_date}_{resolution}.parquet"
        return self.cache_dir / cache_key

    def _load_from_cache(self, ticker: str, start_date: str, end_date: str, resolution: str) -> Optional[pd.DataFrame]:
        """Load data from local cache if available."""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(ticker, start_date, end_date, resolution)
        if cache_path.exists():
            try:
                data = pd.read_parquet(cache_path)
                logger.info(f"Loaded {ticker} data from cache: {cache_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None

        return None

    def _save_to_cache(
        self, data: pd.DataFrame, ticker: str, start_date: str, end_date: str, resolution: str
    ) -> None:
        """Save data to local cache."""
        if not self.cache_enabled or data.empty:
            return

        cache_path = self._get_cache_path(ticker, start_date, end_date, resolution)
        try:
            data.to_parquet(cache_path)
            logger.info(f"Cached {ticker} data ({len(data)} rows): {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data to {cache_path}: {e}")

    def get_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        resolution: str = "daily",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from QuantConnect API.

        Args:
            ticker: Stock ticker symbol (e.g., "SPY", "QQQ")
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            resolution: Data resolution ("minute", "hour", "daily", "weekly", "monthly")

        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume]
            Index: DatetimeIndex with UTC timestamps

        Raises:
            ValueError: If API request fails or data is invalid
            requests.RequestException: If network error occurs
        """
        # Normalize ticker
        ticker = ticker.upper()

        # Check cache first
        cached_data = self._load_from_cache(ticker, start_date, end_date, resolution)
        if cached_data is not None:
            return cached_data

        logger.info(f"Fetching {ticker} data from QC API ({start_date} to {end_date})...")

        try:
            data = self._fetch_from_api(ticker, start_date, end_date, resolution)
            self._save_to_cache(data, ticker, start_date, end_date, resolution)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            raise

    def _fetch_from_api(
        self, ticker: str, start_date: str, end_date: str, resolution: str
    ) -> pd.DataFrame:
        """Fetch data from QuantConnect REST API."""
        endpoint = f"{self.BASE_URL}/data/get"

        params = {
            "ticker": ticker,
            "startDate": start_date,
            "endDate": end_date,
            "resolution": resolution,
            "apikey": self.api_key,
            "api_secret": self.api_secret,
        }

        response = self.session.get(endpoint, params=params, timeout=self.timeout)

        if response.status_code == 401:
            raise ValueError("API authentication failed. Check your API key and secret.")
        elif response.status_code == 403:
            raise ValueError("Access denied. You may need a PRO subscription for this ticker.")
        elif response.status_code == 404:
            raise ValueError(f"No data found for ticker {ticker}")
        elif response.status_code != 200:
            raise requests.RequestException(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        try:
            result = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from API: {response.text}")

        if not result.get("success"):
            raise ValueError(f"API error: {result.get('message', 'Unknown error')}")

        # Parse the data
        data_list = result.get("data", [])
        if not data_list:
            raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}")

        # Convert to DataFrame
        df = pd.DataFrame(data_list)

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime and set as index
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], unit="s")
            df = df.set_index("Time")
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise ValueError("No timestamp column found in API response")

        # Keep only OHLCV columns
        df = df[required_cols]

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with NaN values
        df = df.dropna()

        # Sort by timestamp
        df = df.sort_index()

        logger.info(f"Fetched {len(df)} bars for {ticker} from QC API")
        return df

    def get_available_tickers(self) -> list[str]:
        """
        Get list of available tickers on QuantConnect.

        Note: This requires PRO subscription for full list.
        Free tier has access to major indices: SPY, QQQ, IWM, TLT, GLD, etc.

        Returns:
            List of ticker symbols
        """
        endpoint = f"{self.BASE_URL}/ticker/list"
        params = {
            "apikey": self.api_key,
            "api_secret": self.api_secret,
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch ticker list: {response.status_code}")
                return []

            result = response.json()
            return result.get("data", [])
        except Exception as e:
            logger.warning(f"Failed to fetch available tickers: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache directory: {self.cache_dir}")

    def get_cache_stats(self) -> dict:
        """Get statistics about cached data."""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)

        return {
            "total_files": len(files),
            "total_size_mb": round(total_size, 2),
            "cache_dir": str(self.cache_dir),
        }
