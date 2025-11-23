"""
HTTP session pooling for efficient yfinance data loading.

Implements connection reuse and session pooling to reduce latency
and network overhead when loading historical market data.
"""

from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SessionPool:
    """
    Singleton session pool for HTTP requests to yfinance.

    Features:
    - Reuses HTTP connections across multiple requests
    - Implements automatic retry with exponential backoff
    - Configurable connection pooling
    - Thread-safe lazy initialization
    """

    _instance: Optional["SessionPool"] = None
    _session: Optional[requests.Session] = None

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize session pool if not already done."""
        if not self._initialized:
            self._session = self._create_session()
            self._initialized = True

    @staticmethod
    def _create_session(
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> requests.Session:
        """
        Create a requests session with connection pooling and retry logic.

        Args:
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections in pool
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for exponential backoff

        Returns:
            Configured requests.Session object
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"],
        )

        # Apply retry strategy to both HTTP and HTTPS
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set a reasonable timeout
        session.timeout = 30.0

        return session

    def get_session(self) -> requests.Session:
        """Get the shared session for HTTP requests."""
        return self._session

    @classmethod
    def reset(cls) -> None:
        """Reset the session pool (useful for testing)."""
        if cls._instance is not None and cls._instance._session is not None:
            cls._instance._session.close()
        cls._instance = None

    @classmethod
    def close(cls) -> None:
        """Close the session and clean up resources."""
        if cls._instance is not None and cls._instance._session is not None:
            cls._instance._session.close()
            cls._instance = None

    def __del__(self):
        """Ensure session is closed when pool is garbage collected."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass


def get_pooled_session() -> requests.Session:
    """
    Get the global HTTP session pool for efficient connection reuse.

    Returns:
        Configured requests.Session with connection pooling
    """
    pool = SessionPool()
    return pool.get_session()
