"""
QuantConnect API credentials management.

This module handles secure storage and retrieval of QC API credentials.
Credentials can be provided via:
1. Environment variables (QC_API_KEY, QC_API_SECRET)
2. Config file (~/.qc-credentials.json)
3. Direct initialization (for programmatic use)

Security Notes:
- Never commit credentials to version control
- Use environment variables in production
- Config file should have restricted permissions (0600)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class QCCredentials:
    """Manages QuantConnect API credentials."""

    CREDS_FILE = Path.home() / ".qc-credentials.json"
    ENV_KEY = "QC_API_KEY"
    ENV_SECRET = "QC_API_SECRET"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize credentials handler.

        Priority order:
        1. Provided arguments
        2. Environment variables
        3. Config file

        Args:
            api_key: API key (overrides env/file)
            api_secret: API secret (overrides env/file)
        """
        self.api_key = api_key or os.getenv(self.ENV_KEY)
        self.api_secret = api_secret or os.getenv(self.ENV_SECRET)

        # If not provided, try to load from config file
        if not self.api_key or not self.api_secret:
            file_creds = self._load_from_file()
            if file_creds:
                self.api_key = self.api_key or file_creds.get("api_key")
                self.api_secret = self.api_secret or file_creds.get("api_secret")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "QuantConnect credentials not found. "
                "Set QC_API_KEY and QC_API_SECRET environment variables, "
                "or create ~/.qc-credentials.json with your credentials."
            )

    def _load_from_file(self) -> Optional[dict]:
        """Load credentials from config file."""
        if not self.CREDS_FILE.exists():
            return None

        try:
            with open(self.CREDS_FILE, "r") as f:
                creds = json.load(f)
                logger.debug(f"Loaded credentials from {self.CREDS_FILE}")
                return creds
        except Exception as e:
            logger.warning(f"Failed to load credentials from {self.CREDS_FILE}: {e}")
            return None

    @classmethod
    def save_to_file(cls, api_key: str, api_secret: str) -> None:
        """
        Save credentials to config file with restricted permissions.

        Args:
            api_key: QuantConnect API key
            api_secret: QuantConnect API secret
        """
        creds = {
            "api_key": api_key,
            "api_secret": api_secret,
        }

        try:
            # Create file with restricted permissions (0600)
            cls.CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Write with restricted mode
            with open(cls.CREDS_FILE, "w", opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
                json.dump(creds, f, indent=2)

            logger.info(f"Saved credentials to {cls.CREDS_FILE}")
            logger.warning(
                "Make sure to keep this file private. "
                "File permissions set to 0600 (owner read/write only)."
            )
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise

    def to_dict(self) -> dict:
        """Return credentials as dictionary."""
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
        }
