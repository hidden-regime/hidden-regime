"""
Error handling and taxonomy for Hidden Regime MCP.

Defines standardized error codes, custom exceptions, and error handling utilities
for robust MCP tool execution and user-friendly error messages.

Error Categories:
- VALIDATION_ERROR (1xxx): Input validation failures
- DATA_ERROR (2xxx): Data loading and processing failures
- MODEL_ERROR (3xxx): HMM model training/inference failures
- NETWORK_ERROR (4xxx): Network and connectivity failures
- RESOURCE_ERROR (5xxx): System resource constraints
- UNKNOWN_ERROR (9xxx): Unexpected errors
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class ErrorCode(str, Enum):
    """Standardized error codes for MCP tools."""

    # Validation Errors (1xxx)
    INVALID_TICKER = "1001"
    INVALID_N_STATES = "1002"
    INVALID_DATE_FORMAT = "1003"
    INVALID_DATE_RANGE = "1004"
    INVALID_PARAMETER = "1005"

    # Data Errors (2xxx)
    DATA_NOT_FOUND = "2001"
    INSUFFICIENT_DATA = "2002"
    DATA_PROCESSING_FAILED = "2003"
    MISSING_REQUIRED_FIELD = "2004"
    DATA_VALIDATION_FAILED = "2005"

    # Model Errors (3xxx)
    MODEL_TRAINING_FAILED = "3001"
    MODEL_INFERENCE_FAILED = "3002"
    INVALID_MODEL_STATE = "3003"
    PARAMETER_ESTIMATION_FAILED = "3004"

    # Network Errors (4xxx)
    NETWORK_TIMEOUT = "4001"
    CONNECTION_FAILED = "4002"
    DATA_SOURCE_UNAVAILABLE = "4003"
    YFINANCE_ERROR = "4004"
    RATE_LIMIT_EXCEEDED = "4005"

    # Resource Errors (5xxx)
    OUT_OF_MEMORY = "5001"
    TIMEOUT = "5002"
    RESOURCE_EXHAUSTED = "5003"

    # Unknown Errors (9xxx)
    UNKNOWN_ERROR = "9001"


@dataclass
class ErrorInfo:
    """Detailed error information with recovery suggestions."""

    code: ErrorCode
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    retriable: bool = False
    http_status: int = 400

    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "retriable": self.retriable,
        }


class MCPToolError(Exception):
    """Base exception for MCP tool errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[str] = None,
        suggestion: Optional[str] = None,
        retriable: bool = False,
    ):
        """Initialize MCP tool error."""
        self.code = code
        self.message = message
        self.details = details
        self.suggestion = suggestion
        self.retriable = retriable
        super().__init__(message)

    def to_error_info(self) -> ErrorInfo:
        """Convert to ErrorInfo object."""
        return ErrorInfo(
            code=self.code,
            message=self.message,
            details=self.details,
            suggestion=self.suggestion,
            retriable=self.retriable,
        )


class ValidationError(MCPToolError):
    """Input validation error."""

    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize validation error."""
        super().__init__(
            code=ErrorCode.INVALID_PARAMETER,
            message=message,
            details=details,
            suggestion="Check your input parameters and try again",
            retriable=False,
        )


class DataError(MCPToolError):
    """Data loading or processing error."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        retriable: bool = True,
    ):
        """Initialize data error."""
        super().__init__(
            code=ErrorCode.DATA_NOT_FOUND,
            message=message,
            details=details,
            suggestion="Check the ticker symbol and ensure sufficient trading data exists",
            retriable=retriable,
        )


class NetworkError(MCPToolError):
    """Network connectivity error."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        retriable: bool = True,
    ):
        """Initialize network error."""
        super().__init__(
            code=ErrorCode.NETWORK_TIMEOUT,
            message=message,
            details=details,
            suggestion="Check your internet connection and try again",
            retriable=retriable,
        )


class ModelError(MCPToolError):
    """HMM model training or inference error."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        retriable: bool = False,
    ):
        """Initialize model error."""
        super().__init__(
            code=ErrorCode.MODEL_TRAINING_FAILED,
            message=message,
            details=details,
            suggestion="Try with a different number of states or a longer date range",
            retriable=retriable,
        )


class ResourceError(MCPToolError):
    """System resource constraint error."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        retriable: bool = True,
    ):
        """Initialize resource error."""
        super().__init__(
            code=ErrorCode.RESOURCE_EXHAUSTED,
            message=message,
            details=details,
            suggestion="Try again after a moment or reduce the date range",
            retriable=retriable,
        )


# Error Message Templates

ERROR_MESSAGES: Dict[ErrorCode, Dict[str, str]] = {
    ErrorCode.INVALID_TICKER: {
        "short": "Invalid ticker symbol",
        "long": "The ticker '{ticker}' is not in a valid format. Tickers should be alphanumeric with optional dots or hyphens, max 10 characters.",
        "suggestion": "Use standard ticker symbols (e.g., SPY, AAPL, BRK.A)",
    },
    ErrorCode.INVALID_N_STATES: {
        "short": "Invalid number of regimes",
        "long": "Number of states must be between 2 and 5, got {n_states}",
        "suggestion": "Use n_states between 2 (bear/bull) and 5 (max complexity)",
    },
    ErrorCode.INVALID_DATE_FORMAT: {
        "short": "Invalid date format",
        "long": "Date must be in YYYY-MM-DD format, got '{date}'",
        "suggestion": "Use dates like 2025-01-15 or 2025-12-31",
    },
    ErrorCode.INVALID_DATE_RANGE: {
        "short": "Invalid date range",
        "long": "Start date must be before end date. Got start={start}, end={end}",
        "suggestion": "Ensure start_date < end_date",
    },
    ErrorCode.DATA_NOT_FOUND: {
        "short": "No data available for ticker",
        "long": "Could not find market data for {ticker} in the period {start}-{end}",
        "suggestion": "Check ticker symbol, ensure it's trading, and try a different date range",
    },
    ErrorCode.INSUFFICIENT_DATA: {
        "short": "Insufficient historical data",
        "long": "Only {observations} observations available, need at least 100 for reliable regime detection",
        "suggestion": "Use a longer date range or a more established ticker with more history",
    },
    ErrorCode.NETWORK_TIMEOUT: {
        "short": "Network timeout",
        "long": "Request to {service} timed out after {timeout}s",
        "suggestion": "Check your internet connection and try again",
    },
    ErrorCode.DATA_SOURCE_UNAVAILABLE: {
        "short": "Data source unavailable",
        "long": "Cannot reach {service}. The data provider may be temporarily unavailable.",
        "suggestion": "Wait a moment and try again, or check https://yfinance.io status",
    },
    ErrorCode.MODEL_TRAINING_FAILED: {
        "short": "Model training failed",
        "long": "HMM training failed for {ticker}: {reason}",
        "suggestion": "Try with n_states={suggestion_states} or a different date range",
    },
    ErrorCode.OUT_OF_MEMORY: {
        "short": "System out of memory",
        "long": "Not enough memory to process {ticker} with {observations} observations",
        "suggestion": "Try with fewer observations or restart the MCP server",
    },
}


def get_error_message(
    code: ErrorCode, **kwargs
) -> Dict[str, str]:
    """
    Get error message template for a given error code.

    Args:
        code: ErrorCode enum value
        **kwargs: Format arguments for the message template

    Returns:
        Dictionary with 'short' and 'long' message templates, formatted with kwargs
    """
    template = ERROR_MESSAGES.get(code, ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR])

    return {
        "short": template["short"],
        "long": template["long"].format(**kwargs) if kwargs else template["long"],
        "suggestion": template.get("suggestion", "Try again or contact support"),
    }
