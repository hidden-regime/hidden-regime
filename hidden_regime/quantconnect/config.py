"""
Configuration classes for QuantConnect LEAN integration.

This module provides configuration dataclasses for managing QuantConnect-specific
settings and regime trading parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class QuantConnectConfig:
    """
    Configuration for QuantConnect LEAN integration.

    Attributes:
        lookback_days: Number of days of historical data to maintain
        retrain_frequency: How often to retrain HMM ('daily', 'weekly', 'monthly')
        warmup_days: Number of days for algorithm warm-up period
        use_cache: Whether to cache trained models
        log_regime_changes: Whether to log regime transitions
        min_confidence: Minimum confidence threshold for regime signals
    """

    lookback_days: int = 252  # ~1 year of trading days
    retrain_frequency: str = "weekly"  # 'daily', 'weekly', 'monthly', 'never'
    warmup_days: int = 252
    use_cache: bool = True
    log_regime_changes: bool = True
    min_confidence: float = 0.0  # 0.0 = no threshold

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.lookback_days < 30:
            raise ValueError("lookback_days must be at least 30")

        valid_frequencies = ["daily", "weekly", "monthly", "never"]
        if self.retrain_frequency not in valid_frequencies:
            raise ValueError(
                f"retrain_frequency must be one of {valid_frequencies}, "
                f"got {self.retrain_frequency}"
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass
class RegimeTradingConfig:
    """
    Configuration for regime-based trading rules.

    Attributes:
        regime_allocations: Dict mapping regime names to portfolio allocations
        rebalance_threshold: Minimum allocation change to trigger rebalance
        use_risk_parity: Whether to use risk parity for position sizing
        max_leverage: Maximum portfolio leverage allowed
        cash_regime: Regime that triggers moving to cash (optional)
    """

    regime_allocations: Dict[str, float] = field(
        default_factory=lambda: {
            "Bull": 1.0,  # 100% long
            "Bear": 0.0,  # Cash
            "Sideways": 0.5,  # 50% long
            "Crisis": 0.0,  # Cash
        }
    )
    rebalance_threshold: float = 0.1  # 10% change triggers rebalance
    use_risk_parity: bool = False
    max_leverage: float = 1.0
    cash_regime: Optional[str] = "Bear"

    def __post_init__(self) -> None:
        """Validate configuration."""
        for regime, allocation in self.regime_allocations.items():
            if not -2.0 <= allocation <= 2.0:  # Allow some shorting/leverage
                raise ValueError(
                    f"Allocation for {regime} must be between -2.0 and 2.0, "
                    f"got {allocation}"
                )

        if not 0.0 <= self.rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be between 0.0 and 1.0")

        if self.max_leverage < 0:
            raise ValueError("max_leverage must be positive")

    def get_allocation(self, regime_name: str) -> float:
        """
        Get allocation for a given regime.

        Args:
            regime_name: Name of the regime

        Returns:
            Portfolio allocation (0.0 to 1.0, or negative for shorts)
        """
        return self.regime_allocations.get(regime_name, 0.0)

    @classmethod
    def create_conservative(cls) -> "RegimeTradingConfig":
        """
        Create conservative trading configuration.

        Conservative strategy:
        - Bull: 60% long
        - Sideways: 30% long
        - Bear/Crisis: Cash
        """
        return cls(
            regime_allocations={
                "Bull": 0.6,
                "Bear": 0.0,
                "Sideways": 0.3,
                "Crisis": 0.0,
            },
            rebalance_threshold=0.15,
            use_risk_parity=False,
            max_leverage=1.0,
        )

    @classmethod
    def create_aggressive(cls) -> "RegimeTradingConfig":
        """
        Create aggressive trading configuration.

        Aggressive strategy:
        - Bull: 100% long
        - Sideways: 80% long
        - Bear: 20% long (or cash)
        - Crisis: Cash
        """
        return cls(
            regime_allocations={
                "Bull": 1.0,
                "Bear": 0.2,
                "Sideways": 0.8,
                "Crisis": 0.0,
            },
            rebalance_threshold=0.05,
            use_risk_parity=False,
            max_leverage=1.0,
        )

    @classmethod
    def create_market_neutral(cls) -> "RegimeTradingConfig":
        """
        Create market-neutral trading configuration.

        Market-neutral strategy:
        - Bull: 50% long
        - Bear: -50% short
        - Sideways: 0% (cash)
        - Crisis: 0% (cash)
        """
        return cls(
            regime_allocations={
                "Bull": 0.5,
                "Bear": -0.5,
                "Sideways": 0.0,
                "Crisis": 0.0,
            },
            rebalance_threshold=0.1,
            use_risk_parity=False,
            max_leverage=1.5,
        )


@dataclass
class MultiAssetRegimeConfig:
    """
    Configuration for multi-asset regime-based allocation.

    Attributes:
        assets: Dict mapping ticker symbols to their configurations
        equal_weight: Whether to equally weight assets in same regime
        concentration_limit: Maximum allocation to single asset
        rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
    """

    assets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    equal_weight: bool = True
    concentration_limit: float = 0.5  # Max 50% in one asset
    rebalance_frequency: str = "weekly"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.concentration_limit <= 1.0:
            raise ValueError("concentration_limit must be between 0.0 and 1.0")

        valid_frequencies = ["daily", "weekly", "monthly"]
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(
                f"rebalance_frequency must be one of {valid_frequencies}"
            )

    @classmethod
    def create_risk_parity(cls, tickers: list) -> "MultiAssetRegimeConfig":
        """
        Create risk parity multi-asset configuration.

        Args:
            tickers: List of ticker symbols

        Returns:
            MultiAssetRegimeConfig for risk parity allocation
        """
        assets = {ticker: {} for ticker in tickers}
        return cls(
            assets=assets,
            equal_weight=False,  # Use volatility-based weighting
            concentration_limit=0.4,
            rebalance_frequency="weekly",
        )
