"""
Signal generator interface and implementations for trading simulation.

Provides unified interface for generating buy/sell/hold signals from various
sources including HMM regime detection, technical indicators, and buy-and-hold.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""

    BUY = 1
    SELL = -1
    HOLD = 0


class SignalGenerator(ABC):
    """
    Abstract base class for all signal generators.

    Provides unified interface for generating trading signals that can be
    consumed by the TradingSimulationEngine.
    """

    def __init__(self, name: str):
        """
        Initialize signal generator.

        Args:
            name: Unique name for this signal generator
        """
        self.name = name

    @abstractmethod
    def generate_signals(
        self, price_data: pd.DataFrame, additional_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate trading signals for the given price data.

        Args:
            price_data: DataFrame with OHLC price data
            additional_data: Optional additional data (regime predictions, indicators, etc.)

        Returns:
            Series with trading signals (1=BUY, -1=SELL, 0=HOLD) indexed by date
        """
        pass

    def validate_data(self, price_data: pd.DataFrame) -> bool:
        """
        Validate that required data columns are present.

        Args:
            price_data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["open", "high", "low", "close"]
        return all(col in price_data.columns for col in required_columns)


class BuyHoldSignalGenerator(SignalGenerator):
    """
    Buy-and-hold signal generator.

    Generates a buy signal at the start and holds until the end.
    Should result in exactly 2 trades: buy at start, sell at end.
    """

    def __init__(self):
        """Initialize buy-and-hold signal generator."""
        super().__init__("buy_and_hold")

    def generate_signals(
        self, price_data: pd.DataFrame, additional_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate buy-and-hold signals.

        Args:
            price_data: DataFrame with OHLC price data
            additional_data: Ignored for buy-and-hold

        Returns:
            Series with buy signal on first day, hold thereafter
        """
        if not self.validate_data(price_data):
            raise ValueError("Invalid price data for buy-and-hold signal generation")

        # Create signal series filled with HOLD
        signals = pd.Series(SignalType.HOLD.value, index=price_data.index)

        # Set first signal to BUY
        if len(signals) > 0:
            signals.iloc[0] = SignalType.BUY.value

        return signals


class HMMSignalGenerator(SignalGenerator):
    """
    HMM regime-following signal generator.

    Generates signals based on regime predictions:
    - Bull regime -> Buy signal
    - Bear regime -> Sell signal
    - Sideways regime -> Hold signal
    """

    def __init__(self, strategy_type: str = "regime_following"):
        """
        Initialize HMM signal generator.

        Args:
            strategy_type: Type of HMM strategy ('regime_following', 'regime_contrarian', 'confidence_weighted')
        """
        super().__init__(f"hmm_{strategy_type}")
        self.strategy_type = strategy_type

    def generate_signals(
        self, price_data: pd.DataFrame, additional_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate HMM-based trading signals.

        Args:
            price_data: DataFrame with OHLC price data
            additional_data: DataFrame with regime predictions and confidence

        Returns:
            Series with trading signals based on regime predictions
        """
        if not self.validate_data(price_data):
            raise ValueError("Invalid price data for HMM signal generation")

        if additional_data is None or "predicted_state" not in additional_data.columns:
            raise ValueError(
                "HMM signal generator requires regime predictions in additional_data"
            )

        # Align regime data with price data
        aligned_data = price_data.join(
            additional_data[["predicted_state", "confidence"]], how="inner"
        )

        if len(aligned_data) == 0:
            raise ValueError("No aligned data between price and regime predictions")

        # Determine number of states
        n_states = int(aligned_data["predicted_state"].max()) + 1

        # Generate signals based on strategy type
        if self.strategy_type == "regime_following":
            signals = self._generate_regime_following_signals(aligned_data, n_states)
        elif self.strategy_type == "regime_contrarian":
            signals = self._generate_regime_contrarian_signals(aligned_data, n_states)
        elif self.strategy_type == "confidence_weighted":
            signals = self._generate_confidence_weighted_signals(aligned_data, n_states)
        else:
            raise ValueError(f"Unknown HMM strategy type: {self.strategy_type}")

        return signals

    def _generate_regime_following_signals(
        self, data: pd.DataFrame, n_states: int
    ) -> pd.Series:
        """Generate regime-following signals (long in bull, short in bear)."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        for i, (_, row) in enumerate(data.iterrows()):
            regime = row["predicted_state"]

            if regime == n_states - 1:  # Highest regime = bull
                signals.iloc[i] = SignalType.BUY.value
            elif regime == 0:  # Lowest regime = bear
                signals.iloc[i] = SignalType.SELL.value
            # Middle regimes remain HOLD

        return signals

    def _generate_regime_contrarian_signals(
        self, data: pd.DataFrame, n_states: int
    ) -> pd.Series:
        """Generate contrarian signals (short in bull, long in bear)."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        for i, (_, row) in enumerate(data.iterrows()):
            regime = row["predicted_state"]

            if regime == n_states - 1:  # Bull -> sell
                signals.iloc[i] = SignalType.SELL.value
            elif regime == 0:  # Bear -> buy
                signals.iloc[i] = SignalType.BUY.value
            # Middle regimes remain HOLD

        return signals

    def _generate_confidence_weighted_signals(
        self, data: pd.DataFrame, n_states: int
    ) -> pd.Series:
        """Generate confidence-weighted signals."""
        # Start with regime-following signals
        base_signals = self._generate_regime_following_signals(data, n_states)

        # Weight by confidence (scale signal strength)
        if "confidence" in data.columns:
            confidence_weights = data["confidence"].fillna(0.5)
            # Scale signals by confidence but keep them as integers for consistency
            weighted_signals = base_signals * confidence_weights
            # Convert back to integer signals based on magnitude
            signals = pd.Series(SignalType.HOLD.value, index=data.index)
            signals[weighted_signals > 0.5] = SignalType.BUY.value
            signals[weighted_signals < -0.5] = SignalType.SELL.value
            return signals
        else:
            return base_signals


class TechnicalIndicatorSignalGenerator(SignalGenerator):
    """
    Technical indicator signal generator.

    Generates signals based on technical indicator values using standard
    crossover and threshold-based rules.
    """

    def __init__(
        self, indicator_name: str, indicator_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize technical indicator signal generator.

        Args:
            indicator_name: Name of the technical indicator (e.g., 'rsi', 'macd', 'sma')
            indicator_params: Optional parameters for indicator calculation
        """
        super().__init__(f"ta_{indicator_name}")
        self.indicator_name = indicator_name
        self.indicator_params = indicator_params or {}

    def generate_signals(
        self, price_data: pd.DataFrame, additional_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate signals from technical indicator.

        Args:
            price_data: DataFrame with OHLC price data
            additional_data: Optional pre-calculated indicator values

        Returns:
            Series with trading signals based on indicator rules
        """
        if not self.validate_data(price_data):
            raise ValueError(
                "Invalid price data for technical indicator signal generation"
            )

        # Use pre-calculated indicator if provided, otherwise calculate
        if (
            additional_data is not None
            and self.indicator_name in additional_data.columns
        ):
            indicator_values = additional_data[self.indicator_name]
        else:
            indicator_values = self._calculate_indicator(price_data)

        # Generate signals based on indicator type
        if self.indicator_name in ["rsi"]:
            return self._generate_rsi_signals(indicator_values)
        elif self.indicator_name in ["macd"]:
            return self._generate_macd_signals(price_data, indicator_values)
        elif self.indicator_name in ["sma", "ema"]:
            return self._generate_ma_signals(price_data, indicator_values)
        elif self.indicator_name in ["bollinger_bands"]:
            return self._generate_bollinger_signals(price_data, additional_data)
        else:
            # Generic momentum-based signals
            return self._generate_generic_momentum_signals(indicator_values)

    def _calculate_indicator(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values (placeholder - would use TA library)."""
        # This is a simplified placeholder
        # In practice, this would use the TA library to calculate indicators
        if self.indicator_name == "sma":
            period = self.indicator_params.get("period", 20)
            return price_data["close"].rolling(window=period).mean()
        else:
            # Placeholder: return close prices
            return price_data["close"]

    def _generate_rsi_signals(self, rsi_values: pd.Series) -> pd.Series:
        """Generate signals based on RSI indicator."""
        signals = pd.Series(SignalType.HOLD.value, index=rsi_values.index)

        # RSI overbought/oversold signals
        overbought = self.indicator_params.get("overbought", 70)
        oversold = self.indicator_params.get("oversold", 30)

        # Buy when RSI crosses above oversold
        signals[rsi_values.shift(1) <= oversold] = SignalType.BUY.value
        # Sell when RSI crosses above overbought
        signals[rsi_values.shift(1) >= overbought] = SignalType.SELL.value

        return signals

    def _generate_macd_signals(
        self, price_data: pd.DataFrame, macd_values: pd.Series
    ) -> pd.Series:
        """Generate signals based on MACD indicator."""
        signals = pd.Series(SignalType.HOLD.value, index=macd_values.index)

        # MACD crossover signals (simplified)
        macd_change = macd_values.diff()

        # Buy on positive MACD momentum
        signals[macd_change > 0] = SignalType.BUY.value
        # Sell on negative MACD momentum
        signals[macd_change < 0] = SignalType.SELL.value

        return signals

    def _generate_ma_signals(
        self, price_data: pd.DataFrame, ma_values: pd.Series
    ) -> pd.Series:
        """Generate signals based on moving average crossover."""
        signals = pd.Series(SignalType.HOLD.value, index=ma_values.index)

        # Price vs MA crossover
        price = price_data["close"]
        aligned_price = price.reindex(ma_values.index)

        # Buy when price crosses above MA
        signals[
            (aligned_price > ma_values) & (aligned_price.shift(1) <= ma_values.shift(1))
        ] = SignalType.BUY.value
        # Sell when price crosses below MA
        signals[
            (aligned_price < ma_values) & (aligned_price.shift(1) >= ma_values.shift(1))
        ] = SignalType.SELL.value

        return signals

    def _generate_bollinger_signals(
        self, price_data: pd.DataFrame, additional_data: pd.DataFrame
    ) -> pd.Series:
        """Generate signals based on Bollinger Bands."""
        signals = pd.Series(SignalType.HOLD.value, index=price_data.index)

        if additional_data is None:
            return signals

        # Expect upper_band and lower_band in additional_data
        if (
            "upper_band" in additional_data.columns
            and "lower_band" in additional_data.columns
        ):
            price = price_data["close"]
            upper_band = additional_data["upper_band"]
            lower_band = additional_data["lower_band"]

            # Buy when price touches lower band
            signals[price <= lower_band] = SignalType.BUY.value
            # Sell when price touches upper band
            signals[price >= upper_band] = SignalType.SELL.value

        return signals

    def _generate_generic_momentum_signals(
        self, indicator_values: pd.Series
    ) -> pd.Series:
        """Generate generic momentum-based signals."""
        signals = pd.Series(SignalType.HOLD.value, index=indicator_values.index)

        # Simple momentum: buy on upward movement, sell on downward
        indicator_change = indicator_values.pct_change()

        # Use thresholds to avoid noise
        buy_threshold = self.indicator_params.get("buy_threshold", 0.01)  # 1% increase
        sell_threshold = self.indicator_params.get(
            "sell_threshold", -0.01
        )  # 1% decrease

        signals[indicator_change > buy_threshold] = SignalType.BUY.value
        signals[indicator_change < sell_threshold] = SignalType.SELL.value

        return signals


class MultiSignalGenerator:
    """
    Combines multiple signal generators into a unified signal DataFrame.

    Creates the daily signal DataFrame as specified in simulation.md showing
    buy/sell signals for each strategy on each day.
    """

    def __init__(self):
        """Initialize multi-signal generator."""
        self.generators: Dict[str, SignalGenerator] = {}

    def add_generator(self, generator: SignalGenerator) -> None:
        """Add a signal generator to the collection."""
        self.generators[generator.name] = generator

    def add_hmm_generator(self, strategy_type: str = "regime_following") -> None:
        """Add HMM signal generator."""
        generator = HMMSignalGenerator(strategy_type)
        self.add_generator(generator)

    def add_buy_hold_generator(self) -> None:
        """Add buy-and-hold signal generator."""
        generator = BuyHoldSignalGenerator()
        self.add_generator(generator)

    def add_technical_indicator_generator(
        self, indicator_name: str, indicator_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add technical indicator signal generator."""
        generator = TechnicalIndicatorSignalGenerator(indicator_name, indicator_params)
        self.add_generator(generator)

    def generate_all_signals(
        self, price_data: pd.DataFrame, additional_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate signals from all registered generators.

        Args:
            price_data: DataFrame with OHLC price data
            additional_data: Optional additional data for HMM and indicators

        Returns:
            DataFrame with columns for each signal generator and rows for each date
        """
        if not self.generators:
            raise ValueError("No signal generators registered")

        all_signals = {}

        for name, generator in self.generators.items():
            try:
                signals = generator.generate_signals(price_data, additional_data)
                all_signals[name] = signals
            except Exception as e:
                print(f"⚠️ Warning: Failed to generate signals for {name}: {e}")
                # Create empty signal series for failed generators
                all_signals[name] = pd.Series(
                    SignalType.HOLD.value, index=price_data.index
                )

        # Combine into DataFrame
        signals_df = pd.DataFrame(all_signals)

        # Fill any missing values with HOLD
        signals_df = signals_df.fillna(SignalType.HOLD.value)

        return signals_df
