"""
QuantConnect Framework Integration Example

This template demonstrates full QuantConnect Framework usage:
- Alpha Model: HiddenRegimeAlphaModel for signal generation
- Portfolio Construction: Equal weighting with regime filtering
- Execution: Immediate execution
- Risk Management: Maximum drawdown per security
- Universe Selection: Dynamic based on liquidity

Author: hidden-regime
License: MIT
"""

# Note: When running in QuantConnect LEAN, uncomment these lines:
# from AlgorithmImports import *
# from hidden_regime.quantconnect import HiddenRegimeAlphaModel

# For local testing (mocking):
import sys
sys.path.insert(0, '..')

try:
    from AlgorithmImports import *
    from hidden_regime.quantconnect import HiddenRegimeAlphaModel
    QC_AVAILABLE = True
except ImportError:
    # Mock for development
    QC_AVAILABLE = False
    class QCAlgorithm:
        pass


class RegimeFrameworkStrategy(QCAlgorithm):
    """
    Full Framework algorithm using Hidden-Regime for alpha generation.

    Framework Components:
        - Universe: Top liquid stocks (QQQ components)
        - Alpha: HiddenRegimeAlphaModel
        - Portfolio: Equal weight construction
        - Execution: Immediate execution
        - Risk: Max drawdown risk management

    Strategy:
        1. Select liquid stocks from QQQ
        2. Generate regime-based insights
        3. Equal weight allocation to stocks in Bull regime
        4. Exit stocks entering Bear regime
        5. Risk management with per-security drawdown limits
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Set benchmark
        self.SetBenchmark("QQQ")

        # Universe: QQQ components (top liquid tech stocks)
        self.SetUniverseSelection(
            ManualUniverseSelectionModel([
                Symbol.Create("AAPL", SecurityType.Equity, Market.USA),
                Symbol.Create("MSFT", SecurityType.Equity, Market.USA),
                Symbol.Create("GOOGL", SecurityType.Equity, Market.USA),
                Symbol.Create("AMZN", SecurityType.Equity, Market.USA),
                Symbol.Create("NVDA", SecurityType.Equity, Market.USA),
                Symbol.Create("META", SecurityType.Equity, Market.USA),
                Symbol.Create("TSLA", SecurityType.Equity, Market.USA),
            ])
        )

        # Alpha Model: Hidden-Regime insights
        self.SetAlpha(
            HiddenRegimeAlphaModel(
                n_states=3,  # Bull, Sideways, Bear
                lookback_days=90,  # 90 days for faster adaptation
                confidence_threshold=0.65,  # Only high-confidence signals
                insight_period_days=5,  # 5-day insight validity
            )
        )

        # Portfolio Construction: Equal weight among positive insights
        self.SetPortfolioConstruction(
            EqualWeightingPortfolioConstructionModel(
                rebalance=Resolution.Daily
            )
        )

        # Execution: Immediate execution model
        self.SetExecution(ImmediateExecutionModel())

        # Risk Management: Maximum 10% drawdown per security
        self.SetRiskManagement(
            MaximumDrawdownPercentPerSecurity(0.10)
        )

        # Settings
        self.SetWarmUp(timedelta(days=90))

        self.Debug("RegimeFrameworkStrategy initialized")
        self.Debug("Framework components configured")

    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(
                f"Order filled: {orderEvent.Symbol} "
                f"{orderEvent.FillQuantity} @ ${orderEvent.FillPrice:.2f}"
            )

    def OnEndOfDay(self):
        """End of day summary."""
        # Weekly summary
        if self.Time.weekday() == 4:  # Friday
            portfolio_value = self.Portfolio.TotalPortfolioValue

            # Count holdings
            holdings = [
                (s, h.HoldingsValue / portfolio_value)
                for s, h in self.Portfolio.items()
                if h.Invested
            ]

            self.Log("="*60)
            self.Log(f"Weekly Summary - {self.Time.strftime('%Y-%m-%d')}")
            self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.Log(f"Number of Holdings: {len(holdings)}")

            if holdings:
                for symbol, weight in holdings:
                    self.Log(f"  {symbol}: {weight:.1%}")

            self.Log("="*60)


if not QC_AVAILABLE:
    # Mock for testing without QC
    class ManualUniverseSelectionModel:
        def __init__(self, symbols):
            pass

    class EqualWeightingPortfolioConstructionModel:
        def __init__(self, rebalance=None):
            pass

    class ImmediateExecutionModel:
        pass

    class MaximumDrawdownPercentPerSecurity:
        def __init__(self, max_drawdown):
            pass

    class Symbol:
        @staticmethod
        def Create(ticker, sec_type, market):
            return ticker

    class SecurityType:
        Equity = "Equity"

    class Market:
        USA = "USA"

    class Resolution:
        Daily = "Daily"

    class OrderStatus:
        Filled = "Filled"

    from datetime import timedelta
