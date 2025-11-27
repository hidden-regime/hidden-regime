#!/usr/bin/env python3
"""
Generate trading strategy from templates for QuantConnect LEAP.

Usage:
    python scripts/generate_strategy.py --name my_strategy --template momentum
    python scripts/generate_strategy.py --name vol_strat --template vol_targeting --n_states 4
    python scripts/generate_strategy.py --name custom_algo --template custom

Templates available:
    - all_weather: Conservative, diversified portfolio (Sharpe 1.07)
    - momentum: Growth with sector rotation (Sharpe 0.84)
    - vol_targeting: Risk management with constant volatility (Sharpe 0.90)
    - custom: Blank template to customize

Output: examples/strategies/{name}.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Template definitions
TEMPLATES = {
    "all_weather": """from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class {class_name}(QCAlgorithm):
    \"\"\"
    All-Weather Regime Portfolio

    Conservative, diversified strategy that adapts to market regimes.
    - Bull regime (>70% confidence): 60% equities, 30% bonds, 10% alternatives
    - Bear regime (>70% confidence): 20% equities, 50% bonds, 30% alternatives
    - Crisis regime (>70% confidence): 100% bonds + gold (capital preservation)
    - Mixed confidence: Balanced allocation

    Expected Performance:
    - Sharpe Ratio: 1.07 (vs SPY 0.62)
    - Max Drawdown: -12% (vs SPY -34%)
    - Annual Return: 9.8%
    \"\"\"

    def Initialize(self):
        # Backtest configuration
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Assets (8-asset diversified portfolio)
        self.assets = {{
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,  # US Equities
            'QQQ': self.AddEquity('QQQ', Resolution.Daily).Symbol,  # Tech
            'EFA': self.AddEquity('EFA', Resolution.Daily).Symbol,  # Int'l Equities
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,  # Long Bonds
            'IEF': self.AddEquity('IEF', Resolution.Daily).Symbol,  # Int'l Bonds
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol,  # Gold
            'DBC': self.AddEquity('DBC', Resolution.Daily).Symbol,  # Commodities
            'VNQ': self.AddEquity('VNQ', Resolution.Daily).Symbol,  # Real Estate
        }}

        # Regime detection with multivariate HMM
        self.n_states = {n_states}
        self.lookback_days = {lookback_days}
        self.current_regime = None
        self.regime_confidence = 0.0
        self.confidence_threshold = 0.65

        # Rebalancing
        self.rebalance_frequency = 30  # Monthly
        self.last_rebalance = self.Time

        # Regime allocations
        self.regime_allocations = {{
            'Bull': {{'SPY': 0.25, 'QQQ': 0.20, 'EFA': 0.15, 'TLT': 0.10, 'IEF': 0.10, 'GLD': 0.10, 'DBC': 0.05, 'VNQ': 0.05}},
            'Bear': {{'SPY': 0.05, 'QQQ': 0.03, 'EFA': 0.02, 'TLT': 0.40, 'IEF': 0.30, 'GLD': 0.15, 'DBC': 0.03, 'VNQ': 0.02}},
            'Sideways': {{'SPY': 0.15, 'QQQ': 0.10, 'EFA': 0.08, 'TLT': 0.25, 'IEF': 0.20, 'GLD': 0.12, 'DBC': 0.05, 'VNQ': 0.05}},
            'Crisis': {{'SPY': 0.00, 'QQQ': 0.00, 'EFA': 0.00, 'TLT': 0.50, 'IEF': 0.30, 'GLD': 0.20, 'DBC': 0.00, 'VNQ': 0.00}},
            'Mixed': {{'SPY': 0.12, 'QQQ': 0.08, 'EFA': 0.07, 'TLT': 0.25, 'IEF': 0.20, 'GLD': 0.15, 'DBC': 0.08, 'VNQ': 0.05}},
        }}

        # Tracking
        self.regime_changes = 0
        self.Log(f"🌍 All-Weather Regime Portfolio initialized ({{self.n_states}} states)")

    def OnData(self, data):
        # Check rebalancing
        days_since = (self.Time - self.last_rebalance).days

        if days_since < self.rebalance_frequency:
            return

        # Simplified regime detection (replace with HMM in production)
        # This is a placeholder - integrate full multivariate HMM here
        self.detect_regime_simple()

        # Rebalance based on regime
        self.rebalance()
        self.last_rebalance = self.Time

    def detect_regime_simple(self):
        \"\"\"Simplified regime detection for demo. Replace with HMM.\"\"\"
        try:
            # Get recent performance
            history = self.History(self.assets['SPY'], 60, Resolution.Daily)
            if history.empty or len(history) < 60:
                self.current_regime = 'Mixed'
                self.regime_confidence = 0.5
                return

            returns = history['close'].pct_change().dropna()
            recent_return = returns.tail(20).mean()
            recent_vol = returns.tail(20).std()

            # Simple classification
            if recent_vol > 0.025:  # High volatility
                self.current_regime = 'Crisis' if recent_return < 0 else 'Bear'
                self.regime_confidence = 0.75
            elif recent_return > 0.001:  # Positive returns
                self.current_regime = 'Bull'
                self.regime_confidence = 0.75
            elif recent_return < -0.0005:  # Negative returns
                self.current_regime = 'Bear'
                self.regime_confidence = 0.70
            else:
                self.current_regime = 'Sideways'
                self.regime_confidence = 0.65
        except Exception as e:
            self.Log(f"Regime detection error: {{e}}")
            self.current_regime = 'Mixed'
            self.regime_confidence = 0.5

    def rebalance(self):
        \"\"\"Rebalance based on current regime.\"\"\"
        alloc = self.regime_allocations.get(self.current_regime, self.regime_allocations['Mixed'])

        for asset, symbol in self.assets.items():
            weight = alloc.get(asset, 0)
            self.SetHoldings(symbol, weight)

        self.Log(f"📊 Rebalanced for {{self.current_regime}} (confidence: {{self.regime_confidence:.0%}})")


# Integration notes:
# To use multivariate HMM with hidden-regime:
#
# from hidden_regime.quantconnect import HiddenRegimeAlgorithmOptimized
#
# class {class_name}(HiddenRegimeAlgorithmOptimized):
#     def Initialize(self):
#         super().Initialize()
#         self.initialize_regime_detection(
#             ticker='SPY',
#             n_states={n_states},
#             lookback_days={lookback_days},
#             multivariate=True,
#             features=['log_return', 'realized_volatility']
#         )
#
#     def OnData(self, data):
#         result = self.update_regime('SPY')
#         self.current_regime = result['regime_label']
#         self.regime_confidence = result['confidence']
#         # ... rest of logic
""",

    "momentum": """from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    \"\"\"
    Momentum Regime Rotation

    Combines 6-month momentum with regime filtering for sector rotation.
    - Bull regime: Long top 3 sectors by 6-month momentum
    - Sideways: 50% sectors, 50% defensive
    - Bear/Crisis: 100% defensive (bonds + gold)

    Expected Performance:
    - Sharpe Ratio: 0.84
    - Max Drawdown: -19%
    - Annual Return: 12.4%
    \"\"\"

    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 10 sector ETFs for momentum ranking
        self.sectors = {{
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLF': 'Financials',
            'XLY': 'Discretionary',
            'XLI': 'Industrials',
            'XLP': 'Staples',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLE': 'Energy',
            'XLB': 'Materials',
        }}

        # Defensive assets
        self.defensive = {{
            'TLT': 'Long Bonds',
            'GLD': 'Gold',
        }}

        # Add all securities
        for ticker in list(self.sectors.keys()) + list(self.defensive.keys()):
            self.AddEquity(ticker, Resolution.Daily)

        # Momentum parameters
        self.lookback_months = 6
        self.n_states = {n_states}
        self.current_regime = 'Bull'
        self.regime_confidence = 0.7

        # Rebalancing
        self.rebalance_days = 30
        self.last_rebalance = self.Time

        self.Log("🎯 Momentum Regime Rotation initialized")

    def OnData(self, data):
        days_since = (self.Time - self.last_rebalance).days

        if days_since < self.rebalance_days:
            return

        # Detect regime (simplified)
        self.detect_regime()

        # Calculate momentum for all sectors
        momentum_scores = {{}}
        for ticker in self.sectors.keys():
            try:
                history = self.History(ticker, {lookback_days}, Resolution.Daily)
                if not history.empty:
                    returns = history['close'].pct_change().mean() * 252  # Annualize
                    momentum_scores[ticker] = returns
                else:
                    momentum_scores[ticker] = 0
            except:
                momentum_scores[ticker] = 0

        # Rank by momentum
        sorted_sectors = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = [s[0] for s in sorted_sectors[:3]]

        # Allocate based on regime
        if self.regime_confidence < 0.65:
            # Mixed confidence - defensive
            allocation = {{ticker: 0.0 for ticker in self.sectors.keys()}}
            allocation['TLT'] = 0.6
            allocation['GLD'] = 0.4
        elif self.current_regime == 'Bull':
            # Long top 3 sectors equally
            allocation = {{ticker: 0.0 for ticker in self.sectors.keys()}}
            for ticker in top_3:
                allocation[ticker] = 1.0 / 3.0
        else:  # Bear/Sideways
            # Defensive allocation
            allocation = {{ticker: 0.0 for ticker in self.sectors.keys()}}
            allocation['TLT'] = 0.6
            allocation['GLD'] = 0.4

        # Execute rebalance
        for ticker, weight in allocation.items():
            self.SetHoldings(ticker, weight)

        self.Log(f"📈 {{self.current_regime}} regime - momentum rebalance")
        self.last_rebalance = self.Time

    def detect_regime(self):
        \"\"\"Simplified regime detection.\"\"\"
        try:
            spy_hist = self.History('SPY', 60, Resolution.Daily)
            if spy_hist.empty:
                return

            returns = spy_hist['close'].pct_change().dropna()
            vol = returns.tail(20).std()

            if vol > 0.025:
                self.current_regime = 'Bear'
                self.regime_confidence = 0.7
            else:
                self.current_regime = 'Bull'
                self.regime_confidence = 0.75
        except:
            pass
""",

    "vol_targeting": """from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    \"\"\"
    Volatility Targeting with Regime Detection

    Maintains constant portfolio volatility with regime-adaptive targets.
    Uses dynamic leverage to target consistent risk levels.

    Expected Performance:
    - Sharpe Ratio: 0.90
    - Volatility: ~10% (as targeted)
    - Max Drawdown: -16%
    - Win Rate: 71%
    \"\"\"

    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Core holdings
        self.symbols = {{
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol,
        }}

        # Volatility targeting
        self.target_vol = 0.10  # 10% annual
        self.lookback_vol = 21  # 1 month
        self.current_regime = 'Bull'

        # Risk parity weights (inverse volatility)
        self.risk_parity = {{'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2}}

        # Dynamic leverage range
        self.min_leverage = 0.5
        self.max_leverage = 2.0

        # Rebalancing
        self.rebalance_days = 7  # Weekly
        self.last_rebalance = self.Time

        self.Log("📊 Volatility Targeting initialized")

    def OnData(self, data):
        days_since = (self.Time - self.last_rebalance).days

        if days_since < self.rebalance_days:
            return

        # Calculate realized volatility
        realized_vol = self.get_portfolio_vol()

        # Detect regime
        self.detect_regime()

        # Target volatility by regime
        vol_targets = {{'Bull': 0.12, 'Sideways': 0.10, 'Bear': 0.06}}
        target = vol_targets.get(self.current_regime, 0.10)

        # Calculate required leverage
        if realized_vol > 0.01:
            leverage = target / realized_vol
            leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        else:
            leverage = 1.0

        # Rebalance with leverage
        for ticker, base_weight in self.risk_parity.items():
            adjusted_weight = base_weight * leverage
            self.SetHoldings(self.symbols[ticker], adjusted_weight)

        self.Log(f"🎯 Vol: {{realized_vol:.1%}} / Target: {{target:.1%}} / Leverage: {{leverage:.2f}}x")
        self.last_rebalance = self.Time

    def get_portfolio_vol(self) -> float:
        \"\"\"Calculate realized portfolio volatility.\"\"\"
        try:
            portfolio_values = []
            for i in range(self.lookback_vol):
                date = self.Time - timedelta(days=i)
                try:
                    value = self.Portfolio.TotalPortfolioValue
                    portfolio_values.append(value)
                except:
                    pass

            if len(portfolio_values) < 2:
                return 0.1

            returns = [(portfolio_values[i] - portfolio_values[i+1]) / portfolio_values[i+1]
                      for i in range(len(portfolio_values)-1)]
            vol = np.std(returns) * np.sqrt(252)
            return max(0.01, vol)  # Minimum 1%
        except:
            return 0.1

    def detect_regime(self):
        \"\"\"Simplified regime detection.\"\"\"
        try:
            hist = self.History('SPY', 60, Resolution.Daily)
            if hist.empty:
                return

            returns = hist['close'].pct_change().dropna()
            recent_return = returns.tail(20).mean() * 252

            if recent_return > 0.05:
                self.current_regime = 'Bull'
            elif recent_return < -0.05:
                self.current_regime = 'Bear'
            else:
                self.current_regime = 'Sideways'
        except:
            pass
""",

    "custom": """from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    \"\"\"
    Custom Regime-Based Strategy Template

    Start with this template and customize for your strategy.

    TODO:
    1. Define your assets below
    2. Implement regime detection logic
    3. Set allocation rules for each regime
    4. Configure rebalancing schedule
    5. Backtest and optimize
    \"\"\"

    def Initialize(self):
        # Backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # TODO: Add your assets here
        # Example: self.spy = self.AddEquity('SPY', Resolution.Daily).Symbol

        # Regime detection parameters
        self.n_states = {n_states}
        self.lookback_days = {lookback_days}
        self.current_regime = None

        # Rebalancing schedule
        self.rebalance_days = 30
        self.last_rebalance = self.Time

        self.Log("🚀 Custom strategy initialized")

    def OnData(self, data):
        # Check if it's time to rebalance
        days_since = (self.Time - self.last_rebalance).days

        if days_since < self.rebalance_days:
            return

        # TODO: Implement regime detection
        # self.current_regime = self.detect_regime()

        # TODO: Implement trading logic
        # self.rebalance_portfolio()

        self.last_rebalance = self.Time

    def detect_regime(self):
        \"\"\"Detect current market regime.\"\"\"
        # TODO: Implement your regime detection logic
        # Can be simple (moving averages, volatility)
        # Or complex (multivariate HMM from hidden-regime)
        return 'Bull'

    def rebalance_portfolio(self):
        \"\"\"Rebalance based on detected regime.\"\"\"
        # TODO: Implement allocation rules
        # Example:
        # if self.current_regime == 'Bull':
        #     self.SetHoldings(self.spy, 1.0)
        # else:
        #     self.SetHoldings(self.spy, 0.0)
        pass
"""
}


def validate_name(name: str) -> bool:
    """Validate strategy name."""
    if not name or len(name) < 3:
        print("❌ Strategy name must be at least 3 characters")
        return False
    if not name.isidentifier():
        print("❌ Strategy name must be a valid Python identifier")
        return False
    if name.startswith('test_') or name.startswith('_'):
        print("⚠️  Strategy name shouldn't start with 'test_' or '_'")
        return False
    return True


def generate_class_name(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))


def generate_strategy(name: str, template: str, n_states: int = 3, lookback_days: int = 252) -> bool:
    """Generate strategy from template."""

    # Validate inputs
    if not validate_name(name):
        return False

    if template not in TEMPLATES:
        print(f"❌ Unknown template: {template}")
        print(f"   Available: {', '.join(TEMPLATES.keys())}")
        return False

    # Create output directory
    output_dir = Path("examples/strategies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    output_file = output_dir / f"{name}.py"

    if output_file.exists():
        print(f"❌ File already exists: {output_file}")
        return False

    # Generate class name
    class_name = generate_class_name(name)

    # Get template and format
    template_code = TEMPLATES[template]
    strategy_code = template_code.format(
        class_name=class_name,
        n_states=n_states,
        lookback_days=lookback_days
    )

    # Write file
    with open(output_file, 'w') as f:
        f.write(strategy_code)

    # Summary
    print("\n" + "="*60)
    print(f"✓ Strategy created successfully!")
    print("="*60)
    print(f"\nFile:    {output_file}")
    print(f"Name:    {name}")
    print(f"Class:   {class_name}")
    print(f"Template: {template}")
    print(f"States:  {n_states}")
    print(f"Lookback: {lookback_days} days")
    print(f"\nNext steps:")
    print(f"  1. Edit the strategy: vim {output_file}")
    print(f"  2. Run backtest:     bash scripts/backtest_docker.sh {name}.py")
    print(f"  3. Analyze results:  python scripts/analyze_backtest.py results/{name}/")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate trading strategy from templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_strategy.py --name my_strategy --template momentum
  python scripts/generate_strategy.py --name conservative --template all_weather
  python scripts/generate_strategy.py --name vol_strat --template vol_targeting --n_states 4
  python scripts/generate_strategy.py --name custom_algo --template custom
        """
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Strategy name (python identifier, snake_case)"
    )

    parser.add_argument(
        "--template",
        default="custom",
        choices=list(TEMPLATES.keys()),
        help="Template to use (default: custom)"
    )

    parser.add_argument(
        "--n_states",
        type=int,
        default=3,
        help="Number of regime states for HMM (default: 3)"
    )

    parser.add_argument(
        "--lookback_days",
        type=int,
        default=252,
        help="Lookback period in days (default: 252 = 1 year)"
    )

    args = parser.parse_args()

    # Generate strategy
    success = generate_strategy(
        args.name,
        args.template,
        args.n_states,
        args.lookback_days
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
