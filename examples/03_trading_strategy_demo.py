#!/usr/bin/env python3
"""
Trading Strategy Demo

This example demonstrates how to build a practical trading strategy based on
HMM regime detection. Shows how to:

- Implement regime-based position sizing
- Create entry and exit signals
- Calculate performance metrics
- Simulate realistic trading with transaction costs
- Generate strategy performance reports

This is a practical example for traders and portfolio managers.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import using current working architecture
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.observations.financial import FinancialObservationGenerator
from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.config.analysis import FinancialAnalysisConfig

def create_trading_sample_data(n_days=500):
    """Create sample data optimized for trading strategy demonstration."""
    print("Creating sample trading data with realistic regime patterns...")
    
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # Create regime patterns that are good for trading
    regime_states = []
    current_regime = 1  # Start neutral
    days_in_regime = 0
    
    for i in range(n_days):
        regime_states.append(current_regime)
        days_in_regime += 1
        
        # Create more predictable regime patterns for trading demo
        if days_in_regime > 15:  # Minimum regime duration
            transition_prob = 0.03
            if days_in_regime > 30:
                transition_prob = 0.08
            if days_in_regime > 50:
                transition_prob = 0.15
                
            if np.random.random() < transition_prob:
                # Create somewhat predictable transitions
                if current_regime == 0:  # Bear -> Sideways (recovery)
                    current_regime = 1
                elif current_regime == 1:  # Sideways -> Bull or Bear
                    current_regime = np.random.choice([0, 2], p=[0.3, 0.7])
                else:  # Bull -> Sideways (correction)
                    current_regime = 1
                
                days_in_regime = 0
    
    # Generate prices with clear regime characteristics
    prices = [100.0]  # Starting price
    
    for i in range(1, n_days):
        regime = regime_states[i]
        
        if regime == 0:  # Bear market
            daily_return = np.random.normal(-0.0015, 0.025)  # Clear downtrend
        elif regime == 1:  # Sideways market  
            daily_return = np.random.normal(0.0001, 0.012)   # Range-bound
        else:  # Bull market
            daily_return = np.random.normal(0.0020, 0.018)   # Clear uptrend
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 5.0))  # Prevent very low prices
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 3000000, n_days)
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    
    return data, regime_states

class RegimeTradingStrategy:
    """A simple regime-based trading strategy."""
    
    def __init__(self, transaction_cost=0.001, max_position=1.0):
        self.transaction_cost = transaction_cost  # 0.1% per trade
        self.max_position = max_position
        self.position = 0.0  # Current position (-1 to 1)
        self.cash = 100000.0  # Starting cash
        self.shares = 0.0
        self.trade_log = []
        
    def get_target_position(self, regime, confidence):
        """Determine target position based on regime and confidence."""
        if regime == 0:  # Bear market
            if confidence > 0.7:
                return -0.8 * self.max_position  # Strong short
            elif confidence > 0.5:
                return -0.4 * self.max_position  # Moderate short
            else:
                return 0.0  # Neutral if low confidence
                
        elif regime == 1:  # Sideways market
            return 0.0  # Stay in cash
            
        else:  # Bull market
            if confidence > 0.7:
                return 0.9 * self.max_position   # Strong long
            elif confidence > 0.5:
                return 0.5 * self.max_position   # Moderate long
            else:
                return 0.2 * self.max_position   # Small long position
    
    def execute_trade(self, date, price, target_position):
        """Execute a trade to reach target position."""
        current_value = self.cash + self.shares * price
        target_shares = (target_position * current_value) / price
        
        shares_to_trade = target_shares - self.shares
        
        if abs(shares_to_trade) > 0.01:  # Only trade if meaningful change
            trade_value = abs(shares_to_trade * price)
            cost = trade_value * self.transaction_cost
            
            self.cash -= shares_to_trade * price + cost
            self.shares += shares_to_trade
            
            # Log the trade
            self.trade_log.append({
                'date': date,
                'price': price,
                'shares_traded': shares_to_trade,
                'new_position': target_position,
                'cost': cost,
                'portfolio_value': self.cash + self.shares * price
            })
            
            return True
        return False
    
    def get_portfolio_value(self, price):
        """Get current portfolio value."""
        return self.cash + self.shares * price
    
    def get_current_position(self, price):
        """Get current position as fraction of portfolio value."""
        portfolio_value = self.get_portfolio_value(price)
        if portfolio_value > 0:
            return (self.shares * price) / portfolio_value
        return 0.0

def backtest_strategy(data, regime_predictions, strategy):
    """Backtest the regime trading strategy."""
    portfolio_values = []
    positions = []
    
    for i, (date, row) in enumerate(data.iterrows()):
        price = row['close']
        
        if i < len(regime_predictions):
            regime = regime_predictions.iloc[i]
            # Use a fixed confidence for demo (in practice, use model confidence)
            confidence = 0.8  # High confidence for demo purposes
            
            target_position = strategy.get_target_position(regime, confidence)
            strategy.execute_trade(date, price, target_position)
        
        portfolio_value = strategy.get_portfolio_value(price)
        current_position = strategy.get_current_position(price)
        
        portfolio_values.append(portfolio_value)
        positions.append(current_position)
    
    return pd.Series(portfolio_values, index=data.index), pd.Series(positions, index=data.index)

def calculate_performance_metrics(portfolio_values, benchmark_values):
    """Calculate comprehensive performance metrics."""
    # Convert to returns
    portfolio_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()
    
    # Align the series
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    portfolio_returns = aligned_data['portfolio']
    benchmark_returns = aligned_data['benchmark']
    
    # Calculate metrics
    total_return_portfolio = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    total_return_benchmark = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
    
    annual_return_portfolio = portfolio_returns.mean() * 252
    annual_return_benchmark = benchmark_returns.mean() * 252
    
    volatility_portfolio = portfolio_returns.std() * np.sqrt(252)
    volatility_benchmark = benchmark_returns.std() * np.sqrt(252)
    
    sharpe_portfolio = annual_return_portfolio / volatility_portfolio if volatility_portfolio > 0 else 0
    sharpe_benchmark = annual_return_benchmark / volatility_benchmark if volatility_benchmark > 0 else 0
    
    # Calculate max drawdown
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate excess return
    excess_returns = portfolio_returns - benchmark_returns
    alpha = excess_returns.mean() * 252
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0
    
    return {
        'total_return_portfolio': total_return_portfolio,
        'total_return_benchmark': total_return_benchmark,
        'annual_return_portfolio': annual_return_portfolio,
        'annual_return_benchmark': annual_return_benchmark,
        'volatility_portfolio': volatility_portfolio,
        'volatility_benchmark': volatility_benchmark,
        'sharpe_portfolio': sharpe_portfolio,
        'sharpe_benchmark': sharpe_benchmark,
        'max_drawdown': max_drawdown,
        'alpha': alpha,
        'information_ratio': information_ratio,
        'total_trades': len([t for t in excess_returns if abs(t) > 0])
    }

def main():
    """Main trading strategy demonstration."""
    print("📊 Trading Strategy Demo")
    print("=" * 40)
    
    # Generate or load data
    try:
        # Try real data first
        print("\\n📈 Attempting to load real market data...")
        data_loader = FinancialDataLoader(FinancialDataConfig(
            ticker="SPY",
            start_date="2022-01-01",
            end_date="2024-01-01"
        ))
        
        raw_data = data_loader.update()
        
        if raw_data.empty:
            print("❌ No real data available, using sample data")
            raw_data, true_regimes = create_trading_sample_data()
            ticker = "DEMO"
        else:
            print(f"✅ Loaded {len(raw_data)} days of SPY data")
            ticker = "SPY"
            true_regimes = None
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("📝 Using sample trading data")
        raw_data, true_regimes = create_trading_sample_data()
        ticker = "DEMO"
    
    # Perform regime analysis
    print("\\n🤖 Training HMM for regime detection...")
    
    # Create observations
    observation_config = FinancialObservationConfig(generators=['log_return'])
    observation_component = FinancialObservationGenerator(observation_config)
    observations = observation_component.update(raw_data)
    
    # Train HMM model
    model_config = HMMConfig(n_states=3, random_seed=42)
    hmm_model = HiddenMarkovModel(model_config)
    model_output = hmm_model.update(observations)
    
    print(f"✅ Model trained successfully")
    print(f"📊 Generated regime predictions for {len(model_output)} days")
    
    # Run trading strategy
    print("\\n💰 Running regime-based trading strategy...")
    
    strategy = RegimeTradingStrategy(transaction_cost=0.001, max_position=1.0)
    portfolio_values, positions = backtest_strategy(raw_data, model_output['predicted_state'], strategy)
    
    # Calculate buy-and-hold benchmark
    initial_value = 100000.0
    benchmark_values = initial_value * (raw_data['close'] / raw_data['close'].iloc[0])
    
    # Calculate performance
    print("\\n📊 Calculating performance metrics...")
    metrics = calculate_performance_metrics(portfolio_values, benchmark_values)
    
    # Generate trading report
    print("\\n📝 Generating trading strategy report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    report_filename = os.path.join(output_dir, 'reports', f"trading_strategy_report_{ticker}_{timestamp}.md")
    
    with open(report_filename, 'w') as f:
        f.write("# Regime-Based Trading Strategy Report\\n\\n")
        f.write(f"**Asset**: {ticker}\\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"**Analysis Period**: {len(raw_data)} trading days\\n\\n")
        
        f.write("## Strategy Overview\\n\\n")
        f.write("This strategy uses Hidden Markov Model regime detection to make trading decisions:\\n\\n")
        f.write("- **Bear Market**: Short position (up to -80% of portfolio)\\n")
        f.write("- **Sideways Market**: Cash position (0% equity exposure)\\n")
        f.write("- **Bull Market**: Long position (up to 90% of portfolio)\\n\\n")
        f.write(f"- **Transaction Cost**: {strategy.transaction_cost:.1%} per trade\\n")
        f.write(f"- **Maximum Position**: {strategy.max_position:.0%} of portfolio\\n\\n")
        
        f.write("## Performance Summary\\n\\n")
        f.write("### Returns\\n")
        f.write(f"- **Strategy Total Return**: {metrics['total_return_portfolio']:.2%}\\n")
        f.write(f"- **Buy & Hold Total Return**: {metrics['total_return_benchmark']:.2%}\\n")
        f.write(f"- **Excess Return**: {metrics['total_return_portfolio'] - metrics['total_return_benchmark']:.2%}\\n\\n")
        
        f.write(f"- **Strategy Annual Return**: {metrics['annual_return_portfolio']:.2%}\\n")
        f.write(f"- **Buy & Hold Annual Return**: {metrics['annual_return_benchmark']:.2%}\\n")
        f.write(f"- **Alpha (Excess Annual Return)**: {metrics['alpha']:.2%}\\n\\n")
        
        f.write("### Risk Metrics\\n")
        f.write(f"- **Strategy Volatility**: {metrics['volatility_portfolio']:.2%}\\n")
        f.write(f"- **Buy & Hold Volatility**: {metrics['volatility_benchmark']:.2%}\\n")
        f.write(f"- **Maximum Drawdown**: {metrics['max_drawdown']:.2%}\\n\\n")
        
        f.write("### Risk-Adjusted Performance\\n")
        f.write(f"- **Strategy Sharpe Ratio**: {metrics['sharpe_portfolio']:.3f}\\n")
        f.write(f"- **Buy & Hold Sharpe Ratio**: {metrics['sharpe_benchmark']:.3f}\\n")
        f.write(f"- **Information Ratio**: {metrics['information_ratio']:.3f}\\n\\n")
        
        f.write("## Trade Analysis\\n\\n")
        f.write(f"- **Total Trades**: {len(strategy.trade_log)}\\n")
        if strategy.trade_log:
            total_costs = sum(trade['cost'] for trade in strategy.trade_log)
            f.write(f"- **Total Transaction Costs**: ${total_costs:,.2f}\\n")
            f.write(f"- **Average Trade Size**: ${np.mean([abs(t['shares_traded'] * t['price']) for t in strategy.trade_log]):,.2f}\\n\\n")
        
        f.write("## Regime Distribution\\n\\n")
        regime_counts = model_output['predicted_state'].value_counts().sort_index()
        regime_names = ['Bear Market', 'Sideways Market', 'Bull Market']
        total_days = len(model_output)
        
        for regime, count in regime_counts.items():
            percentage = count / total_days * 100
            f.write(f"- **{regime_names[regime]}**: {count} days ({percentage:.1f}%)\\n")
        
        f.write("\\n## Strategy Interpretation\\n\\n")
        
        if metrics['total_return_portfolio'] > metrics['total_return_benchmark']:
            f.write("✅ **Strategy outperformed** buy-and-hold benchmark.\\n\\n")
        else:
            f.write("❌ **Strategy underperformed** buy-and-hold benchmark.\\n\\n")
        
        if metrics['sharpe_portfolio'] > metrics['sharpe_benchmark']:
            f.write("✅ **Better risk-adjusted returns** than benchmark.\\n\\n")
        else:
            f.write("❌ **Worse risk-adjusted returns** than benchmark.\\n\\n")
        
        if metrics['max_drawdown'] > -0.1:
            f.write("✅ **Low maximum drawdown** - good downside protection.\\n\\n")
        elif metrics['max_drawdown'] > -0.2:
            f.write("⚠️ **Moderate maximum drawdown** - acceptable risk level.\\n\\n")
        else:
            f.write("❌ **High maximum drawdown** - significant downside risk.\\n\\n")
    
    print(f"✅ Report saved as: {report_filename}")
    
    # Create performance visualization
    print("\\n🎨 Creating performance visualization...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Portfolio value comparison
        ax1.plot(portfolio_values.index, portfolio_values.values, 
                linewidth=2, label='Regime Strategy', color='blue')
        ax1.plot(benchmark_values.index, benchmark_values.values, 
                linewidth=2, label='Buy & Hold', color='gray', alpha=0.7)
        ax1.set_title('Portfolio Value Comparison', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position over time
        ax2.plot(positions.index, positions.values, linewidth=1.5, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Strategy Position Over Time', fontweight='bold')
        ax2.set_ylabel('Position (% of Portfolio)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime detection with price
        ax3_twin = ax3.twinx()
        ax3_twin.plot(raw_data.index, raw_data['close'], color='black', alpha=0.5, linewidth=1)
        ax3_twin.set_ylabel('Price ($)', color='gray')
        
        regime_colors = ['red', 'orange', 'green']
        regime_names = ['Bear', 'Sideways', 'Bull']
        
        for regime in [0, 1, 2]:
            mask = model_output['predicted_state'] == regime
            if mask.sum() > 0:
                ax3.scatter(model_output.index[mask], [regime] * mask.sum(),
                           c=regime_colors[regime], alpha=0.8, s=8,
                           label=f"{regime_names[regime]}")
        
        ax3.set_title('Regime Detection', fontweight='bold')
        ax3.set_ylabel('Market Regime')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Bear', 'Sideways', 'Bull'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown comparison
        portfolio_dd = (portfolio_values / portfolio_values.expanding().max() - 1) * 100
        benchmark_dd = (benchmark_values / benchmark_values.expanding().max() - 1) * 100
        
        ax4.fill_between(portfolio_dd.index, portfolio_dd.values, 0, 
                        alpha=0.7, color='blue', label='Strategy Drawdown')
        ax4.fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                        alpha=0.5, color='gray', label='Benchmark Drawdown')
        ax4.set_title('Drawdown Comparison', fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        plot_filename = os.path.join(output_dir, 'plots', f'trading_strategy_performance_{ticker}_{timestamp}.png')
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Visualization saved as: {plot_filename}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        plot_filename = None
    
    # Display summary results
    print(f"\\n📊 Trading Strategy Results for {ticker}:")
    print("=" * 50)
    
    print(f"Strategy Performance:")
    print(f"  Total Return: {metrics['total_return_portfolio']:.2%}")
    print(f"  Annual Return: {metrics['annual_return_portfolio']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_portfolio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print(f"\\nBenchmark (Buy & Hold):")
    print(f"  Total Return: {metrics['total_return_benchmark']:.2%}")
    print(f"  Annual Return: {metrics['annual_return_benchmark']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_benchmark']:.3f}")
    
    print(f"\\nStrategy vs Benchmark:")
    excess_return = metrics['total_return_portfolio'] - metrics['total_return_benchmark']
    print(f"  Excess Return: {excess_return:.2%}")
    print(f"  Alpha: {metrics['alpha']:.2%}")
    print(f"  Information Ratio: {metrics['information_ratio']:.3f}")
    
    print(f"\\nTrade Statistics:")
    print(f"  Total Trades: {len(strategy.trade_log)}")
    if strategy.trade_log:
        total_costs = sum(trade['cost'] for trade in strategy.trade_log)
        print(f"  Transaction Costs: ${total_costs:,.2f}")
    
    print(f"\\n🎉 Trading Strategy Demo Complete!")
    print(f"📁 Generated files:")
    print(f"   • Report: {report_filename}")
    if plot_filename:
        print(f"   • Performance Chart: {plot_filename}")
    
    return {
        'metrics': metrics,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'strategy': strategy,
        'regime_predictions': model_output,
        'raw_data': raw_data
    }

if __name__ == "__main__":
    try:
        results = main()
        print("\\n" + "="*50)
        print("🚀 Trading Strategy Demo: SUCCESS")
        print("="*50)
    except Exception as e:
        print(f"\\n❌ Error running trading demo: {e}")
        import traceback
        traceback.print_exc()