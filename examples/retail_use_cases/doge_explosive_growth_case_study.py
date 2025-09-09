"""
DOGE Explosive Growth Case Study: Crypto Trading with Hidden Regime Models

This case study demonstrates how Hidden Regime's online learning capabilities
handle extreme cryptocurrency volatility and explosive growth scenarios using
Dogecoin (DOGE) during the 2020-2021 bull run.

Trading Scenario:
- Train initial model on 2020 data (pre-explosion baseline)
- Starting January 1, 2021, perform daily online updates
- Each day: update model → predict next day → execute trade → record performance
- Show how regime detection captures meme stock/crypto dynamics

Key Learning Objectives:
1. Handling extreme volatility in crypto markets (+10,000% gains)
2. Regime transition detection during parabolic moves
3. Managing position sizing during explosive growth
4. Performance vs. HODL strategy during crypto bull run
5. Risk management during euphoric bubble periods
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hidden_regime import HiddenMarkovModel, HMMConfig
from hidden_regime.models.online_hmm import OnlineHMM, OnlineHMMConfig
from retail_trading_framework import RetailTradingSimulator, RegimeTrader, TradingConfig
from performance_analysis import RetailPerformanceAnalyzer

warnings.filterwarnings('ignore')


class DOGEExplosiveGrowthCaseStudy:
    """DOGE explosive growth case study with online learning for crypto trading"""
    
    def __init__(self):
        self.ticker = "DOGE-USD"
        self.training_start = "2020-01-01"
        self.training_end = "2020-12-31"
        self.trading_start = "2021-01-01"
        self.trading_end = "2021-12-31"
        
        # Crypto trading parameters (more aggressive for meme coins)
        self.initial_capital = 50000   # $50K starting capital
        self.max_position_size = 0.9   # Max 90% allocation (crypto risk tolerance)
        self.stop_loss_pct = 0.15      # 15% stop loss (wider for crypto volatility)
        
        # Results storage
        self.results = {
            'dates': [],
            'returns': [],
            'prices': [],
            'regime_probs': [],
            'regime_predictions': [],
            'positions': [],
            'portfolio_values': [],
            'model_confidence': [],
            'regime_names': [],
            'volatility': []
        }
        
    def load_data(self):
        """Load DOGE data for training and trading periods"""
        print(f"🐕 Loading {self.ticker} data...")
        
        # Extended date range to ensure we have enough data
        start_date = "2019-12-01"  # Buffer before 2020
        end_date = "2022-01-31"    # Buffer after 2021
        
        try:
            # DOGE-USD ticker for crypto
            crypto = yf.Ticker(self.ticker)
            data = crypto.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data retrieved for {self.ticker}")
            
            # Calculate log returns and rolling volatility
            data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['rolling_vol'] = data['log_return'].rolling(window=30).std()
            data = data.dropna()
            
            print(f"   🚀 Retrieved {len(data)} days of data")
            print(f"   📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
            print(f"   💰 Price range: ${data['Close'].min():.4f} to ${data['Close'].max():.2f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def prepare_training_data(self, data):
        """Extract and validate training data"""
        training_data = data[self.training_start:self.training_end].copy()
        
        if training_data.empty:
            raise ValueError("No training data found for specified period")
        
        returns = training_data['log_return'].values
        
        print(f"📚 Training period: {training_data.index.min().date()} to {training_data.index.max().date()}")
        print(f"   📊 Training samples: {len(returns)}")
        print(f"   📈 Mean return: {returns.mean():.4f}")
        print(f"   📉 Volatility: {returns.std():.4f}")
        print(f"   🎢 Min/Max return: {returns.min():.4f} / {returns.max():.4f}")
        print(f"   🎯 Sharpe ratio: {returns.mean() / returns.std() * np.sqrt(365):.2f}")
        
        return returns, training_data
    
    def initialize_models(self, training_returns):
        """Initialize models optimized for crypto volatility"""
        print("\n🤖 Initializing crypto-optimized models...")
        
        # Use 4-state model for crypto (includes euphoria state)
        config = HMMConfig.for_standardized_regimes(
            regime_type='4_state',  # Bear, Sideways, Bull, Euphoria
            conservative=False      # Aggressive for crypto
        )
        
        self.batch_hmm = HiddenMarkovModel(config=config)
        
        # Online HMM with faster adaptation for crypto volatility
        online_config = OnlineHMMConfig(
            forgetting_factor=0.95,   # Faster forgetting for volatile crypto
            adaptation_rate=0.08,     # Higher learning rate
            min_observations_for_update=20,  # More frequent updates
            parameter_smoothing=True,
            smoothing_weight=0.7      # Less smoothing for rapid adaptation
        )
        
        self.online_hmm = OnlineHMM(config=config, online_config=online_config)
        
        # Train initial models
        print("   🔄 Training batch HMM on 2020 data...")
        self.batch_hmm.fit(training_returns, verbose=False)
        
        print("   🔄 Training and initializing online HMM...")
        self.online_hmm.fit(training_returns, verbose=False)
        
        # Display initial regime characteristics
        self._display_initial_regimes()
        
        print("   ✅ Models initialized successfully")
    
    def _display_initial_regimes(self):
        """Display initial regime characteristics from batch model"""
        print("\n📋 Initial Regime Characteristics (from 2020 data):")
        
        if hasattr(self.batch_hmm, '_state_standardizer') and self.batch_hmm._state_standardizer:
            config = self.batch_hmm._state_standardizer.current_config
            if config:
                for i in range(self.batch_hmm.n_states):
                    mean, std = self.batch_hmm.emission_params_[i]
                    
                    # Get regime name
                    regime_name = f"State {i}"
                    if hasattr(self.batch_hmm, '_state_mapping') and self.batch_hmm._state_mapping:
                        mapped = self.batch_hmm._state_mapping.get(i, i)
                        if isinstance(mapped, str):
                            regime_name = mapped
                        elif isinstance(mapped, (int, np.integer)) and int(mapped) < len(config.state_names):
                            regime_name = config.state_names[int(mapped)]
                    
                    annual_return = mean * 365  # Crypto trades 365 days
                    annual_vol = std * np.sqrt(365)
                    
                    print(f"   🎯 {regime_name}:")
                    print(f"      Daily: μ={mean:.4f}, σ={std:.4f}")
                    print(f"      Annual: {annual_return:.1%} ± {annual_vol:.1%}")
    
    def run_online_trading_simulation(self, data):
        """Run day-by-day crypto trading simulation"""
        print(f"\n🚀 Starting crypto trading simulation...")
        print(f"   📅 Period: {self.trading_start} to {self.trading_end}")
        print(f"   💰 Initial capital: ${self.initial_capital:,.2f}")
        
        # Initialize crypto trading simulation
        trading_config = TradingConfig(
            initial_capital=self.initial_capital,
            max_position_size=self.max_position_size,
            commission_rate=0.0025,     # 0.25% crypto exchange fees
            daily_risk_limit=self.stop_loss_pct,
            max_drawdown_limit=0.4,     # 40% max drawdown for crypto
            confidence_threshold=0.5    # Lower threshold for crypto volatility
        )
        simulator = RetailTradingSimulator(config=trading_config)
        
        # Get trading period data
        trading_data = data[self.trading_start:self.trading_end].copy()
        
        if trading_data.empty:
            raise ValueError("No trading data found for specified period")
        
        print(f"   📊 Trading days: {len(trading_data)}")
        
        # Daily simulation loop
        portfolio_value = self.initial_capital
        
        for i, (date, row) in enumerate(trading_data.iterrows()):
            current_return = row['log_return']
            current_price = row['Close']
            current_vol = row['rolling_vol'] if 'rolling_vol' in row else 0.1
            
            # Update online model with new observation and get regime analysis
            regime_info = self.online_hmm.add_observation(current_return)
            
            # Get position recommendation (crypto-optimized)
            position_pct = self._get_crypto_position_recommendation(
                regime_info, current_vol, current_price
            )
            
            # Execute trade through simulator
            regime_str = regime_info.get('regime_name', f"State {regime_info.get('regime', 0)}")
            confidence = regime_info.get('confidence', 0.5)
            
            trade_result = simulator.execute_trade(
                target_position=position_pct,
                current_price=current_price,
                timestamp=date,
                regime=regime_str,
                confidence=confidence
            )
            
            # Update portfolio value
            portfolio_update = simulator.update_portfolio(current_price, date)
            
            # Record results
            self.results['dates'].append(date)
            self.results['returns'].append(current_return)
            self.results['prices'].append(current_price)
            self.results['regime_probs'].append(regime_info.get('state_probabilities', [0.25, 0.25, 0.25, 0.25]))
            self.results['regime_predictions'].append(regime_info.get('regime', 0))
            self.results['positions'].append(position_pct)
            self.results['portfolio_values'].append(portfolio_update['total_capital'])
            self.results['model_confidence'].append(confidence)
            self.results['regime_names'].append(regime_str)
            self.results['volatility'].append(current_vol)
            
            # Progress update (more frequent for crypto excitement)
            if (i + 1) % 30 == 0 or (i + 1) == len(trading_data) or current_return > 0.5 or current_return < -0.5:
                progress = (i + 1) / len(trading_data) * 100
                price_change = "📈" if current_return > 0 else "📉"
                
                print(f"   {price_change} Day {i+1:3d} ({progress:5.1f}%): {date.strftime('%Y-%m-%d')} | "
                      f"${current_price:8.4f} ({current_return:+6.1%}) | "
                      f"{regime_str:8s} ({confidence:.1%}) | "
                      f"Pos: {position_pct:5.1%} | "
                      f"Portfolio: ${portfolio_update['total_capital']:8,.0f}")
        
        print(f"   ✅ Simulation completed!")
        
        return simulator
    
    def _get_crypto_position_recommendation(self, regime_info, volatility, current_price):
        """Get crypto-optimized position size recommendation"""
        regime = regime_info.get('regime', 0)
        confidence = regime_info.get('confidence', 0.5)
        regime_name = regime_info.get('regime_name', f"State {regime}")
        
        # Base position sizes by regime type (crypto-optimized)
        if 'bear' in regime_name.lower() or 'crisis' in regime_name.lower():
            base_position = 0.1   # Small long position (crypto rarely goes to zero)
        elif 'sideways' in regime_name.lower():
            base_position = 0.4   # Moderate position during consolidation
        elif 'bull' in regime_name.lower():
            base_position = 0.8   # Large position during bull runs
        else:  # Euphoria or unknown state
            base_position = 0.6   # Moderate position (bubble risk)
        
        # Adjust by confidence
        confidence_adjusted = base_position * (0.5 + 0.5 * confidence)
        
        # Volatility adjustment (reduce position during extreme volatility)
        vol_adjustment = 1.0
        if volatility > 0.15:  # Very high volatility
            vol_adjustment = 0.7
        elif volatility > 0.25:  # Extreme volatility
            vol_adjustment = 0.5
        
        position_adjusted = confidence_adjusted * vol_adjustment
        
        # Apply maximum position limits
        return np.clip(position_adjusted, 0.0, self.max_position_size)
    
    def analyze_crypto_performance(self, simulator):
        """Comprehensive crypto performance analysis"""
        print(f"\n📊 CRYPTO PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame({
            'date': self.results['dates'],
            'return': self.results['returns'],
            'price': self.results['prices'],
            'position': self.results['positions'],
            'portfolio_value': self.results['portfolio_values'],
            'regime': self.results['regime_predictions'],
            'regime_name': self.results['regime_names'],
            'confidence': self.results['model_confidence'],
            'volatility': self.results['volatility']
        })
        
        # Calculate performance metrics
        initial_value = self.initial_capital
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # DOGE price performance
        initial_price = results_df['price'].iloc[0]
        final_price = results_df['price'].iloc[-1]
        doge_return = (final_price - initial_price) / initial_price
        
        # Daily portfolio returns
        portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
        
        # Performance metrics
        annual_return = portfolio_returns.mean() * 365
        annual_vol = portfolio_returns.std() * np.sqrt(365)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Crypto-specific metrics
        max_daily_gain = portfolio_returns.max()
        max_daily_loss = portfolio_returns.min()
        days_over_10pct = len(portfolio_returns[portfolio_returns > 0.10])
        days_under_minus_10pct = len(portfolio_returns[portfolio_returns < -0.10])
        
        print(f"💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Capital:    ${initial_value:10,.2f}")
        print(f"   Final Value:        ${final_value:10,.2f}")
        print(f"   Total Return:       {total_return:10.1%}")
        print(f"   Annual Return:      {annual_return:10.1%}")
        print(f"   Annual Volatility:  {annual_vol:10.1%}")
        print(f"   Sharpe Ratio:       {sharpe_ratio:10.2f}")
        print(f"   Maximum Drawdown:   {max_drawdown:10.1%}")
        
        print(f"\n🚀 CRYPTO-SPECIFIC METRICS:")
        print(f"   Max Daily Gain:     {max_daily_gain:10.1%}")
        print(f"   Max Daily Loss:     {max_daily_loss:10.1%}")
        print(f"   Days >+10%:         {days_over_10pct:10.0f}")
        print(f"   Days <-10%:         {days_under_minus_10pct:10.0f}")
        
        print(f"\n📈 DOGE PRICE PERFORMANCE:")
        print(f"   Initial Price:      ${initial_price:10.6f}")
        print(f"   Final Price:        ${final_price:10.4f}")
        print(f"   DOGE HODL Return:   {doge_return:10.1%}")
        print(f"   Excess Return:      {total_return - doge_return:10.1%}")
        
        # Regime analysis
        print(f"\n🎯 REGIME ANALYSIS:")
        regime_stats = results_df.groupby('regime_name').agg({
            'date': 'count',
            'confidence': 'mean',
            'position': 'mean',
            'return': ['mean', 'std'],
            'price': ['min', 'max']
        }).round(4)
        
        for regime_name in regime_stats.index:
            days = regime_stats.loc[regime_name, ('date', 'count')]
            avg_conf = regime_stats.loc[regime_name, ('confidence', 'mean')]
            avg_pos = regime_stats.loc[regime_name, ('position', 'mean')]
            avg_ret = regime_stats.loc[regime_name, ('return', 'mean')]
            vol = regime_stats.loc[regime_name, ('return', 'std')]
            min_price = regime_stats.loc[regime_name, ('price', 'min')]
            max_price = regime_stats.loc[regime_name, ('price', 'max')]
            
            print(f"   {regime_name:12s}: {days:3.0f} days ({days/len(results_df):.1%}) | "
                  f"Conf: {avg_conf:.1%} | Pos: {avg_pos:+.1%} | "
                  f"Ret: {avg_ret:+.4f} ± {vol:.4f} | "
                  f"Price: ${min_price:.4f}-${max_price:.4f}")
        
        # Key crypto events analysis
        self._analyze_crypto_key_events(results_df)
        
        return results_df
    
    def _analyze_crypto_key_events(self, results_df):
        """Analyze performance during key crypto events"""
        print(f"\n🐕 DOGE KEY EVENTS ANALYSIS:")
        
        # Define key DOGE events in 2021
        key_events = [
            ("GameStop Rally", "2021-01-25", "2021-02-08"),
            ("Elon Tweet Spree", "2021-02-04", "2021-02-14"),
            ("First Major Pump", "2021-04-01", "2021-04-20"),
            ("Doge Day (4/20)", "2021-04-16", "2021-04-24"),
            ("SNL Appearance", "2021-05-04", "2021-05-12"),
            ("Summer Correction", "2021-05-20", "2021-07-20"),
            ("Fall Recovery", "2021-10-01", "2021-10-31"),
            ("November Spike", "2021-11-01", "2021-11-15")
        ]
        
        for event_name, start_date, end_date in key_events:
            event_data = results_df[
                (results_df['date'] >= start_date) & 
                (results_df['date'] <= end_date)
            ]
            
            if not event_data.empty:
                start_value = event_data['portfolio_value'].iloc[0]
                end_value = event_data['portfolio_value'].iloc[-1]
                event_return = (end_value - start_value) / start_value
                
                start_price = event_data['price'].iloc[0]
                end_price = event_data['price'].iloc[-1]
                price_return = (end_price - start_price) / start_price
                
                dominant_regime = event_data['regime_name'].mode().iloc[0] if not event_data['regime_name'].mode().empty else "Unknown"
                avg_confidence = event_data['confidence'].mean()
                avg_position = event_data['position'].mean()
                
                print(f"   {event_name:15s}: Portfolio {event_return:+7.1%} | DOGE {price_return:+7.1%} | "
                      f"{dominant_regime:8s} ({avg_confidence:.1%}) | "
                      f"Pos: {avg_position:+5.1%}")
    
    def create_crypto_visualizations(self, results_df):
        """Create comprehensive crypto visualization dashboard"""
        print(f"\n📊 Creating crypto visualization dashboard...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'{self.ticker} Explosive Growth Case Study: Online Learning Performance', 
                     fontsize=16, fontweight='bold')
        
        # 1. Portfolio vs DOGE performance
        ax1 = axes[0, 0]
        
        # Portfolio value (normalized)
        portfolio_norm = np.array(results_df['portfolio_value']) / self.initial_capital
        ax1.plot(results_df['date'], portfolio_norm, label='HMM Strategy', linewidth=2, color='blue')
        
        # DOGE HODL performance (normalized)
        doge_norm = np.array(results_df['price']) / results_df['price'].iloc[0]
        ax1.plot(results_df['date'], doge_norm, label='DOGE HODL', linewidth=2, color='orange')
        
        ax1.set_title('Portfolio vs HODL Performance')
        ax1.set_ylabel('Normalized Value (Log Scale)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. DOGE price with regime overlay
        ax2 = axes[0, 1]
        
        # Price chart
        ax2.plot(results_df['date'], results_df['price'], linewidth=1, color='black', alpha=0.7)
        
        # Color background by regime
        regime_colors = {'State 0': 'red', 'State 1': 'orange', 'State 2': 'green', 'State 3': 'purple'}
        prev_date = None
        prev_regime = None
        
        for i, (date, regime) in enumerate(zip(results_df['date'], results_df['regime_name'])):
            if prev_date is not None and prev_regime != regime:
                color = regime_colors.get(prev_regime, 'gray')
                ax2.axvspan(prev_date, date, alpha=0.2, color=color)
            prev_date = date
            prev_regime = regime
        
        ax2.set_title('DOGE Price with Regime Overlay')
        ax2.set_ylabel('DOGE Price ($)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime probabilities over time
        ax3 = axes[1, 0]
        
        regime_probs = np.array(self.results['regime_probs'])
        if regime_probs.shape[1] >= 4:
            bottom = np.zeros(len(results_df))
            colors = ['red', 'orange', 'green', 'purple']
            labels = ['State 0', 'State 1', 'State 2', 'State 3']
            
            for i in range(min(4, regime_probs.shape[1])):
                ax3.fill_between(
                    results_df['date'], 
                    bottom, 
                    bottom + regime_probs[:, i],
                    label=labels[i],
                    color=colors[i],
                    alpha=0.7
                )
                bottom += regime_probs[:, i]
        
        ax3.set_title('Regime Probability Evolution')
        ax3.set_ylabel('Regime Probability')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Position sizing and volatility
        ax4 = axes[1, 1]
        
        # Position sizing
        ax4_twin = ax4.twinx()
        ax4.plot(results_df['date'], results_df['position'], color='blue', alpha=0.7, label='Position Size')
        ax4_twin.plot(results_df['date'], results_df['volatility'], color='red', alpha=0.7, label='Volatility')
        
        ax4.set_title('Position Sizing vs Volatility')
        ax4.set_ylabel('Position Size', color='blue')
        ax4_twin.set_ylabel('30-Day Volatility', color='red')
        ax4.grid(True, alpha=0.3)
        
        # 5. Daily returns distribution
        ax5 = axes[2, 0]
        
        portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
        doge_returns = results_df['price'].pct_change().dropna()
        
        ax5.hist(portfolio_returns, bins=50, alpha=0.7, label='Portfolio Returns', color='blue')
        ax5.hist(doge_returns, bins=50, alpha=0.7, label='DOGE Returns', color='orange')
        ax5.axvline(portfolio_returns.mean(), color='blue', linestyle='--', alpha=0.7)
        ax5.axvline(doge_returns.mean(), color='orange', linestyle='--', alpha=0.7)
        
        ax5.set_title('Daily Returns Distribution')
        ax5.set_xlabel('Daily Return')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown analysis
        ax6 = axes[2, 1]
        
        # Portfolio drawdown
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        portfolio_rolling_max = portfolio_cumulative.expanding().max()
        portfolio_drawdown = (portfolio_cumulative - portfolio_rolling_max) / portfolio_rolling_max
        
        # DOGE drawdown
        doge_cumulative = (1 + doge_returns).cumprod()
        doge_rolling_max = doge_cumulative.expanding().max()
        doge_drawdown = (doge_cumulative - doge_rolling_max) / doge_rolling_max
        
        ax6.fill_between(results_df['date'][1:], portfolio_drawdown, 0, alpha=0.7, color='blue', label='Portfolio DD')
        ax6.fill_between(results_df['date'][1:], doge_drawdown, 0, alpha=0.7, color='orange', label='DOGE DD')
        
        ax6.set_title('Drawdown Comparison')
        ax6.set_ylabel('Drawdown')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = Path(__file__).parent / 'visualizations'
        viz_path.mkdir(exist_ok=True)
        
        plt.savefig(viz_path / 'doge_explosive_growth_case_study.png', dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {viz_path}/doge_explosive_growth_case_study.png")
        
        plt.show()
    
    def generate_crypto_insights(self, results_df):
        """Generate key insights for crypto traders"""
        print(f"\n💡 KEY CRYPTO TRADING INSIGHTS")
        print("=" * 60)
        
        # Extreme moves analysis
        portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
        extreme_gains = portfolio_returns[portfolio_returns > 0.20]
        extreme_losses = portfolio_returns[portfolio_returns < -0.20]
        
        print(f"🎢 Extreme Movement Analysis:")
        print(f"   Days with >20% gains: {len(extreme_gains)}")
        print(f"   Days with >20% losses: {len(extreme_losses)}")
        print(f"   Largest daily gain: {portfolio_returns.max():.1%}")
        print(f"   Largest daily loss: {portfolio_returns.min():.1%}")
        
        # Best/worst performing periods
        if not portfolio_returns.empty:
            best_day = results_df.loc[portfolio_returns.idxmax()]
            worst_day = results_df.loc[portfolio_returns.idxmin()]
            
            print(f"\n📈 Best Trading Day: {best_day['date'].strftime('%Y-%m-%d')}")
            print(f"   Portfolio Return: {portfolio_returns.max():.2%}")
            print(f"   DOGE Price: ${best_day['price']:.4f}")
            print(f"   Regime: {best_day['regime_name']} (confidence: {best_day['confidence']:.1%})")
            print(f"   Position: {best_day['position']:+.1%}")
            
            print(f"\n📉 Worst Trading Day: {worst_day['date'].strftime('%Y-%m-%d')}")
            print(f"   Portfolio Return: {portfolio_returns.min():.2%}")
            print(f"   DOGE Price: ${worst_day['price']:.4f}")
            print(f"   Regime: {worst_day['regime_name']} (confidence: {worst_day['confidence']:.1%})")
            print(f"   Position: {worst_day['position']:+.1%}")
        
        # Regime transition analysis
        regime_changes = results_df['regime_name'] != results_df['regime_name'].shift(1)
        transition_dates = results_df[regime_changes]['date']
        
        print(f"\n🔄 Regime Transitions:")
        print(f"   Total transitions: {len(transition_dates) - 1}")
        print(f"   Average days per regime: {len(results_df) / len(transition_dates):.1f}")
        
        # Volatility impact
        high_vol_days = results_df[results_df['volatility'] > results_df['volatility'].quantile(0.9)]
        if not high_vol_days.empty:
            avg_position_high_vol = high_vol_days['position'].mean()
            avg_position_low_vol = results_df[results_df['volatility'] <= results_df['volatility'].quantile(0.9)]['position'].mean()
            
            print(f"\n📊 Volatility Impact:")
            print(f"   Avg position during high volatility: {avg_position_high_vol:.1%}")
            print(f"   Avg position during normal volatility: {avg_position_low_vol:.1%}")
        
        # Top recommendations for crypto trading
        print(f"\n🎯 CRYPTO TRADER RECOMMENDATIONS:")
        print("   1. Use 4-state models to capture euphoria/bubble phases")
        print("   2. Increase adaptation rate for faster regime detection in volatile crypto")
        print("   3. Implement volatility-based position sizing to manage extreme moves")
        print("   4. Monitor social media sentiment for meme coin regime changes")
        print("   5. Set wider stop-losses and drawdown limits for crypto volatility")
        print("   6. Consider partial profit-taking during euphoric regimes")
        print("   7. Online learning critical for capturing rapid crypto regime shifts")


def main():
    """Run the complete DOGE explosive growth case study"""
    print("🐕 DOGE Explosive Growth Case Study: Crypto Trading with Hidden Regimes")
    print("=" * 80)
    print("Demonstrating Hidden Regime's online learning capabilities")
    print("during extreme cryptocurrency volatility and explosive growth.\n")
    
    try:
        # Initialize case study
        study = DOGEExplosiveGrowthCaseStudy()
        
        # 1. Load data
        data = study.load_data()
        
        # 2. Prepare training data
        training_returns, training_data = study.prepare_training_data(data)
        
        # 3. Initialize models
        study.initialize_models(training_returns)
        
        # 4. Run online trading simulation
        simulator = study.run_online_trading_simulation(data)
        
        # 5. Analyze performance
        results_df = study.analyze_crypto_performance(simulator)
        
        # 6. Create visualizations
        study.create_crypto_visualizations(results_df)
        
        # 7. Generate insights
        study.generate_crypto_insights(results_df)
        
        print(f"\n✅ CRYPTO CASE STUDY COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("This demonstration shows how online learning can help crypto")
        print("traders navigate extreme volatility and explosive growth periods")
        print("with regime-based position sizing and risk management.")
        
        return results_df, study
        
    except Exception as e:
        print(f"\n❌ Crypto case study failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, study = main()