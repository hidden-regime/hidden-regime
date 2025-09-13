#!/usr/bin/env python3
"""
Custom Screening Strategies
===========================

This example demonstrates how to build custom stock screening strategies
using HMM regime detection combined with various technical and fundamental criteria.

Key features:
- Multiple pre-built screening strategies
- Custom strategy builder framework
- Multi-factor scoring system
- Strategy performance backtesting
- Portfolio construction utilities

Included strategies:
1. Regime Momentum Strategy
2. Regime Reversal Strategy  
3. Volatility Breakout Strategy
4. Defensive Regime Strategy
5. Growth Regime Strategy

Use cases:
- Strategy development and testing
- Portfolio construction and optimization
- Risk-adjusted stock selection
- Multi-factor model development

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.models import BayesianHMM
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.utils import setup_logging
from hidden_regime.config import Config

# Technical analysis library
try:
    import ta
except ImportError:
    print("Installing ta library for technical analysis...")
    os.system("pip install ta")
    import ta

# Setup logging
logger = setup_logging()

class CustomScreeningStrategies:
    """Framework for building and executing custom screening strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.analyzer = RegimeAnalyzer(config)
        
        # Initialize strategy registry
        self.strategies = {}
        self.register_built_in_strategies()
        
    def register_built_in_strategies(self):
        """Register all built-in screening strategies"""
        self.strategies.update({
            'regime_momentum': self.regime_momentum_strategy,
            'regime_reversal': self.regime_reversal_strategy,
            'volatility_breakout': self.volatility_breakout_strategy,
            'defensive_regime': self.defensive_regime_strategy,
            'growth_regime': self.growth_regime_strategy
        })
    
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """Get basic fundamental data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'profit_margins': info.get('profitMargins', np.nan),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {symbol}: {str(e)}")
            return {}
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Price-based indicators
            data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # Momentum indicators
            data['rsi'] = ta.momentum.rsi(data['Close'], window=14)
            data['macd'] = ta.trend.macd_diff(data['Close'])
            data['stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            
            # Volatility indicators
            data['bb_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['bb_lower'] = ta.volatility.bollinger_lband(data['Close'])
            data['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume'])
            data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Support/Resistance levels
            data['resistance'] = data['High'].rolling(window=20).max()
            data['support'] = data['Low'].rolling(window=20).min()
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def analyze_stock_complete(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Complete stock analysis including regime, technical, and fundamental data"""
        try:
            # Get regime analysis
            regime_analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            if not regime_analysis:
                return None
            
            # Load price data
            data = self.data_loader.load_stock_data(symbol, start_date, end_date)
            if data is None or len(data) < 50:
                return None
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Get fundamental data
            fundamentals = self.get_stock_fundamentals(symbol)
            
            # Calculate additional metrics
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            current_price = data['Close'].iloc[-1]
            
            # Technical analysis summary
            technical_summary = {
                'price_vs_sma20': (current_price / data['sma_20'].iloc[-1] - 1) if not pd.isna(data['sma_20'].iloc[-1]) else 0,
                'price_vs_sma50': (current_price / data['sma_50'].iloc[-1] - 1) if not pd.isna(data['sma_50'].iloc[-1]) else 0,
                'rsi': data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50,
                'macd_signal': 1 if data['macd'].iloc[-1] > 0 else -1 if not pd.isna(data['macd'].iloc[-1]) else 0,
                'bb_position': self.calculate_bb_position(data),
                'volume_trend': self.calculate_volume_trend(data),
                'volatility': returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) > 20 else 0,
                'momentum_1m': (current_price / data['Close'].iloc[-21] - 1) if len(data) > 21 else 0,
                'momentum_3m': (current_price / data['Close'].iloc[-63] - 1) if len(data) > 63 else 0
            }
            
            # Combine all analysis
            complete_analysis = {
                'symbol': symbol,
                'regime_analysis': regime_analysis,
                'technical_summary': technical_summary,
                'fundamentals': fundamentals,
                'current_price': current_price,
                'data': data
            }
            
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Error in complete analysis for {symbol}: {str(e)}")
            return None
    
    def calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            current_price = data['Close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            
            if pd.isna(bb_upper) or pd.isna(bb_lower) or bb_upper == bb_lower:
                return 0.5
            
            position = (current_price - bb_lower) / (bb_upper - bb_lower)
            return np.clip(position, 0, 1)
            
        except Exception:
            return 0.5
    
    def calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """Calculate volume trend (positive for increasing, negative for decreasing)"""
        try:
            volume_sma = data['volume_sma'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            if pd.isna(volume_sma) or volume_sma == 0:
                return 0
            
            return (current_volume / volume_sma - 1)
            
        except Exception:
            return 0
    
    # Strategy Implementation Functions
    
    def regime_momentum_strategy(self, analysis: Dict) -> Dict:
        """Strategy: Buy stocks in strong bull regimes with technical momentum"""
        try:
            regime = analysis['regime_analysis']
            technical = analysis['technical_summary']
            fundamentals = analysis['fundamentals']
            
            score = 0.0
            signals = []
            
            # Regime momentum factors
            if regime.get('current_regime') == 'Bull':
                score += 0.4
                signals.append('Bull regime detected')
                
                # Higher score for strong bull confidence
                confidence = regime.get('confidence', 0)
                score += confidence * 0.2
                
                # Bonus for recent regime change (fresh momentum)
                days_in_regime = regime.get('days_in_regime', float('inf'))
                if days_in_regime <= 10:
                    score += 0.1
                    signals.append('Recent bull regime start')
            
            # Technical momentum confirmation
            if technical['price_vs_sma20'] > 0.02:  # 2% above 20-day SMA
                score += 0.15
                signals.append('Above 20-day SMA')
            
            if technical['price_vs_sma50'] > 0.05:  # 5% above 50-day SMA
                score += 0.1
                signals.append('Above 50-day SMA')
            
            if technical['macd_signal'] > 0:
                score += 0.1
                signals.append('MACD bullish')
            
            if 30 < technical['rsi'] < 70:  # Not overbought/oversold
                score += 0.05
                signals.append('RSI neutral zone')
            
            # Volume confirmation
            if technical['volume_trend'] > 0.2:  # 20% above average volume
                score += 0.1
                signals.append('Above average volume')
            
            # Fundamental filters (if available)
            if fundamentals:
                if fundamentals.get('revenue_growth', 0) > 0.1:  # 10% revenue growth
                    score += 0.05
                    signals.append('Strong revenue growth')
                
                if 10 < fundamentals.get('pe_ratio', 0) < 30:  # Reasonable valuation
                    score += 0.05
                    signals.append('Reasonable valuation')
            
            return {
                'strategy': 'Regime Momentum',
                'score': min(score, 1.0),
                'signals': signals,
                'recommendation': 'BUY' if score > 0.6 else 'HOLD' if score > 0.3 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error in regime momentum strategy: {str(e)}")
            return {'strategy': 'Regime Momentum', 'score': 0, 'signals': [], 'recommendation': 'PASS'}
    
    def regime_reversal_strategy(self, analysis: Dict) -> Dict:
        """Strategy: Buy stocks showing signs of regime reversal from bear to bull"""
        try:
            regime = analysis['regime_analysis']
            technical = analysis['technical_summary']
            
            score = 0.0
            signals = []
            
            # Look for potential reversals
            current_regime = regime.get('current_regime')
            days_in_regime = regime.get('days_in_regime', 0)
            
            # Recent transition bonus
            if days_in_regime <= 5:
                if current_regime == 'Bull':
                    score += 0.4
                    signals.append('Recent transition to bull regime')
                elif current_regime == 'Sideways':
                    score += 0.2
                    signals.append('Recent transition to sideways regime')
            
            # Technical reversal signals
            if technical['bb_position'] < 0.2:  # Near lower Bollinger Band
                score += 0.15
                signals.append('Near oversold levels')
            
            if technical['rsi'] < 35:  # RSI oversold
                score += 0.15
                signals.append('RSI oversold')
                
            if technical['rsi'] > 30 and days_in_regime <= 3:  # RSI starting to recover
                score += 0.1
                signals.append('RSI recovering')
            
            # Price action reversal
            if technical['momentum_1m'] < -0.1 and technical['price_vs_sma20'] > -0.05:
                score += 0.1
                signals.append('Price stabilizing after decline')
            
            # Volume surge confirmation
            if technical['volume_trend'] > 0.5:  # 50% above average volume
                score += 0.15
                signals.append('Volume surge detected')
            
            return {
                'strategy': 'Regime Reversal',
                'score': min(score, 1.0),
                'signals': signals,
                'recommendation': 'BUY' if score > 0.5 else 'WATCH' if score > 0.3 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error in regime reversal strategy: {str(e)}")
            return {'strategy': 'Regime Reversal', 'score': 0, 'signals': [], 'recommendation': 'PASS'}
    
    def volatility_breakout_strategy(self, analysis: Dict) -> Dict:
        """Strategy: Capitalize on volatility regime changes and breakouts"""
        try:
            regime = analysis['regime_analysis']
            technical = analysis['technical_summary']
            
            score = 0.0
            signals = []
            
            # Volatility expansion opportunities
            volatility = technical['volatility']
            if volatility > 0.3:  # High volatility (>30% annualized)
                score += 0.2
                signals.append('High volatility environment')
                
                # Technical breakout confirmation
                if technical['bb_position'] > 0.8:  # Breaking upper Bollinger Band
                    score += 0.25
                    signals.append('Bollinger Band breakout')
                
                if technical['price_vs_sma20'] > 0.05:  # Strong price momentum
                    score += 0.2
                    signals.append('Strong price momentum')
                
                if technical['volume_trend'] > 0.3:  # Volume confirmation
                    score += 0.15
                    signals.append('Volume confirmation')
                
                # Recent regime change
                days_in_regime = regime.get('days_in_regime', float('inf'))
                if days_in_regime <= 7:
                    score += 0.2
                    signals.append('Recent regime change')
            
            # Low volatility compression (potential breakout setup)
            elif volatility < 0.15:  # Low volatility (<15% annualized)
                score += 0.1
                signals.append('Low volatility compression')
                
                if 0.3 < technical['bb_position'] < 0.7:  # Consolidating
                    score += 0.1
                    signals.append('Price consolidation')
                
                if technical['volume_trend'] < -0.2:  # Decreasing volume
                    score += 0.05
                    signals.append('Volume drying up')
            
            return {
                'strategy': 'Volatility Breakout',
                'score': min(score, 1.0),
                'signals': signals,
                'recommendation': 'BUY' if score > 0.6 else 'WATCH' if score > 0.3 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error in volatility breakout strategy: {str(e)}")
            return {'strategy': 'Volatility Breakout', 'score': 0, 'signals': [], 'recommendation': 'PASS'}
    
    def defensive_regime_strategy(self, analysis: Dict) -> Dict:
        """Strategy: Focus on defensive stocks during uncertain regimes"""
        try:
            regime = analysis['regime_analysis']
            technical = analysis['technical_summary']
            fundamentals = analysis['fundamentals']
            
            score = 0.0
            signals = []
            
            # Prefer defensive characteristics
            current_regime = regime.get('current_regime')
            
            # Regime stability bonus
            days_in_regime = regime.get('days_in_regime', 0)
            if days_in_regime > 20:  # Stable regime
                score += 0.15
                signals.append('Stable regime')
            
            # Low volatility preference
            volatility = technical['volatility']
            if volatility < 0.25:  # Low volatility
                score += 0.2
                signals.append('Low volatility')
            
            # Technical stability
            if -0.02 < technical['price_vs_sma20'] < 0.02:  # Close to 20-day SMA
                score += 0.1
                signals.append('Price stability')
            
            if 40 < technical['rsi'] < 60:  # Neutral RSI
                score += 0.1
                signals.append('Neutral momentum')
            
            # Fundamental quality (if available)
            if fundamentals:
                if fundamentals.get('debt_to_equity', float('inf')) < 0.5:  # Low debt
                    score += 0.15
                    signals.append('Low debt ratio')
                
                if fundamentals.get('roe', 0) > 0.15:  # Good ROE
                    score += 0.1
                    signals.append('Strong ROE')
                
                sector = fundamentals.get('sector', '')
                if sector in ['Utilities', 'Consumer Staples', 'Healthcare']:
                    score += 0.15
                    signals.append('Defensive sector')
            
            # Downside protection
            if technical['bb_position'] > 0.4:  # Not oversold
                score += 0.05
                signals.append('Not oversold')
            
            return {
                'strategy': 'Defensive Regime',
                'score': min(score, 1.0),
                'signals': signals,
                'recommendation': 'BUY' if score > 0.5 else 'HOLD' if score > 0.3 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error in defensive regime strategy: {str(e)}")
            return {'strategy': 'Defensive Regime', 'score': 0, 'signals': [], 'recommendation': 'PASS'}
    
    def growth_regime_strategy(self, analysis: Dict) -> Dict:
        """Strategy: Target high-growth stocks in favorable regimes"""
        try:
            regime = analysis['regime_analysis']
            technical = analysis['technical_summary']
            fundamentals = analysis['fundamentals']
            
            score = 0.0
            signals = []
            
            # Growth-favorable regimes
            current_regime = regime.get('current_regime')
            if current_regime == 'Bull':
                score += 0.25
                signals.append('Bull regime favorable for growth')
            elif current_regime == 'Sideways':
                score += 0.1
                signals.append('Neutral regime')
            
            # Technical growth signals
            if technical['momentum_3m'] > 0.1:  # 10% gain over 3 months
                score += 0.2
                signals.append('Strong 3-month momentum')
            
            if technical['price_vs_sma50'] > 0.1:  # 10% above 50-day SMA
                score += 0.15
                signals.append('Strong trend')
            
            if technical['rsi'] > 50:  # Positive momentum
                score += 0.1
                signals.append('Positive momentum')
            
            # Fundamental growth (if available)
            if fundamentals:
                revenue_growth = fundamentals.get('revenue_growth', 0)
                if revenue_growth > 0.2:  # 20% revenue growth
                    score += 0.2
                    signals.append('Exceptional revenue growth')
                elif revenue_growth > 0.1:  # 10% revenue growth
                    score += 0.1
                    signals.append('Strong revenue growth')
                
                sector = fundamentals.get('sector', '')
                if sector in ['Technology', 'Healthcare', 'Communication Services']:
                    score += 0.1
                    signals.append('Growth sector')
                
                # Growth at reasonable price
                pe_ratio = fundamentals.get('pe_ratio', 0)
                if 15 < pe_ratio < 40 and revenue_growth > 0.15:
                    score += 0.1
                    signals.append('Growth at reasonable price')
            
            return {
                'strategy': 'Growth Regime',
                'score': min(score, 1.0),
                'signals': signals,
                'recommendation': 'BUY' if score > 0.6 else 'WATCH' if score > 0.4 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error in growth regime strategy: {str(e)}")
            return {'strategy': 'Growth Regime', 'score': 0, 'signals': [], 'recommendation': 'PASS'}
    
    def run_strategy_screen(self, symbols: List[str], strategy_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Run a specific strategy screen on a list of symbols"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(self.strategies.keys())}")
        
        strategy_func = self.strategies[strategy_name]
        results = []
        total_symbols = len(symbols)
        
        print(f"Running {strategy_name} strategy on {total_symbols} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"Processing {i+1}/{total_symbols}: {symbol}")
            
            # Get complete analysis
            analysis = self.analyze_stock_complete(symbol, start_date, end_date)
            if not analysis:
                continue
            
            # Run strategy
            strategy_result = strategy_func(analysis)
            
            # Compile results
            result = {
                'Symbol': symbol,
                'Strategy': strategy_result['strategy'],
                'Score': strategy_result['score'],
                'Recommendation': strategy_result['recommendation'],
                'Signals': '; '.join(strategy_result['signals']),
                'Current_Regime': analysis['regime_analysis'].get('current_regime', 'Unknown'),
                'Regime_Confidence': analysis['regime_analysis'].get('confidence', 0),
                'Current_Price': analysis['current_price'],
                'RSI': analysis['technical_summary']['rsi'],
                'Volatility': analysis['technical_summary']['volatility'],
                'Momentum_1M': analysis['technical_summary']['momentum_1m'],
                'Momentum_3M': analysis['technical_summary']['momentum_3m']
            }
            
            # Add fundamental data if available
            if analysis['fundamentals']:
                result.update({
                    'Sector': analysis['fundamentals'].get('sector', 'Unknown'),
                    'PE_Ratio': analysis['fundamentals'].get('pe_ratio', np.nan),
                    'Revenue_Growth': analysis['fundamentals'].get('revenue_growth', np.nan)
                })
            
            results.append(result)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('Score', ascending=False)
        
        return df
    
    def run_multi_strategy_screen(self, symbols: List[str], strategies: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Run multiple strategies and return combined results"""
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"RUNNING {strategy.upper()} STRATEGY")
            print(f"{'='*60}")
            
            strategy_results = self.run_strategy_screen(symbols, strategy, start_date, end_date)
            results[strategy] = strategy_results
            
            if len(strategy_results) > 0:
                print(f"Found {len(strategy_results)} opportunities")
                print(f"Top 3 picks:")
                top_3 = strategy_results.head(3)
                for _, row in top_3.iterrows():
                    print(f"  {row['Symbol']}: {row['Score']:.3f} ({row['Recommendation']})")
            else:
                print("No opportunities found")
        
        return results
    
    def create_strategy_comparison_chart(self, multi_results: Dict[str, pd.DataFrame], output_dir: str = './output'):
        """Create comparison charts across multiple strategies"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not multi_results:
            print("No data to visualize")
            return
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create comprehensive comparison chart
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Strategy Score Distributions
        ax1 = plt.subplot(2, 3, 1)
        score_data = []
        strategy_names = []
        for strategy, df in multi_results.items():
            if len(df) > 0:
                score_data.append(df['Score'])
                strategy_names.append(strategy.replace('_', ' ').title())
        
        if score_data:
            ax1.boxplot(score_data, labels=strategy_names)
            ax1.set_title('Score Distributions by Strategy', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Strategy Score')
            plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # 2. Top Picks Heatmap
        ax2 = plt.subplot(2, 3, 2)
        top_picks_data = []
        for strategy, df in multi_results.items():
            if len(df) > 0:
                top_5 = df.head(5)
                strategy_scores = top_5.set_index('Symbol')['Score']
                top_picks_data.append(strategy_scores)
        
        if top_picks_data:
            combined_scores = pd.concat(top_picks_data, axis=1, keys=strategy_names)
            sns.heatmap(combined_scores.fillna(0), annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2)
            ax2.set_title('Top Picks by Strategy', fontsize=14, fontweight='bold')
        
        # 3. Recommendation Breakdown
        ax3 = plt.subplot(2, 3, 3)
        rec_data = {}
        for strategy, df in multi_results.items():
            if len(df) > 0:
                rec_counts = df['Recommendation'].value_counts()
                rec_data[strategy.replace('_', ' ').title()] = rec_counts
        
        if rec_data:
            rec_df = pd.DataFrame(rec_data).fillna(0)
            rec_df.plot(kind='bar', ax=ax3, alpha=0.8)
            ax3.set_title('Recommendations by Strategy', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Recommendation Type')
            ax3.set_ylabel('Number of Stocks')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(ax3.get_xticklabels(), rotation=0)
        
        # 4. Sector Distribution for Top Picks
        ax4 = plt.subplot(2, 3, 4)
        all_top_picks = []
        for strategy, df in multi_results.items():
            if len(df) > 0 and 'Sector' in df.columns:
                top_picks = df[df['Recommendation'] == 'BUY'].head(10)
                all_top_picks.extend(top_picks['Sector'].tolist())
        
        if all_top_picks:
            sector_counts = pd.Series(all_top_picks).value_counts().head(8)
            sector_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
            ax4.set_title('Sector Distribution (Top Picks)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('')
        
        # 5. Risk-Return Scatter
        ax5 = plt.subplot(2, 3, 5)
        for strategy, df in multi_results.items():
            if len(df) > 0 and 'Volatility' in df.columns and 'Momentum_3M' in df.columns:
                df_clean = df.dropna(subset=['Volatility', 'Momentum_3M'])
                if len(df_clean) > 0:
                    ax5.scatter(df_clean['Volatility'], df_clean['Momentum_3M'], 
                              alpha=0.6, s=50, label=strategy.replace('_', ' ').title())
        
        ax5.set_title('Risk vs Return (3M Momentum)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Volatility')
        ax5.set_ylabel('3-Month Momentum')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Strategy Overlap Analysis
        ax6 = plt.subplot(2, 3, 6)
        if len(multi_results) >= 2:
            # Find stocks that appear in multiple strategies' top picks
            strategy_top_picks = {}
            for strategy, df in multi_results.items():
                if len(df) > 0:
                    top_picks = df[df['Score'] > 0.5]['Symbol'].tolist()
                    strategy_top_picks[strategy] = set(top_picks)
            
            if len(strategy_top_picks) >= 2:
                # Calculate overlap matrix
                strategies = list(strategy_top_picks.keys())
                overlap_matrix = np.zeros((len(strategies), len(strategies)))
                
                for i, strat1 in enumerate(strategies):
                    for j, strat2 in enumerate(strategies):
                        if i != j:
                            overlap = len(strategy_top_picks[strat1] & strategy_top_picks[strat2])
                            total = len(strategy_top_picks[strat1] | strategy_top_picks[strat2])
                            overlap_matrix[i][j] = overlap / total if total > 0 else 0
                
                sns.heatmap(overlap_matrix, annot=True, fmt='.2f', 
                           xticklabels=[s.replace('_', ' ').title() for s in strategies],
                           yticklabels=[s.replace('_', ' ').title() for s in strategies],
                           cmap='Blues', ax=ax6)
                ax6.set_title('Strategy Overlap Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, 'multi_strategy_comparison.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Strategy comparison charts saved to {chart_file}")
    
    def generate_multi_strategy_report(self, multi_results: Dict[str, pd.DataFrame], output_dir: str = './output') -> str:
        """Generate comprehensive multi-strategy screening report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate summary statistics
        total_strategies = len(multi_results)
        total_opportunities = sum(len(df) for df in multi_results.values())
        
        # Generate report
        report = f"""# Multi-Strategy Stock Screening Report
*Generated on {timestamp}*

## Executive Summary

This report presents results from **{total_strategies}** custom screening strategies using Hidden Markov Model regime detection combined with technical and fundamental analysis. A total of **{total_opportunities}** opportunities were identified across all strategies.

## Strategy Overview

The following strategies were analyzed:
"""
        
        # Strategy summaries
        for strategy_name, df in multi_results.items():
            strategy_display = strategy_name.replace('_', ' ').title()
            num_opportunities = len(df)
            num_buy_recs = len(df[df['Recommendation'] == 'BUY']) if len(df) > 0 else 0
            avg_score = df['Score'].mean() if len(df) > 0 else 0
            
            report += f"""
### {strategy_display}
- **Opportunities Found**: {num_opportunities}
- **Buy Recommendations**: {num_buy_recs} 
- **Average Score**: {avg_score:.3f}
"""
        
        # Top picks across all strategies
        report += f"""
## Top Picks Across All Strategies

### Highest Scoring Opportunities
"""
        
        # Combine all results and find top picks
        all_results = []
        for strategy, df in multi_results.items():
            if len(df) > 0:
                df_copy = df.copy()
                df_copy['Strategy'] = strategy.replace('_', ' ').title()
                all_results.append(df_copy)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            top_overall = combined_df.nlargest(10, 'Score')
            
            report += """
| Rank | Symbol | Strategy | Score | Recommendation | Current Regime | Signals |
|------|--------|----------|-------|----------------|----------------|---------|"""
            
            for i, (_, row) in enumerate(top_overall.iterrows(), 1):
                signals_short = row['Signals'][:100] + "..." if len(row['Signals']) > 100 else row['Signals']
                report += f"""
| {i} | {row['Symbol']} | {row['Strategy']} | {row['Score']:.3f} | {row['Recommendation']} | {row['Current_Regime']} | {signals_short} |"""
        
        # Strategy-specific top picks
        for strategy_name, df in multi_results.items():
            if len(df) == 0:
                continue
                
            strategy_display = strategy_name.replace('_', ' ').title()
            
            report += f"""

## {strategy_display} Strategy Results

### Strategy Description
"""
            
            # Add strategy descriptions
            strategy_descriptions = {
                'regime_momentum': 'Identifies stocks in strong bull regimes with technical momentum confirmation.',
                'regime_reversal': 'Targets stocks showing signs of regime reversal from bear to bull markets.',
                'volatility_breakout': 'Capitalizes on volatility regime changes and technical breakouts.',
                'defensive_regime': 'Focuses on defensive stocks with stable characteristics during uncertain regimes.',
                'growth_regime': 'Targets high-growth stocks in favorable market regimes.'
            }
            
            description = strategy_descriptions.get(strategy_name, 'Custom screening strategy.')
            report += f"{description}\n"
            
            # Top picks for this strategy
            top_5 = df.head(5)
            buy_recs = df[df['Recommendation'] == 'BUY']
            
            report += f"""
### Performance Summary
- **Total Opportunities**: {len(df)}
- **Buy Recommendations**: {len(buy_recs)}
- **Watch Recommendations**: {len(df[df['Recommendation'] == 'WATCH'])}
- **Average Score**: {df['Score'].mean():.3f}
- **Top Score**: {df['Score'].max():.3f}

### Top 5 Picks

| Symbol | Score | Recommendation | Current Regime | Key Signals |
|--------|-------|----------------|----------------|-------------|"""
            
            for _, row in top_5.iterrows():
                signals_short = row['Signals'][:80] + "..." if len(row['Signals']) > 80 else row['Signals']
                report += f"""
| {row['Symbol']} | {row['Score']:.3f} | {row['Recommendation']} | {row['Current_Regime']} | {signals_short} |"""
        
        # Cross-strategy analysis
        if len(multi_results) > 1:
            report += f"""

## Cross-Strategy Analysis

### Multi-Strategy Consensus Picks
Stocks that score well across multiple strategies often represent the highest-conviction opportunities:
"""
            
            # Find stocks that appear in multiple strategies
            symbol_strategies = {}
            for strategy, df in multi_results.items():
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    if symbol not in symbol_strategies:
                        symbol_strategies[symbol] = []
                    symbol_strategies[symbol].append({
                        'strategy': strategy.replace('_', ' ').title(),
                        'score': row['Score'],
                        'recommendation': row['Recommendation']
                    })
            
            # Find consensus picks (appearing in 2+ strategies)
            consensus_picks = []
            for symbol, strategies in symbol_strategies.items():
                if len(strategies) >= 2:
                    avg_score = np.mean([s['score'] for s in strategies])
                    consensus_picks.append({
                        'symbol': symbol,
                        'num_strategies': len(strategies),
                        'avg_score': avg_score,
                        'strategies': strategies
                    })
            
            consensus_picks.sort(key=lambda x: (x['num_strategies'], x['avg_score']), reverse=True)
            
            if consensus_picks:
                report += """
| Symbol | # Strategies | Avg Score | Strategy Details |
|--------|-------------|-----------|------------------|"""
                
                for pick in consensus_picks[:10]:
                    strategy_details = "; ".join([f"{s['strategy']} ({s['score']:.2f})" for s in pick['strategies']])
                    report += f"""
| {pick['symbol']} | {pick['num_strategies']} | {pick['avg_score']:.3f} | {strategy_details} |"""
        
        # Risk considerations
        report += f"""

## Risk Considerations and Trading Guidelines

### Strategy-Specific Risks
- **Regime Momentum**: Risk of trend reversal, especially in overextended markets
- **Regime Reversal**: Risk of false signals during volatile periods
- **Volatility Breakout**: Risk of whipsaws in choppy markets
- **Defensive Regime**: Risk of underperformance in strong bull markets
- **Growth Regime**: Risk of multiple compression during market stress

### General Risk Management Guidelines
1. **Position Sizing**: Use regime confidence scores to adjust position sizes
2. **Diversification**: Combine strategies for balanced exposure
3. **Stop Losses**: Implement regime-based stop loss levels
4. **Rebalancing**: Monitor regime changes for portfolio adjustments
5. **Market Context**: Consider overall market regime for strategy selection

### Backtesting Recommendations
Before deploying these strategies:
1. Test on historical data for at least 3-5 years
2. Include transaction costs and slippage
3. Account for regime detection lags
4. Validate during different market cycles
5. Monitor live performance vs. backtests

## Disclaimer

This analysis is for educational and research purposes only. It does not constitute investment advice or recommendations. Past performance does not guarantee future results. Market regimes can change rapidly, and quantitative models have inherent limitations. Always conduct thorough due diligence and consult with qualified financial advisors before making investment decisions.

---
*Analysis performed using Hidden Regime framework with custom screening strategies*
"""
        
        # Save report
        report_file = os.path.join(output_dir, 'multi_strategy_screening_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Multi-strategy screening report saved to {report_file}")
        return report_file

def main():
    """Main execution function demonstrating all custom screening strategies"""
    # Configuration
    config = Config()
    screener = CustomScreeningStrategies(config)
    
    # Define stock universe
    stock_universe = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'BMY', 'GILD', 'BIIB', 'TMO', 'DHR',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'KMI',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'HON', 'DE'
    ]
    
    # Analysis period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("CUSTOM SCREENING STRATEGIES ANALYSIS")
    print("=" * 80)
    
    # Define strategies to run
    strategies_to_run = [
        'regime_momentum',
        'regime_reversal', 
        'volatility_breakout',
        'defensive_regime',
        'growth_regime'
    ]
    
    # Run multi-strategy screening
    print(f"Running {len(strategies_to_run)} strategies on {len(stock_universe)} stocks...")
    multi_results = screener.run_multi_strategy_screen(
        stock_universe, strategies_to_run, start_date, end_date
    )
    
    # Create output directory
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    print("\nGenerating multi-strategy screening report...")
    report_file = screener.generate_multi_strategy_report(multi_results, output_dir)
    
    # Create comparison charts
    print("Creating strategy comparison visualizations...")
    screener.create_strategy_comparison_chart(multi_results, output_dir)
    
    # Save individual strategy results
    for strategy_name, df in multi_results.items():
        if len(df) > 0:
            csv_file = os.path.join(output_dir, f'{strategy_name}_results.csv')
            df.to_csv(csv_file, index=False)
            print(f"Saved {strategy_name} results to {csv_file}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("MULTI-STRATEGY SCREENING SUMMARY")
    print("=" * 80)
    
    for strategy_name, df in multi_results.items():
        strategy_display = strategy_name.replace('_', ' ').title()
        buy_count = len(df[df['Recommendation'] == 'BUY']) if len(df) > 0 else 0
        
        print(f"\n{strategy_display}:")
        print(f"  Total Opportunities: {len(df)}")
        print(f"  Buy Recommendations: {buy_count}")
        
        if len(df) > 0:
            print(f"  Top Pick: {df.iloc[0]['Symbol']} (Score: {df.iloc[0]['Score']:.3f})")
    
    print(f"\nAnalysis complete! Check the output directory for detailed reports and charts.")

if __name__ == "__main__":
    main()