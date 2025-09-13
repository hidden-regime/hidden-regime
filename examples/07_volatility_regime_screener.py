#!/usr/bin/env python3
"""
Volatility Regime Screener
==========================

This example demonstrates how to screen stocks based on volatility regime patterns
and identify opportunities where volatility regimes diverge from price regimes.

Key features:
- Volatility-based regime detection
- Volatility breakout identification
- Low/high volatility regime screening
- Volatility-price regime divergence analysis
- Risk-adjusted opportunity scoring

Use cases:
- Options trading strategy development
- Risk management and portfolio construction
- Volatility arbitrage opportunities
- Mean reversion strategy identification

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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.models import BayesianHMM
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.utils import setup_logging
from hidden_regime.config import Config

# Setup logging
logger = setup_logging()

class VolatilityRegimeScreener:
    """Screen stocks based on volatility regime patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.analyzer = RegimeAnalyzer(config)
        
        # Screening parameters
        self.volatility_lookback = 20  # Days for volatility calculation
        self.regime_threshold = 0.7   # Minimum confidence for regime classification
        self.divergence_threshold = 0.5  # Threshold for price-volatility divergence
        
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate realized volatility using rolling window"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def detect_volatility_regimes(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Detect volatility-based regimes for a stock"""
        try:
            # Load price data
            data = self.data_loader.load_stock_data(symbol, start_date, end_date)
            if data is None or len(data) < 100:
                return None
                
            # Calculate returns and volatility
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            volatility = self.calculate_realized_volatility(returns, self.volatility_lookback)
            
            # Create volatility-based features for HMM
            vol_features = pd.DataFrame({
                'log_volatility': np.log(volatility),
                'vol_change': volatility.pct_change(),
                'vol_zscore': (volatility - volatility.rolling(60).mean()) / volatility.rolling(60).std()
            }).dropna()
            
            # Fit HMM on volatility features
            model = BayesianHMM(
                n_states=3,
                n_iterations=self.config.model_params['n_iterations']
            )
            
            # Use log volatility as primary feature
            vol_data = vol_features['log_volatility'].values.reshape(-1, 1)
            model.fit(vol_data)
            
            # Get regime predictions
            states = model.predict(vol_data)
            state_probs = model.predict_proba(vol_data)
            
            # Classify volatility regimes
            vol_means = []
            for state in range(3):
                state_mask = states == state
                if state_mask.sum() > 0:
                    mean_vol = volatility[vol_features.index[state_mask]].mean()
                    vol_means.append((state, mean_vol))
            
            vol_means.sort(key=lambda x: x[1])
            low_vol_state = vol_means[0][0]
            med_vol_state = vol_means[1][0]
            high_vol_state = vol_means[2][0]
            
            # Map states to regime names
            regime_names = {
                low_vol_state: 'Low_Volatility',
                med_vol_state: 'Medium_Volatility', 
                high_vol_state: 'High_Volatility'
            }
            
            # Current regime analysis
            current_state = states[-1]
            current_confidence = state_probs[-1, current_state]
            current_regime = regime_names[current_state]
            current_volatility = volatility.iloc[-1]
            
            # Regime statistics
            regime_stats = {}
            for state_num, regime_name in regime_names.items():
                state_mask = states == state_num
                if state_mask.sum() > 0:
                    regime_indices = vol_features.index[state_mask]
                    regime_vol = volatility[regime_indices]
                    regime_returns = returns[regime_indices]
                    
                    regime_stats[regime_name] = {
                        'avg_volatility': regime_vol.mean(),
                        'avg_return': regime_returns.mean() * 252,
                        'regime_days': len(regime_indices),
                        'frequency': len(regime_indices) / len(states)
                    }
            
            # Volatility breakout detection
            vol_percentile_90 = volatility.rolling(252).quantile(0.9).iloc[-1]
            vol_percentile_10 = volatility.rolling(252).quantile(0.1).iloc[-1]
            
            is_vol_breakout = current_volatility > vol_percentile_90
            is_vol_breakdown = current_volatility < vol_percentile_10
            
            # Recent regime changes
            regime_changes = (pd.Series(states).diff() != 0).sum()
            days_in_regime = 1
            for i in range(len(states) - 2, -1, -1):
                if states[i] == current_state:
                    days_in_regime += 1
                else:
                    break
            
            return {
                'symbol': symbol,
                'current_regime': current_regime,
                'current_confidence': current_confidence,
                'current_volatility': current_volatility,
                'days_in_regime': days_in_regime,
                'regime_changes': regime_changes,
                'is_vol_breakout': is_vol_breakout,
                'is_vol_breakdown': is_vol_breakdown,
                'vol_percentile_90': vol_percentile_90,
                'vol_percentile_10': vol_percentile_10,
                'regime_stats': regime_stats,
                'vol_data': volatility,
                'price_data': data['Close'],
                'states': states,
                'state_probs': state_probs,
                'vol_features_index': vol_features.index
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def detect_price_volatility_divergence(self, symbol: str, vol_result: Dict) -> Dict:
        """Detect divergence between price and volatility regimes"""
        try:
            # Get price regime analysis
            price_analysis = self.analyzer.analyze_stock(
                symbol, 
                (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            if not price_analysis:
                return {'divergence_score': 0, 'divergence_type': 'Unknown'}
            
            price_regime = price_analysis.get('current_regime', 'Unknown')
            vol_regime = vol_result['current_regime']
            
            # Define expected relationships
            expected_relationships = {
                'Bear': 'High_Volatility',
                'Sideways': 'Medium_Volatility',
                'Bull': 'Low_Volatility'
            }
            
            # Calculate divergence
            expected_vol_regime = expected_relationships.get(price_regime, 'Medium_Volatility')
            is_divergent = vol_regime != expected_vol_regime
            
            # Divergence scoring
            divergence_types = {
                ('Bear', 'Low_Volatility'): {'score': 0.9, 'type': 'Low_Vol_Bear'},
                ('Bear', 'Medium_Volatility'): {'score': 0.3, 'type': 'Mild_Divergence'},
                ('Bull', 'High_Volatility'): {'score': 0.8, 'type': 'High_Vol_Bull'},
                ('Bull', 'Medium_Volatility'): {'score': 0.2, 'type': 'Mild_Divergence'},
                ('Sideways', 'High_Volatility'): {'score': 0.6, 'type': 'Volatile_Consolidation'},
                ('Sideways', 'Low_Volatility'): {'score': 0.4, 'type': 'Quiet_Consolidation'}
            }
            
            divergence_info = divergence_types.get(
                (price_regime, vol_regime),
                {'score': 0.1, 'type': 'Normal_Relationship'}
            )
            
            return {
                'price_regime': price_regime,
                'vol_regime': vol_regime,
                'expected_vol_regime': expected_vol_regime,
                'is_divergent': is_divergent,
                'divergence_score': divergence_info['score'],
                'divergence_type': divergence_info['type'],
                'price_confidence': price_analysis.get('confidence', 0),
                'vol_confidence': vol_result['current_confidence']
            }
            
        except Exception as e:
            logger.error(f"Error detecting divergence for {symbol}: {str(e)}")
            return {'divergence_score': 0, 'divergence_type': 'Error'}
    
    def calculate_volatility_opportunity_score(self, vol_result: Dict, divergence_result: Dict) -> float:
        """Calculate opportunity score based on volatility patterns"""
        try:
            score = 0.0
            
            # Base score from regime confidence
            score += vol_result['current_confidence'] * 0.3
            
            # Volatility breakout bonus
            if vol_result['is_vol_breakout']:
                score += 0.4
            elif vol_result['is_vol_breakdown']:
                score += 0.3
            
            # Divergence bonus
            score += divergence_result['divergence_score'] * 0.3
            
            # Recent regime change bonus
            if vol_result['days_in_regime'] <= 5:
                score += 0.2
            
            # Regime stability penalty
            if vol_result['regime_changes'] > 20:  # Too many changes
                score *= 0.8
            
            # Volatility level adjustments
            current_vol = vol_result['current_volatility']
            if current_vol > 0.6:  # Very high volatility
                score += 0.1
            elif current_vol < 0.1:  # Very low volatility
                score += 0.15
                
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0
    
    def screen_volatility_opportunities(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Screen multiple stocks for volatility-based opportunities"""
        results = []
        total_symbols = len(symbols)
        
        print(f"Screening {total_symbols} stocks for volatility regime opportunities...")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                print(f"Processing {i+1}/{total_symbols}: {symbol}")
            
            # Analyze volatility regimes
            vol_result = self.detect_volatility_regimes(symbol, start_date, end_date)
            if not vol_result:
                continue
            
            # Check regime confidence threshold
            if vol_result['current_confidence'] < self.regime_threshold:
                continue
            
            # Detect price-volatility divergence
            divergence_result = self.detect_price_volatility_divergence(symbol, vol_result)
            
            # Calculate opportunity score
            opportunity_score = self.calculate_volatility_opportunity_score(vol_result, divergence_result)
            
            # Compile results
            results.append({
                'Symbol': symbol,
                'Current_Vol_Regime': vol_result['current_regime'],
                'Vol_Confidence': vol_result['current_confidence'],
                'Current_Volatility': vol_result['current_volatility'],
                'Days_In_Regime': vol_result['days_in_regime'],
                'Vol_Breakout': vol_result['is_vol_breakout'],
                'Vol_Breakdown': vol_result['is_vol_breakdown'],
                'Price_Regime': divergence_result.get('price_regime', 'Unknown'),
                'Divergence_Type': divergence_result['divergence_type'],
                'Divergence_Score': divergence_result['divergence_score'],
                'Is_Divergent': divergence_result['is_divergent'],
                'Opportunity_Score': opportunity_score,
                'Regime_Changes': vol_result['regime_changes']
            })
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('Opportunity_Score', ascending=False)
        
        return df
    
    def create_volatility_screening_charts(self, results_df: pd.DataFrame, output_dir: str = './output'):
        """Create visualization charts for volatility screening results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if len(results_df) == 0:
            print("No data to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Volatility Regime Distribution
        ax1 = plt.subplot(2, 3, 1)
        regime_counts = results_df['Current_Vol_Regime'].value_counts()
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(regime_counts)]
        regime_counts.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
        ax1.set_title('Current Volatility Regime Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Volatility Regime')
        ax1.set_ylabel('Number of Stocks')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # 2. Opportunity Score Distribution  
        ax2 = plt.subplot(2, 3, 2)
        results_df['Opportunity_Score'].hist(bins=20, ax=ax2, alpha=0.7, color='gold', edgecolor='black')
        ax2.set_title('Opportunity Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Opportunity Score')
        ax2.set_ylabel('Number of Stocks')
        ax2.axvline(results_df['Opportunity_Score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {results_df["Opportunity_Score"].mean():.2f}')
        ax2.legend()
        
        # 3. Volatility vs Opportunity Score Scatter
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(results_df['Current_Volatility'], results_df['Opportunity_Score'], 
                            c=results_df['Vol_Confidence'], cmap='viridis', alpha=0.6, s=50)
        ax3.set_title('Volatility vs Opportunity Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Current Volatility')
        ax3.set_ylabel('Opportunity Score')
        plt.colorbar(scatter, ax=ax3, label='Volatility Confidence')
        
        # 4. Divergence Analysis
        ax4 = plt.subplot(2, 3, 4)
        divergence_counts = results_df['Divergence_Type'].value_counts()
        divergence_counts.plot(kind='barh', ax=ax4, color='lightsteelblue', alpha=0.8)
        ax4.set_title('Price-Volatility Divergence Types', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Stocks')
        
        # 5. Top Opportunities Heatmap
        ax5 = plt.subplot(2, 3, 5)
        top_20 = results_df.head(20)
        heatmap_data = top_20[['Vol_Confidence', 'Opportunity_Score', 'Divergence_Score', 'Current_Volatility']].T
        sns.heatmap(heatmap_data, ax=ax5, cmap='RdYlGn', annot=False, cbar_kws={'label': 'Score'})
        ax5.set_title('Top 20 Opportunities Heatmap', fontsize=14, fontweight='bold')
        ax5.set_xticklabels([f"{sym[:4]}" for sym in top_20['Symbol']], rotation=45)
        
        # 6. Volatility Breakout Analysis
        ax6 = plt.subplot(2, 3, 6)
        breakout_data = pd.DataFrame({
            'Volatility Breakout': results_df['Vol_Breakout'].sum(),
            'Volatility Breakdown': results_df['Vol_Breakdown'].sum(),
            'Normal Volatility': len(results_df) - results_df['Vol_Breakout'].sum() - results_df['Vol_Breakdown'].sum()
        }, index=[0])
        breakout_data.T.plot(kind='pie', ax=ax6, autopct='%1.1f%%', 
                           colors=['crimson', 'dodgerblue', 'lightgray'])
        ax6.set_title('Volatility Breakout/Breakdown Analysis', fontsize=14, fontweight='bold')
        ax6.set_ylabel('')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, 'volatility_screening_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Charts saved to {chart_file}")
    
    def generate_volatility_screening_report(self, results_df: pd.DataFrame, output_dir: str = './output') -> str:
        """Generate comprehensive volatility screening report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate summary statistics
        total_stocks = len(results_df)
        high_opportunity_stocks = len(results_df[results_df['Opportunity_Score'] > 0.7])
        vol_breakout_stocks = results_df['Vol_Breakout'].sum()
        vol_breakdown_stocks = results_df['Vol_Breakdown'].sum()
        divergent_stocks = results_df['Is_Divergent'].sum()
        
        avg_opportunity_score = results_df['Opportunity_Score'].mean()
        avg_volatility = results_df['Current_Volatility'].mean()
        
        # Get top opportunities
        top_opportunities = results_df.head(10)
        
        # Regime distribution
        regime_distribution = results_df['Current_Vol_Regime'].value_counts()
        divergence_distribution = results_df['Divergence_Type'].value_counts()
        
        # Generate markdown report
        report = f"""# Volatility Regime Screening Report
*Generated on {timestamp}*

## Executive Summary

This report analyzes **{total_stocks}** stocks for volatility-based trading opportunities using Hidden Markov Model regime detection. The analysis identifies stocks with unusual volatility patterns, price-volatility divergences, and regime transition opportunities.

### Key Findings

- **High-Opportunity Stocks**: {high_opportunity_stocks} stocks with opportunity scores > 0.7
- **Volatility Breakouts**: {vol_breakout_stocks} stocks experiencing volatility breakouts  
- **Volatility Breakdowns**: {vol_breakdown_stocks} stocks in volatility breakdown mode
- **Price-Vol Divergences**: {divergent_stocks} stocks showing price-volatility divergence patterns
- **Average Opportunity Score**: {avg_opportunity_score:.3f}
- **Average Current Volatility**: {avg_volatility:.1%}

## Top Volatility Opportunities

The following stocks show the highest opportunity scores based on volatility regime analysis:

| Rank | Symbol | Vol Regime | Confidence | Current Vol | Opportunity Score | Divergence Type |
|------|--------|------------|------------|-------------|-------------------|-----------------|"""

        for i, (_, row) in enumerate(top_opportunities.iterrows(), 1):
            report += f"""
| {i} | {row['Symbol']} | {row['Current_Vol_Regime']} | {row['Vol_Confidence']:.1%} | {row['Current_Volatility']:.1%} | {row['Opportunity_Score']:.3f} | {row['Divergence_Type']} |"""

        report += f"""

## Volatility Regime Analysis

### Current Regime Distribution
"""
        for regime, count in regime_distribution.items():
            pct = (count / total_stocks) * 100
            report += f"- **{regime}**: {count} stocks ({pct:.1f}%)\n"

        report += f"""
### Price-Volatility Divergence Analysis
"""
        for divergence, count in divergence_distribution.items():
            pct = (count / total_stocks) * 100
            report += f"- **{divergence}**: {count} stocks ({pct:.1f}%)\n"

        report += f"""

## Special Situation Stocks

### Volatility Breakout Candidates
Stocks experiencing volatility expansion above 90th percentile:
"""
        breakout_stocks = results_df[results_df['Vol_Breakout'] == True].head(5)
        for _, row in breakout_stocks.iterrows():
            report += f"- **{row['Symbol']}**: {row['Current_Volatility']:.1%} volatility ({row['Current_Vol_Regime']} regime)\n"

        report += f"""
### Volatility Breakdown Candidates  
Stocks experiencing volatility compression below 10th percentile:
"""
        breakdown_stocks = results_df[results_df['Vol_Breakdown'] == True].head(5)
        for _, row in breakdown_stocks.iterrows():
            report += f"- **{row['Symbol']}**: {row['Current_Volatility']:.1%} volatility ({row['Current_Vol_Regime']} regime)\n"

        report += f"""
### High Divergence Opportunities
Stocks with significant price-volatility regime divergence:
"""
        divergent_stocks = results_df[results_df['Divergence_Score'] > 0.6].head(5)
        for _, row in divergent_stocks.iterrows():
            report += f"- **{row['Symbol']}**: {row['Price_Regime']} price vs {row['Current_Vol_Regime']} volatility (Score: {row['Divergence_Score']:.2f})\n"

        report += f"""

## Trading Strategy Implications

### Options Trading Opportunities
1. **High Volatility Regimes**: Consider selling premium in stocks with sustained high volatility
2. **Low Volatility Regimes**: Look for volatility expansion plays and long options strategies
3. **Volatility Breakouts**: Momentum strategies and breakout trades
4. **Volatility Breakdowns**: Mean reversion and range-bound strategies

### Risk Management Considerations
1. **Regime Confidence**: Focus on stocks with >70% regime confidence
2. **Recent Changes**: Monitor stocks with <5 days in current regime
3. **Divergence Patterns**: Use price-volatility divergence for contrarian positions
4. **Volatility Clustering**: Expect volatility persistence in current regimes

### Portfolio Construction
1. **Low Vol Regime Stocks**: Core holdings with steady growth potential
2. **High Vol Regime Stocks**: Tactical positions with tight stop losses  
3. **Divergent Patterns**: Hedge positions and market-neutral strategies
4. **Breakout/Breakdown**: Momentum and mean-reversion allocations

## Methodology Notes

- **Volatility Calculation**: 20-day realized volatility, annualized
- **Regime Detection**: 3-state HMM on log-volatility features
- **Confidence Threshold**: Minimum 70% for regime classification
- **Divergence Analysis**: Comparison between price and volatility regimes
- **Opportunity Scoring**: Weighted combination of confidence, breakouts, and divergence

## Disclaimers

This analysis is for educational and informational purposes only. It does not constitute investment advice or recommendations. Past performance does not guarantee future results. Volatility patterns can change rapidly, and regime detection models have inherent limitations. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

---
*Analysis performed using Hidden Regime framework with Bayesian Hidden Markov Models*
"""
        
        # Save report
        report_file = os.path.join(output_dir, 'volatility_screening_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Volatility screening report saved to {report_file}")
        return report_file

def main():
    """Main execution function"""
    # Configuration
    config = Config()
    screener = VolatilityRegimeScreener(config)
    
    # Define stock universe - focus on liquid, high-volume stocks
    stock_universe = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
        # Healthcare  
        'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'BMY', 'GILD', 'BIIB',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX',
        # ETFs for sector analysis
        'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI'
    ]
    
    # Analysis period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("VOLATILITY REGIME SCREENING ANALYSIS")
    print("=" * 60)
    
    # Run volatility screening
    print(f"\nAnalyzing {len(stock_universe)} stocks for volatility regime opportunities...")
    results_df = screener.screen_volatility_opportunities(stock_universe, start_date, end_date)
    
    if len(results_df) == 0:
        print("No qualifying opportunities found.")
        return
    
    print(f"\nFound {len(results_df)} qualifying opportunities")
    print(f"Average opportunity score: {results_df['Opportunity_Score'].mean():.3f}")
    print(f"Top opportunity: {results_df.iloc[0]['Symbol']} (Score: {results_df.iloc[0]['Opportunity_Score']:.3f})")
    
    # Create output directory
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive volatility screening report...")
    report_file = screener.generate_volatility_screening_report(results_df, output_dir)
    
    # Create visualization charts
    print("Creating volatility screening visualizations...")
    screener.create_volatility_screening_charts(results_df, output_dir)
    
    # Save detailed results to CSV
    csv_file = os.path.join(output_dir, 'volatility_screening_results.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to {csv_file}")
    
    # Display top opportunities
    print("\n" + "=" * 80)
    print("TOP 10 VOLATILITY OPPORTUNITIES")
    print("=" * 80)
    print(results_df[['Symbol', 'Current_Vol_Regime', 'Vol_Confidence', 'Current_Volatility', 
                     'Divergence_Type', 'Opportunity_Score']].head(10).to_string(index=False))
    
    print(f"\nAnalysis complete! Check the output directory for detailed reports and charts.")

if __name__ == "__main__":
    main()