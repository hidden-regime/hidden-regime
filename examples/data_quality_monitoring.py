"""
Data Quality Monitoring Example for hidden-regime package.

Demonstrates continuous data quality monitoring including:
- Quality score tracking over time
- Data quality degradation detection
- Automated quality reporting
- Quality threshold alerts
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import time

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import DataConfig, ValidationConfig, PreprocessingConfig


class DataQualityMonitor:
    """Monitor data quality over time with trend analysis."""
    
    def __init__(self, validation_config=None):
        """Initialize quality monitor."""
        self.validation_config = validation_config or ValidationConfig()
        self.validator = hr.DataValidator(self.validation_config)
        self.quality_history = defaultdict(list)
        self.alert_thresholds = {
            'quality_score_min': 0.6,      # Alert if quality drops below 60%
            'quality_drop_threshold': 0.2,  # Alert if quality drops by 20%
            'consecutive_failures': 3        # Alert after 3 consecutive low-quality periods
        }
    
    def monitor_stock_quality(self, ticker, data, timestamp=None):
        """Monitor quality for a single stock."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate data
        validation_result = self.validator.validate_data(data, ticker)
        
        # Store quality history
        quality_record = {
            'timestamp': timestamp,
            'quality_score': validation_result.quality_score,
            'is_valid': validation_result.is_valid,
            'n_issues': len(validation_result.issues),
            'n_warnings': len(validation_result.warnings),
            'n_observations': len(data),
            'issues': validation_result.issues.copy(),
            'warnings': validation_result.warnings.copy(),
            'metrics': validation_result.metrics.copy()
        }
        
        self.quality_history[ticker].append(quality_record)
        
        # Check for alerts
        alerts = self._check_quality_alerts(ticker, quality_record)
        
        return validation_result, alerts
    
    def _check_quality_alerts(self, ticker, current_record):
        """Check for quality alerts based on current and historical data."""
        alerts = []
        
        history = self.quality_history[ticker]
        current_quality = current_record['quality_score']
        
        # Alert 1: Quality below minimum threshold
        if current_quality < self.alert_thresholds['quality_score_min']:
            alerts.append({
                'type': 'LOW_QUALITY',
                'severity': 'HIGH',
                'message': f"Quality score {current_quality:.3f} below threshold {self.alert_thresholds['quality_score_min']:.3f}",
                'ticker': ticker,
                'timestamp': current_record['timestamp']
            })
        
        # Alert 2: Quality drop from previous measurement
        if len(history) >= 2:
            previous_quality = history[-2]['quality_score']
            quality_drop = previous_quality - current_quality
            
            if quality_drop > self.alert_thresholds['quality_drop_threshold']:
                alerts.append({
                    'type': 'QUALITY_DROP',
                    'severity': 'MEDIUM',
                    'message': f"Quality dropped by {quality_drop:.3f} ({quality_drop/previous_quality:.1%}) from previous measurement",
                    'ticker': ticker,
                    'timestamp': current_record['timestamp']
                })
        
        # Alert 3: Consecutive low-quality periods
        if len(history) >= self.alert_thresholds['consecutive_failures']:
            recent_qualities = [record['quality_score'] for record in history[-self.alert_thresholds['consecutive_failures']:]]
            
            if all(q < self.alert_thresholds['quality_score_min'] for q in recent_qualities):
                alerts.append({
                    'type': 'PERSISTENT_LOW_QUALITY',
                    'severity': 'HIGH',
                    'message': f"Quality below threshold for {self.alert_thresholds['consecutive_failures']} consecutive periods",
                    'ticker': ticker,
                    'timestamp': current_record['timestamp']
                })
        
        # Alert 4: Significant increase in issues
        if current_record['n_issues'] > 0:
            alerts.append({
                'type': 'DATA_ISSUES',
                'severity': 'MEDIUM',
                'message': f"{current_record['n_issues']} data issues detected: {', '.join(current_record['issues'][:2])}",
                'ticker': ticker,
                'timestamp': current_record['timestamp']
            })
        
        return alerts
    
    def get_quality_trend(self, ticker, periods=10):
        """Get quality trend analysis for a ticker."""
        history = self.quality_history[ticker]
        
        if len(history) < 2:
            return None
        
        # Get recent quality scores
        recent_history = history[-periods:] if len(history) >= periods else history
        quality_scores = [record['quality_score'] for record in recent_history]
        timestamps = [record['timestamp'] for record in recent_history]
        
        # Calculate trend
        if len(quality_scores) >= 2:
            # Simple linear trend (slope)
            x = np.arange(len(quality_scores))
            trend_slope = np.polyfit(x, quality_scores, 1)[0]
            
            trend_analysis = {
                'current_quality': quality_scores[-1],
                'average_quality': np.mean(quality_scores),
                'quality_std': np.std(quality_scores),
                'trend_slope': trend_slope,
                'trend_direction': 'improving' if trend_slope > 0.01 else 'degrading' if trend_slope < -0.01 else 'stable',
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores),
                'periods_analyzed': len(quality_scores),
                'time_span': timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else timedelta(0)
            }
            
            return trend_analysis
        
        return None
    
    def generate_quality_report(self, tickers=None):
        """Generate comprehensive quality report."""
        if tickers is None:
            tickers = list(self.quality_history.keys())
        
        report = {
            'generated_at': datetime.now(),
            'summary': {},
            'ticker_details': {},
            'alerts': [],
            'recommendations': []
        }
        
        all_current_qualities = []
        all_alerts = []
        
        for ticker in tickers:
            if ticker not in self.quality_history or not self.quality_history[ticker]:
                continue
            
            latest_record = self.quality_history[ticker][-1]
            trend_analysis = self.get_quality_trend(ticker)
            
            ticker_details = {
                'latest_quality': latest_record['quality_score'],
                'is_valid': latest_record['is_valid'],
                'n_issues': latest_record['n_issues'],
                'n_warnings': latest_record['n_warnings'],
                'trend_analysis': trend_analysis,
                'total_measurements': len(self.quality_history[ticker])
            }
            
            report['ticker_details'][ticker] = ticker_details
            all_current_qualities.append(latest_record['quality_score'])
            
            # Collect recent alerts
            recent_alerts = []
            for record in self.quality_history[ticker][-5:]:  # Last 5 measurements
                if hasattr(record, 'alerts'):
                    recent_alerts.extend(record.get('alerts', []))
            
            all_alerts.extend(recent_alerts)
        
        # Generate summary statistics
        if all_current_qualities:
            report['summary'] = {
                'n_tickers_monitored': len(tickers),
                'average_quality': np.mean(all_current_qualities),
                'min_quality': min(all_current_qualities),
                'max_quality': max(all_current_qualities),
                'quality_std': np.std(all_current_qualities),
                'tickers_below_threshold': sum(1 for q in all_current_qualities 
                                             if q < self.alert_thresholds['quality_score_min']),
                'total_active_alerts': len(all_alerts)
            }
        
        # Generate recommendations
        avg_quality = report['summary'].get('average_quality', 1.0)
        if avg_quality < 0.5:
            report['recommendations'].append("Critical: Portfolio-wide data quality issues detected. Review data sources and collection processes.")
        elif avg_quality < 0.7:
            report['recommendations'].append("Warning: Below-average data quality. Consider implementing data cleaning procedures.")
        else:
            report['recommendations'].append("Good: Data quality within acceptable ranges. Continue monitoring.")
        
        if report['summary'].get('tickers_below_threshold', 0) > 0:
            report['recommendations'].append(f"Action needed: {report['summary']['tickers_below_threshold']} tickers below quality threshold. Investigate specific data issues.")
        
        return report


def simulate_quality_monitoring(monitor, tickers, n_periods=10):
    """Simulate quality monitoring over time with varying data quality."""
    print(f"Simulating quality monitoring for {len(tickers)} stocks over {n_periods} periods...")
    
    # Simulate different data quality scenarios
    base_date = datetime.now() - timedelta(days=n_periods)
    
    for period in range(n_periods):
        current_date = base_date + timedelta(days=period)
        print(f"\nPeriod {period + 1}: {current_date.strftime('%Y-%m-%d')}")
        
        period_alerts = []
        
        for ticker in tickers:
            # Generate mock data with varying quality
            data_size = np.random.randint(50, 200)  # Variable data size
            
            # Base data quality varies by period to simulate trends
            base_quality_factor = 1.0
            if period > n_periods * 0.7:  # Quality degrades in later periods
                base_quality_factor = 0.7 + period * 0.02  # Gradual degradation
            elif period < n_periods * 0.3:  # Good quality early on
                base_quality_factor = 0.9
            
            # Create sample data with intentional quality issues
            dates = pd.date_range('2024-01-01', periods=data_size)
            prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, data_size))
            
            # Add quality issues based on period
            missing_pct = max(0, np.random.normal(0.02 * (1 - base_quality_factor), 0.01))
            n_missing = int(data_size * missing_pct)
            
            if n_missing > 0:
                missing_indices = np.random.choice(data_size, min(n_missing, data_size//2), replace=False)
                prices[missing_indices] = np.nan
            
            # Add outliers
            if np.random.random() > base_quality_factor:
                outlier_indices = np.random.choice(data_size, max(1, data_size//20), replace=False)
                prices[outlier_indices] *= np.random.choice([0.5, 1.5], len(outlier_indices))
            
            # Create DataFrame
            mock_data = pd.DataFrame({
                'date': dates,
                'price': prices,
                'log_return': np.random.normal(0.001, 0.02, data_size),
                'volume': np.random.randint(1000000, 5000000, data_size)
            })
            
            # Monitor quality
            validation_result, alerts = monitor.monitor_stock_quality(ticker, mock_data, current_date)
            
            print(f"  {ticker}: Quality {validation_result.quality_score:.3f} "
                  f"({'Valid' if validation_result.is_valid else 'Invalid'})")
            
            if alerts:
                period_alerts.extend(alerts)
                for alert in alerts:
                    severity_symbol = "🚨" if alert['severity'] == 'HIGH' else "⚠️"
                    print(f"    {severity_symbol} {alert['type']}: {alert['message']}")
        
        # Show period summary
        if not period_alerts:
            print("  ✅ No quality alerts this period")
        else:
            high_alerts = sum(1 for alert in period_alerts if alert['severity'] == 'HIGH')
            med_alerts = len(period_alerts) - high_alerts
            print(f"  🚨 Period alerts: {high_alerts} high, {med_alerts} medium severity")


def demonstrate_quality_reporting(monitor, tickers):
    """Demonstrate quality reporting capabilities."""
    print(f"\nGenerating comprehensive quality report...")
    
    report = monitor.generate_quality_report(tickers)
    
    print(f"\n📊 DATA QUALITY REPORT")
    print(f"Generated: {report['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Summary section
    summary = report['summary']
    print(f"\n📈 PORTFOLIO SUMMARY")
    print(f"  Stocks monitored: {summary['n_tickers_monitored']}")
    print(f"  Average quality: {summary['average_quality']:.3f}")
    print(f"  Quality range: {summary['min_quality']:.3f} - {summary['max_quality']:.3f}")
    print(f"  Quality std dev: {summary['quality_std']:.3f}")
    print(f"  Below threshold: {summary['tickers_below_threshold']} stocks")
    
    # Individual stock details
    print(f"\n📋 INDIVIDUAL STOCK ANALYSIS")
    for ticker, details in report['ticker_details'].items():
        trend = details['trend_analysis']
        status = "✅" if details['is_valid'] else "❌"
        
        print(f"  {status} {ticker}:")
        print(f"    Current quality: {details['latest_quality']:.3f}")
        
        if trend:
            trend_symbol = "📈" if trend['trend_direction'] == 'improving' else "📉" if trend['trend_direction'] == 'degrading' else "➡️"
            print(f"    Trend: {trend_symbol} {trend['trend_direction']} (slope: {trend['trend_slope']:.4f})")
            print(f"    Range: {trend['min_quality']:.3f} - {trend['max_quality']:.3f}")
        
        if details['n_issues'] > 0:
            print(f"    Issues: {details['n_issues']} active")
        if details['n_warnings'] > 0:
            print(f"    Warnings: {details['n_warnings']} active")
    
    # Recommendations
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report


def main():
    """Run data quality monitoring example."""
    print("Hidden Regime Data Quality Monitoring Example")
    print("=" * 55)
    
    # Set up monitoring configuration
    monitoring_config = ValidationConfig(
        outlier_method='iqr',
        outlier_threshold=2.5,      # Slightly sensitive for early detection
        max_daily_return=0.25,      # 25% max daily return
        max_consecutive_missing=3,  # Strict missing data tolerance
        interpolation_method='linear'
    )
    
    # Initialize monitor
    monitor = DataQualityMonitor(monitoring_config)
    
    # Define stocks to monitor
    monitored_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'JPM']
    
    print(f"Monitoring configuration:")
    print(f"  Outlier method: {monitoring_config.outlier_method}")
    print(f"  Quality threshold: {monitor.alert_thresholds['quality_score_min']:.1%}")
    print(f"  Monitored stocks: {', '.join(monitored_stocks)}")
    
    try:
        # Simulate quality monitoring over time
        simulate_quality_monitoring(monitor, monitored_stocks, n_periods=8)
        
        # Generate and display comprehensive report
        report = demonstrate_quality_reporting(monitor, monitored_stocks)
        
        # Show trend analysis examples
        print(f"\n🔍 DETAILED TREND ANALYSIS")
        for ticker in monitored_stocks[:3]:  # Show first 3 stocks
            trend = monitor.get_quality_trend(ticker, periods=5)
            if trend:
                print(f"\n  {ticker} Trend Analysis (last 5 periods):")
                print(f"    Direction: {trend['trend_direction']}")
                print(f"    Current: {trend['current_quality']:.3f}")
                print(f"    Average: {trend['average_quality']:.3f}")
                print(f"    Volatility: {trend['quality_std']:.3f}")
                print(f"    Time span: {trend['time_span'].days} days")
        
        print(f"\n" + "=" * 55)
        print("Quality Monitoring Example Complete!")
        print("\nKey Features Demonstrated:")
        print("✅ Real-time quality score tracking")
        print("✅ Quality trend analysis")
        print("✅ Automated alert generation")
        print("✅ Comprehensive reporting")
        print("✅ Portfolio-level quality insights")
        
        print(f"\nIntegration Options:")
        print("• Use with actual data streams for live monitoring")
        print("• Set up scheduled quality checks (daily/weekly)")
        print("• Integrate with alerting systems (email/Slack)")
        print("• Store quality history in databases for long-term analysis")
        
    except Exception as e:
        print(f"✗ Error during quality monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()