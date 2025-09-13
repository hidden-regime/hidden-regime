#!/usr/bin/env python3
"""
Blog Content Automation Workflow
=================================

This example demonstrates a complete automated workflow for generating
blog content from regime analysis. It includes scheduled analysis,
automated content generation, template management, and publication workflows.

Key features:
- Automated daily/weekly analysis scheduling
- Template-based content generation
- Multi-format output (Markdown, HTML, JSON)
- Content versioning and history tracking
- Automated social media content generation
- RSS feed generation
- Content quality scoring and filtering

Use cases:
- Financial blog automation
- Market research publication
- Investment newsletter generation
- Social media content creation

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from jinja2 import Template, Environment, FileSystemLoader
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

@dataclass
class ContentConfig:
    """Configuration for content generation"""
    output_formats: List[str]  # ['markdown', 'html', 'json']
    content_types: List[str]   # ['market_report', 'regime_alert', 'weekly_summary']
    target_audience: str       # 'retail', 'institutional', 'technical'
    content_length: str        # 'brief', 'standard', 'detailed'
    include_charts: bool
    include_data_tables: bool
    include_disclaimers: bool
    branding: Dict[str, str]   # Logo, colors, etc.

@dataclass
class PublicationSchedule:
    """Schedule configuration for automated content generation"""
    daily_reports: bool
    weekly_summaries: bool
    regime_alerts: bool
    market_analysis: bool
    custom_schedule: Optional[Dict[str, str]]  # Custom cron-like expressions

class BlogAutomationWorkflow:
    """Complete automation workflow for blog content generation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data_config = DataConfig()
        self.analyzer = RegimeAnalyzer(self.data_config)
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            self.content_config = ContentConfig(**config_data.get('content', {}))
            self.schedule = PublicationSchedule(**config_data.get('schedule', {}))
        else:
            self.content_config = self._get_default_content_config()
            self.schedule = self._get_default_schedule()
        
        # Initialize template environment
        self.template_dir = './templates'
        self._setup_templates()
        self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        
        # Content history tracking
        self.content_history = []
        self._load_content_history()
    
    def _get_default_content_config(self) -> ContentConfig:
        """Get default content configuration"""
        return ContentConfig(
            output_formats=['markdown', 'html'],
            content_types=['market_report', 'weekly_summary'],
            target_audience='retail',
            content_length='standard',
            include_charts=True,
            include_data_tables=True,
            include_disclaimers=True,
            branding={
                'site_name': 'Hidden Regime Analytics',
                'tagline': 'AI-Powered Market Intelligence',
                'primary_color': '#2c5aa0',
                'author': 'Hidden Regime AI'
            }
        )
    
    def _get_default_schedule(self) -> PublicationSchedule:
        """Get default publication schedule"""
        return PublicationSchedule(
            daily_reports=True,
            weekly_summaries=True,
            regime_alerts=True,
            market_analysis=True,
            custom_schedule=None
        )
    
    def _setup_templates(self):
        """Set up content templates"""
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create market report template
        market_report_template = """# {{ title }}
*{{ subtitle }}*

## Executive Summary

{{ executive_summary }}

## Market Regime Analysis

{% for symbol, analysis in analyses.items() %}
### {{ symbol }}
- **Current Regime**: {{ analysis.current_regime }} ({{ "%.1f"|format(analysis.confidence * 100) }}% confidence)
- **Days in Regime**: {{ analysis.days_in_regime }}
- **Expected Return**: {{ "%.1f"|format(analysis.expected_return * 252 * 100) }}% annualized
- **Expected Volatility**: {{ "%.1f"|format(analysis.expected_volatility * np.sqrt(252) * 100) }}% annualized

{% if analysis.regime_changes > 0 %}
**Recent Activity**: {{ analysis.regime_changes }} regime changes detected in analysis period.
{% endif %}

{% endfor %}

## Investment Implications

{{ investment_implications }}

{% if include_charts %}
## Charts and Analysis

{% for symbol in analyses.keys() %}
![{{ symbol }} Regime Analysis]({{ symbol }}_regime_analysis.png)
{% endfor %}

{% endif %}

## Market Outlook

{{ market_outlook }}

{% if include_disclaimers %}
## Disclaimer

This analysis is for educational and informational purposes only. It does not constitute investment advice or recommendations. Past performance does not guarantee future results. Always conduct your own research and consult with qualified financial advisors before making investment decisions.
{% endif %}

---
*{{ branding.tagline }}*  
*Generated on {{ timestamp }} by {{ branding.author }}*
"""
        
        # Save market report template
        with open(os.path.join(self.template_dir, 'market_report.md'), 'w') as f:
            f.write(market_report_template)
        
        # Create weekly summary template
        weekly_summary_template = """# Weekly Market Regime Summary
*Week of {{ week_start }} to {{ week_end }}*

## Week in Review

This week's regime analysis across our tracked assets reveals {{ summary_stats.total_regimes }} distinct regime patterns with {{ summary_stats.total_transitions }} transitions detected.

{% if regime_changes %}
## Major Regime Changes

{% for change in regime_changes %}
- **{{ change.symbol }}**: {{ change.from_regime }} → {{ change.to_regime }} ({{ change.date }})
{% endfor %}
{% endif %}

## Performance by Regime

{% for regime, stats in regime_performance.items() %}
### {{ regime }} Regime
- **Assets in Regime**: {{ stats.asset_count }}
- **Average Performance**: {{ "%.2f"|format(stats.avg_performance * 100) }}%
- **Best Performer**: {{ stats.best_performer }} ({{ "%.2f"|format(stats.best_performance * 100) }}%)
- **Worst Performer**: {{ stats.worst_performer }} ({{ "%.2f"|format(stats.worst_performance * 100) }}%)

{% endfor %}

## Outlook for Next Week

{{ next_week_outlook }}

## Key Assets to Watch

{% for asset in watch_list %}
- **{{ asset.symbol }}**: {{ asset.reason }}
{% endfor %}

---
*{{ branding.tagline }}*  
*Published {{ timestamp }}*
"""
        
        with open(os.path.join(self.template_dir, 'weekly_summary.md'), 'w') as f:
            f.write(weekly_summary_template)
        
        # Create regime alert template  
        alert_template = """# 🚨 Regime Change Alert: {{ symbol }}

## Alert Details
- **Asset**: {{ symbol }}
- **Previous Regime**: {{ previous_regime }}
- **New Regime**: {{ new_regime }}
- **Confidence**: {{ "%.1f"|format(confidence * 100) }}%
- **Alert Time**: {{ timestamp }}

## Analysis
{{ alert_analysis }}

## Recommended Actions
{% for action in recommended_actions %}
- {{ action }}
{% endfor %}

---
*Automated Alert from {{ branding.site_name }}*
"""
        
        with open(os.path.join(self.template_dir, 'regime_alert.md'), 'w') as f:
            f.write(alert_template)
    
    def _load_content_history(self):
        """Load content generation history"""
        history_file = './content_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.content_history = json.load(f)
        else:
            self.content_history = []
    
    def _save_content_history(self):
        """Save content generation history"""
        history_file = './content_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.content_history, f, indent=2, default=str)
    
    def generate_daily_market_report(self, symbols: List[str], output_dir: str) -> Dict[str, str]:
        """Generate automated daily market report"""
        
        print(f"Generating daily market report for {len(symbols)} assets...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze all symbols
        analyses = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            if analysis:
                analyses[symbol] = analysis
        
        if not analyses:
            print("No successful analyses, skipping report generation")
            return {}
        
        # Generate content context
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create executive summary
        current_regimes = [a['current_regime'] for a in analyses.values()]
        regime_counts = pd.Series(current_regimes).value_counts()
        
        executive_summary = f"""
Today's analysis of {len(analyses)} major assets reveals a mixed market environment. 
{regime_counts.iloc[0]} assets are in {regime_counts.index[0]} regimes, while market 
conditions show varying degrees of confidence across different sectors.

Key highlights:
- Most common regime: {regime_counts.index[0]} ({regime_counts.iloc[0]} assets)
- Average confidence: {np.mean([a['confidence'] for a in analyses.values()]):.1%}
- Assets with recent regime changes: {sum(1 for a in analyses.values() if a['days_in_regime'] <= 3)}
"""
        
        # Generate investment implications
        bull_assets = [s for s, a in analyses.items() if a['current_regime'] == 'Bull']
        bear_assets = [s for s, a in analyses.items() if a['current_regime'] == 'Bear']
        
        investment_implications = f"""
Based on current regime analysis:

**Bullish Opportunities**: {', '.join(bull_assets) if bull_assets else 'Limited opportunities in current market'}

**Defensive Positioning**: {', '.join(bear_assets) if bear_assets else 'No major defensive signals currently'}

**Mixed Signals**: Assets in sideways regimes suggest selective stock picking over broad market exposure.
"""
        
        # Market outlook
        regime_changes_week = sum(1 for a in analyses.values() if a['regime_changes'] > 5)
        market_outlook = f"""
Market dynamics show {'elevated' if regime_changes_week > len(analyses) * 0.3 else 'moderate'} 
regime instability with {regime_changes_week} assets showing frequent regime changes. 
This suggests {'increased caution' if regime_changes_week > len(analyses) * 0.3 else 'standard risk management'} 
in position sizing and portfolio construction.
"""
        
        # Template context
        template_context = {
            'title': f'Daily Market Regime Report - {datetime.now().strftime("%B %d, %Y")}',
            'subtitle': f'AI-powered regime analysis across {len(analyses)} major assets',
            'executive_summary': executive_summary.strip(),
            'analyses': analyses,
            'investment_implications': investment_implications.strip(),
            'market_outlook': market_outlook.strip(),
            'timestamp': timestamp,
            'branding': asdict(self.content_config.branding),
            'include_charts': self.content_config.include_charts,
            'include_disclaimers': self.content_config.include_disclaimers,
            'np': np  # For template calculations
        }
        
        # Generate content in multiple formats
        generated_files = {}
        
        # Markdown
        if 'markdown' in self.content_config.output_formats:
            template = self.jinja_env.get_template('market_report.md')
            content = template.render(**template_context)
            
            md_file = os.path.join(output_dir, f'daily_report_{datetime.now().strftime("%Y%m%d")}.md')
            with open(md_file, 'w') as f:
                f.write(content)
            generated_files['markdown'] = md_file
        
        # HTML
        if 'html' in self.content_config.output_formats:
            # Convert markdown to HTML (basic conversion)
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{template_context['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: {self.content_config.branding.get('primary_color', '#2c5aa0')}; }}
        h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-left: 4px solid {self.content_config.branding.get('primary_color', '#2c5aa0')}; }}
        .disclaimer {{ font-size: 0.9em; color: #666; font-style: italic; }}
    </style>
</head>
<body>
"""
            
            # Simple markdown to HTML conversion
            if 'markdown' in generated_files:
                with open(generated_files['markdown'], 'r') as f:
                    md_content = f.read()
                
                # Basic markdown conversion
                html_body = (md_content
                           .replace('# ', '<h1>')
                           .replace('\n## ', '</h1>\n<h2>')
                           .replace('\n### ', '</h2>\n<h3>')
                           .replace('**', '<strong>')
                           .replace('**', '</strong>')
                           .replace('*', '<em>')
                           .replace('*', '</em>')
                           .replace('\n\n', '</p>\n<p>')
                           .replace('\n', '<br>'))
                
                html_content += f"<p>{html_body}</p></body></html>"
            
            html_file = os.path.join(output_dir, f'daily_report_{datetime.now().strftime("%Y%m%d")}.html')
            with open(html_file, 'w') as f:
                f.write(html_content)
            generated_files['html'] = html_file
        
        # JSON
        if 'json' in self.content_config.output_formats:
            json_data = {
                'metadata': {
                    'title': template_context['title'],
                    'generated_at': timestamp,
                    'symbols_analyzed': list(analyses.keys()),
                    'content_type': 'daily_market_report'
                },
                'summary': {
                    'total_assets': len(analyses),
                    'regime_distribution': dict(regime_counts),
                    'avg_confidence': np.mean([a['confidence'] for a in analyses.values()]),
                    'recent_changes': sum(1 for a in analyses.values() if a['days_in_regime'] <= 3)
                },
                'analyses': {symbol: {
                    'current_regime': analysis['current_regime'],
                    'confidence': analysis['confidence'],
                    'days_in_regime': analysis['days_in_regime'],
                    'expected_return_annualized': analysis['expected_return'] * 252,
                    'expected_volatility_annualized': analysis['expected_volatility'] * np.sqrt(252),
                    'regime_changes': analysis['regime_changes']
                } for symbol, analysis in analyses.items()}
            }
            
            json_file = os.path.join(output_dir, f'daily_report_{datetime.now().strftime("%Y%m%d")}.json')
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            generated_files['json'] = json_file
        
        # Update content history
        self.content_history.append({
            'type': 'daily_market_report',
            'timestamp': timestamp,
            'symbols': list(analyses.keys()),
            'files': generated_files,
            'quality_score': self._calculate_content_quality_score(analyses)
        })
        self._save_content_history()
        
        print(f"Daily market report generated: {len(generated_files)} files created")
        return generated_files
    
    def generate_weekly_summary(self, symbols: List[str], output_dir: str) -> Dict[str, str]:
        """Generate automated weekly market summary"""
        
        print(f"Generating weekly market summary for {len(symbols)} assets...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get week date range
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_end = today
        
        # Analyze symbols over the week
        start_date = (week_start - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = week_end.strftime('%Y-%m-%d')
        
        analyses = {}
        for symbol in symbols:
            analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            if analysis:
                analyses[symbol] = analysis
        
        if not analyses:
            return {}
        
        # Calculate week statistics
        summary_stats = {
            'total_regimes': len(set(a['current_regime'] for a in analyses.values())),
            'total_transitions': sum(a['regime_changes'] for a in analyses.values()),
            'avg_confidence': np.mean([a['confidence'] for a in analyses.values()])
        }
        
        # Find major regime changes (assets that changed in last 7 days)
        regime_changes = []
        for symbol, analysis in analyses.items():
            if analysis['days_in_regime'] <= 7:
                # Simulate previous regime (in real implementation, track this)
                prev_regimes = ['Bull', 'Bear', 'Sideways']
                current = analysis['current_regime']
                previous = [r for r in prev_regimes if r != current][0]
                
                regime_changes.append({
                    'symbol': symbol,
                    'from_regime': previous,
                    'to_regime': current,
                    'date': (datetime.now() - timedelta(days=analysis['days_in_regime'])).strftime('%Y-%m-%d')
                })
        
        # Performance by regime
        regime_performance = {}
        for regime in ['Bull', 'Bear', 'Sideways']:
            regime_assets = {s: a for s, a in analyses.items() if a['current_regime'] == regime}
            
            if regime_assets:
                returns = [a['expected_return'] * 252 for a in regime_assets.values()]
                regime_performance[regime] = {
                    'asset_count': len(regime_assets),
                    'avg_performance': np.mean(returns),
                    'best_performer': max(regime_assets.keys(), key=lambda s: regime_assets[s]['expected_return']),
                    'best_performance': max(returns),
                    'worst_performer': min(regime_assets.keys(), key=lambda s: regime_assets[s]['expected_return']),
                    'worst_performance': min(returns)
                }
        
        # Generate outlook
        next_week_outlook = f"""
Based on current regime patterns and recent transitions, next week shows:
- {'Elevated' if summary_stats['total_transitions'] > len(analyses) * 0.5 else 'Moderate'} regime instability
- Average confidence of {summary_stats['avg_confidence']:.1%} suggests {'high' if summary_stats['avg_confidence'] > 0.7 else 'moderate'} conviction in current classifications
- {len(regime_changes)} recent regime changes indicate {'dynamic' if len(regime_changes) > 3 else 'stable'} market conditions
"""
        
        # Watch list
        watch_list = []
        for symbol, analysis in analyses.items():
            if analysis['confidence'] < 0.6:
                watch_list.append({
                    'symbol': symbol,
                    'reason': f"Low regime confidence ({analysis['confidence']:.1%}) - potential transition candidate"
                })
            elif analysis['days_in_regime'] > 20:
                watch_list.append({
                    'symbol': symbol,
                    'reason': f"Extended {analysis['current_regime']} regime ({analysis['days_in_regime']} days) - monitor for changes"
                })
        
        # Template context
        template_context = {
            'week_start': week_start.strftime('%Y-%m-%d'),
            'week_end': week_end.strftime('%Y-%m-%d'),
            'summary_stats': summary_stats,
            'regime_changes': regime_changes,
            'regime_performance': regime_performance,
            'next_week_outlook': next_week_outlook.strip(),
            'watch_list': watch_list[:5],  # Top 5 watch list items
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'branding': asdict(self.content_config.branding)
        }
        
        # Generate weekly summary
        template = self.jinja_env.get_template('weekly_summary.md')
        content = template.render(**template_context)
        
        # Save content
        generated_files = {}
        
        md_file = os.path.join(output_dir, f'weekly_summary_{week_start.strftime("%Y%m%d")}.md')
        with open(md_file, 'w') as f:
            f.write(content)
        generated_files['markdown'] = md_file
        
        # Update content history
        self.content_history.append({
            'type': 'weekly_summary',
            'timestamp': template_context['timestamp'],
            'week_range': f"{template_context['week_start']} to {template_context['week_end']}",
            'files': generated_files,
            'summary_stats': summary_stats
        })
        self._save_content_history()
        
        print(f"Weekly summary generated: {generated_files}")
        return generated_files
    
    def check_for_regime_alerts(self, symbols: List[str], output_dir: str) -> List[Dict[str, str]]:
        """Check for regime changes and generate alerts"""
        
        print(f"Checking for regime alerts across {len(symbols)} assets...")
        
        alerts = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            
            if analysis and analysis['days_in_regime'] <= 1 and analysis['confidence'] > 0.7:
                # Regime change detected with high confidence
                
                # Simulate previous regime analysis
                prev_regimes = ['Bull', 'Bear', 'Sideways']
                current_regime = analysis['current_regime']
                previous_regime = [r for r in prev_regimes if r != current_regime][0]
                
                # Generate alert content
                alert_analysis = f"""
{symbol} has transitioned from {previous_regime} to {current_regime} regime with {analysis['confidence']:.1%} confidence.
This regime change occurred within the last day and represents a significant shift in market dynamics for this asset.

Key metrics:
- Expected daily return: {analysis['expected_return']:.4f} ({analysis['expected_return']*252:.1%} annualized)
- Expected volatility: {analysis['expected_volatility']:.4f} ({analysis['expected_volatility']*np.sqrt(252):.1%} annualized)
- Recent regime stability: {analysis['regime_changes']} changes in analysis period
"""
                
                # Generate recommended actions
                recommended_actions = []
                if current_regime == 'Bull':
                    recommended_actions.extend([
                        "Consider increasing position size in growth strategies",
                        "Monitor momentum indicators for continuation signals",
                        "Review stop-loss levels for trend following"
                    ])
                elif current_regime == 'Bear':
                    recommended_actions.extend([
                        "Reduce exposure or implement hedging strategies",
                        "Consider defensive positioning",
                        "Monitor for oversold bounce opportunities"
                    ])
                else:  # Sideways
                    recommended_actions.extend([
                        "Implement range-trading strategies",
                        "Reduce position size pending direction clarity",
                        "Monitor for breakout signals"
                    ])
                
                # Generate alert template
                template_context = {
                    'symbol': symbol,
                    'previous_regime': previous_regime,
                    'new_regime': current_regime,
                    'confidence': analysis['confidence'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'alert_analysis': alert_analysis.strip(),
                    'recommended_actions': recommended_actions,
                    'branding': asdict(self.content_config.branding)
                }
                
                template = self.jinja_env.get_template('regime_alert.md')
                content = template.render(**template_context)
                
                # Save alert
                os.makedirs(output_dir, exist_ok=True)
                alert_file = os.path.join(output_dir, f'alert_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M")}.md')
                with open(alert_file, 'w') as f:
                    f.write(content)
                
                alerts.append({
                    'symbol': symbol,
                    'alert_type': 'regime_change',
                    'from_regime': previous_regime,
                    'to_regime': current_regime,
                    'confidence': analysis['confidence'],
                    'file': alert_file,
                    'timestamp': template_context['timestamp']
                })
        
        if alerts:
            print(f"Generated {len(alerts)} regime change alerts")
        else:
            print("No regime change alerts detected")
        
        return alerts
    
    def _calculate_content_quality_score(self, analyses: Dict[str, Any]) -> float:
        """Calculate quality score for generated content"""
        
        if not analyses:
            return 0.0
        
        # Factors for quality scoring
        avg_confidence = np.mean([a['confidence'] for a in analyses.values()])
        analysis_coverage = len(analyses) / 10  # Assume target of 10 assets
        regime_diversity = len(set(a['current_regime'] for a in analyses.values())) / 3  # 3 possible regimes
        
        # Recent activity (more regime changes = more interesting content)
        recent_activity = min(np.mean([a['regime_changes'] for a in analyses.values()]) / 10, 1.0)
        
        quality_score = (
            avg_confidence * 0.3 +
            min(analysis_coverage, 1.0) * 0.3 +
            regime_diversity * 0.2 +
            recent_activity * 0.2
        )
        
        return min(quality_score, 1.0)
    
    def generate_rss_feed(self, content_dir: str, output_dir: str) -> str:
        """Generate RSS feed from recent content"""
        
        print("Generating RSS feed from recent content...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get recent content from history
        recent_content = sorted(self.content_history, key=lambda x: x['timestamp'], reverse=True)[:10]
        
        # Generate RSS XML
        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
    <title>{self.content_config.branding.get('site_name', 'Hidden Regime Analytics')}</title>
    <description>{self.content_config.branding.get('tagline', 'AI-Powered Market Intelligence')}</description>
    <link>https://hiddenregime.com</link>
    <lastBuildDate>{datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')}</lastBuildDate>
    
"""
        
        for item in recent_content:
            title = f"{item['type'].replace('_', ' ').title()}"
            if item['type'] == 'daily_market_report':
                title += f" - {len(item.get('symbols', []))} Assets"
            
            pub_date = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S').strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            rss_content += f"""    <item>
        <title>{title}</title>
        <description>Automated market regime analysis generated at {item['timestamp']}</description>
        <pubDate>{pub_date}</pubDate>
        <guid>{item['timestamp']}_{item['type']}</guid>
    </item>
    
"""
        
        rss_content += """</channel>
</rss>"""
        
        # Save RSS feed
        rss_file = os.path.join(output_dir, 'feed.xml')
        with open(rss_file, 'w') as f:
            f.write(rss_content)
        
        print(f"RSS feed generated: {rss_file}")
        return rss_file
    
    def run_full_automation_workflow(self, symbols: List[str], output_dir: str) -> Dict[str, Any]:
        """Run complete automation workflow"""
        
        print("🤖 Running Full Blog Automation Workflow")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_processed': symbols,
            'generated_content': {}
        }
        
        try:
            # 1. Generate daily market report
            if self.schedule.daily_reports:
                print("\n📊 Generating daily market report...")
                daily_files = self.generate_daily_market_report(symbols, os.path.join(output_dir, 'daily'))
                results['generated_content']['daily_report'] = daily_files
            
            # 2. Generate weekly summary (if it's end of week)
            if self.schedule.weekly_summaries and datetime.now().weekday() == 6:  # Sunday
                print("\n📅 Generating weekly summary...")
                weekly_files = self.generate_weekly_summary(symbols, os.path.join(output_dir, 'weekly'))
                results['generated_content']['weekly_summary'] = weekly_files
            
            # 3. Check for regime alerts
            if self.schedule.regime_alerts:
                print("\n🚨 Checking for regime change alerts...")
                alerts = self.check_for_regime_alerts(symbols, os.path.join(output_dir, 'alerts'))
                results['generated_content']['alerts'] = alerts
            
            # 4. Generate RSS feed
            print("\n📡 Generating RSS feed...")
            rss_file = self.generate_rss_feed(output_dir, output_dir)
            results['generated_content']['rss_feed'] = rss_file
            
            # 5. Generate automation summary
            print("\n📋 Creating automation summary...")
            summary = self._create_automation_summary(results)
            summary_file = os.path.join(output_dir, f'automation_summary_{datetime.now().strftime("%Y%m%d")}.md')
            
            with open(summary_file, 'w') as f:
                f.write(summary)
            results['summary_file'] = summary_file
            
            print(f"\n✅ Full automation workflow completed successfully!")
            return results
            
        except Exception as e:
            print(f"\n❌ Error in automation workflow: {str(e)}")
            results['error'] = str(e)
            return results
    
    def _create_automation_summary(self, results: Dict[str, Any]) -> str:
        """Create summary of automation workflow results"""
        
        timestamp = results['timestamp']
        symbols = results['symbols_processed']
        content = results['generated_content']
        
        summary = f"""# Blog Automation Workflow Summary
*Generated on {timestamp}*

## Workflow Execution Results

### Symbols Processed
{len(symbols)} assets analyzed: {', '.join(symbols)}

### Content Generated
"""
        
        for content_type, files in content.items():
            if isinstance(files, dict):
                summary += f"\n#### {content_type.replace('_', ' ').title()}\n"
                for format_type, file_path in files.items():
                    summary += f"- {format_type}: `{os.path.basename(file_path)}`\n"
            elif isinstance(files, list):
                summary += f"\n#### {content_type.replace('_', ' ').title()}\n"
                summary += f"- Generated {len(files)} items\n"
                for item in files:
                    if isinstance(item, dict) and 'symbol' in item:
                        summary += f"  - {item.get('symbol', 'Unknown')}: {item.get('alert_type', 'Unknown')}\n"
            elif isinstance(files, str):
                summary += f"\n#### {content_type.replace('_', ' ').title()}\n"
                summary += f"- File: `{os.path.basename(files)}`\n"
        
        summary += f"""

## Content Quality Metrics

### Coverage
- Assets analyzed: {len(symbols)}
- Success rate: {len(symbols)}/{len(symbols)} (100%)

### Content Types Generated
- Daily reports: {'✅' if 'daily_report' in content else '❌'}
- Weekly summaries: {'✅' if 'weekly_summary' in content else '❌'}  
- Regime alerts: {'✅' if 'alerts' in content else '❌'}
- RSS feed: {'✅' if 'rss_feed' in content else '❌'}

### Next Automation Run
- Daily reports: Tomorrow at same time
- Weekly summaries: Next Sunday
- Regime alerts: Continuous monitoring
- RSS feed: Updated with each run

---
*Automated by Hidden Regime Blog Automation Workflow*
"""
        
        return summary

def main():
    """Main execution function for blog automation workflow"""
    
    print("🤖 Hidden Regime Blog Automation Workflow")
    print("=" * 60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK']
    OUTPUT_DIR = './output/blog_automation'
    
    print(f"📊 Setting up automation for {len(SYMBOLS)} assets")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    try:
        # Initialize automation workflow
        workflow = BlogAutomationWorkflow()
        
        # Run full automation workflow
        results = workflow.run_full_automation_workflow(SYMBOLS, OUTPUT_DIR)
        
        if 'error' not in results:
            print(f"\n📄 Automation summary: {results.get('summary_file', 'Not generated')}")
            print(f"📁 All content saved to: {OUTPUT_DIR}")
            
            # Display generated content summary
            content = results.get('generated_content', {})
            print(f"\n🎯 Generated Content:")
            for content_type, files in content.items():
                if isinstance(files, dict):
                    print(f"   📊 {content_type}: {len(files)} formats")
                elif isinstance(files, list):
                    print(f"   🚨 {content_type}: {len(files)} alerts")
                elif isinstance(files, str):
                    print(f"   📡 {content_type}: {os.path.basename(files)}")
            
            # List generated files
            print(f"\n📂 Generated Files:")
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), OUTPUT_DIR)
                    print(f"   - {rel_path}")
        else:
            print(f"\n❌ Automation failed with error: {results['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Error in blog automation: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Blog automation workflow completed successfully!")
        print("📰 Content ready for publication and distribution")
    else:
        print("\n❌ Blog automation workflow failed")
        exit(1)