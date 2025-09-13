"""
Content Templates

Template system for generating consistent, professional blog content and reports.
Provides standardized formats for different types of analysis and ensures
publication-ready output.
"""

from typing import Dict, Optional, List


# Template definitions for different content types
ANALYSIS_TEMPLATES = {
    'market_report': {
        'title': "{ticker} Market Regime Analysis - {date}",
        'sections': [
            'executive_summary',
            'current_regime_status', 
            'regime_analysis',
            'key_insights',
            'statistical_summary',
            'charts_and_visualizations',
            'conclusion'
        ],
        'format': 'markdown'
    },
    
    'historical_analysis': {
        'title': "HMM Analysis: {event_name}",
        'sections': [
            'event_overview',
            'expected_vs_detected',
            'regime_timeline',
            'validation_results',
            'key_learnings',
            'implications'
        ],
        'format': 'markdown'
    },
    
    'comparative_study': {
        'title': "HMM vs Traditional Indicators: {ticker} Analysis",
        'sections': [
            'methodology',
            'hmm_results',
            'indicator_results',
            'timing_comparison',
            'performance_comparison',
            'conclusions'
        ],
        'format': 'markdown'
    },
    
    'regime_update': {
        'title': "Market Regime Update - {date}",
        'sections': [
            'market_overview',
            'regime_highlights',
            'notable_changes',
            'sector_analysis',
            'outlook'
        ],
        'format': 'markdown'
    }
}


def get_template(template_name: str) -> Dict:
    """
    Get template configuration for specified content type.
    
    Args:
        template_name: Name of template ('market_report', 'historical_analysis', etc.)
        
    Returns:
        Template configuration dictionary
        
    Raises:
        ValueError: If template_name not found
    """
    if template_name not in ANALYSIS_TEMPLATES:
        available = list(ANALYSIS_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return ANALYSIS_TEMPLATES[template_name]


def format_analysis_for_blog(
    analysis_data: Dict,
    template: Optional[Dict] = None,
    template_name: Optional[str] = None
) -> str:
    """
    Format analysis data using specified template for blog publication.
    
    Args:
        analysis_data: Analysis results dictionary
        template: Template configuration (optional if template_name provided)
        template_name: Name of template to use (optional if template provided)
        
    Returns:
        Formatted content string ready for publication
    """
    if template is None and template_name is None:
        raise ValueError("Must provide either template or template_name")
    
    if template is None:
        template = get_template(template_name)
    
    # Determine content type and format accordingly
    if 'metadata' in analysis_data:
        if 'event_name' in analysis_data['metadata']:
            return _format_historical_analysis(analysis_data, template)
        elif 'indicators_compared' in analysis_data['metadata']:
            return _format_comparative_analysis(analysis_data, template)
        elif 'ticker' in analysis_data['metadata']:
            return _format_market_report(analysis_data, template)
    
    # Default to regime update format
    return _format_regime_update(analysis_data, template)


def _format_market_report(analysis_data: Dict, template: Dict) -> str:
    """Format market regime report."""
    metadata = analysis_data['metadata']
    current_regime = analysis_data['current_regime']
    summary_stats = analysis_data['summary_statistics']
    insights = analysis_data['key_insights']
    
    content = f"""# {metadata['ticker']} Market Regime Analysis

**Analysis Date:** {metadata['report_date']}  
**Period:** {metadata['analysis_period']}  
**Data Points:** {metadata['data_points']}

## Executive Summary

{metadata['ticker']} is currently in a **{current_regime['name']} regime** with {current_regime['confidence']:.1%} confidence. The stock has been in this regime for {current_regime['days_in_regime']} consecutive days.

## Current Regime Status

### Regime Probabilities
"""
    
    for i, prob in enumerate(current_regime['regime_probabilities']):
        regime_names = ["Bear", "Sideways", "Bull"]
        regime_name = regime_names[i] if i < len(regime_names) else f"State {i}"
        content += f"- **{regime_name}:** {prob:.1%}\n"
    
    content += f"""
## Key Insights

"""
    
    for insight in insights:
        content += f"- {insight}\n"
    
    content += f"""
## Statistical Summary

- **Total Return:** {summary_stats['total_return']:.2%}
- **Annualized Return:** {summary_stats['annualized_return']:.2%}
- **Volatility:** {summary_stats['volatility']:.2%}
- **Sharpe Ratio:** {summary_stats['sharpe_ratio']:.2f}
- **Max Drawdown:** {summary_stats['max_drawdown']:.2%}
- **Regime Transitions:** {summary_stats['regime_transitions']}

### Price Levels
- **Current Price:** ${summary_stats['current_price']:.2f}
- **Period High:** ${summary_stats['period_high']:.2f}
- **Period Low:** ${summary_stats['period_low']:.2f}

## Charts and Visualizations

*[Charts will be embedded here when generated]*

## Conclusion

The HMM regime detection system provides valuable insights into {metadata['ticker']}'s current market behavior. The {current_regime['confidence']:.1%} confidence level in the current {current_regime['name']} regime suggests {_get_confidence_interpretation(current_regime['confidence'])}.

---

*Analysis generated using the Hidden Regime HMM system. For more information, visit [hiddenregime.com](https://hiddenregime.com)*
"""
    
    return content


def _format_historical_analysis(analysis_data: Dict, template: Dict) -> str:
    """Format historical event analysis."""
    metadata = analysis_data['metadata']
    event_info = analysis_data['event_info']
    narrative = analysis_data['narrative_insights']
    
    content = f"""# HMM Analysis: {event_info['name']}

**Period:** {metadata['period']}  
**Ticker:** {metadata['ticker']}  
**Analysis Date:** {metadata['analysis_date']}

## Event Overview

{event_info['description']}

**Expected Regime:** {event_info['expected_regime']}  
**Key Characteristics:**
"""
    
    if 'characteristics' in event_info:
        chars = event_info['characteristics']
        if isinstance(chars, dict):
            for key, value in chars.items():
                if isinstance(value, (int, float)):
                    if 'return' in key.lower():
                        content += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
                    elif 'volatility' in key.lower():
                        content += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
                    else:
                        content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
    
    content += f"""
## Key Findings

"""
    
    for insight in narrative:
        content += f"- {insight}\n"
    
    # Add validation results if available
    if analysis_data.get('validation_results'):
        validation = analysis_data['validation_results']['validation_metrics']
        content += f"""
## Validation Results

The HMM regime detection was validated against the known characteristics of this historical event:

- **Match Score:** {validation['regime_match_score']:.3f} out of 1.0
- **Regime Consistency:** {validation['regime_consistency']:.3f}
- **Validation Status:** {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}

"""
        
        if validation['validation_passed']:
            content += "The HMM successfully identified the expected regime characteristics during this historical period, demonstrating its effectiveness for regime detection.\n"
        else:
            content += "The HMM detection did not fully match expected characteristics, suggesting potential areas for model improvement.\n"
    
    content += f"""
## Implications

This analysis demonstrates how HMM regime detection performs during significant market events. Understanding these patterns helps validate the model's effectiveness and provides insights for future market analysis.

---

*Historical analysis powered by the Hidden Regime HMM system.*
"""
    
    return content


def _format_comparative_analysis(analysis_data: Dict, template: Dict) -> str:
    """Format comparative analysis report."""
    metadata = analysis_data['metadata']
    narrative = analysis_data['narrative_summary']
    
    content = f"""# HMM vs Traditional Indicators: {metadata['ticker']} Analysis

**Period:** {metadata['period']}  
**Analysis Date:** {metadata['analysis_date']}  
**Indicators Compared:** {', '.join(metadata['indicators_compared'])}

## Methodology

This analysis compares Hidden Markov Model (HMM) regime detection with traditional technical indicators to evaluate:

- Signal timing and accuracy
- Performance during different market conditions
- Practical implementation advantages

## Key Findings

"""
    
    for insight in narrative:
        content += f"- {insight}\n"
    
    content += f"""
## HMM Results

*[HMM analysis results will be populated here]*

## Traditional Indicator Results

*[Indicator analysis results will be populated here]*

## Performance Comparison

*[Backtesting and performance comparison results will be available in Phase B]*

## Conclusions

The comparison between HMM regime detection and traditional technical indicators reveals important differences in approach and effectiveness. HMM provides:

1. **Probabilistic Framework:** Unlike binary signals, HMM provides confidence levels
2. **Multiple Regime Detection:** Simultaneously identifies different market states
3. **Statistical Rigor:** Based on proven statistical models rather than heuristics

*Note: Full quantitative comparison including backtesting results will be available upon completion of technical indicators integration.*

---

*Comparative analysis powered by the Hidden Regime system.*
"""
    
    return content


def _format_regime_update(update_data: Dict, template: Dict) -> str:
    """Format regime update report."""
    metadata = update_data['metadata']
    market_summary = update_data['market_summary']
    highlights = update_data['highlights']
    
    content = f"""# Market Regime Update - {metadata['update_date'][:10]}

**Tickers Analyzed:** {metadata['tickers_analyzed']} of {metadata['tickers_requested']} requested  
**Lookback Period:** {metadata['lookback_days']} days

## Market Overview

"""
    
    if 'regime_distribution' in market_summary:
        dist = market_summary['regime_distribution']
        content += f"""Current regime distribution across analyzed stocks:

- **🐂 Bull Regimes:** {dist['bull']:.1%} of stocks
- **📈 Sideways Regimes:** {dist['sideways']:.1%} of stocks  
- **🐻 Bear Regimes:** {dist['bear']:.1%} of stocks

**Market Activity:**
- Total regime changes in period: {market_summary['total_regime_changes']}
- High-alert tickers: {market_summary['high_alert_tickers']}
- Overall market stability: {market_summary['market_stability']}

"""
    
    content += """## Notable Highlights

"""
    
    for highlight in highlights:
        if highlight['type'] == 'regime_change':
            content += f"🔄 **{highlight['ticker']}**: {highlight['description']}\n\n"
        elif highlight['type'] == 'high_confidence':
            content += f"⭐ **{highlight['ticker']}**: {highlight['description']}\n\n"
        else:
            content += f"📊 **{highlight['ticker']}**: {highlight['description']}\n\n"
    
    content += f"""
## Individual Ticker Updates

"""
    
    for ticker_update in update_data['ticker_updates'][:10]:  # Show top 10
        ticker = ticker_update['ticker']
        current = ticker_update['current_regime']
        
        regime_names = {0: "Bear 🐻", 1: "Sideways 📈", 2: "Bull 🐂"}
        regime_name = regime_names.get(current['state'], f"State {current['state']}")
        
        content += f"### {ticker}\n"
        content += f"- **Current Regime:** {regime_name} ({current['confidence']:.1%} confidence)\n"
        
        if ticker_update['recent_changes']:
            content += f"- **Recent Changes:** {len(ticker_update['recent_changes'])} in lookback period\n"
        
        if 'performance_summary' in ticker_update and not ticker_update['performance_summary'].get('insufficient_data'):
            perf = ticker_update['performance_summary']
            content += f"- **Period Return:** {perf['period_return']:.2%}\n"
        
        content += f"- **Alert Level:** {ticker_update['alert_level']}\n\n"
    
    content += f"""
---

*Market regime update generated by the Hidden Regime system. Data reflects regime detection over the past {metadata['lookback_days']} days.*
"""
    
    return content


def _get_confidence_interpretation(confidence: float) -> str:
    """Get human-readable interpretation of confidence level."""
    if confidence >= 0.9:
        return "very high confidence in the regime classification"
    elif confidence >= 0.8:
        return "high confidence in the regime classification"
    elif confidence >= 0.7:
        return "moderate confidence in the regime classification"
    elif confidence >= 0.6:
        return "reasonable confidence in the regime classification"
    else:
        return "low confidence, suggesting potential regime transition"


def format_for_social_media(analysis_data: Dict, platform: str = 'twitter') -> str:
    """
    Format analysis data for social media posts.
    
    Args:
        analysis_data: Analysis results
        platform: Social media platform ('twitter', 'linkedin', 'reddit')
        
    Returns:
        Formatted content for specified platform
    """
    if platform.lower() == 'twitter':
        return _format_twitter_post(analysis_data)
    elif platform.lower() == 'linkedin':
        return _format_linkedin_post(analysis_data)
    elif platform.lower() == 'reddit':
        return _format_reddit_post(analysis_data)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def _format_twitter_post(analysis_data: Dict) -> str:
    """Format for Twitter (character limit)."""
    if 'metadata' in analysis_data and 'ticker' in analysis_data['metadata']:
        ticker = analysis_data['metadata']['ticker']
        current_regime = analysis_data.get('current_regime', {})
        
        if current_regime:
            regime_name = current_regime.get('name', 'Unknown')
            confidence = current_regime.get('confidence', 0)
            
            post = f"📊 ${ticker} Regime Update:\n"
            post += f"Current: {regime_name} ({confidence:.0%} confidence)\n"
            post += f"#MarketRegimes #HMM #TradingAnalysis"
            
            return post
    
    return "📊 Market regime analysis available #HiddenRegime #MarketAnalysis"


def _format_linkedin_post(analysis_data: Dict) -> str:
    """Format for LinkedIn (professional tone)."""
    if 'metadata' in analysis_data and 'ticker' in analysis_data['metadata']:
        ticker = analysis_data['metadata']['ticker']
        
        post = f"Market Regime Analysis Update: ${ticker}\n\n"
        post += "Our Hidden Markov Model analysis reveals current market regime characteristics and transition probabilities. "
        post += "This quantitative approach provides institutional-grade insights into market behavior patterns.\n\n"
        post += "#QuantitativeFinance #MarketAnalysis #AlgorithmicTrading #FinTech"
        
        return post
    
    return "Advanced market regime detection using Hidden Markov Models. #QuantitativeFinance #MarketAnalysis"


def _format_reddit_post(analysis_data: Dict) -> str:
    """Format for Reddit (detailed, community-focused)."""
    if 'metadata' in analysis_data and 'ticker' in analysis_data['metadata']:
        ticker = analysis_data['metadata']['ticker']
        current_regime = analysis_data.get('current_regime', {})
        
        post = f"HMM Regime Analysis: ${ticker}\n\n"
        post += "Just ran my Hidden Markov Model on recent price action. Here's what the math is telling us:\n\n"
        
        if current_regime:
            regime_name = current_regime.get('name', 'Unknown')
            confidence = current_regime.get('confidence', 0)
            
            post += f"Current Regime: {regime_name} ({confidence:.1%} confidence)\n"
            post += f"Days in regime: {current_regime.get('days_in_regime', 'Unknown')}\n\n"
        
        post += "The HMM approach is pretty neat - it uses statistical models to identify distinct market 'regimes' "
        post += "rather than relying on traditional indicators. Thoughts?\n\n"
        post += "Happy to share the methodology if anyone's interested!"
        
        return post
    
    return "Sharing some Hidden Markov Model results for market regime detection. Interesting approach vs traditional TA!"


def create_newsletter_template(
    featured_analysis: Dict,
    market_updates: List[Dict],
    educational_content: str
) -> str:
    """
    Create newsletter template combining multiple analysis pieces.
    
    Args:
        featured_analysis: Main analysis to feature
        market_updates: List of market update summaries
        educational_content: Educational content about HMM/regimes
        
    Returns:
        Complete newsletter content
    """
    newsletter_date = datetime.now().strftime('%B %d, %Y')
    
    content = f"""# Hidden Regime Weekly Newsletter
## Market Intelligence Through Bayesian HMMs

**{newsletter_date}**

---

## 🎯 Featured Analysis

{_create_newsletter_featured_section(featured_analysis)}

---

## 📊 Market Regime Updates

"""
    
    for update in market_updates:
        content += _create_newsletter_update_section(update)
    
    content += f"""
---

## 🎓 Educational Corner

{educational_content}

---

## 📈 Coming Next Week

- Comparative analysis: HMM vs MACD performance during volatility spikes
- Sector regime analysis: Technology vs Financial regimes
- Historical deep-dive: How HMM detected the 2018 Volmageddon

---

**About Hidden Regime**

Advanced market regime detection using sophisticated Hidden Markov Models with Bayesian uncertainty quantification. 

Visit [hiddenregime.com](https://hiddenregime.com) for more analysis and tools.

*This newsletter is for educational purposes only and does not constitute financial advice.*
"""
    
    return content


def _create_newsletter_featured_section(analysis: Dict) -> str:
    """Create featured analysis section for newsletter."""
    if 'metadata' in analysis and 'ticker' in analysis['metadata']:
        ticker = analysis['metadata']['ticker']
        return f"This week we dive deep into ${ticker}'s regime characteristics and what they signal for the coming weeks..."
    
    return "Featured analysis of current market regime dynamics and transition probabilities..."


def _create_newsletter_update_section(update: Dict) -> str:
    """Create market update section for newsletter."""
    return f"### Market Update Summary\n\nKey regime changes and notable transitions across our monitored universe...\n\n"