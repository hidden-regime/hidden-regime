"""
Market Screening Engine

Main orchestration engine for running comprehensive stock screens using HMM
regime detection, technical indicators, and custom criteria. Provides high-level
interface for complex screening operations with reporting capabilities.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .batch import BatchHMMProcessor, ScreeningConfig
from .criteria import ScreeningCriteria, apply_screening_criteria
from .universes import get_custom_universe, get_sp500_universe


@dataclass
class ScreeningResult:
    """
    Result of a stock screening operation.
    
    Contains both the filtered results and metadata about the screening process.
    """
    
    # Core results
    passed_stocks: Dict[str, Dict]
    failed_stocks: Dict[str, Dict]
    screening_metadata: Dict
    
    # Summary statistics
    total_stocks: int
    passed_count: int
    failed_count: int
    success_rate: float
    
    # Criteria information
    criteria_name: str
    criteria_description: str
    
    # Processing information
    processing_time: float
    analysis_date: str


class MarketScreener:
    """
    Comprehensive market screening engine.
    
    Orchestrates the entire screening process from universe selection through
    HMM analysis to criteria application and result generation.
    """
    
    def __init__(self, 
                 config: Optional[ScreeningConfig] = None,
                 cache_results: bool = True):
        """
        Initialize market screener.
        
        Args:
            config: Screening configuration
            cache_results: Whether to cache HMM analysis results
        """
        self.config = config or ScreeningConfig()
        self.cache_results = cache_results
        self.batch_processor = BatchHMMProcessor(self.config)
        
        # Results cache
        self._analysis_cache = {}
        self._last_screen_results = None
    
    def screen_universe(self,
                       universe: Union[str, List[str]],
                       criteria: ScreeningCriteria,
                       force_reprocess: bool = False) -> ScreeningResult:
        """
        Screen a universe of stocks using specified criteria.
        
        Args:
            universe: Universe name or list of tickers to screen
            criteria: Screening criteria to apply
            force_reprocess: Force reprocessing even if cached results exist
            
        Returns:
            ScreeningResult with filtered stocks and metadata
        """
        import time
        start_time = time.time()
        
        # Get ticker list
        if isinstance(universe, str):
            if universe == 'sp500':
                tickers = get_sp500_universe()
            else:
                tickers = get_custom_universe(universe)
        else:
            tickers = universe
        
        if self.config.verbose:
            print(f"Screening {len(tickers)} stocks using criteria: {criteria.name}")
        
        # Get or perform HMM analysis
        if not force_reprocess and universe in self._analysis_cache:
            if self.config.verbose:
                print("Using cached analysis results...")
            batch_results = self._analysis_cache[universe]
        else:
            if self.config.verbose:
                print("Performing HMM analysis...")
            batch_results = self.batch_processor.process_stock_list(tickers)
            
            if self.cache_results:
                self._analysis_cache[universe] = batch_results
        
        # Apply screening criteria
        passed_stocks = {}
        failed_stocks = {}
        criteria_details = {}
        
        if 'results' in batch_results:
            for ticker, analysis in batch_results['results'].items():
                passes, details = apply_screening_criteria(analysis, criteria)
                criteria_details[ticker] = details
                
                if passes:
                    passed_stocks[ticker] = analysis
                else:
                    failed_stocks[ticker] = analysis
        
        # Create result object
        processing_time = time.time() - start_time
        
        result = ScreeningResult(
            passed_stocks=passed_stocks,
            failed_stocks=failed_stocks,
            screening_metadata={
                'criteria_details': criteria_details,
                'batch_results': batch_results,
                'universe_name': universe if isinstance(universe, str) else 'custom',
                'original_universe_size': len(tickers)
            },
            total_stocks=len(tickers),
            passed_count=len(passed_stocks),
            failed_count=len(failed_stocks),
            success_rate=len(passed_stocks) / len(tickers) if tickers else 0.0,
            criteria_name=criteria.name,
            criteria_description=criteria.description,
            processing_time=processing_time,
            analysis_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        self._last_screen_results = result
        
        if self.config.verbose:
            print(f"Screening complete: {result.passed_count}/{result.total_stocks} stocks passed ({result.success_rate:.1%})")
            print(f"Processing time: {processing_time:.1f} seconds")
        
        return result
    
    def multi_criteria_screen(self,
                            universe: Union[str, List[str]],
                            criteria_list: List[ScreeningCriteria],
                            combine_with_and: bool = True) -> Dict[str, ScreeningResult]:
        """
        Screen universe using multiple criteria sets.
        
        Args:
            universe: Universe to screen
            criteria_list: List of screening criteria
            combine_with_and: Whether to combine criteria with AND logic
            
        Returns:
            Dictionary mapping criteria names to ScreeningResults
        """
        results = {}
        
        # Run first screening to get base analysis
        base_result = self.screen_universe(universe, criteria_list[0])
        results[criteria_list[0].name] = base_result
        
        # Apply additional criteria to the same base analysis
        for criteria in criteria_list[1:]:
            # Use cached analysis from first run
            result = self._apply_criteria_to_cached_analysis(criteria, base_result.screening_metadata['batch_results'])
            results[criteria.name] = result
        
        return results
    
    def _apply_criteria_to_cached_analysis(self, criteria: ScreeningCriteria, batch_results: Dict) -> ScreeningResult:
        """Apply criteria to already-computed HMM analysis."""
        import time
        start_time = time.time()
        
        passed_stocks = {}
        failed_stocks = {}
        criteria_details = {}
        
        if 'results' in batch_results:
            for ticker, analysis in batch_results['results'].items():
                passes, details = apply_screening_criteria(analysis, criteria)
                criteria_details[ticker] = details
                
                if passes:
                    passed_stocks[ticker] = analysis
                else:
                    failed_stocks[ticker] = analysis
        
        processing_time = time.time() - start_time
        total_stocks = len(batch_results.get('results', {}))
        
        return ScreeningResult(
            passed_stocks=passed_stocks,
            failed_stocks=failed_stocks,
            screening_metadata={
                'criteria_details': criteria_details,
                'batch_results': batch_results
            },
            total_stocks=total_stocks,
            passed_count=len(passed_stocks),
            failed_count=len(failed_stocks),
            success_rate=len(passed_stocks) / total_stocks if total_stocks > 0 else 0.0,
            criteria_name=criteria.name,
            criteria_description=criteria.description,
            processing_time=processing_time,
            analysis_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def get_screening_summary(self, result: ScreeningResult) -> Dict:
        """
        Generate comprehensive summary of screening results.
        
        Args:
            result: ScreeningResult to summarize
            
        Returns:
            Dictionary with summary statistics and insights
        """
        summary = {
            'overview': {
                'total_stocks': result.total_stocks,
                'passed_count': result.passed_count,
                'failed_count': result.failed_count,
                'success_rate': result.success_rate,
                'criteria': result.criteria_name
            },
            'regime_analysis': {},
            'performance_metrics': {},
            'top_candidates': []
        }
        
        if result.passed_stocks:
            # Analyze regime distribution in passed stocks
            regime_distribution = {}
            confidence_scores = []
            recent_returns = []
            volatilities = []
            
            for ticker, analysis in result.passed_stocks.items():
                regime_id = analysis['current_regime']['regime']
                confidence = analysis['current_regime']['confidence']
                
                regime_distribution[regime_id] = regime_distribution.get(regime_id, 0) + 1
                confidence_scores.append(confidence)
                
                # Collect performance metrics
                recent_returns.append(analysis['recent_metrics']['return_20d_annualized'])
                volatilities.append(analysis['recent_metrics']['volatility_20d_annualized'])
            
            summary['regime_analysis'] = {
                'regime_distribution': regime_distribution,
                'avg_confidence': np.mean(confidence_scores),
                'confidence_std': np.std(confidence_scores),
                'min_confidence': np.min(confidence_scores),
                'max_confidence': np.max(confidence_scores)
            }
            
            summary['performance_metrics'] = {
                'avg_return_20d': np.mean(recent_returns),
                'avg_volatility_20d': np.mean(volatilities),
                'return_std': np.std(recent_returns),
                'volatility_std': np.std(volatilities),
                'sharpe_estimate': np.mean(recent_returns) / np.mean(volatilities) if np.mean(volatilities) > 0 else 0
            }
            
            # Get top candidates by confidence
            top_candidates = sorted(
                result.passed_stocks.items(),
                key=lambda x: x[1]['current_regime']['confidence'],
                reverse=True
            )[:10]
            
            summary['top_candidates'] = [
                {
                    'ticker': ticker,
                    'confidence': analysis['current_regime']['confidence'],
                    'regime': analysis['current_regime']['regime'],
                    'days_in_regime': analysis['current_regime']['days_in_regime'],
                    'recent_return': analysis['recent_metrics']['return_20d_annualized'],
                    'volatility': analysis['recent_metrics']['volatility_20d_annualized']
                }
                for ticker, analysis in top_candidates
            ]
        
        return summary
    
    def export_results(self, 
                      result: ScreeningResult,
                      filepath: str,
                      format: str = 'csv',
                      include_details: bool = False) -> None:
        """
        Export screening results to file.
        
        Args:
            result: ScreeningResult to export
            filepath: Path for output file
            format: Export format ('csv', 'json', 'excel')
            include_details: Whether to include detailed analysis
        """
        if format == 'csv':
            self._export_csv(result, filepath, include_details)
        elif format == 'json':
            self._export_json(result, filepath, include_details)
        elif format == 'excel':
            self._export_excel(result, filepath, include_details)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, result: ScreeningResult, filepath: str, include_details: bool):
        """Export results to CSV format."""
        data = []
        
        for ticker, analysis in result.passed_stocks.items():
            row = {
                'ticker': ticker,
                'current_regime': analysis['current_regime']['regime'],
                'confidence': analysis['current_regime']['confidence'],
                'days_in_regime': analysis['current_regime']['days_in_regime'],
                'recent_return_20d': analysis['recent_metrics']['return_20d_annualized'],
                'volatility_20d': analysis['recent_metrics']['volatility_20d_annualized'],
                'last_price': analysis['recent_metrics']['last_price'],
                'data_points': analysis['data_points']
            }
            
            if include_details:
                regime_stats = analysis['regime_analysis']['regime_statistics']['regime_stats']
                current_regime_id = analysis['current_regime']['regime']
                regime_characteristics = regime_stats[current_regime_id]
                
                row.update({
                    'regime_mean_return': regime_characteristics['mean_return'],
                    'regime_volatility': regime_characteristics['volatility'],
                    'regime_frequency': regime_characteristics['frequency'],
                    'regime_avg_duration': regime_characteristics['avg_duration'],
                    'model_converged': analysis['hmm_model_info']['converged'],
                    'model_iterations': analysis['hmm_model_info']['iterations']
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        if self.config.verbose:
            print(f"Results exported to: {filepath}")
    
    def _export_json(self, result: ScreeningResult, filepath: str, include_details: bool):
        """Export results to JSON format."""
        import json
        
        export_data = {
            'screening_summary': {
                'criteria_name': result.criteria_name,
                'criteria_description': result.criteria_description,
                'analysis_date': result.analysis_date,
                'total_stocks': result.total_stocks,
                'passed_count': result.passed_count,
                'success_rate': result.success_rate,
                'processing_time': result.processing_time
            },
            'passed_stocks': {}
        }
        
        for ticker, analysis in result.passed_stocks.items():
            if include_details:
                # Convert numpy types for JSON serialization
                serialized_analysis = self.batch_processor._make_json_serializable(analysis)
                export_data['passed_stocks'][ticker] = serialized_analysis
            else:
                # Simplified data
                export_data['passed_stocks'][ticker] = {
                    'current_regime': analysis['current_regime'],
                    'recent_metrics': analysis['recent_metrics']
                }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        if self.config.verbose:
            print(f"Results exported to: {filepath}")
    
    def _export_excel(self, result: ScreeningResult, filepath: str, include_details: bool):
        """Export results to Excel format with multiple sheets."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary = self.get_screening_summary(result)
            
            summary_data = []
            for key, value in summary['overview'].items():
                summary_data.append({'Metric': key, 'Value': value})
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Results sheet (same as CSV)
            self._create_results_dataframe(result, include_details).to_excel(
                writer, sheet_name='Results', index=False
            )
            
            # Top candidates sheet
            if summary['top_candidates']:
                pd.DataFrame(summary['top_candidates']).to_excel(
                    writer, sheet_name='Top_Candidates', index=False
                )
        
        if self.config.verbose:
            print(f"Results exported to: {filepath}")
    
    def _create_results_dataframe(self, result: ScreeningResult, include_details: bool) -> pd.DataFrame:
        """Create DataFrame from screening results."""
        data = []
        
        for ticker, analysis in result.passed_stocks.items():
            row = {
                'ticker': ticker,
                'current_regime': analysis['current_regime']['regime'],
                'confidence': analysis['current_regime']['confidence'],
                'days_in_regime': analysis['current_regime']['days_in_regime'],
                'recent_return_20d': analysis['recent_metrics']['return_20d_annualized'],
                'volatility_20d': analysis['recent_metrics']['volatility_20d_annualized'],
                'last_price': analysis['recent_metrics']['last_price'],
                'data_points': analysis['data_points']
            }
            data.append(row)
        
        return pd.DataFrame(data)


def screen_stock_universe(universe: Union[str, List[str]],
                         criteria: ScreeningCriteria,
                         config: Optional[ScreeningConfig] = None) -> ScreeningResult:
    """
    Convenience function for quick stock universe screening.
    
    Args:
        universe: Universe name or ticker list
        criteria: Screening criteria
        config: Optional screening configuration
        
    Returns:
        ScreeningResult
    """
    screener = MarketScreener(config)
    return screener.screen_universe(universe, criteria)


def create_screening_report(result: ScreeningResult, 
                          save_path: Optional[str] = None) -> str:
    """
    Create comprehensive text report from screening results.
    
    Args:
        result: ScreeningResult to report on
        save_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    lines = [
        "# Stock Screening Report",
        "",
        f"**Analysis Date**: {result.analysis_date}",
        f"**Screening Criteria**: {result.criteria_name}",
        f"**Description**: {result.criteria_description}",
        "",
        "## Summary",
        "",
        f"- **Total Stocks Analyzed**: {result.total_stocks:,}",
        f"- **Stocks Passed Screening**: {result.passed_count:,}",
        f"- **Success Rate**: {result.success_rate:.1%}",
        f"- **Processing Time**: {result.processing_time:.1f} seconds",
        "",
        "## Top Candidates",
        ""
    ]
    
    if result.passed_stocks:
        # Sort by confidence
        top_stocks = sorted(
            result.passed_stocks.items(),
            key=lambda x: x[1]['current_regime']['confidence'],
            reverse=True
        )[:20]  # Top 20
        
        lines.extend([
            "| Ticker | Regime | Confidence | Days in Regime | 20d Return | 20d Volatility |",
            "|--------|--------|------------|----------------|------------|----------------|"
        ])
        
        for ticker, analysis in top_stocks:
            regime = analysis['current_regime']['regime']
            confidence = analysis['current_regime']['confidence']
            days = analysis['current_regime']['days_in_regime']
            ret = analysis['recent_metrics']['return_20d_annualized']
            vol = analysis['recent_metrics']['volatility_20d_annualized']
            
            lines.append(
                f"| {ticker} | {regime} | {confidence:.1%} | {days} | "
                f"{ret:.1%} | {vol:.1%} |"
            )
    else:
        lines.append("No stocks passed the screening criteria.")
    
    lines.extend([
        "",
        "---",
        "*Report generated by Hidden Regime Market Screener*"
    ])
    
    report_content = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_content)
        print(f"Report saved to: {save_path}")
    
    return report_content