"""
Batch HMM Processing Engine

High-performance batch processing system for running HMM regime detection
across large stock universes. Optimized for parallel execution with proper
error handling and progress monitoring.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import DataConfig, DataLoader
from ..models import HiddenMarkovModel, HMMConfig


@dataclass
class ScreeningConfig:
    """Configuration for stock screening parameters."""
    
    # Data loading settings
    period_days: int = 252  # 1 year of data
    min_data_points: int = 100  # Minimum valid data points
    data_config: Optional[DataConfig] = None
    
    # HMM settings
    n_states: int = 3
    hmm_config: Optional[HMMConfig] = None
    
    # Processing settings
    max_workers: int = 8
    use_multiprocessing: bool = True
    batch_size: int = 50
    timeout_per_stock: int = 30  # seconds
    
    # Output settings
    save_individual_results: bool = False
    results_directory: Optional[str] = None
    verbose: bool = True


class BatchHMMProcessor:
    """
    High-performance batch processor for HMM regime detection across stock universes.
    
    Handles parallel processing, error management, and progress tracking for
    large-scale stock screening operations.
    """
    
    def __init__(self, config: Optional[ScreeningConfig] = None):
        """
        Initialize batch processor with configuration.
        
        Args:
            config: Screening configuration (uses defaults if None)
        """
        self.config = config or ScreeningConfig()
        self.results_cache = {}
        self._setup_data_loader()
        self._setup_hmm_config()
    
    def _setup_data_loader(self):
        """Setup data loader with optimized configuration for batch processing."""
        if self.config.data_config is None:
            self.data_config = DataConfig(
                use_ohlc_average=True,
                include_volume=True,
                cache_enabled=True,
                max_missing_data_pct=0.1  # Allow some missing data for speed
            )
        else:
            self.data_config = self.config.data_config
        
        self.data_loader = DataLoader(self.data_config)
    
    def _setup_hmm_config(self):
        """Setup HMM configuration optimized for batch processing."""
        if self.config.hmm_config is None:
            self.hmm_config = HMMConfig(
                n_states=self.config.n_states,
                max_iterations=100,  # Reduced for speed
                tolerance=1e-3,  # Slightly relaxed for speed
                random_seed=42,
                initialization_method='kmeans'
            )
        else:
            self.hmm_config = self.config.hmm_config
    
    def process_stock_list(self, 
                          tickers: List[str],
                          progress_callback: Optional[callable] = None) -> Dict[str, Dict]:
        """
        Process a list of stock tickers using HMM regime detection.
        
        Args:
            tickers: List of stock symbols to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping tickers to their analysis results
        """
        if self.config.verbose:
            print(f"Processing {len(tickers)} stocks with {self.config.max_workers} workers...")
        
        results = {}
        failed_tickers = []
        
        # Process in batches to manage memory
        total_batches = (len(tickers) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(tickers))
            batch_tickers = tickers[start_idx:end_idx]
            
            if self.config.verbose:
                print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_tickers)} stocks)")
            
            batch_results, batch_failures = self._process_batch(batch_tickers)
            
            results.update(batch_results)
            failed_tickers.extend(batch_failures)
            
            if progress_callback:
                progress = (batch_idx + 1) / total_batches
                progress_callback(progress, len(results), len(failed_tickers))
        
        if self.config.verbose:
            success_rate = len(results) / len(tickers) if tickers else 0
            print(f"Batch processing complete: {len(results)} successful, {len(failed_tickers)} failed")
            print(f"Success rate: {success_rate:.1%}")
        
        return {
            'results': results,
            'failed_tickers': failed_tickers,
            'success_rate': success_rate,
            'total_processed': len(tickers)
        }
    
    def _process_batch(self, tickers: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
        """Process a batch of tickers in parallel."""
        results = {}
        failed_tickers = []
        
        if self.config.use_multiprocessing:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_ticker = {
                executor.submit(self._process_single_stock, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results with progress bar
            if self.config.verbose:
                futures = tqdm(as_completed(future_to_ticker), total=len(tickers), 
                              desc="Processing stocks", leave=False)
            else:
                futures = as_completed(future_to_ticker)
            
            for future in futures:
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_stock)
                    if result is not None:
                        results[ticker] = result
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Failed to process {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        return results, failed_tickers
    
    def _process_single_stock(self, ticker: str) -> Optional[Dict]:
        """
        Process a single stock ticker.
        
        This method is designed to be serializable for multiprocessing.
        
        Args:
            ticker: Stock symbol to process
            
        Returns:
            Analysis results dictionary or None if processing failed
        """
        try:
            # Load data
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=self.config.period_days + 30)  # Extra buffer
            
            data = self.data_loader.load_stock_data(
                ticker, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if data is None or len(data) < self.config.min_data_points:
                return None
            
            # Prepare returns
            returns = data['log_return'].dropna()
            if len(returns) < self.config.min_data_points:
                return None
            
            # Take most recent period_days of data
            returns = returns.tail(self.config.period_days)
            dates = data['date'].tail(len(returns))
            
            # Train HMM
            hmm = HiddenMarkovModel(config=self.hmm_config)
            hmm.fit(returns, verbose=False)
            
            # Get regime analysis
            states = hmm.predict(returns)
            regime_analysis = hmm.analyze_regimes(returns, dates)
            
            # Calculate additional metrics
            recent_volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
            recent_return = returns.tail(20).mean() * 252  # Annualized
            
            # Package results
            result = {
                'ticker': ticker,
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'data_points': len(returns),
                'regime_analysis': regime_analysis,
                'current_regime': regime_analysis['current_regime'],
                'recent_metrics': {
                    'volatility_20d_annualized': recent_volatility,
                    'return_20d_annualized': recent_return,
                    'last_price': data['close'].iloc[-1] if 'close' in data.columns else data['price'].iloc[-1],
                },
                'hmm_model_info': {
                    'n_states': self.hmm_config.n_states,
                    'log_likelihood': hmm.log_likelihood_,
                    'converged': hmm.monitor_.converged,
                    'iterations': hmm.monitor_.iter,
                },
            }
            
            # Save individual result if requested
            if self.config.save_individual_results and self.config.results_directory:
                self._save_individual_result(ticker, result)
            
            return result
            
        except Exception as e:
            # Log error but don't raise to avoid breaking the batch
            if self.config.verbose:
                warnings.warn(f"Error processing {ticker}: {str(e)}")
            return None
    
    def _save_individual_result(self, ticker: str, result: Dict):
        """Save individual stock result to file."""
        import json
        import os
        
        if not os.path.exists(self.config.results_directory):
            os.makedirs(self.config.results_directory)
        
        # Convert numpy types to JSON-serializable types
        serializable_result = self._make_json_serializable(result)
        
        filename = f"{ticker}_hmm_analysis.json"
        filepath = os.path.join(self.config.results_directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def get_processing_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from batch processing results."""
        if 'results' not in results:
            return {}
        
        stock_results = results['results']
        
        if not stock_results:
            return {
                'total_stocks': 0,
                'success_rate': 0.0,
                'regime_distribution': {},
                'avg_confidence': 0.0
            }
        
        # Analyze regime distribution
        regime_counts = {}
        confidences = []
        data_points = []
        
        for ticker, analysis in stock_results.items():
            current_regime = analysis['current_regime']
            regime_id = current_regime['regime']
            confidence = current_regime['confidence']
            
            regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1
            confidences.append(confidence)
            data_points.append(analysis['data_points'])
        
        summary = {
            'total_stocks': len(stock_results),
            'success_rate': results.get('success_rate', 0.0),
            'failed_count': len(results.get('failed_tickers', [])),
            'regime_distribution': regime_counts,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'avg_data_points': np.mean(data_points) if data_points else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0,
        }
        
        return summary
    
    def filter_by_regime_change(self, 
                               results: Dict, 
                               days_threshold: int = 5,
                               confidence_threshold: float = 0.6) -> Dict[str, Dict]:
        """
        Filter results to find stocks with recent regime changes.
        
        Args:
            results: Batch processing results
            days_threshold: Maximum days since regime change
            confidence_threshold: Minimum confidence in regime detection
            
        Returns:
            Filtered results dictionary
        """
        if 'results' not in results:
            return {}
        
        regime_change_stocks = {}
        
        for ticker, analysis in results['results'].items():
            current_regime = analysis['current_regime']
            
            if (current_regime['days_in_regime'] <= days_threshold and
                current_regime['confidence'] >= confidence_threshold):
                
                regime_change_stocks[ticker] = analysis
        
        return regime_change_stocks
    
    def get_top_performers(self, 
                          results: Dict, 
                          metric: str = 'confidence',
                          n_top: int = 10) -> List[Tuple[str, Dict]]:
        """
        Get top performing stocks by specified metric.
        
        Args:
            results: Batch processing results
            metric: Metric to rank by ('confidence', 'volatility', 'return')
            n_top: Number of top stocks to return
            
        Returns:
            List of (ticker, analysis) tuples sorted by metric
        """
        if 'results' not in results:
            return []
        
        stock_results = results['results']
        scored_stocks = []
        
        for ticker, analysis in stock_results.items():
            if metric == 'confidence':
                score = analysis['current_regime']['confidence']
            elif metric == 'volatility':
                score = analysis['recent_metrics']['volatility_20d_annualized']
            elif metric == 'return':
                score = analysis['recent_metrics']['return_20d_annualized']
            else:
                score = 0.0
            
            scored_stocks.append((ticker, analysis, score))
        
        # Sort by score (descending)
        scored_stocks.sort(key=lambda x: x[2], reverse=True)
        
        # Return top N
        return [(ticker, analysis) for ticker, analysis, score in scored_stocks[:n_top]]


def create_batch_processor(period_days: int = 252,
                          max_workers: int = 8,
                          n_states: int = 3,
                          verbose: bool = True) -> BatchHMMProcessor:
    """
    Convenience function to create a batch processor with common settings.
    
    Args:
        period_days: Days of data to analyze per stock
        max_workers: Number of parallel workers
        n_states: Number of HMM states
        verbose: Enable verbose output
        
    Returns:
        Configured BatchHMMProcessor instance
    """
    config = ScreeningConfig(
        period_days=period_days,
        max_workers=max_workers,
        n_states=n_states,
        verbose=verbose
    )
    
    return BatchHMMProcessor(config)


def quick_screen(tickers: List[str], **kwargs) -> Dict:
    """
    Quick screening function for a list of tickers.
    
    Args:
        tickers: List of stock symbols
        **kwargs: Additional arguments for ScreeningConfig
        
    Returns:
        Screening results dictionary
    """
    processor = create_batch_processor(**kwargs)
    return processor.process_stock_list(tickers)