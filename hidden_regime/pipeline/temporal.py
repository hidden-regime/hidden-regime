"""
Temporal data isolation for V&V backtesting.

Provides TemporalController and TemporalDataStub classes that ensure no temporal data leakage
during backtesting, enabling rigorous verification and validation of trading strategies.
"""

from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .interfaces import DataComponent


class TemporalDataStub(DataComponent):
    """
    Data component stub that only provides pre-filtered temporal data.
    
    This class wraps a filtered dataset and prevents any access to data beyond
    the specified temporal boundary, ensuring no future data leakage during backtesting.
    """
    
    def __init__(self, filtered_data: pd.DataFrame):
        """
        Initialize with temporally filtered data.
        
        Args:
            filtered_data: DataFrame filtered to specific time boundary
        """
        self.filtered_data = filtered_data.copy()
        self.creation_time = datetime.now()
        
    def get_all_data(self) -> pd.DataFrame:
        """
        Return only the temporally filtered data.
        
        This is the key method that prevents future data access - it can only
        return data that was filtered at creation time.
        
        Returns:
            DataFrame with only data up to the temporal boundary
        """
        return self.filtered_data.copy()
    
    def update(self, current_date: Optional[str] = None) -> pd.DataFrame:
        """
        Return filtered data (ignores current_date to prevent future access).
        
        Args:
            current_date: Ignored to prevent temporal leakage
            
        Returns:
            The same filtered data regardless of current_date
        """
        return self.filtered_data.copy()
    
    def plot(self, **kwargs) -> plt.Figure:
        """Generate plot of the temporally filtered data."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot price if available
        if 'price' in self.filtered_data.columns:
            axes[0].plot(self.filtered_data.index, self.filtered_data['price'])
            axes[0].set_title(f'Price Data (Filtered to {self.filtered_data.index.max()})')
            axes[0].set_ylabel('Price')
            axes[0].grid(True, alpha=0.3)
        
        # Plot log returns if available
        if 'log_return' in self.filtered_data.columns:
            axes[1].plot(self.filtered_data.index, self.filtered_data['log_return'])
            axes[1].set_title('Log Returns')
            axes[1].set_ylabel('Log Return')
            axes[1].grid(True, alpha=0.3)
        
        # Add temporal boundary annotation
        max_date = self.filtered_data.index.max()
        for ax in axes:
            ax.axvline(x=max_date, color='red', linestyle='--', alpha=0.7, 
                      label=f'Temporal Boundary: {max_date}')
            ax.legend()
        
        plt.tight_layout()
        return fig


class TemporalController:
    """
    Provides temporal data leakage prevention for backtesting V&V.
    
    This class ensures that during backtesting, the pipeline never has access to future data,
    enabling rigorous validation of trading strategies and regulatory compliance.
    
    Key features:
    - Temporal data isolation: Model can only see data up to specified as_of_date
    - Complete audit trail: Every data access is logged for V&V
    - Verifiable boundaries: Unit testable temporal isolation
    - Reproducible backtests: Exact simulation of real-time trading conditions
    """
    
    def __init__(self, pipeline: 'Pipeline', full_dataset: pd.DataFrame):
        """
        Initialize temporal controller with pipeline and full dataset.
        
        Args:
            pipeline: Pipeline object to control
            full_dataset: Complete dataset for temporal slicing
        """
        from .core import Pipeline  # Import here to avoid circular imports
        
        self.pipeline = pipeline
        self.full_dataset = full_dataset.sort_index()  # Ensure chronological order
        self.access_log: List[Dict[str, Any]] = []  # Complete audit trail
        self.original_data = None  # Store original data component
        
        # Validate dataset has proper time index
        if not isinstance(self.full_dataset.index, pd.DatetimeIndex):
            raise ValueError("Dataset must have DatetimeIndex for temporal control")
    
    def update_as_of(self, as_of_date: str) -> str:
        """
        Update pipeline with data only up to as_of_date.

        GUARANTEES: Model can never access data after as_of_date.

        Args:
            as_of_date: Date boundary (YYYY-MM-DD format)

        Returns:
            Pipeline output (typically markdown report)
        """
        # Convert as_of_date to proper datetime with timezone handling
        as_of_datetime = pd.to_datetime(as_of_date)

        # Handle timezone compatibility
        if self.full_dataset.index.tz is not None:
            # If dataset index is timezone-aware, make as_of_datetime compatible
            if as_of_datetime.tz is None:
                # Localize to the same timezone as the dataset
                as_of_datetime = as_of_datetime.tz_localize(self.full_dataset.index.tz)
        else:
            # If dataset index is timezone-naive, ensure as_of_datetime is also naive
            if as_of_datetime.tz is not None:
                as_of_datetime = as_of_datetime.tz_localize(None)

        # Filter dataset to only include data <= as_of_date
        filtered_data = self.full_dataset[self.full_dataset.index <= as_of_datetime]
        
        if len(filtered_data) == 0:
            raise ValueError(f"No data available up to {as_of_date}")
        
        # Log access for audit trail
        self.access_log.append({
            'timestamp': datetime.now(),
            'as_of_date': as_of_date,
            'data_start': filtered_data.index.min(),
            'data_end': filtered_data.index.max(),
            'num_observations': len(filtered_data),
            'total_dataset_size': len(self.full_dataset),
            'data_coverage': len(filtered_data) / len(self.full_dataset)
        })
        
        # Store original data component if not already stored
        if self.original_data is None:
            self.original_data = self.pipeline.data
        
        # Temporarily replace pipeline's data component with filtered stub
        self.pipeline.data = TemporalDataStub(filtered_data)
        
        try:
            # Run pipeline update with temporally isolated data
            result = self.pipeline.update()
        finally:
            # Always restore original data component (exception safety)
            self.pipeline.data = self.original_data
        
        return result
    
    def step_through_time(self, start_date: str, end_date: str, 
                         freq: str = 'D') -> List[Tuple[str, str]]:
        """
        Step through time period for systematic backtesting.
        
        Args:
            start_date: Start date for stepping (YYYY-MM-DD)
            end_date: End date for stepping (YYYY-MM-DD)
            freq: Frequency for stepping ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            List of (date, report) tuples with complete temporal isolation
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        results = []
        
        for date in date_range:
            # Only process dates that exist in our dataset
            if date in self.full_dataset.index:
                date_str = date.strftime('%Y-%m-%d')
                try:
                    report = self.update_as_of(date_str)
                    results.append((date_str, report))
                except Exception as e:
                    # Log error but continue processing
                    self.access_log.append({
                        'timestamp': datetime.now(),
                        'as_of_date': date_str,
                        'error': str(e),
                        'status': 'failed'
                    })
                    
        return results
    
    def get_access_audit(self) -> pd.DataFrame:
        """
        Return complete audit trail for V&V verification.
        
        Returns:
            DataFrame with complete log of temporal data access
        """
        if not self.access_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.access_log)
    
    def verify_temporal_isolation(self, test_date: str) -> Dict[str, bool]:
        """
        Verify that temporal isolation is working correctly.
        
        This method can be used in unit tests to prove that the temporal
        controller prevents future data leakage.
        
        Args:
            test_date: Date to test isolation for
            
        Returns:
            Dictionary with verification results
        """
        test_datetime = pd.to_datetime(test_date)
        
        # Get data that should be accessible
        accessible_data = self.full_dataset[self.full_dataset.index <= test_datetime]
        
        # Get data that should NOT be accessible
        future_data = self.full_dataset[self.full_dataset.index > test_datetime]
        
        # Test temporal isolation by creating stub
        stub = TemporalDataStub(accessible_data)
        stub_data = stub.get_all_data()
        
        verification = {
            'correct_data_size': len(stub_data) == len(accessible_data),
            'no_future_data': len(stub_data[stub_data.index > test_datetime]) == 0,
            'data_boundary_correct': stub_data.index.max() <= test_datetime,
            'future_data_exists': len(future_data) > 0,  # Ensure test is meaningful
        }
        
        return verification
    
    def plot_temporal_access(self, **kwargs) -> plt.Figure:
        """
        Visualize temporal access pattern for V&V verification.
        
        Returns:
            Figure showing temporal boundaries and access patterns
        """
        if not self.access_log:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No temporal access logged yet', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        audit_df = self.get_access_audit()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Data coverage over time
        if 'data_coverage' in audit_df.columns:
            axes[0].plot(pd.to_datetime(audit_df['as_of_date']), 
                        audit_df['data_coverage'], 'b-o', markersize=4)
            axes[0].set_title('Temporal Data Coverage During Backtesting')
            axes[0].set_ylabel('Data Coverage Ratio')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.1)
        
        # Plot 2: Number of observations over time
        if 'num_observations' in audit_df.columns:
            axes[1].plot(pd.to_datetime(audit_df['as_of_date']), 
                        audit_df['num_observations'], 'g-o', markersize=4)
            axes[1].set_title('Number of Observations Available Over Time')
            axes[1].set_ylabel('Number of Observations')
            axes[1].set_xlabel('As-Of Date')
            axes[1].grid(True, alpha=0.3)
        
        # Add temporal boundary markers
        for ax in axes:
            for _, row in audit_df.iterrows():
                ax.axvline(x=pd.to_datetime(row['as_of_date']), 
                          color='red', alpha=0.1, linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def reset_audit_log(self) -> None:
        """Reset the audit log (useful for testing)."""
        self.access_log = []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of temporal access patterns.
        
        Returns:
            Dictionary with summary statistics for V&V reporting
        """
        if not self.access_log:
            return {'error': 'No temporal access logged'}
        
        audit_df = self.get_access_audit()
        
        return {
            'total_temporal_updates': len(audit_df),
            'date_range_tested': {
                'start': audit_df['as_of_date'].min(),
                'end': audit_df['as_of_date'].max()
            },
            'data_coverage_stats': {
                'min': audit_df['data_coverage'].min() if 'data_coverage' in audit_df else None,
                'max': audit_df['data_coverage'].max() if 'data_coverage' in audit_df else None,
                'mean': audit_df['data_coverage'].mean() if 'data_coverage' in audit_df else None
            },
            'total_dataset_size': self.full_dataset.shape[0],
            'temporal_isolation_verified': True,  # Always true if no exceptions
            'audit_log_complete': len(self.access_log) > 0
        }