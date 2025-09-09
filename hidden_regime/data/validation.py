"""
Data validation functionality for hidden-regime package.

Provides DataValidator class for comprehensive data quality assessment,
anomaly detection, and validation reporting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

from ..config.settings import ValidationConfig
from ..utils.exceptions import ValidationError, DataQualityError


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str] 
    recommendations: List[str]
    quality_score: float
    metrics: Dict[str, Any]


class DataValidator:
    """
    Comprehensive data validation for market data.
    
    Performs quality checks, anomaly detection, and generates
    detailed validation reports with recommendations.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize DataValidator with configuration.
        
        Args:
            config: ValidationConfig instance. Uses defaults if None.
        """
        self.config = config or ValidationConfig()
    
    def validate_data(
        self, 
        data: pd.DataFrame, 
        ticker: Optional[str] = None
    ) -> ValidationResult:
        """
        Perform comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            ticker: Optional ticker symbol for context
            
        Returns:
            ValidationResult with detailed findings and recommendations
        """
        issues = []
        warnings_list = []
        recommendations = []
        metrics = {}
        
        try:
            # Add basic metrics
            metrics['n_observations'] = len(data)
            
            # Basic structure validation
            structure_issues = self._validate_structure(data)
            issues.extend(structure_issues)
            
            # Date validation
            date_issues, date_warnings, date_metrics = self._validate_dates(data)
            issues.extend(date_issues)
            warnings_list.extend(date_warnings)
            metrics.update(date_metrics)
            
            # Price validation
            price_issues, price_warnings, price_metrics = self._validate_prices(data)
            issues.extend(price_issues)
            warnings_list.extend(price_warnings)
            metrics.update(price_metrics)
            
            # Return validation
            if 'log_return' in data.columns:
                return_issues, return_warnings, return_metrics = self._validate_returns(data)
                issues.extend(return_issues)
                warnings_list.extend(return_warnings)
                metrics.update(return_metrics)
            
            # Missing data validation
            missing_issues, missing_warnings, missing_metrics = self._validate_missing_data(data)
            issues.extend(missing_issues)
            warnings_list.extend(missing_warnings)
            metrics.update(missing_metrics)
            
            # Outlier detection
            outlier_warnings, outlier_metrics = self._detect_outliers(data)
            warnings_list.extend(outlier_warnings)
            metrics.update(outlier_metrics)
            
            # Volume validation (if present)
            if 'volume' in data.columns:
                volume_warnings, volume_metrics = self._validate_volume(data)
                warnings_list.extend(volume_warnings)
                metrics.update(volume_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                data, issues, warnings_list, metrics
            )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(issues, warnings_list, metrics)
            
        except Exception as e:
            issues.append(f"Validation failed with error: {str(e)}")
            quality_score = 0.0
        
        is_valid = len([issue for issue in issues if not issue.startswith("Warning:")]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings_list,
            recommendations=recommendations,
            quality_score=quality_score,
            metrics=metrics
        )
    
    def validate_ticker_format(self, ticker: str) -> bool:
        """
        Validate ticker format.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ticker format is valid
        """
        if not isinstance(ticker, str):
            return False
        
        # Basic validation - alphanumeric, dots, hyphens
        import re
        pattern = r'^[A-Z0-9.-]+$'
        return bool(re.match(pattern, ticker.upper())) and len(ticker) <= 10
    
    def validate_date_range(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Tuple[bool, List[str]]:
        """
        Validate date range parameters.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            issues.append(f"Invalid date format: {e}")
            return False, issues
        
        if start_dt >= end_dt:
            issues.append("Start date must be before end date")
        
        if end_dt > datetime.now():
            issues.append("End date cannot be in the future")
        
        # Check for reasonable date range
        days_diff = (end_dt - start_dt).days
        if days_diff < 7:
            issues.append("Date range too short (minimum 7 days recommended)")
        elif days_diff > 365 * 10:  # 10 years
            issues.append("Date range very long (may cause performance issues)")
        
        # Check if start date is too far in the past
        if start_dt < datetime(1970, 1, 1):
            issues.append("Start date too far in the past")
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic DataFrame structure."""
        issues = []
        
        if data.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Check for required columns
        required_columns = ['price']
        for column in required_columns:
            if column not in data.columns:
                issues.append(f"Required column '{column}' is missing")
        
        # Check for reasonable number of columns
        if len(data.columns) > 20:
            issues.append(f"Too many columns ({len(data.columns)}) - may indicate data issues")
        
        return issues
    
    def _validate_dates(self, data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate date column and temporal structure."""
        issues = []
        warnings_list = []
        metrics = {}
        
        if 'date' not in data.columns:
            warnings_list.append("No date column found - temporal analysis limited")
            return issues, warnings_list, metrics
        
        # Ensure dates are datetime
        try:
            dates = pd.to_datetime(data['date'])
        except Exception as e:
            issues.append(f"Invalid date format in date column: {e}")
            return issues, warnings_list, metrics
        
        # Check for duplicate dates
        duplicates = dates.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate dates")
        
        # Check date ordering
        if not dates.is_monotonic_increasing:
            warnings_list.append("Dates are not in chronological order")
        
        # Check for reasonable date gaps
        if len(dates) > 1:
            date_diffs = dates.diff().dt.days.dropna()
            max_gap = date_diffs.max()
            avg_gap = date_diffs.mean()
            
            if max_gap > 14:  # More than 2 weeks gap
                warnings_list.append(f"Large gap in dates detected: {max_gap} days")
            
            # Check for weekend/holiday patterns
            business_days_expected = pd.bdate_range(dates.min(), dates.max())
            coverage_ratio = len(dates) / len(business_days_expected)
            
            if coverage_ratio < 0.8:  # Less than 80% of business days
                warnings_list.append(
                    f"Low date coverage: {coverage_ratio:.1%} of business days"
                )
        
        # Calculate metrics
        metrics.update({
            'date_range_days': (dates.max() - dates.min()).days if len(dates) > 1 else 0,
            'n_dates': len(dates),
            'duplicate_dates': duplicates,
            'date_coverage_ratio': coverage_ratio if len(dates) > 1 else 1.0
        })
        
        return issues, warnings_list, metrics
    
    def _validate_prices(self, data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate price data."""
        issues = []
        warnings_list = []
        metrics = {}
        
        if 'price' not in data.columns:
            issues.append("Price column is missing")
            return issues, warnings_list, metrics
        
        prices = data['price']
        
        # Check for non-numeric prices
        if not pd.api.types.is_numeric_dtype(prices):
            issues.append("Price column contains non-numeric values")
            return issues, warnings_list, metrics
        
        # Check for negative or zero prices
        invalid_prices = (prices <= 0).sum()
        if invalid_prices > 0:
            issues.append(f"Found {invalid_prices} non-positive prices")
        
        # Check for extremely low prices (potential data errors)
        very_low_prices = (prices < self.config.min_price).sum()
        if very_low_prices > 0:
            warnings_list.append(
                f"Found {very_low_prices} prices below ${self.config.min_price}"
            )
        
        # Check for price jumps/drops (potential stock splits or data errors)
        if len(prices) > 1:
            price_changes = prices.pct_change(fill_method=None).abs()
            large_changes = (price_changes > 0.5).sum()  # >50% change
            
            if large_changes > 0:
                warnings_list.append(
                    f"Found {large_changes} large price changes (>50%) - "
                    "may indicate stock splits or data errors"
                )
        
        # Calculate price statistics
        metrics.update({
            'price_min': prices.min(),
            'price_max': prices.max(),
            'price_mean': prices.mean(),
            'price_std': prices.std(),
            'invalid_prices': invalid_prices,
            'very_low_prices': very_low_prices,
            'large_price_changes': large_changes if len(prices) > 1 else 0
        })
        
        return issues, warnings_list, metrics
    
    def _validate_returns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate return data."""
        issues = []
        warnings_list = []
        metrics = {}
        
        returns = data['log_return'].dropna()
        
        if len(returns) == 0:
            issues.append("No valid returns found")
            return issues, warnings_list, metrics
        
        # Check for infinite returns
        infinite_returns = np.isinf(returns).sum()
        if infinite_returns > 0:
            issues.append(f"Found {infinite_returns} infinite returns")
        
        # Check for extremely large returns
        extreme_returns = (returns.abs() > self.config.max_daily_return).sum()
        if extreme_returns > 0:
            warnings_list.append(
                f"Found {extreme_returns} extreme returns "
                f"(>{self.config.max_daily_return:.1%})"
            )
        
        # Check return distribution properties
        return_std = returns.std()
        if return_std > 0.1:  # >10% daily volatility
            warnings_list.append(f"Very high volatility: {return_std:.2%} daily")
        elif return_std < 0.005:  # <0.5% daily volatility
            warnings_list.append(f"Very low volatility: {return_std:.2%} daily")
        
        # Calculate return statistics
        from scipy import stats
        metrics.update({
            'return_mean': returns.mean(),
            'return_std': return_std,
            'return_skewness': stats.skew(returns),
            'return_kurtosis': stats.kurtosis(returns),
            'extreme_returns': extreme_returns,
            'infinite_returns': infinite_returns
        })
        
        return issues, warnings_list, metrics
    
    def _validate_missing_data(self, data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate missing data patterns."""
        issues = []
        warnings_list = []
        metrics = {}
        
        # Calculate missing data statistics
        missing_counts = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        total_missing = missing_counts.sum()
        missing_pct = total_missing / total_cells
        
        # Check overall missing data percentage
        if missing_pct > 0.1:  # >10% missing
            issues.append(f"High percentage of missing data: {missing_pct:.1%}")
        elif missing_pct > 0.05:  # >5% missing
            warnings_list.append(f"Moderate missing data: {missing_pct:.1%}")
        
        # Check missing data by column
        for column, missing_count in missing_counts.items():
            if missing_count > 0:
                column_missing_pct = missing_count / len(data)
                if column_missing_pct > 0.2:  # >20% missing in column
                    issues.append(
                        f"Column '{column}' has {column_missing_pct:.1%} missing data"
                    )
                elif column_missing_pct > 0.1:  # >10% missing in column
                    warnings_list.append(
                        f"Column '{column}' has {column_missing_pct:.1%} missing data"
                    )
        
        # Check for consecutive missing values
        for column in ['price']:
            if column in data.columns:
                consecutive_missing = self._find_max_consecutive_missing(data[column])
                if consecutive_missing > self.config.max_consecutive_missing:
                    issues.append(
                        f"Too many consecutive missing values in {column}: "
                        f"{consecutive_missing} > {self.config.max_consecutive_missing}"
                    )
        
        metrics.update({
            'total_missing': total_missing,
            'missing_percentage': missing_pct,
            'columns_with_missing': (missing_counts > 0).sum(),
            'max_consecutive_missing': max([
                self._find_max_consecutive_missing(data[col]) 
                for col in ['price'] if col in data.columns
            ]) if any(col in data.columns for col in ['price']) else 0
        })
        
        return issues, warnings_list, metrics
    
    def _detect_outliers(self, data: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Detect outliers in the data."""
        warnings_list = []
        metrics = {}
        
        if 'log_return' not in data.columns:
            return warnings_list, metrics
        
        returns = data['log_return'].dropna()
        if len(returns) == 0:
            return warnings_list, metrics
        
        # Use configured outlier detection method
        if self.config.outlier_method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(returns))
            outliers = z_scores > self.config.outlier_threshold
        elif self.config.outlier_method == 'iqr':
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.iqr_multiplier * IQR
            upper_bound = Q3 + self.config.iqr_multiplier * IQR
            outliers = (returns < lower_bound) | (returns > upper_bound)
        else:
            # Fallback to simple threshold
            outliers = returns.abs() > 0.1  # >10% daily return
        
        n_outliers = outliers.sum()
        outlier_pct = n_outliers / len(returns)
        
        if outlier_pct > 0.1:  # >10% outliers
            warnings_list.append(f"High percentage of outliers: {outlier_pct:.1%}")
        elif outlier_pct > 0.05:  # >5% outliers
            warnings_list.append(f"Moderate outliers detected: {outlier_pct:.1%}")
        
        metrics.update({
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_pct,
            'outlier_method': self.config.outlier_method
        })
        
        return warnings_list, metrics
    
    def _validate_volume(self, data: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Validate volume data."""
        warnings_list = []
        metrics = {}
        
        volume = data['volume']
        
        # Check for negative volume
        negative_volume = (volume < 0).sum()
        if negative_volume > 0:
            warnings_list.append(f"Found {negative_volume} negative volume values")
        
        # Check for zero volume days
        zero_volume = (volume == 0).sum()
        zero_volume_pct = zero_volume / len(volume)
        if zero_volume_pct > 0.1:  # >10% zero volume
            warnings_list.append(f"High percentage of zero volume days: {zero_volume_pct:.1%}")
        
        # Check for extremely high volume (potential data errors)
        volume_median = volume.median()
        volume_99th = volume.quantile(0.99)
        if volume_99th > volume_median * 100:  # 99th percentile > 100x median
            warnings_list.append("Extremely high volume spikes detected")
        
        metrics.update({
            'volume_median': volume_median,
            'volume_99th_percentile': volume_99th,
            'zero_volume_days': zero_volume,
            'negative_volume_days': negative_volume
        })
        
        return warnings_list, metrics
    
    def _generate_recommendations(
        self,
        data: pd.DataFrame,
        issues: List[str],
        warnings_list: List[str],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Recommendations based on issues
        if any('missing' in issue.lower() for issue in issues):
            recommendations.append("Consider data imputation or filtering periods with excessive missing data")
        
        if any('price' in issue.lower() for issue in issues):
            recommendations.append("Review price data for splits, dividends, or data quality issues")
        
        # Recommendations based on warnings
        if any('outlier' in warning.lower() for warning in warnings_list):
            recommendations.append("Consider outlier treatment (winsorization, removal, or robust modeling)")
        
        if any('volatility' in warning.lower() for warning in warnings_list):
            recommendations.append("High volatility detected - consider regime-specific modeling")
        
        if any('gap' in warning.lower() for warning in warnings_list):
            recommendations.append("Data gaps detected - ensure proper handling of non-trading periods")
        
        # Recommendations based on data characteristics
        if len(data) < 100:
            recommendations.append("Limited data available - consider longer time period for robust analysis")
        
        if 'return_kurtosis' in metrics and metrics['return_kurtosis'] > 3:
            recommendations.append("High kurtosis detected - consider fat-tailed distributions for modeling")
        
        return recommendations
    
    def _calculate_quality_score(
        self,
        issues: List[str],
        warnings_list: List[str],
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0
        
        # Deduct points for issues (more severe)
        score -= len(issues) * 0.2
        
        # Deduct points for warnings (less severe)
        score -= len(warnings_list) * 0.05
        
        # Adjust based on specific metrics
        if 'missing_percentage' in metrics:
            score -= metrics['missing_percentage'] * 2  # Heavy penalty for missing data
        
        if 'outlier_percentage' in metrics:
            score -= metrics['outlier_percentage'] * 1.0  # More strict penalty for outliers
        
        # Extra penalty for extreme returns
        if 'extreme_returns' in metrics and metrics['extreme_returns'] > 0:
            score -= metrics['extreme_returns'] * 0.1  # 0.1 per extreme return
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _find_max_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum number of consecutive missing values."""
        is_null = series.isnull()
        if not is_null.any():
            return 0
        
        groups = (is_null != is_null.shift()).cumsum()
        consecutive_counts = is_null.groupby(groups).sum()
        return consecutive_counts.max()