"""
Streaming Data Interfaces for Real-Time Market Data Processing

This module provides interfaces and adapters for connecting Online HMM models
to various real-time market data sources, enabling continuous regime detection
and analysis on live data streams.

Features:
- Unified streaming data interface
- Rate limiting and connection management  
- Data validation and error recovery
- Buffering and batch processing
- Multiple data source adapters

Author: aoaustin  
Created: 2025-09-03
"""

import asyncio
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from dataclasses import dataclass
from collections import deque
import logging
from enum import Enum

from .online_hmm import OnlineHMM
from ..data.loader import DataLoader
from ..utils.exceptions import HiddenRegimeError


class StreamingMode(Enum):
    """Streaming processing modes."""
    REAL_TIME = "real_time"      # Process each observation immediately
    MICRO_BATCH = "micro_batch"   # Process small batches (e.g., 5-10 observations)
    TIME_WINDOW = "time_window"   # Process observations in time windows


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing."""
    
    # Processing mode
    mode: StreamingMode = StreamingMode.REAL_TIME
    batch_size: int = 5                    # For micro-batch mode
    time_window_seconds: float = 60.0      # For time-window mode
    
    # Rate limiting
    max_updates_per_second: float = 10.0   # Maximum parameter updates per second
    max_observations_per_second: float = 100.0  # Maximum observations per second
    
    # Buffer management
    input_buffer_size: int = 1000          # Input observation buffer
    output_buffer_size: int = 500          # Output results buffer
    
    # Error handling
    max_consecutive_errors: int = 5        # Max errors before stopping
    retry_delay_seconds: float = 1.0       # Delay between retries
    
    # Data validation
    validate_observations: bool = True      # Validate each observation
    outlier_threshold: float = 5.0         # Standard deviations for outlier detection
    
    # Logging
    log_level: str = "INFO"
    log_performance_metrics: bool = True
    performance_log_interval: int = 100    # Log every N observations


@dataclass
class StreamingObservation:
    """Container for streaming market data observations."""
    
    timestamp: pd.Timestamp
    symbol: str
    price: float
    log_return: Optional[float] = None
    volume: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate observation data."""
        if not np.isfinite(self.price) or self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if self.log_return is not None and not np.isfinite(self.log_return):
            raise ValueError(f"Invalid log_return: {self.log_return}")


@dataclass  
class StreamingResult:
    """Results from streaming regime detection."""
    
    timestamp: pd.Timestamp
    symbol: str
    regime: int
    regime_probabilities: List[float]
    confidence: float
    regime_interpretation: str
    regime_characteristics: Dict[str, float]
    diagnostics: Dict[str, Any]
    processing_time_ms: float


class StreamingDataSource(ABC):
    """Abstract base class for streaming data sources."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    async def stream_observations(self, symbols: List[str]) -> AsyncIterator[StreamingObservation]:
        """Stream observations for given symbols."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""
        pass


class SimulatedDataSource(StreamingDataSource):
    """Simulated data source for testing and backtesting."""
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 speed_multiplier: float = 1.0,
                 add_noise: bool = False,
                 noise_std: float = 0.001):
        """
        Initialize simulated data source.
        
        Args:
            historical_data: DataFrame with columns: date, symbol, price, log_return
            speed_multiplier: Speed up factor (1.0 = real-time, 10.0 = 10x faster)
            add_noise: Whether to add random noise to historical data
            noise_std: Standard deviation of noise to add
        """
        self.historical_data = historical_data.copy()
        self.speed_multiplier = speed_multiplier
        self.add_noise = add_noise
        self.noise_std = noise_std
        self._connected = False
        self._current_index = 0
        
    async def connect(self) -> bool:
        """Establish connection (no-op for simulated source)."""
        self._connected = True
        self._current_index = 0
        return True
        
    async def disconnect(self) -> bool:
        """Close connection."""
        self._connected = False
        return True
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
        
    async def stream_observations(self, symbols: List[str]) -> AsyncIterator[StreamingObservation]:
        """Stream historical observations as if they were live."""
        if not self._connected:
            raise RuntimeError("Data source not connected")
            
        # Filter data for requested symbols
        filtered_data = self.historical_data[
            self.historical_data['symbol'].isin(symbols) if 'symbol' in self.historical_data.columns
            else self.historical_data  # Assume single symbol if no symbol column
        ].sort_values('date')
        
        for _, row in filtered_data.iterrows():
            if not self._connected:
                break
                
            # Calculate sleep time based on speed multiplier
            sleep_time = 1.0 / self.speed_multiplier  # Base: 1 observation per second
            
            # Add noise if enabled
            price = row['price']
            log_return = row['log_return'] 
            if self.add_noise:
                noise = np.random.normal(0, self.noise_std)
                log_return += noise
                
            # Create observation
            observation = StreamingObservation(
                timestamp=pd.Timestamp(row['date']),
                symbol=symbols[0] if len(symbols) == 1 else row.get('symbol', 'UNKNOWN'),
                price=price,
                log_return=log_return,
                volume=row.get('volume'),
                metadata={'source': 'simulated', 'index': self._current_index}
            )
            
            yield observation
            
            self._current_index += 1
            await asyncio.sleep(sleep_time)


class YFinanceLiveDataSource(StreamingDataSource):
    """Live data source using yfinance (polling-based pseudo-streaming)."""
    
    def __init__(self, 
                 poll_interval_seconds: float = 60.0,
                 max_poll_failures: int = 5):
        """
        Initialize yfinance live data source.
        
        Args:
            poll_interval_seconds: How often to poll for new data
            max_poll_failures: Maximum consecutive polling failures
        """
        self.poll_interval = poll_interval_seconds
        self.max_poll_failures = max_poll_failures
        self._connected = False
        self._last_prices: Dict[str, float] = {}
        self._data_loader = DataLoader()
        
    async def connect(self) -> bool:
        """Establish connection."""
        self._connected = True
        return True
        
    async def disconnect(self) -> bool:
        """Close connection."""
        self._connected = False
        return True
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
        
    async def stream_observations(self, symbols: List[str]) -> AsyncIterator[StreamingObservation]:
        """Poll yfinance for latest prices and stream observations."""
        if not self._connected:
            raise RuntimeError("Data source not connected")
            
        consecutive_failures = 0
        
        while self._connected and consecutive_failures < self.max_poll_failures:
            try:
                for symbol in symbols:
                    # Get latest price (last 2 days to calculate return)
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.Timedelta(days=2)
                    
                    data = self._data_loader.load_stock_data(
                        symbol, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if len(data) >= 1:
                        latest = data.iloc[-1]
                        
                        observation = StreamingObservation(
                            timestamp=pd.Timestamp(latest['date']),
                            symbol=symbol,
                            price=latest['price'],
                            log_return=latest['log_return'],
                            volume=latest.get('volume'),
                            metadata={'source': 'yfinance_live'}
                        )
                        
                        yield observation
                        consecutive_failures = 0  # Reset on success
                        
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                consecutive_failures += 1
                logging.warning(f"Polling failed for attempt {consecutive_failures}: {e}")
                await asyncio.sleep(self.poll_interval)


class StreamingProcessor:
    """Main processor for streaming regime detection."""
    
    def __init__(self,
                 online_hmm: OnlineHMM,
                 data_source: StreamingDataSource,
                 config: Optional[StreamingConfig] = None):
        """
        Initialize streaming processor.
        
        Args:
            online_hmm: Fitted OnlineHMM model for regime detection
            data_source: Source of streaming market data
            config: Streaming configuration
        """
        self.online_hmm = online_hmm
        self.data_source = data_source
        self.config = config or StreamingConfig()
        
        # Processing state
        self._running = False
        self._observation_buffer = deque(maxlen=self.config.input_buffer_size)
        self._result_buffer = deque(maxlen=self.config.output_buffer_size)
        
        # Performance tracking
        self._observations_processed = 0
        self._last_update_time = time.time()
        self._processing_times = deque(maxlen=100)
        self._consecutive_errors = 0
        
        # Rate limiting
        self._last_observation_time = 0.0
        self._last_update_time_rate = 0.0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{__name__}.StreamingProcessor")
        
    async def start(self, symbols: List[str], result_callback: Optional[Callable] = None) -> None:
        """
        Start streaming processing.
        
        Args:
            symbols: List of symbols to process
            result_callback: Optional callback for results
        """
        if self._running:
            raise RuntimeError("Processor is already running")
            
        self._running = True
        self.logger.info(f"Starting streaming processor for symbols: {symbols}")
        
        try:
            # Connect to data source
            if not await self.data_source.connect():
                raise RuntimeError("Failed to connect to data source")
                
            # Start processing based on mode
            if self.config.mode == StreamingMode.REAL_TIME:
                await self._process_real_time(symbols, result_callback)
            elif self.config.mode == StreamingMode.MICRO_BATCH:
                await self._process_micro_batch(symbols, result_callback)  
            elif self.config.mode == StreamingMode.TIME_WINDOW:
                await self._process_time_window(symbols, result_callback)
                
        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            raise
        finally:
            await self.data_source.disconnect()
            self._running = False
            
    async def stop(self) -> None:
        """Stop streaming processing."""
        self.logger.info("Stopping streaming processor")
        self._running = False
        
    async def _process_real_time(self, symbols: List[str], result_callback: Optional[Callable]) -> None:
        """Process observations in real-time mode."""
        async for observation in self.data_source.stream_observations(symbols):
            if not self._running:
                break
                
            # Rate limiting
            if not self._check_observation_rate_limit():
                await asyncio.sleep(0.01)  # Small delay
                continue
                
            # Validate observation
            if self.config.validate_observations:
                if not self._validate_observation(observation):
                    continue
                    
            # Process observation
            try:
                start_time = time.time()
                result = await self._process_observation(observation)
                processing_time = (time.time() - start_time) * 1000
                
                result.processing_time_ms = processing_time
                self._processing_times.append(processing_time)
                
                # Store result
                self._result_buffer.append(result)
                
                # Call result callback
                if result_callback:
                    await self._safe_callback(result_callback, result)
                    
                # Performance logging
                self._observations_processed += 1
                if (self.config.log_performance_metrics and 
                    self._observations_processed % self.config.performance_log_interval == 0):
                    await self._log_performance_metrics()
                    
                self._consecutive_errors = 0  # Reset error count on success
                
            except Exception as e:
                self._consecutive_errors += 1
                self.logger.error(f"Processing error: {e}")
                
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    raise RuntimeError(f"Too many consecutive errors: {self._consecutive_errors}")
                    
                await asyncio.sleep(self.config.retry_delay_seconds)
                
    async def _process_micro_batch(self, symbols: List[str], result_callback: Optional[Callable]) -> None:
        """Process observations in micro-batch mode."""
        batch = []
        
        async for observation in self.data_source.stream_observations(symbols):
            if not self._running:
                break
                
            batch.append(observation)
            
            if len(batch) >= self.config.batch_size:
                # Process batch
                for obs in batch:
                    if self.config.validate_observations and not self._validate_observation(obs):
                        continue
                        
                    result = await self._process_observation(obs)
                    self._result_buffer.append(result)
                    
                    if result_callback:
                        await self._safe_callback(result_callback, result)
                        
                batch.clear()
                
        # Process remaining observations
        for obs in batch:
            if self.config.validate_observations and not self._validate_observation(obs):
                continue
            result = await self._process_observation(obs)
            self._result_buffer.append(result)
            if result_callback:
                await self._safe_callback(result_callback, result)
                
    async def _process_time_window(self, symbols: List[str], result_callback: Optional[Callable]) -> None:
        """Process observations in time-window mode."""
        window_observations = []
        window_start_time = time.time()
        
        async for observation in self.data_source.stream_observations(symbols):
            if not self._running:
                break
                
            window_observations.append(observation)
            current_time = time.time()
            
            # Check if window is complete
            if current_time - window_start_time >= self.config.time_window_seconds:
                # Process window
                for obs in window_observations:
                    if self.config.validate_observations and not self._validate_observation(obs):
                        continue
                        
                    result = await self._process_observation(obs)
                    self._result_buffer.append(result)
                    
                    if result_callback:
                        await self._safe_callback(result_callback, result)
                        
                # Reset window
                window_observations.clear()
                window_start_time = current_time
                
    async def _process_observation(self, observation: StreamingObservation) -> StreamingResult:
        """Process a single observation through OnlineHMM."""
        # Check if we should update parameters (rate limiting)
        should_update_params = self._check_update_rate_limit()
        
        # Process through OnlineHMM
        regime_info = self.online_hmm.add_observation(
            observation.log_return, 
            update_parameters=should_update_params
        )
        
        # Create result
        result = StreamingResult(
            timestamp=observation.timestamp,
            symbol=observation.symbol,
            regime=regime_info['regime'],
            regime_probabilities=regime_info['regime_probabilities'],
            confidence=regime_info['confidence'],
            regime_interpretation=regime_info['regime_interpretation'],
            regime_characteristics=regime_info['regime_characteristics'],
            diagnostics=regime_info['diagnostics'],
            processing_time_ms=0.0  # Will be set by caller
        )
        
        return result
        
    def _validate_observation(self, observation: StreamingObservation) -> bool:
        """Validate observation for outliers and data quality."""
        if observation.log_return is None:
            return False
            
        # Check for outliers
        if abs(observation.log_return) > self.config.outlier_threshold * 0.02:  # 2% baseline std
            self.logger.warning(f"Outlier detected: {observation.log_return}")
            return False
            
        return True
        
    def _check_observation_rate_limit(self) -> bool:
        """Check if observation rate limit is exceeded."""
        current_time = time.time()
        time_since_last = current_time - self._last_observation_time
        min_interval = 1.0 / self.config.max_observations_per_second
        
        if time_since_last >= min_interval:
            self._last_observation_time = current_time
            return True
        return False
        
    def _check_update_rate_limit(self) -> bool:
        """Check if parameter update rate limit is exceeded."""
        current_time = time.time()
        time_since_last = current_time - self._last_update_time_rate
        min_interval = 1.0 / self.config.max_updates_per_second
        
        if time_since_last >= min_interval:
            self._last_update_time_rate = current_time
            return True
        return False
        
    async def _safe_callback(self, callback: Callable, result: StreamingResult) -> None:
        """Safely call result callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)
        except Exception as e:
            self.logger.error(f"Result callback failed: {e}")
            
    async def _log_performance_metrics(self) -> None:
        """Log performance metrics."""
        if self._processing_times:
            avg_processing_time = np.mean(list(self._processing_times))
            max_processing_time = np.max(list(self._processing_times))
            
            self.logger.info(
                f"Performance: {self._observations_processed} observations processed, "
                f"avg={avg_processing_time:.2f}ms, max={max_processing_time:.2f}ms, "
                f"buffer_size={len(self._result_buffer)}"
            )
            
    def get_recent_results(self, n: int = 10) -> List[StreamingResult]:
        """Get most recent processing results."""
        return list(self._result_buffer)[-n:]
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._processing_times:
            return {}
            
        return {
            'observations_processed': self._observations_processed,
            'avg_processing_time_ms': np.mean(list(self._processing_times)),
            'max_processing_time_ms': np.max(list(self._processing_times)),
            'min_processing_time_ms': np.min(list(self._processing_times)),
            'buffer_utilization': len(self._result_buffer) / self.config.output_buffer_size,
            'consecutive_errors': self._consecutive_errors
        }