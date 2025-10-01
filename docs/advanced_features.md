# Advanced Features Documentation

*Comprehensive guide to advanced HMM capabilities and extensions*

---

## Table of Contents

1. [Overview](#overview)
2. [Streaming Data Architecture](#streaming-data-architecture)
3. [Change Point Detection](#change-point-detection)
4. [Multi-Asset Analysis](#multi-asset-analysis)
5. [Model Comparison Framework](#model-comparison-framework)
6. [Uncertainty Quantification](#uncertainty-quantification)
7. [Custom Extensions](#custom-extensions)
8. [Advanced Regime Analysis](#advanced-regime-analysis)
9. [Performance Monitoring](#performance-monitoring)
10. [Integration Patterns](#integration-patterns)

---

## Overview

The Hidden Regime framework provides advanced features that extend beyond basic HMM functionality. These features enable sophisticated financial analysis, real-time processing, and integration with complex trading systems.

### Advanced Feature Categories

| Category | Features | Use Cases |
|----------|----------|-----------|
| **Streaming & Real-Time** | Data ingestion, online updates, live processing | High-frequency trading, real-time risk management |
| **Change Detection** | Structural break identification, regime transitions | Crisis detection, market shift analysis |
| **Multi-Asset** | Cross-asset correlations, portfolio analysis | Portfolio management, sector rotation |
| **Model Enhancement** | Ensemble methods, uncertainty quantification | Risk assessment, model validation |
| **Custom Extensions** | User-defined features, plugin architecture | Specialized applications, research |

---

## Streaming Data Architecture

### Real-Time Data Processing Framework

The streaming architecture enables continuous processing of market data with minimal latency:

```python
from hidden_regime.models.streaming import StreamingDataSource, StreamingConfig, StreamingProcessor
from hidden_regime.models import OnlineHMM, OnlineHMMConfig
import asyncio
from datetime import datetime
import numpy as np

class AdvancedStreamingSystem:
    """Advanced streaming system for real-time HMM processing"""
    
    def __init__(self, config):
        self.config = config
        self.streaming_config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            buffer_size=1000,
            processing_interval=0.1,  # 100ms intervals
            max_latency_ms=50,
            enable_heartbeat=True
        )
        
        # Initialize models for different assets
        self.models = {}
        self.processors = {}
        self.performance_monitors = {}
        
    async def initialize_streaming(self, assets):
        """Initialize streaming for multiple assets"""
        
        for asset in assets:
            # Create asset-specific model
            online_config = self.get_asset_config(asset)
            model = OnlineHMM(
                n_states=3,
                online_config=online_config
            )
            
            # Initialize with historical data
            historical_data = await self.load_historical_data(asset)
            model.fit(historical_data['returns'])
            
            self.models[asset] = model
            
            # Create streaming processor
            processor = StreamingProcessor(
                model=model,
                asset=asset,
                config=self.streaming_config
            )
            self.processors[asset] = processor
            
            # Performance monitoring
            monitor = PerformanceMonitor(asset)
            self.performance_monitors[asset] = monitor
            
        print(f"✅ Streaming system initialized for {len(assets)} assets")
    
    async def start_streaming(self):
        """Start the streaming data processing"""
        
        # Create data source connections
        data_sources = []
        for asset in self.models.keys():
            source = await self.create_data_source(asset)
            data_sources.append(source)
        
        # Start processing tasks
        tasks = []
        for source in data_sources:
            task = asyncio.create_task(self.process_data_stream(source))
            tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_system_performance())
        tasks.append(monitor_task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    async def process_data_stream(self, data_source):
        """Process streaming data for a single asset"""
        
        async for data_point in data_source:
            try:
                asset = data_point['symbol']
                price = data_point['price']
                timestamp = data_point['timestamp']
                
                # Calculate return
                last_price = getattr(self, f'last_price_{asset}', None)
                if last_price is not None:
                    log_return = np.log(price / last_price)
                    
                    # Process with streaming processor
                    result = await self.processors[asset].process_observation(
                        observation=log_return,
                        timestamp=timestamp,
                        metadata=data_point
                    )
                    
                    # Update performance monitoring
                    self.performance_monitors[asset].record_processing(
                        processing_time=result['processing_time'],
                        regime_info=result['regime_info']
                    )
                    
                    # Trigger any callbacks
                    await self.handle_regime_update(asset, result)
                
                setattr(self, f'last_price_{asset}', price)
                
            except Exception as e:
                print(f"❌ Error processing {asset}: {e}")
    
    async def handle_regime_update(self, asset, result):
        """Handle regime updates and trigger downstream actions"""
        
        regime_info = result['regime_info']
        
        # Check for regime changes
        if result.get('regime_changed', False):
            await self.on_regime_change(asset, regime_info)
        
        # Update risk systems
        if regime_info['confidence'] < 0.5:
            await self.on_low_confidence(asset, regime_info)
        
        # Real-time notifications
        await self.send_real_time_update(asset, regime_info)
    
    async def on_regime_change(self, asset, regime_info):
        """Handle regime change events"""
        
        print(f"🔄 {asset} regime change to {regime_info['regime_interpretation']}")
        
        # Notify trading systems
        await self.notify_trading_systems(asset, 'regime_change', regime_info)
        
        # Update risk parameters
        await self.update_risk_parameters(asset, regime_info)
        
        # Log event
        await self.log_regime_change(asset, regime_info)

class StreamingProcessor:
    """Process individual asset streams with advanced features"""
    
    def __init__(self, model, asset, config):
        self.model = model
        self.asset = asset
        self.config = config
        
        # Advanced processing state
        self.buffer = StreamingBuffer(config.buffer_size)
        self.change_detector = ChangePointDetector()
        self.latency_monitor = LatencyMonitor()
        
    async def process_observation(self, observation, timestamp, metadata=None):
        """Process single observation with advanced features"""
        
        start_time = asyncio.get_event_loop().time()
        
        # Add to buffer
        self.buffer.add(observation, timestamp, metadata)
        
        # Process with model
        self.model.add_observation(observation)
        regime_info = self.model.get_current_regime_info()
        
        # Change point detection
        change_detected = self.change_detector.check_change_point(
            observation, regime_info
        )
        
        # Check for regime change
        regime_changed = self.check_regime_change(regime_info)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Monitor latency
        self.latency_monitor.record_latency(processing_time)
        
        return {
            'regime_info': regime_info,
            'regime_changed': regime_changed,
            'change_point_detected': change_detected,
            'processing_time': processing_time,
            'buffer_size': len(self.buffer),
            'metadata': metadata
        }

class StreamingBuffer:
    """Efficient buffer for streaming data"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.observations = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.metadata = deque(maxlen=max_size)
        
    def add(self, observation, timestamp, metadata=None):
        """Add observation to buffer"""
        self.observations.append(observation)
        self.timestamps.append(timestamp)
        self.metadata.append(metadata)
    
    def get_recent(self, n=10):
        """Get n most recent observations"""
        return {
            'observations': list(self.observations)[-n:],
            'timestamps': list(self.timestamps)[-n:],
            'metadata': list(self.metadata)[-n:]
        }
    
    def __len__(self):
        return len(self.observations)
```

### WebSocket Integration

Real-time data integration with WebSocket feeds:

```python
import websocket
import json
import threading
from queue import Queue

class WebSocketDataFeed:
    """WebSocket integration for real-time market data"""
    
    def __init__(self, streaming_system):
        self.streaming_system = streaming_system
        self.data_queue = Queue()
        self.ws = None
        self.running = False
        
    def connect(self, ws_url, symbols):
        """Connect to WebSocket feed"""
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Parse market data
                parsed_data = self.parse_market_data(data)
                if parsed_data:
                    self.data_queue.put(parsed_data)
                    
            except Exception as e:
                print(f"❌ Error parsing WebSocket message: {e}")
        
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("🔌 WebSocket connection closed")
            self.running = False
        
        def on_open(ws):
            print("✅ WebSocket connection opened")
            
            # Subscribe to symbols
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
            
            self.running = True
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_data_queue)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Run WebSocket
        self.ws.run_forever()
    
    def parse_market_data(self, data):
        """Parse incoming market data"""
        
        if data.get('e') == '24hrTicker':  # Binance ticker format
            return {
                'symbol': data['s'],
                'price': float(data['c']),
                'timestamp': data['E'],
                'volume': float(data['v']),
                'price_change': float(data['P'])
            }
        
        return None
    
    def process_data_queue(self):
        """Process queued market data"""
        
        while self.running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get(timeout=1)
                    
                    # Send to streaming system
                    asyncio.run_coroutine_threadsafe(
                        self.streaming_system.process_market_data(data),
                        self.streaming_system.loop
                    )
                    
            except Exception as e:
                print(f"❌ Error processing data queue: {e}")

# Usage example
async def run_websocket_streaming():
    """Example of WebSocket streaming integration"""
    
    # Create streaming system
    config = StreamingConfig(mode=StreamingMode.REAL_TIME)
    streaming_system = AdvancedStreamingSystem(config)
    
    # Initialize for crypto assets
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    await streaming_system.initialize_streaming(assets)
    
    # Create WebSocket feed
    ws_feed = WebSocketDataFeed(streaming_system)
    
    # Connect to Binance WebSocket (example)
    ws_url = "wss://stream.binance.com:9443/ws/stream"
    
    # Start streaming (this would run continuously)
    ws_feed.connect(ws_url, assets)
```

---

## Change Point Detection

### Structural Break Identification

Advanced change point detection for identifying market regime shifts:

```python
from scipy import stats
import ruptures as rpt
from collections import deque

class AdvancedChangePointDetector:
    """Advanced change point detection for market regime shifts"""
    
    def __init__(self, config=None):
        self.config = config or ChangeDetectionConfig()
        
        # Multiple detection methods
        self.detectors = {
            'cusum': CUSUMDetector(self.config),
            'bayesian': BayesianChangeDetector(self.config),
            'kernel': KernelChangeDetector(self.config),
            'statistical': StatisticalChangeDetector(self.config)
        }
        
        # Historical data for detection
        self.observation_window = deque(maxlen=self.config.window_size)
        self.detection_history = []
        
    def detect_change_points(self, new_observation, regime_info=None):
        """Detect change points using multiple methods"""
        
        self.observation_window.append(new_observation)
        
        if len(self.observation_window) < self.config.min_window_size:
            return {'change_detected': False, 'confidence': 0.0}
        
        # Run all detection methods
        detection_results = {}
        for method_name, detector in self.detectors.items():
            try:
                result = detector.detect(
                    list(self.observation_window), 
                    regime_info
                )
                detection_results[method_name] = result
            except Exception as e:
                print(f"❌ {method_name} detector error: {e}")
                detection_results[method_name] = {'change_detected': False, 'confidence': 0.0}
        
        # Combine results using ensemble method
        ensemble_result = self.ensemble_detection(detection_results)
        
        # Update detection history
        self.detection_history.append({
            'timestamp': len(self.observation_window),
            'observation': new_observation,
            'individual_results': detection_results,
            'ensemble_result': ensemble_result
        })
        
        return ensemble_result
    
    def ensemble_detection(self, detection_results):
        """Combine multiple detection methods"""
        
        # Weighted voting approach
        method_weights = {
            'cusum': 0.3,
            'bayesian': 0.3,
            'kernel': 0.2,
            'statistical': 0.2
        }
        
        weighted_confidence = 0.0
        change_votes = 0
        
        for method, result in detection_results.items():
            weight = method_weights.get(method, 0.25)
            weighted_confidence += result['confidence'] * weight
            
            if result['change_detected']:
                change_votes += 1
        
        # Decision logic
        change_detected = (
            change_votes >= 2 or  # Majority vote
            weighted_confidence > 0.7  # High confidence
        )
        
        return {
            'change_detected': change_detected,
            'confidence': weighted_confidence,
            'method_votes': change_votes,
            'individual_confidences': {k: v['confidence'] for k, v in detection_results.items()}
        }

class CUSUMDetector:
    """CUSUM (Cumulative Sum) change point detector"""
    
    def __init__(self, config):
        self.config = config
        self.cumsum_pos = 0
        self.cumsum_neg = 0
        
    def detect(self, observations, regime_info=None):
        """Detect change points using CUSUM algorithm"""
        
        # Parameters
        threshold = self.config.cusum_threshold
        drift = self.config.cusum_drift
        
        # Reset if regime changed (external information)
        if regime_info and regime_info.get('regime_changed', False):
            self.reset()
        
        # Calculate CUSUM statistics
        recent_mean = np.mean(observations[-10:]) if len(observations) >= 10 else np.mean(observations)
        overall_mean = np.mean(observations)
        
        deviation = recent_mean - overall_mean
        
        # Update CUSUM
        self.cumsum_pos = max(0, self.cumsum_pos + deviation - drift)
        self.cumsum_neg = max(0, self.cumsum_neg - deviation - drift)
        
        # Check thresholds
        change_detected = (self.cumsum_pos > threshold or self.cumsum_neg > threshold)
        confidence = min(max(self.cumsum_pos, self.cumsum_neg) / threshold, 1.0)
        
        if change_detected:
            self.reset()
        
        return {
            'change_detected': change_detected,
            'confidence': confidence,
            'cusum_pos': self.cumsum_pos,
            'cusum_neg': self.cumsum_neg
        }
    
    def reset(self):
        """Reset CUSUM statistics"""
        self.cumsum_pos = 0
        self.cumsum_neg = 0

class BayesianChangeDetector:
    """Bayesian change point detection"""
    
    def __init__(self, config):
        self.config = config
        self.prior_prob = 0.01  # Prior probability of change
        
    def detect(self, observations, regime_info=None):
        """Bayesian change point detection"""
        
        if len(observations) < 20:
            return {'change_detected': False, 'confidence': 0.0}
        
        # Split observations
        split_point = len(observations) // 2
        before = observations[:split_point]
        after = observations[split_point:]
        
        # Calculate likelihoods
        likelihood_no_change = self.calculate_single_segment_likelihood(observations)
        likelihood_change = (
            self.calculate_single_segment_likelihood(before) +
            self.calculate_single_segment_likelihood(after)
        )
        
        # Bayesian factor
        bayes_factor = likelihood_change - likelihood_no_change
        
        # Posterior probability of change
        posterior_prob = 1 / (1 + ((1 - self.prior_prob) / self.prior_prob) * np.exp(-bayes_factor))
        
        change_detected = posterior_prob > 0.5
        confidence = posterior_prob
        
        return {
            'change_detected': change_detected,
            'confidence': confidence,
            'bayes_factor': bayes_factor,
            'posterior_probability': posterior_prob
        }
    
    def calculate_single_segment_likelihood(self, segment):
        """Calculate likelihood for a single segment"""
        if len(segment) < 2:
            return 0.0
        
        mean = np.mean(segment)
        var = np.var(segment, ddof=1)
        
        if var <= 0:
            return 0.0
        
        # Log-likelihood for normal distribution
        n = len(segment)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * var) - 0.5 * np.sum((segment - mean)**2) / var
        
        return log_likelihood

class StatisticalChangeDetector:
    """Statistical tests for change point detection"""
    
    def __init__(self, config):
        self.config = config
        
    def detect(self, observations, regime_info=None):
        """Statistical change point detection using multiple tests"""
        
        if len(observations) < 30:
            return {'change_detected': False, 'confidence': 0.0}
        
        # Multiple statistical tests
        tests = {
            'variance': self.variance_change_test(observations),
            'mean': self.mean_change_test(observations),
            'distribution': self.distribution_change_test(observations)
        }
        
        # Combine test results
        p_values = [test['p_value'] for test in tests.values()]
        min_p_value = min(p_values)
        
        # Bonferroni correction for multiple tests
        corrected_alpha = 0.05 / len(tests)
        
        change_detected = min_p_value < corrected_alpha
        confidence = 1 - min_p_value
        
        return {
            'change_detected': change_detected,
            'confidence': confidence,
            'test_results': tests,
            'min_p_value': min_p_value
        }
    
    def variance_change_test(self, observations):
        """Test for change in variance"""
        n = len(observations)
        split = n // 2
        
        before = observations[:split]
        after = observations[split:]
        
        if len(before) < 3 or len(after) < 3:
            return {'test_statistic': 0, 'p_value': 1.0}
        
        # F-test for equal variances
        var1 = np.var(before, ddof=1)
        var2 = np.var(after, ddof=1)
        
        if var1 <= 0 or var2 <= 0:
            return {'test_statistic': 0, 'p_value': 1.0}
        
        f_stat = var1 / var2 if var1 > var2 else var2 / var1
        df1 = len(before) - 1
        df2 = len(after) - 1
        
        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
        
        return {
            'test_statistic': f_stat,
            'p_value': p_value
        }
    
    def mean_change_test(self, observations):
        """Test for change in mean"""
        n = len(observations)
        split = n // 2
        
        before = observations[:split]
        after = observations[split:]
        
        if len(before) < 3 or len(after) < 3:
            return {'test_statistic': 0, 'p_value': 1.0}
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(before, after)
        
        return {
            'test_statistic': abs(t_stat),
            'p_value': p_value
        }
    
    def distribution_change_test(self, observations):
        """Test for change in distribution"""
        n = len(observations)
        split = n // 2
        
        before = observations[:split]
        after = observations[split:]
        
        if len(before) < 3 or len(after) < 3:
            return {'test_statistic': 0, 'p_value': 1.0}
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(before, after)
        
        return {
            'test_statistic': ks_stat,
            'p_value': p_value
        }

@dataclass
class ChangeDetectionConfig:
    """Configuration for change point detection"""
    
    window_size: int = 100
    min_window_size: int = 20
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    confidence_threshold: float = 0.7
    enable_ensemble: bool = True

# Usage example
def change_detection_example():
    """Example of advanced change point detection"""
    
    # Create detector
    config = ChangeDetectionConfig()
    detector = AdvancedChangePointDetector(config)
    
    # Simulate data with change point
    np.random.seed(42)
    
    # Normal period
    normal_data = np.random.normal(0.001, 0.02, 50)
    
    # Crisis period (change point)
    crisis_data = np.random.normal(-0.01, 0.05, 30)
    
    # Recovery period
    recovery_data = np.random.normal(0.005, 0.025, 30)
    
    all_data = np.concatenate([normal_data, crisis_data, recovery_data])
    
    # Process data and detect changes
    change_points = []
    
    for i, observation in enumerate(all_data):
        result = detector.detect_change_points(observation)
        
        if result['change_detected']:
            change_points.append({
                'index': i,
                'confidence': result['confidence'],
                'observation': observation
            })
            
            print(f"🚨 Change point detected at index {i}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Observation: {observation:.4f}")
    
    print(f"\nDetected {len(change_points)} change points")
    print(f"Actual change points at indices: 50, 80")

if __name__ == "__main__":
    change_detection_example()
```

---

## Multi-Asset Analysis

### Cross-Asset Regime Correlation

Advanced multi-asset regime analysis with correlation modeling:

```python
class MultiAssetRegimeAnalyzer:
    """Advanced multi-asset regime correlation analysis"""
    
    def __init__(self, assets, regime_window=100):
        self.assets = assets
        self.regime_window = regime_window
        
        # Individual asset models
        self.asset_models = {
            asset: OnlineHMM(n_states=3) for asset in assets
        }
        
        # Cross-asset analysis
        self.regime_correlation_tracker = RegimeCorrelationTracker(assets, regime_window)
        self.sector_analyzer = SectorRegimeAnalyzer()
        self.contagion_detector = ContagionDetector(assets)
        
    def analyze_multi_asset_regimes(self, returns_dict):
        """Comprehensive multi-asset regime analysis"""
        
        results = {}
        
        # Update individual models
        individual_regimes = {}
        for asset, returns in returns_dict.items():
            if asset in self.asset_models:
                self.asset_models[asset].add_observation(returns)
                regime_info = self.asset_models[asset].get_current_regime_info()
                individual_regimes[asset] = regime_info
        
        results['individual_regimes'] = individual_regimes
        
        # Cross-asset correlation analysis
        correlation_analysis = self.regime_correlation_tracker.update_and_analyze(individual_regimes)
        results['regime_correlations'] = correlation_analysis
        
        # Sector-level analysis
        sector_analysis = self.sector_analyzer.analyze_sector_regimes(individual_regimes)
        results['sector_analysis'] = sector_analysis
        
        # Contagion detection
        contagion_analysis = self.contagion_detector.detect_contagion(individual_regimes)
        results['contagion_analysis'] = contagion_analysis
        
        # Portfolio implications
        portfolio_implications = self.calculate_portfolio_implications(results)
        results['portfolio_implications'] = portfolio_implications
        
        return results

class RegimeCorrelationTracker:
    """Track regime correlations across assets"""
    
    def __init__(self, assets, window_size=100):
        self.assets = assets
        self.window_size = window_size
        self.regime_history = {asset: deque(maxlen=window_size) for asset in assets}
        self.correlation_history = deque(maxlen=window_size)
        
    def update_and_analyze(self, current_regimes):
        """Update regime history and analyze correlations"""
        
        # Update regime history
        for asset, regime_info in current_regimes.items():
            if asset in self.regime_history:
                self.regime_history[asset].append(regime_info['current_state'])
        
        # Calculate current correlations
        current_correlations = self.calculate_regime_correlations()
        self.correlation_history.append(current_correlations)
        
        # Analyze correlation patterns
        correlation_analysis = self.analyze_correlation_patterns()
        
        return {
            'current_correlations': current_correlations,
            'correlation_trends': correlation_analysis,
            'regime_synchronization': self.calculate_regime_synchronization(),
            'divergence_signals': self.detect_regime_divergences()
        }
    
    def calculate_regime_correlations(self):
        """Calculate pairwise regime correlations"""
        
        correlations = {}
        
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets[i+1:], i+1):
                if (len(self.regime_history[asset1]) >= 30 and 
                    len(self.regime_history[asset2]) >= 30):
                    
                    seq1 = list(self.regime_history[asset1])
                    seq2 = list(self.regime_history[asset2])
                    
                    # Align sequences
                    min_len = min(len(seq1), len(seq2))
                    seq1 = seq1[-min_len:]
                    seq2 = seq2[-min_len:]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(seq1, seq2)[0, 1]
                    correlations[(asset1, asset2)] = correlation
        
        return correlations
    
    def analyze_correlation_patterns(self):
        """Analyze patterns in correlation over time"""
        
        if len(self.correlation_history) < 20:
            return {'trend': 'insufficient_data'}
        
        # Get correlation time series for analysis
        recent_correlations = list(self.correlation_history)[-20:]
        
        analysis = {}
        
        for asset_pair in recent_correlations[0].keys():
            correlation_series = [corrs.get(asset_pair, 0) for corrs in recent_correlations]
            
            # Trend analysis
            if len(correlation_series) >= 10:
                x = np.arange(len(correlation_series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, correlation_series)
                
                analysis[asset_pair] = {
                    'trend_slope': slope,
                    'current_correlation': correlation_series[-1],
                    'avg_correlation': np.mean(correlation_series),
                    'volatility': np.std(correlation_series),
                    'trend_strength': abs(r_value)
                }
        
        return analysis
    
    def calculate_regime_synchronization(self):
        """Calculate how synchronized regimes are across assets"""
        
        if not all(len(history) >= 10 for history in self.regime_history.values()):
            return 0.0
        
        # Get recent regime states
        recent_regimes = {}
        min_length = min(len(history) for history in self.regime_history.values())
        
        for asset, history in self.regime_history.items():
            recent_regimes[asset] = list(history)[-min_length:]
        
        # Calculate synchronization
        synchronization_scores = []
        
        for i in range(min_length):
            day_regimes = [recent_regimes[asset][i] for asset in self.assets]
            
            # Calculate how many assets are in the same regime
            regime_counts = {}
            for regime in day_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Synchronization = max proportion in same regime
            max_proportion = max(regime_counts.values()) / len(self.assets)
            synchronization_scores.append(max_proportion)
        
        return {
            'current_synchronization': synchronization_scores[-1] if synchronization_scores else 0,
            'avg_synchronization': np.mean(synchronization_scores) if synchronization_scores else 0,
            'synchronization_trend': self.calculate_synchronization_trend(synchronization_scores)
        }
    
    def detect_regime_divergences(self):
        """Detect when assets diverge from common regime patterns"""
        
        divergences = []
        
        if not all(len(history) >= 5 for history in self.regime_history.values()):
            return divergences
        
        # Get most recent regime states
        current_regimes = {asset: list(history)[-1] for asset, history in self.regime_history.items()}
        
        # Find most common regime
        regime_counts = {}
        for regime in current_regimes.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            dominant_regime = max(regime_counts, key=regime_counts.get)
            dominance_ratio = regime_counts[dominant_regime] / len(self.assets)
            
            # Identify divergent assets
            for asset, regime in current_regimes.items():
                if regime != dominant_regime:
                    divergences.append({
                        'asset': asset,
                        'current_regime': regime,
                        'dominant_regime': dominant_regime,
                        'divergence_strength': 1.0 - dominance_ratio
                    })
        
        return divergences

class SectorRegimeAnalyzer:
    """Analyze regimes at sector level"""
    
    def __init__(self):
        # Define sector mappings (example for US equities)
        self.sector_mappings = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE']
        }
        
        self.sector_regime_history = {}
        
    def analyze_sector_regimes(self, individual_regimes):
        """Analyze regimes at sector level"""
        
        sector_analysis = {}
        
        for sector, assets in self.sector_mappings.items():
            sector_regimes = []
            
            # Collect regimes for assets in this sector
            for asset in assets:
                if asset in individual_regimes:
                    sector_regimes.append(individual_regimes[asset]['current_state'])
            
            if sector_regimes:
                # Calculate sector regime characteristics
                regime_distribution = {}
                for regime in sector_regimes:
                    regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
                
                # Dominant regime
                dominant_regime = max(regime_distribution, key=regime_distribution.get)
                dominance_strength = regime_distribution[dominant_regime] / len(sector_regimes)
                
                # Sector consensus
                consensus_strength = self.calculate_sector_consensus(sector_regimes)
                
                sector_analysis[sector] = {
                    'dominant_regime': dominant_regime,
                    'dominance_strength': dominance_strength,
                    'consensus_strength': consensus_strength,
                    'regime_distribution': regime_distribution,
                    'num_assets': len(sector_regimes)
                }
                
                # Update sector history
                if sector not in self.sector_regime_history:
                    self.sector_regime_history[sector] = deque(maxlen=100)
                
                self.sector_regime_history[sector].append({
                    'dominant_regime': dominant_regime,
                    'consensus_strength': consensus_strength
                })
        
        # Cross-sector analysis
        cross_sector_analysis = self.analyze_cross_sector_patterns(sector_analysis)
        
        return {
            'individual_sectors': sector_analysis,
            'cross_sector': cross_sector_analysis,
            'sector_rotation_signals': self.detect_sector_rotation_signals(sector_analysis)
        }
    
    def calculate_sector_consensus(self, regime_list):
        """Calculate how much consensus exists within a sector"""
        
        if not regime_list:
            return 0.0
        
        # Calculate entropy as measure of consensus (lower entropy = higher consensus)
        regime_counts = {}
        for regime in regime_list:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(regime_list)
        entropy = 0
        
        for count in regime_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        # Convert to consensus (0 = no consensus, 1 = perfect consensus)
        max_entropy = np.log2(min(len(regime_counts), 3))  # Max possible entropy
        consensus = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
        
        return consensus

class ContagionDetector:
    """Detect regime contagion between assets"""
    
    def __init__(self, assets):
        self.assets = assets
        self.contagion_history = deque(maxlen=50)
        
    def detect_contagion(self, current_regimes):
        """Detect regime contagion patterns"""
        
        contagion_signals = []
        
        # Check for simultaneous regime changes
        regime_changes = {}
        for asset, regime_info in current_regimes.items():
            if regime_info.get('regime_changed', False):
                regime_changes[asset] = {
                    'new_regime': regime_info['current_state'],
                    'confidence': regime_info['confidence']
                }
        
        # Analyze contagion if multiple assets changed regimes
        if len(regime_changes) >= 2:
            contagion_analysis = self.analyze_simultaneous_changes(regime_changes)
            contagion_signals.append(contagion_analysis)
        
        # Check for lagged contagion
        lagged_contagion = self.check_lagged_contagion(current_regimes)
        if lagged_contagion:
            contagion_signals.extend(lagged_contagion)
        
        return {
            'current_contagion_signals': contagion_signals,
            'contagion_risk_level': self.assess_contagion_risk(contagion_signals),
            'systemic_risk_indicators': self.calculate_systemic_risk_indicators(current_regimes)
        }
    
    def analyze_simultaneous_changes(self, regime_changes):
        """Analyze simultaneous regime changes for contagion"""
        
        # Check if changes are in the same direction
        new_regimes = [info['new_regime'] for info in regime_changes.values()]
        
        # Contagion strength based on:
        # 1. Number of assets changing
        # 2. Confidence levels
        # 3. Direction consistency
        
        avg_confidence = np.mean([info['confidence'] for info in regime_changes.values()])
        
        # Check for consistent direction (all moving to stress regime, etc.)
        stress_regimes = [0]  # Bear market regime
        stress_changes = sum(1 for regime in new_regimes if regime in stress_regimes)
        direction_consistency = stress_changes / len(new_regimes)
        
        contagion_strength = (
            len(regime_changes) / len(self.assets) * 0.5 +  # Breadth
            avg_confidence * 0.3 +                           # Confidence
            direction_consistency * 0.2                      # Direction
        )
        
        return {
            'type': 'simultaneous',
            'affected_assets': list(regime_changes.keys()),
            'contagion_strength': contagion_strength,
            'avg_confidence': avg_confidence,
            'direction_consistency': direction_consistency
        }

# Usage example
def multi_asset_analysis_example():
    """Example of multi-asset regime analysis"""
    
    # Assets for analysis
    assets = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'IWM']
    
    # Create analyzer
    analyzer = MultiAssetRegimeAnalyzer(assets)
    
    # Simulate returns data
    np.random.seed(42)
    
    for day in range(100):
        # Generate correlated returns
        base_return = np.random.normal(0.001, 0.02)
        returns_dict = {}
        
        for asset in assets:
            # Add some correlation and asset-specific noise
            correlation = 0.7 if asset in ['SPY', 'QQQ', 'IWM'] else 0.5
            asset_return = correlation * base_return + (1 - correlation) * np.random.normal(0.001, 0.02)
            returns_dict[asset] = asset_return
        
        # Analyze multi-asset regimes
        results = analyzer.analyze_multi_asset_regimes(returns_dict)
        
        # Print interesting findings
        if day % 20 == 0:  # Every 20 days
            print(f"\n📊 Day {day} Multi-Asset Analysis:")
            
            # Individual regimes
            print("Individual Regimes:")
            for asset, regime_info in results['individual_regimes'].items():
                regime_name = ['Bear', 'Sideways', 'Bull'][regime_info['current_state']]
                print(f"  {asset}: {regime_name} ({regime_info['confidence']:.2%})")
            
            # Sector analysis
            if 'sector_analysis' in results:
                print("\nSector Analysis:")
                for sector, analysis in results['sector_analysis']['individual_sectors'].items():
                    regime_name = ['Bear', 'Sideways', 'Bull'][analysis['dominant_regime']]
                    print(f"  {sector}: {regime_name} (consensus: {analysis['consensus_strength']:.2%})")
            
            # Contagion signals
            contagion = results['contagion_analysis']
            if contagion['current_contagion_signals']:
                print(f"\n🚨 Contagion Risk Level: {contagion['contagion_risk_level']}")

if __name__ == "__main__":
    multi_asset_analysis_example()
```

---

## Model Comparison Framework

### Systematic Model Evaluation

Framework for comparing different HMM configurations and approaches:

```python
class ModelComparisonFramework:
    """Framework for systematic model comparison and validation"""
    
    def __init__(self, comparison_config=None):
        self.config = comparison_config or ModelComparisonConfig()
        self.comparison_results = []
        self.model_registry = {}
        
    def register_model(self, model_id, model_factory, description=""):
        """Register a model for comparison"""
        
        self.model_registry[model_id] = {
            'factory': model_factory,
            'description': description,
            'results': []
        }
    
    def compare_models(self, training_data, validation_data, test_data=None):
        """Compare all registered models"""
        
        print(f"🔍 Starting model comparison with {len(self.model_registry)} models")
        
        comparison_results = {}
        
        for model_id, model_info in self.model_registry.items():
            print(f"📊 Evaluating {model_id}...")
            
            try:
                # Create model instance
                model = model_info['factory']()
                
                # Comprehensive evaluation
                results = self.evaluate_model(
                    model, model_id, training_data, validation_data, test_data
                )
                
                comparison_results[model_id] = results
                model_info['results'].append(results)
                
                print(f"✅ {model_id} evaluation complete")
                
            except Exception as e:
                print(f"❌ {model_id} evaluation failed: {e}")
                comparison_results[model_id] = {'error': str(e)}
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(comparison_results)
        
        return comparison_report
    
    def evaluate_model(self, model, model_id, training_data, validation_data, test_data):
        """Comprehensive model evaluation"""
        
        evaluation_results = {}
        
        # Training performance
        training_results = self.evaluate_training_performance(model, training_data)
        evaluation_results['training'] = training_results
        
        # Validation performance
        validation_results = self.evaluate_prediction_performance(model, validation_data, 'validation')
        evaluation_results['validation'] = validation_results
        
        # Test performance (if provided)
        if test_data is not None:
            test_results = self.evaluate_prediction_performance(model, test_data, 'test')
            evaluation_results['test'] = test_results
        
        # Regime detection quality
        regime_quality = self.evaluate_regime_detection_quality(model, validation_data)
        evaluation_results['regime_quality'] = regime_quality
        
        # Computational efficiency
        efficiency_results = self.evaluate_computational_efficiency(model, validation_data)
        evaluation_results['efficiency'] = efficiency_results
        
        # Stability analysis
        stability_results = self.evaluate_model_stability(model, validation_data)
        evaluation_results['stability'] = stability_results
        
        # Overall score
        overall_score = self.calculate_overall_score(evaluation_results)
        evaluation_results['overall_score'] = overall_score
        
        return evaluation_results
    
    def evaluate_training_performance(self, model, training_data):
        """Evaluate training performance"""
        
        start_time = time.time()
        
        # Train model
        model.fit(training_data, verbose=False)
        
        training_time = time.time() - start_time
        
        # Training metrics
        training_results = {
            'training_time': training_time,
            'converged': model.is_fitted if hasattr(model, 'is_fitted') else True,
            'final_log_likelihood': None,
            'iterations_used': None
        }
        
        # Get training history if available
        if hasattr(model, 'training_history_'):
            history = model.training_history_
            training_results['final_log_likelihood'] = history.get('final_log_likelihood')
            training_results['iterations_used'] = history.get('iterations', 0)
        
        return training_results
    
    def evaluate_prediction_performance(self, model, data, dataset_name):
        """Evaluate prediction performance"""
        
        # Generate predictions
        start_time = time.time()
        
        states = model.predict(data)
        state_probs = model.predict_proba(data)
        likelihood = model.score(data)
        
        prediction_time = time.time() - start_time
        
        # Calculate performance metrics
        performance_metrics = {
            'log_likelihood': likelihood,
            'avg_confidence': np.mean(np.max(state_probs, axis=1)),
            'regime_stability': self.calculate_regime_stability(states),
            'prediction_time': prediction_time,
            'predictions_per_second': len(data) / prediction_time if prediction_time > 0 else float('inf')
        }
        
        # Regime-specific analysis
        regime_analysis = self.analyze_regime_predictions(states, state_probs, data)
        performance_metrics['regime_analysis'] = regime_analysis
        
        return performance_metrics
    
    def evaluate_regime_detection_quality(self, model, data):
        """Evaluate quality of regime detection"""
        
        states = model.predict(data)
        state_probs = model.predict_proba(data)
        
        # Quality metrics
        quality_metrics = {
            'regime_diversity': len(np.unique(states)) / model.n_states,
            'confidence_distribution': {
                'mean': np.mean(np.max(state_probs, axis=1)),
                'std': np.std(np.max(state_probs, axis=1)),
                'min': np.min(np.max(state_probs, axis=1)),
                'q25': np.percentile(np.max(state_probs, axis=1), 25),
                'q75': np.percentile(np.max(state_probs, axis=1), 75)
            },
            'regime_transitions': np.sum(states[1:] != states[:-1]),
            'avg_regime_duration': self.calculate_average_regime_duration(states)
        }
        
        # Regime interpretation quality
        regime_interpretations = self.evaluate_regime_interpretations(model, states, data)
        quality_metrics['interpretations'] = regime_interpretations
        
        return quality_metrics
    
    def evaluate_computational_efficiency(self, model, data):
        """Evaluate computational efficiency"""
        
        efficiency_metrics = {}
        
        # Batch prediction timing
        batch_times = []
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            if len(data) >= batch_size:
                batch_data = data[:batch_size]
                
                start_time = time.time()
                model.predict(batch_data)
                batch_time = time.time() - start_time
                
                batch_times.append({
                    'batch_size': batch_size,
                    'total_time': batch_time,
                    'time_per_observation': batch_time / batch_size
                })
        
        efficiency_metrics['batch_performance'] = batch_times
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            efficiency_metrics['memory_usage_mb'] = memory_mb
        except:
            pass
        
        # Online processing efficiency (for OnlineHMM)
        if hasattr(model, 'add_observation'):
            online_performance = self.test_online_processing_speed(model, data[:100])
            efficiency_metrics['online_performance'] = online_performance
        
        return efficiency_metrics
    
    def evaluate_model_stability(self, model, data):
        """Evaluate model stability"""
        
        stability_metrics = {}
        
        # Parameter stability test
        if hasattr(model, 'transition_matrix_'):
            # Check for well-conditioned transition matrix
            transition_matrix = model.transition_matrix_
            
            # Condition number
            cond_number = np.linalg.cond(transition_matrix)
            stability_metrics['transition_matrix_condition'] = cond_number
            
            # Check for numerical issues
            stability_metrics['has_negative_probabilities'] = np.any(transition_matrix < 0)
            stability_metrics['rows_sum_to_one'] = np.allclose(transition_matrix.sum(axis=1), 1.0)
        
        # Emission parameter stability
        if hasattr(model, 'emission_params_'):
            emission_params = model.emission_params_
            
            # Check for extreme parameters
            means = [param[0] for param in emission_params]
            stds = [param[1] for param in emission_params]
            
            stability_metrics['extreme_means'] = any(abs(mean) > 0.1 for mean in means)  # >10% daily
            stability_metrics['extreme_stds'] = any(std > 0.1 for std in stds)  # >10% daily vol
            stability_metrics['zero_variances'] = any(std <= 0 for std in stds)
        
        # Prediction stability
        prediction_stability = self.test_prediction_stability(model, data)
        stability_metrics['prediction_stability'] = prediction_stability
        
        return stability_metrics
    
    def test_prediction_stability(self, model, data):
        """Test prediction stability with slightly perturbed data"""
        
        # Original predictions
        original_states = model.predict(data)
        
        # Predictions with small perturbations
        perturbed_agreements = []
        
        for _ in range(5):  # 5 perturbation tests
            noise = np.random.normal(0, 0.001, len(data))  # Small noise
            perturbed_data = data + noise
            
            perturbed_states = model.predict(perturbed_data)
            agreement = np.mean(original_states == perturbed_states)
            perturbed_agreements.append(agreement)
        
        return {
            'avg_agreement': np.mean(perturbed_agreements),
            'std_agreement': np.std(perturbed_agreements),
            'min_agreement': np.min(perturbed_agreements)
        }
    
    def generate_comparison_report(self, comparison_results):
        """Generate comprehensive comparison report"""
        
        report = {
            'summary': {},
            'detailed_results': comparison_results,
            'rankings': {},
            'recommendations': []
        }
        
        # Calculate rankings for different criteria
        ranking_criteria = [
            'overall_score',
            'validation.log_likelihood',
            'efficiency.online_performance.avg_time_per_obs',
            'regime_quality.confidence_distribution.mean',
            'stability.prediction_stability.avg_agreement'
        ]
        
        for criterion in ranking_criteria:
            try:
                scores = {}
                for model_id, results in comparison_results.items():
                    if 'error' not in results:
                        score = self.get_nested_value(results, criterion)
                        if score is not None:
                            scores[model_id] = score
                
                # Rank models (higher is better for most criteria)
                reverse_sort = not criterion.endswith('avg_time_per_obs')  # Time should be lower
                ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=reverse_sort)
                
                report['rankings'][criterion] = ranked_models
                
            except Exception as e:
                print(f"❌ Error ranking by {criterion}: {e}")
        
        # Generate recommendations
        recommendations = self.generate_model_recommendations(comparison_results)
        report['recommendations'] = recommendations
        
        # Summary statistics
        summary = self.generate_comparison_summary(comparison_results)
        report['summary'] = summary
        
        return report
    
    def get_nested_value(self, dictionary, key_path):
        """Get nested dictionary value using dot notation"""
        keys = key_path.split('.')
        value = dictionary
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def generate_model_recommendations(self, comparison_results):
        """Generate recommendations based on comparison results"""
        
        recommendations = []
        
        # Find best overall model
        overall_scores = {}
        for model_id, results in comparison_results.items():
            if 'error' not in results and 'overall_score' in results:
                overall_scores[model_id] = results['overall_score']
        
        if overall_scores:
            best_model = max(overall_scores, key=overall_scores.get)
            recommendations.append({
                'type': 'best_overall',
                'model': best_model,
                'rationale': f"Highest overall score: {overall_scores[best_model]:.3f}"
            })
        
        # Find best for specific use cases
        use_case_recommendations = [
            ('high_frequency', 'efficiency.online_performance.avg_time_per_obs', 'Fastest online processing'),
            ('accuracy', 'validation.log_likelihood', 'Highest validation likelihood'),
            ('stability', 'stability.prediction_stability.avg_agreement', 'Most stable predictions')
        ]
        
        for use_case, criterion, description in use_case_recommendations:
            scores = {}
            for model_id, results in comparison_results.items():
                if 'error' not in results:
                    score = self.get_nested_value(results, criterion)
                    if score is not None:
                        scores[model_id] = score
            
            if scores:
                # For time-based criteria, lower is better
                reverse = not criterion.endswith('avg_time_per_obs')
                best_model = max(scores, key=scores.get) if reverse else min(scores, key=scores.get)
                
                recommendations.append({
                    'type': use_case,
                    'model': best_model,
                    'rationale': f"{description}: {scores[best_model]:.6f}"
                })
        
        return recommendations

# Usage example
def model_comparison_example():
    """Example of systematic model comparison"""
    
    # Generate sample data
    np.random.seed(42)
    training_data = np.random.normal(0.001, 0.02, 500)
    validation_data = np.random.normal(0.001, 0.02, 200)
    test_data = np.random.normal(0.001, 0.02, 100)
    
    # Create comparison framework
    framework = ModelComparisonFramework()
    
    # Register different models
    
    # Model 1: Basic HMM
    framework.register_model(
        'basic_hmm',
        lambda: HiddenMarkovModel(
            config=HMMConfig(n_states=3, initialization_method='random')
        ),
        'Basic HMM with random initialization'
    )
    
    # Model 2: K-means initialized HMM
    framework.register_model(
        'kmeans_hmm',
        lambda: HiddenMarkovModel(
            config=HMMConfig(n_states=3, initialization_method='kmeans')
        ),
        'HMM with k-means initialization'
    )
    
    # Model 3: Online HMM
    framework.register_model(
        'online_hmm',
        lambda: OnlineHMM(
            n_states=3,
            online_config=OnlineHMMConfig(forgetting_factor=0.98)
        ),
        'Online HMM with standard configuration'
    )
    
    # Model 4: Conservative Online HMM
    framework.register_model(
        'conservative_online_hmm',
        lambda: OnlineHMM(
            n_states=3,
            online_config=OnlineHMMConfig(
                forgetting_factor=0.99,
                adaptation_rate=0.02,
                smoothing_weight=0.9
            )
        ),
        'Conservative Online HMM'
    )
    
    # Run comparison
    results = framework.compare_models(training_data, validation_data, test_data)
    
    # Display results
    print("\n📋 MODEL COMPARISON RESULTS")
    print("=" * 50)
    
    # Overall rankings
    if 'overall_score' in results['rankings']:
        print("\n🏆 Overall Rankings:")
        for i, (model_id, score) in enumerate(results['rankings']['overall_score'], 1):
            print(f"  {i}. {model_id}: {score:.3f}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"  {rec['type'].upper()}: {rec['model']}")
        print(f"    Rationale: {rec['rationale']}")
    
    # Detailed performance for top model
    if results['recommendations']:
        top_model = results['recommendations'][0]['model']
        top_results = results['detailed_results'][top_model]
        
        print(f"\n📊 Detailed Results for {top_model}:")
        print(f"  Training time: {top_results['training']['training_time']:.3f}s")
        print(f"  Validation likelihood: {top_results['validation']['log_likelihood']:.2f}")
        print(f"  Average confidence: {top_results['validation']['avg_confidence']:.2%}")
        print(f"  Regime stability: {top_results['validation']['regime_stability']:.2%}")

if __name__ == "__main__":
    model_comparison_example()
```

This comprehensive Advanced Features Documentation provides detailed coverage of sophisticated HMM capabilities including streaming architecture, change point detection, multi-asset analysis, and model comparison frameworks. These features enable production-ready financial applications with advanced analytical capabilities.

---

*For additional information, see the [Online HMM Documentation](online_hmm.md), [Trading Applications Guide](trading_applications.md), and [Configuration Guide](configuration_guide.md).*