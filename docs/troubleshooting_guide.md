# Hidden Regime Troubleshooting Guide

**Complete guide for diagnosing and resolving issues with Hidden Regime**

This guide covers common problems, error messages, performance issues, and their solutions for both development and production environments.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Model Training Issues](#model-training-issues)
4. [Regime Detection Problems](#regime-detection-problems)
5. [Performance Issues](#performance-issues)
6. [API and Production Issues](#api-and-production-issues)
7. [Visualization Problems](#visualization-problems)
8. [Common Error Messages](#common-error-messages)
9. [Diagnostic Tools](#diagnostic-tools)
10. [Getting Help](#getting-help)

## Installation Issues

### Problem: pip install fails with compilation errors

**Symptoms:**
```
ERROR: Failed building wheel for numpy
ERROR: Could not build wheels for numpy which use PEP 517
```

**Solution 1: Install system dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
brew install gcc

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

**Solution 2: Use pre-compiled wheels**
```bash
pip install --only-binary=all hidden-regime
```

**Solution 3: Use conda environment**
```bash
conda create -n hidden-regime python=3.10
conda activate hidden-regime
conda install numpy scipy pandas
pip install hidden-regime
```

### Problem: ImportError after installation

**Symptoms:**
```python
ImportError: No module named 'hidden_regime'
```

**Diagnosis:**
```bash
# Check if package is installed
pip list | grep hidden-regime

# Check Python path
python -c "import sys; print(sys.path)"

# Verify virtual environment
which python
```

**Solutions:**
```bash
# Ensure you're in the correct environment
source venv/bin/activate  # or conda activate hidden-regime

# Reinstall package
pip uninstall hidden-regime
pip install hidden-regime

# Check for name conflicts
pip list | grep regime
```

### Problem: Version conflicts

**Symptoms:**
```
ERROR: hidden-regime requires numpy>=1.20.0, but you have numpy 1.19.0
```

**Solution:**
```bash
# Upgrade conflicting packages
pip install --upgrade numpy scipy pandas

# Or create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install hidden-regime
```

## Data Loading Problems

### Problem: No data retrieved from yfinance

**Symptoms:**
```python
ValidationError: No valid data returned from yfinance for AAPL
```

**Diagnosis:**
```python
import yfinance as yf
import pandas as pd

# Test direct yfinance call
ticker = yf.Ticker('AAPL')
data = ticker.history(start='2023-01-01', end='2023-12-31')
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
```

**Solutions:**

1. **Check ticker symbol:**
```python
# Verify ticker exists
import yfinance as yf
ticker = yf.Ticker('AAPL')
info = ticker.info
print(f"Company: {info.get('longName', 'Not found')}")
```

2. **Adjust date range:**
```python
from datetime import datetime, timedelta

# Try more recent dates
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=365)

data = loader.load('AAPL', start_date, end_date)
```

3. **Check internet connection:**
```python
import requests

try:
    response = requests.get('https://query1.finance.yahoo.com/v1/finance/search?q=AAPL', timeout=10)
    print(f"Yahoo Finance accessible: {response.status_code == 200}")
except Exception as e:
    print(f"Connection error: {e}")
```

### Problem: Data quality score too low

**Symptoms:**
```
ValidationError: Data quality score 0.42 is below minimum threshold 0.7
```

**Diagnosis:**
```python
from hidden_regime import DataValidator, ValidationConfig

validator = DataValidator()
result = validator.validate_data(data, 'AAPL')

print(f"Quality score: {result.quality_score}")
print(f"Issues: {result.issues}")
print(f"Warnings: {result.warnings}")
print(f"Missing data %: {result.missing_data_percentage}")
```

**Solutions:**

1. **Lower quality threshold:**
```python
config = ValidationConfig(min_quality_score=0.4)
validator = DataValidator(config)
```

2. **Fix specific issues:**
```python
# Handle missing data
from hidden_regime import DataPreprocessor

preprocessor = DataPreprocessor()
cleaned_data = preprocessor.process_data(data, 'AAPL')

# Re-validate
result = validator.validate_data(cleaned_data, 'AAPL')
```

3. **Use different data source or period:**
```python
# Try different time period
data = loader.load('AAPL', '2020-01-01', '2023-01-01')

# Or different ticker
data = loader.load('SPY', '2022-01-01', '2023-12-31')
```

### Problem: Rate limiting from data provider

**Symptoms:**
```
HTTPError: 429 Too Many Requests
```

**Solutions:**

1. **Adjust rate limiting:**
```python
from hidden_regime import DataConfig

config = DataConfig(
    requests_per_minute=30,  # Reduce from default 60
    retry_delay=5.0          # Increase delay
)
loader = DataLoader(config)
```

2. **Use caching:**
```python
config = DataConfig(
    cache_enabled=True,
    cache_ttl_hours=24
)
```

3. **Implement exponential backoff:**
```python
import time
import random

def load_with_backoff(ticker, start, end, max_retries=5):
    for attempt in range(max_retries):
        try:
            return loader.load(ticker, start, end)
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
```

## Model Training Issues

### Problem: Model fails to converge

**Symptoms:**
```
Warning: Model did not converge after 100 iterations
```

**Diagnosis:**
```python
# Check training progress
config = HMMConfig(max_iterations=100, tolerance=1e-6)
model = HiddenMarkovModel(config)
model.fit(data['log_return'], verbose=True)

# Check log likelihood history
if hasattr(model, 'log_likelihood_history_'):
    import matplotlib.pyplot as plt
    plt.plot(model.log_likelihood_history_)
    plt.title('Log Likelihood During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.show()
```

**Solutions:**

1. **Adjust convergence parameters:**
```python
config = HMMConfig(
    max_iterations=200,      # Increase iterations
    tolerance=1e-5,          # Relax tolerance
    early_stopping=False     # Disable early stopping
)
```

2. **Improve initialization:**
```python
config = HMMConfig(
    initialization_method='kmeans',  # Better than 'random'
    random_seed=42                   # Reproducible results
)
```

3. **Check data quality:**
```python
# Ensure sufficient data
print(f"Data points: {len(data)}")
print(f"Return statistics:")
print(data['log_return'].describe())

# Check for extreme values
import numpy as np
extreme_returns = data[np.abs(data['log_return']) > 0.1]['log_return']
print(f"Extreme returns (>10%): {len(extreme_returns)}")
```

4. **Regularization:**
```python
config = HMMConfig(
    regularization=1e-5,     # Add regularization
    min_variance=1e-6        # Prevent variance collapse
)
```

### Problem: Invalid model parameters

**Symptoms:**
```
ValueError: Transition matrix contains NaN values
RuntimeWarning: invalid value encountered in log
```

**Diagnosis:**
```python
# Check model parameters after training
print("Transition matrix:")
print(model.transition_matrix_)
print("\nEmission parameters:")
print(model.emission_params_)
print("\nInitial probabilities:")
print(model.initial_probs_)

# Check for NaN/inf values
import numpy as np
print(f"NaN in transitions: {np.isnan(model.transition_matrix_).any()}")
print(f"NaN in emissions: {np.isnan(model.emission_params_).any()}")
```

**Solutions:**

1. **Add numerical stability:**
```python
config = HMMConfig(
    regularization=1e-4,           # Higher regularization
    min_variance=1e-5,             # Higher minimum variance
    log_likelihood_threshold=-1e8   # Prevent extreme likelihoods
)
```

2. **Clean input data:**
```python
# Remove extreme outliers
data_clean = data[np.abs(data['log_return']) < 0.2].copy()

# Handle missing values
data_clean = data_clean.dropna()

# Check for infinite values
data_clean = data_clean[np.isfinite(data_clean['log_return'])]
```

3. **Use different initialization:**
```python
# Try different initialization methods
for init_method in ['random', 'kmeans']:
    try:
        config = HMMConfig(initialization_method=init_method, random_seed=42)
        model = HiddenMarkovModel(config)
        model.fit(data['log_return'])
        print(f"Success with {init_method} initialization")
        break
    except Exception as e:
        print(f"Failed with {init_method}: {e}")
```

### Problem: Memory error during training

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce data size:**
```python
# Use smaller date range
data_subset = data.tail(1000)  # Last 1000 days only

# Or sample data
data_sampled = data.sample(n=1000, random_state=42)
```

2. **Reduce model complexity:**
```python
config = HMMConfig(
    n_states=3,              # Use fewer states
    max_iterations=50        # Fewer iterations
)
```

3. **Optimize memory usage:**
```python
# Use float32 instead of float64
data['log_return'] = data['log_return'].astype(np.float32)

# Clear unnecessary variables
import gc
gc.collect()
```

## Regime Detection Problems

### Problem: StateStandardizer validation fails

**Symptoms:**
```
ValueError: Cannot standardize states with regime_type='auto'. Call select_optimal_configuration first.
```

**Solution:**
```python
from hidden_regime.models.state_standardizer import StateStandardizer

# For auto regime type
standardizer = StateStandardizer(regime_type='auto')
best_config, score, details = standardizer.select_optimal_configuration(
    data['log_return'].values, 
    max_states=6
)
print(f"Selected {best_config.n_states} states with score {score}")

# Then use the selected configuration
standardizer = StateStandardizer(regime_type=f"{best_config.n_states}_state")
```

### Problem: Low economic validation confidence

**Symptoms:**
```
Warning: Economic validation confidence 0.23 is below threshold 0.7
```

**Diagnosis:**
```python
# Check regime characteristics
state_mapping = standardizer.standardize_states(model.emission_params_)
states = model.predict(data['log_return'].values)

for state_idx, regime_name in state_mapping.items():
    state_returns = data['log_return'].values[states == state_idx]
    print(f"{regime_name} (State {state_idx}):")
    print(f"  Mean return: {np.mean(state_returns):.4f}")
    print(f"  Std dev: {np.std(state_returns):.4f}")
    print(f"  Count: {len(state_returns)}")
```

**Solutions:**

1. **Use different regime type:**
```python
# Try 3-state instead of 4 or 5-state
standardizer = StateStandardizer(regime_type='3_state')
```

2. **Adjust validation threshold:**
```python
# Lower the validation threshold
confidence = standardizer.validate_interpretation(
    states, data['log_return'].values, model.emission_params_
)
# Accept lower confidence if necessary
if confidence > 0.4:  # Lower threshold
    print("Acceptable validation")
```

3. **Retrain with more data:**
```python
# Use longer time period
data_longer = loader.load('AAPL', '2015-01-01', '2023-12-31')
model.fit(data_longer['log_return'])
```

### Problem: Inconsistent regime predictions

**Symptoms:**
- Regimes change too frequently
- Short-lived regime periods (1-2 days)
- No clear regime persistence

**Solutions:**

1. **Add regime duration constraints:**
```python
config = HMMConfig(
    min_regime_duration=3,    # Minimum 3 days in regime
    force_state_ordering=True # Ensure ordered by return
)
```

2. **Use conservative training:**
```python
config = HMMConfig.for_standardized_regimes(
    regime_type='3_state',
    conservative=True
)
```

3. **Smooth predictions:**
```python
def smooth_regimes(regimes, window=3):
    """Apply simple smoothing to reduce regime switching"""
    import pandas as pd
    regime_series = pd.Series(regimes)
    # Mode filter
    return regime_series.rolling(window=window, center=True).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).fillna(regime_series)

smoothed_regimes = smooth_regimes(regime_names)
```

## Performance Issues

### Problem: Slow model training

**Symptoms:**
- Training takes > 5 minutes for 1000 data points
- CPU usage is low during training

**Solutions:**

1. **Check data size:**
```python
print(f"Training data size: {len(data)}")
print(f"Number of states: {config.n_states}")
print(f"Max iterations: {config.max_iterations}")

# Reduce if necessary
if len(data) > 2000:
    data_subset = data.tail(2000)  # Use last 2000 points
```

2. **Enable early stopping:**
```python
config = HMMConfig(
    early_stopping=True,
    check_convergence_every=5,  # Check every 5 iterations
    tolerance=1e-5              # Less strict tolerance
)
```

3. **Profile the code:**
```python
import cProfile
import pstats

# Profile training
profiler = cProfile.Profile()
profiler.enable()

model.fit(data['log_return'])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### Problem: High memory usage

**Symptoms:**
- Process memory grows during training
- Out of memory errors

**Solutions:**

1. **Monitor memory usage:**
```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before training: {get_memory_usage():.1f} MB")
model.fit(data['log_return'])
print(f"Memory after training: {get_memory_usage():.1f} MB")
```

2. **Optimize data types:**
```python
# Use smaller data types
data['log_return'] = data['log_return'].astype(np.float32)

# Clear intermediate variables
del intermediate_variables
import gc
gc.collect()
```

3. **Process data in chunks:**
```python
def train_incremental(model, data, chunk_size=500):
    """Train model on data chunks to save memory"""
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # This would need custom implementation for incremental training
        pass
```

### Problem: API response times too slow

**Symptoms:**
- API calls take > 5 seconds
- Timeout errors in production

**Solutions:**

1. **Enable caching:**
```python
# Cache model predictions
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_regime_prediction(ticker, data_hash):
    # Implementation with caching
    pass
```

2. **Optimize database queries:**
```sql
-- Add indexes
CREATE INDEX CONCURRENTLY idx_market_data_ticker_date 
ON market_data (ticker, date);

-- Use EXPLAIN to analyze query performance
EXPLAIN ANALYZE SELECT * FROM market_data WHERE ticker = 'AAPL';
```

3. **Use connection pooling:**
```python
# Implement connection pooling as shown in deployment guide
# Avoid creating new connections for each request
```

## API and Production Issues

### Problem: 503 Service Unavailable

**Symptoms:**
```
HTTP 503: Service Unavailable
{"detail": "Service temporarily unavailable"}
```

**Diagnosis:**
```bash
# Check service status
systemctl status hidden-regime

# Check logs
journalctl -u hidden-regime -f

# Check system resources
top
df -h
free -h
```

**Solutions:**

1. **Restart service:**
```bash
sudo systemctl restart hidden-regime
```

2. **Check configuration:**
```bash
# Validate configuration file
python -c "from app.config.production import settings; print('Config OK')"
```

3. **Scale resources:**
```bash
# Increase worker processes
# Edit gunicorn configuration
--workers 8  # Increase from 4
```

### Problem: Database connection errors

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
sqlalchemy.exc.DisconnectionError: Connection invalidated
```

**Solutions:**

1. **Check database status:**
```bash
# PostgreSQL
sudo systemctl status postgresql
pg_isready -h localhost -p 5432

# Redis
redis-cli ping
```

2. **Fix connection parameters:**
```python
# Check connection string
DATABASE_URL = "postgresql://user:pass@localhost:5432/hidden_regime"

# Test connection
import psycopg2
try:
    conn = psycopg2.connect(DATABASE_URL)
    print("Database connection successful")
    conn.close()
except Exception as e:
    print(f"Database connection failed: {e}")
```

3. **Implement connection retry:**
```python
import time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

def create_engine_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            engine = create_engine(url)
            engine.execute("SELECT 1")  # Test connection
            return engine
        except OperationalError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Database connection failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Problem: Memory leaks in production

**Symptoms:**
- Memory usage increases over time
- Application becomes unresponsive
- Out of memory kills

**Solutions:**

1. **Monitor memory usage:**
```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Your application code here

# Get current memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# Get top memory consumers
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

2. **Implement proper cleanup:**
```python
class ModelService:
    def __del__(self):
        # Cleanup resources
        if hasattr(self, 'models'):
            self.models.clear()
```

3. **Configure worker restart:**
```bash
# Restart workers after N requests
gunicorn --max-requests 1000 --max-requests-jitter 100 app:main
```

## Visualization Problems

### Problem: Plots not displaying

**Symptoms:**
```python
fig = loader.plot(data, plot_type='all')
# Nothing appears
```

**Solutions:**

1. **Check backend:**
```python
import matplotlib
print(f"Backend: {matplotlib.get_backend()}")

# For Jupyter notebooks
%matplotlib inline

# For scripts
import matplotlib.pyplot as plt
plt.show()
```

2. **Save plot instead:**
```python
fig = loader.plot(data, plot_type='all', save_path='regime_plot.png')
print("Plot saved to regime_plot.png")
```

3. **Check dependencies:**
```bash
pip install matplotlib seaborn
# For interactive plots
pip install plotly
```

### Problem: Colorblind accessibility warnings

**Symptoms:**
```
Warning: Using red/green colors which may not be colorblind-friendly
```

**Solution:**
```python
# Already fixed in latest version, but ensure you're using updated colors
from hidden_regime.visualization.plotting import REGIME_COLORS
print("Current color scheme:", REGIME_COLORS)

# Should show colorblind-friendly colors:
# Bear: #E69F00 (orange), Bull: #0072B2 (blue), etc.
```

### Problem: Plot performance issues

**Symptoms:**
- Very slow plot generation
- High memory usage during plotting

**Solutions:**

1. **Reduce data points:**
```python
# Sample data for plotting
plot_data = data.sample(n=1000) if len(data) > 1000 else data
fig = loader.plot(plot_data, plot_type='returns')
```

2. **Use lighter plot types:**
```python
# Instead of 'all', use specific types
fig = loader.plot(data, plot_type='returns')  # Faster than 'all'
```

## Common Error Messages

### "Date ordinal converts to invalid date"

**Cause:** Invalid date values in data
**Solution:**
```python
# Check for invalid dates
print(data['date'].dtype)
print(data['date'].min(), data['date'].max())

# Clean invalid dates
data = data[data['date'].notna()]
data = data[data['date'] >= '1900-01-01']
```

### "Transition matrix is not stochastic"

**Cause:** Numerical issues in model training
**Solution:**
```python
# Check and fix transition matrix
import numpy as np

if hasattr(model, 'transition_matrix_'):
    T = model.transition_matrix_
    print("Row sums:", T.sum(axis=1))
    
    # Normalize if needed
    T_normalized = T / T.sum(axis=1, keepdims=True)
    model.transition_matrix_ = T_normalized
```

### "Could not broadcast input array"

**Cause:** Shape mismatch in arrays
**Solution:**
```python
# Check array shapes
print(f"Data shape: {data['log_return'].values.shape}")
print(f"States shape: {states.shape}")

# Ensure 1D arrays
returns = data['log_return'].values.flatten()
states = np.array(states).flatten()
```

### "Model has not been fitted"

**Cause:** Trying to predict before training
**Solution:**
```python
# Always fit before predicting
if not hasattr(model, 'emission_params_'):
    model.fit(data['log_return'].values)

states = model.predict(data['log_return'].values)
```

## Diagnostic Tools

### System Information Script

```python
def system_diagnostics():
    """Comprehensive system diagnostics"""
    import sys
    import platform
    import psutil
    import numpy as np
    import pandas as pd
    
    print("=== SYSTEM DIAGNOSTICS ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    print("\n=== PACKAGE VERSIONS ===")
    packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'yfinance']
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: Not installed")
    
    print("\n=== SYSTEM RESOURCES ===")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"CPU usage: {psutil.cpu_percent()}%")
    
    memory = psutil.virtual_memory()
    print(f"Memory total: {memory.total / 1024**3:.1f} GB")
    print(f"Memory available: {memory.available / 1024**3:.1f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    disk = psutil.disk_usage('/')
    print(f"Disk total: {disk.total / 1024**3:.1f} GB")
    print(f"Disk free: {disk.free / 1024**3:.1f} GB")
    print(f"Disk usage: {(disk.used / disk.total) * 100:.1f}%")

# Run diagnostics
system_diagnostics()
```

### Model Validation Script

```python
def validate_model(model, data):
    """Comprehensive model validation"""
    print("=== MODEL VALIDATION ===")
    
    # Check if model is fitted
    if not hasattr(model, 'emission_params_'):
        print("❌ Model not fitted")
        return False
    
    print("✅ Model is fitted")
    
    # Check parameter validity
    params_valid = True
    
    # Transition matrix
    T = model.transition_matrix_
    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        print("❌ Invalid transition matrix (NaN/Inf)")
        params_valid = False
    elif not np.allclose(T.sum(axis=1), 1.0, rtol=1e-3):
        print("❌ Transition matrix not stochastic")
        params_valid = False
    else:
        print("✅ Transition matrix valid")
    
    # Emission parameters
    emissions = model.emission_params_
    if np.any(np.isnan(emissions)) or np.any(np.isinf(emissions)):
        print("❌ Invalid emission parameters (NaN/Inf)")
        params_valid = False
    elif np.any(emissions[:, 1] <= 0):  # Check variances
        print("❌ Non-positive variances in emission parameters")
        params_valid = False
    else:
        print("✅ Emission parameters valid")
    
    # Test prediction
    try:
        states = model.predict(data['log_return'].values[:100])  # Test with subset
        print("✅ Prediction successful")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        params_valid = False
    
    return params_valid

# Usage
is_valid = validate_model(model, data)
print(f"\nOverall model validity: {'✅ VALID' if is_valid else '❌ INVALID'}")
```

### Performance Profiler

```python
def profile_training(data, config):
    """Profile model training performance"""
    import time
    import tracemalloc
    
    print("=== TRAINING PERFORMANCE PROFILE ===")
    
    # Start profiling
    tracemalloc.start()
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Train model
    model = HiddenMarkovModel(config)
    model.fit(data['log_return'].values, verbose=True)
    
    # End profiling
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Report results
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {end_memory - start_memory:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Data points: {len(data)}")
    print(f"States: {config.n_states}")
    print(f"Iterations: {len(model.log_likelihood_history_) if hasattr(model, 'log_likelihood_history_') else 'Unknown'}")
    
    return model

# Usage
model = profile_training(data, config)
```

## Getting Help

### Before Asking for Help

1. **Run diagnostics:**
```bash
python -c "
import hidden_regime
print(f'Hidden Regime version: {hidden_regime.__version__}')
from docs.troubleshooting_guide import system_diagnostics
system_diagnostics()
"
```

2. **Check logs:**
```bash
# Application logs
tail -n 50 /var/log/hidden-regime.log

# System logs
journalctl -u hidden-regime --since "1 hour ago"
```

3. **Minimal reproducible example:**
```python
# Create minimal example that reproduces the issue
import hidden_regime as hr

try:
    # Minimal code that fails
    data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')
    states = hr.detect_regimes(data['log_return'], n_states=3)
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
    import traceback
    traceback.print_exc()
```

### Where to Get Help

1. **GitHub Issues**: https://github.com/your-org/hidden-regime/issues
2. **Documentation**: Check docs/ directory for detailed guides
3. **Stack Overflow**: Tag questions with `hidden-regime` and `python`
4. **Community Discord/Slack**: [If applicable]

### Information to Include

When reporting issues, include:

1. **Environment information** (from diagnostics script)
2. **Complete error traceback**
3. **Minimal reproducible example**
4. **Expected vs actual behavior**
5. **Steps to reproduce**
6. **Relevant configuration/settings**

### Emergency Production Issues

For critical production issues:

1. **Check health endpoint:** `curl http://your-api/health`
2. **Review monitoring dashboards** (Grafana/Prometheus)
3. **Check resource usage:** `top`, `df -h`, `free -h`
4. **Review recent logs:** `journalctl -f`
5. **Consider rollback** to previous working version

## Conclusion

This troubleshooting guide covers the most common issues you may encounter with Hidden Regime. Most problems fall into these categories:

- **Installation/Environment**: Version conflicts, missing dependencies
- **Data Issues**: Quality problems, API limitations, missing data
- **Model Training**: Convergence issues, parameter problems, performance
- **Production**: Resource issues, configuration problems, scaling

Remember to:
- Check the basics first (environment, data, configuration)
- Use the diagnostic tools provided
- Keep logs and monitoring in place for production systems
- Test fixes in development before applying to production

The package is designed to be robust, but financial data and machine learning models can be complex. Most issues have straightforward solutions once properly diagnosed.