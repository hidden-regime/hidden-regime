# Hidden Regime Examples

This directory contains comprehensive examples demonstrating the visualization capabilities of the Hidden Regime package.

## 🎨 Visualization Examples

### 1. Comprehensive Visualization Demo
**File**: `comprehensive_visualization_demo.py`

A complete showcase of all visualization capabilities across the three main components:

- **DataLoader Visualizations**: Market data analysis (price, returns, distributions, volume)
- **HiddenMarkovModel Visualizations**: Regime detection analysis (states, transitions, probabilities)
- **StateStandardizer Visualizations**: Regime characteristics and validation analysis
- **Multi-Configuration Comparison**: Different regime setups (3-state, 4-state, 5-state)
- **Style Consistency Demo**: Unified color schemes and formatting

**Features Demonstrated**:
- ✅ Professional financial styling
- ✅ Consistent regime color coding
- ✅ Date-aware time series plotting
- ✅ Comprehensive error handling
- ✅ High-quality output suitable for publications

### 2. Quick Visualization Demo  
**File**: `quick_visualization_demo.py`

A streamlined demonstration perfect for quick testing and getting started:

- Essential plotting capabilities from all three components
- Individual plot type examples
- Synthetic data fallback for testing
- Clear step-by-step progression

### 3. Enhanced Basic HMM Demo
**File**: `basic_hmm_demo.py` *(Updated)*

The original HMM demo enhanced with integrated visualization capabilities:

- Complete HMM workflow with plotting integration
- Demonstrates how to add visualizations to existing analysis workflows
- Shows individual plot types and comprehensive analysis views
- Performance comparison and accuracy visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install matplotlib seaborn
```

### Running the Examples

**For comprehensive demonstration**:
```bash
python examples/comprehensive_visualization_demo.py
```

**For quick testing**:
```bash
python examples/quick_visualization_demo.py
```

**For workflow integration example**:
```bash
python examples/basic_hmm_demo.py
```

## 📊 Available Visualization Types

### DataLoader Plots
- **'all'**: Comprehensive market data dashboard
- **'price'**: Price evolution with moving averages
- **'returns'**: Daily log returns with rolling volatility
- **'distribution'**: Returns distribution analysis with statistical overlays
- **'volume'**: Trading volume analysis (when available)

### HiddenMarkovModel Plots  
- **'all'**: Complete regime analysis dashboard
- **'regimes'**: Regime classification timeline
- **'probabilities'**: State probabilities heatmap over time
- **'transitions'**: Regime transition matrix heatmap
- **'statistics'**: Regime statistics comparison
- **'convergence'**: Training convergence analysis
- **'duration'**: Regime duration analysis

### StateStandardizer Plots
- **'all'**: Comprehensive regime characteristics dashboard
- **'characteristics'**: Regime characteristics matrix heatmap
- **'validation'**: Regime validation confidence matrix
- **'comparison'**: Risk-return profile scatter plot
- **'economic'**: Economic validation dashboard

## 🎨 Key Visualization Features

### Professional Styling
- Consistent financial chart styling
- Clean, publication-ready aesthetics
- Proper grid lines and axis formatting
- Professional color schemes

### Colorblind-Friendly Regime Color Coding
- **Bear Markets**: Dark Orange (`#E69F00`) - colorblind-safe alternative to red
- **Crisis Periods**: Pink/Magenta (`#CC79A7`) - distinctive and alarming
- **Sideways Markets**: Yellow (`#F0E442`) - neutral and accessible
- **Bull Markets**: Blue (`#0072B2`) - positive and universally accessible
- **Euphoric Periods**: Purple (`#9467BD`) - extreme and distinctive

**Accessibility Features:**
- ✅ **Colorblind-Safe**: Based on Okabe-Ito research palette
- ✅ **Pattern Support**: Different line styles (solid, dashed, dotted)
- ✅ **Shape Differentiation**: Unique markers (triangles, squares, stars)
- ✅ **High Contrast**: Dark edges on markers for better visibility

### Date-Aware Plotting
- Intelligent date formatting based on data range
- Proper handling of matplotlib date limitations
- Graceful fallback to integer indices when needed

### Comprehensive Error Handling
- Automatic detection of missing visualization dependencies
- Informative error messages and installation instructions
- Graceful degradation when data issues occur

## 📈 Usage Patterns

### Basic Usage
```python
from hidden_regime.data.loader import DataLoader
from hidden_regime.models.base_hmm import HiddenMarkovModel
from hidden_regime.models.state_standardizer import StateStandardizer

# Load data and create plots
loader = DataLoader()
data = loader.load_stock_data("AAPL", "2023-01-01", "2024-01-01")
fig = loader.plot(data, plot_type='all')

# Train HMM and visualize regime analysis
hmm = HiddenMarkovModel()
hmm.fit(data['log_return'].values)
fig = hmm.plot(data['log_return'].values, dates=data['date'].values, plot_type='all')

# Analyze regime characteristics
standardizer = StateStandardizer(regime_type='3_state')
fig = standardizer.plot(hmm.emission_params_, plot_type='all')
```

### Saving Plots
```python
# Save individual plots
fig = hmm.plot(returns, dates=dates, plot_type='regimes', save_path='regime_analysis.png')

# Save with custom settings  
fig = loader.plot(data, plot_type='all', figsize=(16, 12))
plt.savefig('market_analysis.png', dpi=300, bbox_inches='tight')
```

### Customization
```python
# Adjust figure size
fig = hmm.plot(returns, plot_type='all', figsize=(20, 14))

# Focus on specific analysis
fig = standardizer.plot(emission_params, plot_type='characteristics', figsize=(10, 8))

# Individual component analysis
fig = hmm.plot(returns, dates=dates, plot_type='transitions', figsize=(8, 6))
```

## 🏆 Best Practices

1. **Use 'all' plot types** for comprehensive analysis and presentations
2. **Individual plot types** for focused analysis and reports
3. **Consistent figsize** across related plots for professional presentations
4. **Save plots** with high DPI (150-300) for publication quality
5. **Combine components** (DataLoader → HMM → StateStandardizer) for complete analysis

## ♿ Accessibility Features

### Colorblind-Friendly Design
The visualization framework is designed to be accessible to users with color vision deficiencies:

- **Scientific Color Palette**: Based on Okabe-Ito colorblind-safe research
- **Multiple Visual Cues**: Colors + shapes + patterns for regime differentiation  
- **High Contrast**: Dark edges and borders for better visibility
- **Meaningful Colors**: Blue for positive (bull), yellow for neutral (sideways)

### Accessibility Options
```python
# Enable accessibility features (default: True)
fig = hmm.plot(returns, dates=dates, plot_type='regimes', 
               use_accessibility_features=True)

# Get accessibility styling helpers
colors = get_regime_colors(['Bear', 'Bull', 'Sideways'])
markers = get_regime_markers(['Bear', 'Bull', 'Sideways'])  
line_styles = get_regime_line_styles(['Bear', 'Bull', 'Sideways'])
```

### Testing Your Visualizations
- View plots in grayscale to test accessibility
- Use colorblind simulation tools to verify distinction
- Ensure legends include both color and text labels

## 🔧 Troubleshooting

### Common Issues
- **Missing matplotlib/seaborn**: Install with `pip install matplotlib seaborn`
- **Date conversion errors**: The package automatically handles invalid dates
- **Memory issues with large datasets**: Use individual plot types instead of 'all'
- **Display issues**: Ensure you have a display backend configured for matplotlib

### Performance Tips
- Use synthetic data for testing and development
- Individual plot types are faster than comprehensive dashboards
- Consider data size when using 'all' plot types

## 🌟 Next Steps

After running the examples:
1. Try with your own financial data
2. Experiment with different regime configurations
3. Integrate plotting into your analysis workflows
4. Customize styling and colors for your specific needs
5. Explore saving plots for reports and presentations

The visualization framework is designed to be both powerful and easy to use, providing professional-quality financial charts with minimal code.