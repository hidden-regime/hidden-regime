#!/usr/bin/env python3
"""
Advanced Regime Visualization Example
=====================================

This example demonstrates advanced visualization techniques for regime analysis
including interactive dashboards, 3D plots, regime heatmaps, and comparative
analysis across multiple timeframes and assets.

Key features:
- Interactive plotly dashboards for regime exploration
- 3D regime space visualization
- Multi-asset regime correlation heatmaps
- Regime transition flow diagrams
- Advanced statistical visualizations
- Time-series decomposition with regime overlays

Use cases:
- Portfolio management dashboards
- Research and analysis presentations
- Interactive regime exploration tools
- Multi-asset regime correlation analysis

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from hidden_regime.data import DataLoader
from hidden_regime.analysis import RegimeAnalyzer
from hidden_regime.config import DataConfig

# Advanced plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Installing plotly for interactive visualizations...")
    os.system("pip install plotly")
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
        print("Warning: Plotly not available, some visualizations will be skipped")

class AdvancedRegimeVisualizer:
    """Advanced visualization toolkit for regime analysis"""
    
    def __init__(self, data_config: Optional[DataConfig] = None):
        self.data_config = data_config or DataConfig()
        self.analyzer = RegimeAnalyzer(self.data_config)
        
        # Color schemes for different visualizations
        self.regime_colors = {
            'Bear': '#d62728',     # Red
            'Sideways': '#7f7f7f', # Gray
            'Bull': '#2ca02c'      # Green
        }
        
        self.regime_colors_plotly = {
            'Bear': 'rgb(214, 39, 40)',
            'Sideways': 'rgb(127, 127, 127)', 
            'Bull': 'rgb(44, 160, 44)'
        }
    
    def create_interactive_regime_dashboard(self, symbol: str, start_date: str, 
                                          end_date: str, output_dir: str) -> str:
        """Create an interactive plotly dashboard for regime analysis"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available, skipping interactive dashboard")
            return ""
        
        print(f"Creating interactive dashboard for {symbol}...")
        
        # Get analysis data
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            print(f"Could not analyze {symbol}")
            return ""
        
        # Load full data
        data_loader = DataLoader(self.data_config)
        data = data_loader.load_stock_data(symbol, start_date, end_date)
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Price Action with Regime Detection',
                'Daily Returns by Regime',
                'Regime Probabilities Over Time',
                'Regime Statistics Comparison',
                'Regime Transition Heatmap',
                'Volatility Analysis by Regime'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Data preparation
        states = analysis['states']
        probabilities = analysis['probabilities']
        regime_stats = analysis['regime_stats']
        
        # Map states to regime names
        state_to_regime = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        regime_sequence = [state_to_regime[s] for s in states]
        
        # 1. Price chart with regime backgrounds
        dates = pd.to_datetime(data['date'])
        prices = data['price']
        
        fig.add_trace(
            go.Scatter(
                x=dates.iloc[-len(states):],
                y=prices.iloc[-len(states):],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add regime background colors
        current_regime = regime_sequence[0]
        start_idx = 0
        
        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] != current_regime or i == len(regime_sequence) - 1:
                end_idx = i if regime_sequence[i] != current_regime else i + 1
                
                fig.add_vrect(
                    x0=dates.iloc[-len(states):].iloc[start_idx],
                    x1=dates.iloc[-len(states):].iloc[end_idx-1],
                    fillcolor=self.regime_colors_plotly[current_regime],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
                
                current_regime = regime_sequence[i]
                start_idx = i
        
        # 2. Returns scatter by regime
        returns = data['log_return'].iloc[-len(states):]
        return_dates = dates.iloc[-len(states):]
        
        for regime in ['Bear', 'Sideways', 'Bull']:
            regime_mask = np.array(regime_sequence) == regime
            if np.any(regime_mask):
                fig.add_trace(
                    go.Scatter(
                        x=return_dates[regime_mask],
                        y=returns[regime_mask],
                        mode='markers',
                        name=f'{regime} Returns',
                        marker=dict(
                            color=self.regime_colors_plotly[regime],
                            size=6,
                            opacity=0.7
                        )
                    ),
                    row=1, col=2
                )
        
        # Add zero line for returns
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
        
        # 3. Regime probabilities over time
        prob_dates = dates.iloc[-len(probabilities):]
        
        for i, regime in enumerate(['Bear', 'Sideways', 'Bull']):
            fig.add_trace(
                go.Scatter(
                    x=prob_dates,
                    y=probabilities[:, i],
                    mode='lines',
                    name=f'{regime} Probability',
                    line=dict(color=self.regime_colors_plotly[regime], width=2),
                    fill='tonexty' if i > 0 else 'tozeroy',
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # 4. Regime statistics bar chart
        regime_names = []
        annual_returns = []
        annual_volatilities = []
        
        for state_key, stats in regime_stats.items():
            state_num = int(state_key.split('_')[1])
            regime_name = state_to_regime[state_num]
            regime_names.append(regime_name)
            annual_returns.append(stats['annualized_return'] * 100)
            annual_volatilities.append(stats['annualized_volatility'] * 100)
        
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=annual_returns,
                name='Annual Return (%)',
                marker_color=[self.regime_colors_plotly[r] for r in regime_names],
                opacity=0.8
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=annual_volatilities,
                name='Annual Volatility (%)',
                marker_color=[self.regime_colors_plotly[r] for r in regime_names],
                opacity=0.5,
                yaxis='y2'
            ),
            row=3, col=1
        )
        
        # 5. Regime transition matrix heatmap
        transition_matrix = self._calculate_transition_matrix(states)
        
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=['Bear', 'Sideways', 'Bull'],
                y=['Bear', 'Sideways', 'Bull'],
                colorscale='Blues',
                showscale=True,
                text=np.round(transition_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Advanced Regime Analysis Dashboard: {symbol}",
            height=1200,
            showlegend=True,
            template="plotly_white",
            font=dict(size=10)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Daily Returns", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        
        fig.update_xaxes(title_text="Regime", row=3, col=1)
        fig.update_yaxes(title_text="Annual Return (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="To Regime", row=3, col=2)
        fig.update_yaxes(title_text="From Regime", row=3, col=2)
        
        # Save interactive HTML
        os.makedirs(output_dir, exist_ok=True)
        dashboard_file = os.path.join(output_dir, f'{symbol}_interactive_dashboard.html')
        fig.write_html(dashboard_file)
        
        print(f"Interactive dashboard saved to {dashboard_file}")
        return dashboard_file
    
    def create_3d_regime_space_visualization(self, symbols: List[str], start_date: str,
                                           end_date: str, output_dir: str) -> str:
        """Create 3D visualization of regime space across multiple assets"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available, skipping 3D visualization")
            return ""
        
        print(f"Creating 3D regime space visualization for {len(symbols)} assets...")
        
        # Analyze all symbols
        analyses = {}
        for symbol in symbols:
            analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            if analysis:
                analyses[symbol] = analysis
        
        if len(analyses) < 2:
            print("Need at least 2 successful analyses for 3D visualization")
            return ""
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (symbol, analysis) in enumerate(analyses.items()):
            regime_stats = analysis['regime_stats']
            
            # Extract regime characteristics for 3D plotting
            x_values = []  # Expected returns
            y_values = []  # Volatilities
            z_values = []  # Frequencies
            regime_names = []
            
            state_to_regime = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
            
            for state_key, stats in regime_stats.items():
                state_num = int(state_key.split('_')[1])
                regime_name = state_to_regime[state_num]
                
                x_values.append(stats['annualized_return'])
                y_values.append(stats['annualized_volatility'])
                z_values.append(stats['frequency'])
                regime_names.append(f"{symbol}_{regime_name}")
            
            fig.add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                text=[f"{symbol}<br>{r.split('_')[1]}" for r in regime_names],
                textposition="middle center",
                name=symbol,
                hovertemplate="<b>%{text}</b><br>" +
                            "Return: %{x:.1%}<br>" +
                            "Volatility: %{y:.1%}<br>" +
                            "Frequency: %{z:.1%}<extra></extra>"
            ))
        
        # Update layout for 3D
        fig.update_layout(
            title="3D Regime Space Analysis",
            scene=dict(
                xaxis_title="Annualized Return",
                yaxis_title="Annualized Volatility", 
                zaxis_title="Regime Frequency",
                xaxis=dict(tickformat=".1%"),
                yaxis=dict(tickformat=".1%"),
                zaxis=dict(tickformat=".1%")
            ),
            height=800,
            template="plotly_white"
        )
        
        # Save 3D plot
        os.makedirs(output_dir, exist_ok=True)
        plot_3d_file = os.path.join(output_dir, '3d_regime_space_analysis.html')
        fig.write_html(plot_3d_file)
        
        print(f"3D regime space visualization saved to {plot_3d_file}")
        return plot_3d_file
    
    def create_regime_correlation_heatmap(self, symbols: List[str], start_date: str,
                                        end_date: str, output_dir: str) -> str:
        """Create advanced correlation heatmap across multiple assets' regimes"""
        
        print(f"Creating regime correlation analysis for {len(symbols)} assets...")
        
        # Analyze all symbols
        analyses = {}
        for symbol in symbols:
            analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
            if analysis:
                analyses[symbol] = analysis
        
        if len(analyses) < 2:
            print("Need at least 2 successful analyses for correlation analysis")
            return ""
        
        # Create regime correlation matrix
        regime_data = {}
        min_length = min(len(analysis['states']) for analysis in analyses.values())
        
        # Align all regime sequences to same length
        for symbol, analysis in analyses.items():
            states = analysis['states'][-min_length:]  # Take last min_length observations
            regime_data[symbol] = states
        
        # Convert to DataFrame
        regime_df = pd.DataFrame(regime_data)
        
        # Calculate correlations
        correlation_matrix = regime_df.corr()
        
        # Create advanced heatmap visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Asset Regime Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Basic correlation heatmap
        ax1 = axes[0, 0]
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Regime State Correlations', fontweight='bold')
        
        # 2. Regime synchronization analysis
        ax2 = axes[0, 1]
        sync_data = []
        symbol_pairs = []
        
        for i, sym1 in enumerate(regime_df.columns):
            for j, sym2 in enumerate(regime_df.columns):
                if i < j:
                    # Calculate synchronization (same regime at same time)
                    sync_pct = (regime_df[sym1] == regime_df[sym2]).mean()
                    sync_data.append(sync_pct)
                    symbol_pairs.append(f"{sym1}-{sym2}")
        
        if sync_data:
            bars = ax2.bar(range(len(symbol_pairs)), sync_data, alpha=0.7)
            ax2.set_title('Regime Synchronization', fontweight='bold')
            ax2.set_ylabel('Synchronization Rate')
            ax2.set_xticks(range(len(symbol_pairs)))
            ax2.set_xticklabels(symbol_pairs, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Color bars based on synchronization level
            for bar, sync in zip(bars, sync_data):
                if sync > 0.6:
                    bar.set_color('green')
                elif sync > 0.4:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # 3. Regime transition timing analysis
        ax3 = axes[1, 0]
        transition_data = {}
        
        for symbol in regime_df.columns:
            transitions = []
            states = regime_df[symbol].values
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    transitions.append(i)
            transition_data[symbol] = transitions
        
        # Plot transition timing
        for i, (symbol, transitions) in enumerate(transition_data.items()):
            if transitions:
                ax3.scatter([symbol] * len(transitions), transitions, 
                          alpha=0.6, s=50, label=symbol)
        
        ax3.set_title('Regime Transition Timing', fontweight='bold')
        ax3.set_ylabel('Time Period')
        ax3.set_xlabel('Asset')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. Regime distribution comparison
        ax4 = axes[1, 1]
        regime_distributions = {}
        
        for symbol in regime_df.columns:
            states = regime_df[symbol]
            dist = states.value_counts(normalize=True).sort_index()
            regime_distributions[symbol] = dist
        
        # Create stacked bar chart
        regime_names = ['Bear', 'Sideways', 'Bull']
        bottom = np.zeros(len(regime_distributions))
        
        for regime_idx, regime_name in enumerate(regime_names):
            values = [regime_distributions[sym].get(regime_idx, 0) 
                     for sym in regime_distributions.keys()]
            ax4.bar(list(regime_distributions.keys()), values, 
                   bottom=bottom, label=regime_name, 
                   color=self.regime_colors[regime_name], alpha=0.8)
            bottom += values
        
        ax4.set_title('Regime Distribution by Asset', fontweight='bold')
        ax4.set_ylabel('Proportion')
        ax4.legend()
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save correlation analysis
        os.makedirs(output_dir, exist_ok=True)
        correlation_file = os.path.join(output_dir, 'regime_correlation_analysis.png')
        plt.savefig(correlation_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save correlation matrix as CSV
        csv_file = os.path.join(output_dir, 'regime_correlations.csv')
        correlation_matrix.to_csv(csv_file)
        
        print(f"Regime correlation analysis saved to {correlation_file}")
        print(f"Correlation matrix saved to {csv_file}")
        
        return correlation_file
    
    def create_regime_flow_diagram(self, symbol: str, start_date: str, end_date: str,
                                 output_dir: str) -> str:
        """Create regime transition flow diagram using Sankey plot"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available, skipping flow diagram")
            return ""
        
        print(f"Creating regime flow diagram for {symbol}...")
        
        analysis = self.analyzer.analyze_stock(symbol, start_date, end_date)
        if not analysis:
            print(f"Could not analyze {symbol}")
            return ""
        
        states = analysis['states']
        
        # Calculate transition flows
        transitions = {}
        state_names = ['Bear', 'Sideways', 'Bull']
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            
            key = (from_state, to_state)
            transitions[key] = transitions.get(key, 0) + 1
        
        # Prepare Sankey diagram data
        source = []
        target = []
        value = []
        
        for (from_state, to_state), count in transitions.items():
            source.append(from_state)
            target.append(to_state + 3)  # Offset target nodes
            value.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=state_names + [f"{name} (Next)" for name in state_names],
                color=["#d62728", "#7f7f7f", "#2ca02c", "#d62728", "#7f7f7f", "#2ca02c"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=["rgba(214, 39, 40, 0.4)", "rgba(127, 127, 127, 0.4)", 
                       "rgba(44, 160, 44, 0.4)"] * len(source)
            )
        )])
        
        fig.update_layout(
            title=f"Regime Transition Flow: {symbol}",
            font_size=12,
            height=600,
            template="plotly_white"
        )
        
        # Save flow diagram
        os.makedirs(output_dir, exist_ok=True)
        flow_file = os.path.join(output_dir, f'{symbol}_regime_flow.html')
        fig.write_html(flow_file)
        
        print(f"Regime flow diagram saved to {flow_file}")
        return flow_file
    
    def _calculate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Calculate transition probability matrix from state sequence"""
        n_states = len(np.unique(states))
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def generate_comprehensive_visualization_report(self, symbols: List[str], 
                                                  start_date: str, end_date: str,
                                                  output_dir: str) -> str:
        """Generate comprehensive visualization report with all advanced techniques"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("GENERATING COMPREHENSIVE ADVANCED VISUALIZATION REPORT")
        print("=" * 80)
        
        # Create all visualizations
        reports = []
        
        # 1. Interactive dashboards for each symbol
        print("\n1. Creating interactive dashboards...")
        for symbol in symbols[:3]:  # Limit to first 3 for performance
            dashboard = self.create_interactive_regime_dashboard(
                symbol, start_date, end_date, output_dir
            )
            if dashboard:
                reports.append(f"- Interactive dashboard: {os.path.basename(dashboard)}")
        
        # 2. 3D regime space visualization
        print("\n2. Creating 3D regime space visualization...")
        plot_3d = self.create_3d_regime_space_visualization(
            symbols, start_date, end_date, output_dir
        )
        if plot_3d:
            reports.append(f"- 3D regime space: {os.path.basename(plot_3d)}")
        
        # 3. Correlation heatmap analysis
        print("\n3. Creating regime correlation analysis...")
        correlation_plot = self.create_regime_correlation_heatmap(
            symbols, start_date, end_date, output_dir
        )
        if correlation_plot:
            reports.append(f"- Correlation analysis: {os.path.basename(correlation_plot)}")
        
        # 4. Flow diagrams
        print("\n4. Creating regime flow diagrams...")
        for symbol in symbols[:2]:  # Limit to first 2 for performance
            flow_diagram = self.create_regime_flow_diagram(
                symbol, start_date, end_date, output_dir
            )
            if flow_diagram:
                reports.append(f"- Flow diagram ({symbol}): {os.path.basename(flow_diagram)}")
        
        # Generate summary report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_content = f"""# Advanced Regime Visualization Report
*Generated on {timestamp}*

## Overview

This report showcases advanced visualization techniques for Hidden Markov Model regime analysis across multiple assets and timeframes. The visualizations provide deep insights into regime behavior, correlations, and transitions.

## Generated Visualizations

{chr(10).join(reports)}

## Visualization Descriptions

### Interactive Dashboards
Interactive HTML dashboards built with Plotly that allow:
- Real-time exploration of regime changes over time
- Interactive price charts with regime backgrounds
- Dynamic probability visualizations
- Regime statistics comparisons

### 3D Regime Space Analysis  
Three-dimensional visualization showing:
- Return vs Volatility vs Frequency relationships
- Asset clustering in regime space
- Comparative regime characteristics

### Regime Correlation Analysis
Advanced correlation analysis including:
- Cross-asset regime correlation heatmaps
- Regime synchronization analysis
- Transition timing comparisons
- Regime distribution breakdowns

### Regime Flow Diagrams
Sankey diagrams showing:
- Regime transition patterns
- Flow volumes between states
- Transition probability visualization

## Technical Implementation

- **Interactive Elements**: Built with Plotly for web-based exploration
- **Static Analysis**: High-resolution matplotlib/seaborn visualizations  
- **3D Analysis**: Three-dimensional scatter plots with regime clustering
- **Flow Analysis**: Sankey diagrams for transition visualization

## Usage Guidelines

1. **Interactive Dashboards**: Open HTML files in web browser for exploration
2. **Static Images**: Use PNG files for presentations and reports
3. **Data Files**: CSV correlation matrices available for further analysis
4. **Integration**: All visualizations can be embedded in larger analysis workflows

## Next Steps

Consider extending these visualizations with:
- Real-time streaming data updates
- Machine learning regime prediction overlays
- Risk management integration
- Portfolio optimization visualizations

---
*Generated using Hidden Regime Advanced Visualization Framework*
"""
        
        # Save comprehensive report
        report_file = os.path.join(output_dir, 'advanced_visualization_report.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\nComprehensive visualization report saved to: {report_file}")
        return report_file

def main():
    """Main execution function for advanced regime visualization examples"""
    
    print("🎨 Hidden Regime Advanced Visualization Example")
    print("=" * 60)
    
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    OUTPUT_DIR = './output/advanced_visualizations'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📊 Creating advanced visualizations for {len(SYMBOLS)} assets")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        # Initialize visualizer
        data_config = DataConfig()
        visualizer = AdvancedRegimeVisualizer(data_config)
        
        # Generate comprehensive visualization report
        print("\n🎨 Generating comprehensive visualization suite...")
        report_file = visualizer.generate_comprehensive_visualization_report(
            SYMBOLS, start_date, end_date, OUTPUT_DIR
        )
        
        print(f"\n✅ Advanced visualization generation completed!")
        print(f"📄 Report file: {report_file}")
        print(f"📁 All visualizations saved to: {OUTPUT_DIR}")
        
        print(f"\n🎯 Key Deliverables:")
        print(f"   📊 Interactive dashboards for regime exploration")
        print(f"   🔍 3D regime space analysis")
        print(f"   📈 Multi-asset correlation heatmaps")
        print(f"   🌊 Regime transition flow diagrams")
        print(f"   📋 Comprehensive analysis report")
        
        # List generated files
        print(f"\n📂 Generated Files:")
        for file in sorted(os.listdir(OUTPUT_DIR)):
            print(f"   - {file}")
        
    except Exception as e:
        print(f"❌ Error creating advanced visualizations: {str(e)}")
        print("💥 Example failed - check error messages above")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Advanced visualization example completed successfully!")
        print("🔗 Open the HTML files in your browser for interactive exploration")
    else:
        print("\n❌ Advanced visualization example failed")
        exit(1)