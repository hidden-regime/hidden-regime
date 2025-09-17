"""
Core Pipeline implementation for hidden-regime.

Provides the main Pipeline class that orchestrates the Data → Observation → Model → Analysis → Report flow.
The Pipeline serves as the primary user interface and coordinates all components through their standardized interfaces.
"""

from typing import Optional, Dict, Any, Type
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from .interfaces import (
    DataComponent, 
    ObservationComponent, 
    ModelComponent, 
    AnalysisComponent, 
    ReportComponent
)


class Pipeline:
    """
    Main Pipeline orchestrator for hidden-regime analysis.
    
    Coordinates Data → Observation → Model → Analysis → Report flow and serves
    as the primary user interface. All components follow standardized interfaces
    ensuring consistent behavior and extensibility.
    
    Example usage:
        pipeline = Pipeline(
            data=data_component,
            observation=observation_component, 
            model=model_component,
            analysis=analysis_component,
            report=report_component
        )
        
        result = pipeline.update()
    """
    
    def __init__(
        self,
        data: DataComponent,
        observation: ObservationComponent,
        model: ModelComponent,
        analysis: AnalysisComponent,
        report: Optional[ReportComponent] = None
    ):
        """
        Initialize pipeline with all required components.
        
        Args:
            data: Data loading and management component
            observation: Observation generation component  
            model: Model training and prediction component
            analysis: Analysis and interpretation component
            report: Optional report generation component
        """
        self.data = data
        self.observation = observation
        self.model = model
        self.analysis = analysis
        self.report = report
        
        # Pipeline state tracking
        self.last_update = None
        self.update_count = 0
        self.component_outputs = {}
        
        # Configure logging
        self.logger = logging.getLogger(f"Pipeline-{id(self)}")
        
    def update(self, current_date: Optional[str] = None) -> str:
        """
        Execute complete pipeline flow: Data → Observation → Model → Analysis → Report.
        
        Args:
            current_date: Optional date for data updates (YYYY-MM-DD format)
            
        Returns:
            Report output (typically markdown string) or analysis output if no report component
        """
        try:
            self.logger.info(f"Starting pipeline update {self.update_count + 1}")
            update_start = datetime.now()
            
            # Step 1: Update data
            self.logger.debug("Step 1: Updating data component")
            data_output = self.data.update(current_date=current_date)
            self.component_outputs['data'] = data_output
            
            if len(data_output) == 0:
                raise ValueError("No data available from data component")
            
            # Step 2: Generate observations
            self.logger.debug("Step 2: Generating observations")
            observation_output = self.observation.update(data_output)
            self.component_outputs['observations'] = observation_output
            
            if len(observation_output) == 0:
                raise ValueError("No observations generated from data")
            
            # Step 3: Model processing (fit if first time, then predict)
            self.logger.debug("Step 3: Processing model")
            model_output = self.model.update(observation_output)
            self.component_outputs['model'] = model_output
            
            if len(model_output) == 0:
                raise ValueError("No predictions generated from model")
            
            # Step 4: Analysis and interpretation
            self.logger.debug("Step 4: Running analysis")
            # Pass raw data to analysis for indicator calculations if analysis component supports it
            try:
                # Try to pass raw data as a parameter
                analysis_output = self.analysis.update(model_output, raw_data=data_output)
            except TypeError:
                # Fallback for analysis components that don't accept raw_data parameter
                analysis_output = self.analysis.update(model_output)
            self.component_outputs['analysis'] = analysis_output
            
            # Step 5: Generate report (if report component provided)
            if self.report is not None:
                self.logger.debug("Step 5: Generating report")
                report_output = self.report.update(
                    data=data_output,
                    observations=observation_output,
                    model_output=model_output,
                    analysis=analysis_output
                )
                self.component_outputs['report'] = report_output
                result = report_output
            else:
                # Return analysis output if no report component
                result = self._format_analysis_output(analysis_output)
            
            # Update pipeline state
            self.last_update = update_start
            self.update_count += 1
            
            update_duration = (datetime.now() - update_start).total_seconds()
            self.logger.info(f"Pipeline update completed in {update_duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline update failed: {str(e)}")
            raise
    
    def _format_analysis_output(self, analysis_output: pd.DataFrame) -> str:
        """Format analysis output as simple markdown when no report component."""
        if analysis_output.empty:
            return "# Analysis Results\n\nNo analysis results available."
        
        # Get current state info (last row)
        current = analysis_output.iloc[-1]
        
        report_lines = [
            "# Analysis Results",
            "",
            f"**Current Date**: {current.name}",
        ]
        
        # Add regime info if available
        if 'predicted_state' in current:
            report_lines.extend([
                f"**Current Regime**: {current.get('predicted_state', 'Unknown')}",
                f"**Confidence**: {current.get('confidence', 0):.1%}" if 'confidence' in current else "",
            ])
        
        # Add return info if available  
        if 'expected_return' in current:
            report_lines.append(f"**Expected Return**: {current.get('expected_return', 0):.3f}")
        
        # Add days in regime if available
        if 'days_in_state' in current:
            report_lines.append(f"**Days in Current Regime**: {current.get('days_in_state', 0)}")
        
        return "\n".join(filter(None, report_lines))
    
    def get_component_output(self, component_name: str) -> Optional[Any]:
        """
        Get the last output from a specific component.
        
        Args:
            component_name: Name of component ('data', 'observations', 'model', 'analysis', 'report')
            
        Returns:
            Last output from the specified component, or None if not available
        """
        return self.component_outputs.get(component_name)
    
    def plot(self, components: Optional[list] = None, **kwargs) -> plt.Figure:
        """
        Generate visualization showing outputs from all or specified components.
        
        Args:
            components: List of component names to plot. If None, plot all components.
            **kwargs: Additional plotting arguments passed to component plot methods
            
        Returns:
            matplotlib Figure with subplots for each component
        """
        if components is None:
            components = ['data', 'observations', 'model', 'analysis']
        
        n_components = len(components)
        if n_components == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No components to plot', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Create subplots
        fig = plt.figure(figsize=(14, 4 * n_components))
        
        plot_idx = 1
        for component_name in components:
            if component_name == 'data' and hasattr(self.data, 'plot'):
                ax = plt.subplot(n_components, 1, plot_idx)
                self.data.plot(**kwargs)
                plt.title(f'{component_name.title()} Component')
                plot_idx += 1
                
            elif component_name == 'observations' and hasattr(self.observation, 'plot'):
                ax = plt.subplot(n_components, 1, plot_idx)
                self.observation.plot(**kwargs)
                plt.title(f'{component_name.title()} Component')
                plot_idx += 1
                
            elif component_name == 'model' and hasattr(self.model, 'plot'):
                ax = plt.subplot(n_components, 1, plot_idx)
                self.model.plot(**kwargs)
                plt.title(f'{component_name.title()} Component')
                plot_idx += 1
                
            elif component_name == 'analysis' and hasattr(self.analysis, 'plot'):
                ax = plt.subplot(n_components, 1, plot_idx)
                self.analysis.plot(**kwargs)
                plt.title(f'{component_name.title()} Component')
                plot_idx += 1
        
        plt.tight_layout()
        return fig
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about pipeline state and performance.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            'update_count': self.update_count,
            'last_update': self.last_update,
            'components': {
                'data': type(self.data).__name__,
                'observation': type(self.observation).__name__,
                'model': type(self.model).__name__,
                'analysis': type(self.analysis).__name__,
                'report': type(self.report).__name__ if self.report else None,
            }
        }
        
        # Add data shape if available
        if 'data' in self.component_outputs:
            data_output = self.component_outputs['data']
            if hasattr(data_output, 'shape'):
                stats['data_shape'] = data_output.shape
        
        # Add model info if available
        if 'model' in self.component_outputs:
            model_output = self.component_outputs['model']
            if hasattr(model_output, 'shape'):
                stats['model_output_shape'] = model_output.shape
        
        return stats
    
    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return self.__dict__.copy()
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.__dict__.update(state)
        # Recreate logger after unpickling
        self.logger = logging.getLogger(f"Pipeline-{id(self)}")
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        components = []
        components.append(f"Data: {type(self.data).__name__}")
        components.append(f"Observation: {type(self.observation).__name__}")
        components.append(f"Model: {type(self.model).__name__}")
        components.append(f"Analysis: {type(self.analysis).__name__}")
        if self.report:
            components.append(f"Report: {type(self.report).__name__}")
        
        return f"Pipeline({', '.join(components)}, updates={self.update_count})"