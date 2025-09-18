"""
Unit tests for Pipeline core functionality.

Tests the main Pipeline orchestrator including component coordination,
data flow, state management, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from hidden_regime.pipeline import Pipeline
from hidden_regime.pipeline.interfaces import (
    DataComponent,
    ObservationComponent,
    ModelComponent,
    AnalysisComponent,
    ReportComponent
)
from hidden_regime.utils.exceptions import ValidationError


class MockDataComponent(DataComponent):
    """Mock data component for testing."""
    
    def __init__(self, should_fail=False, return_empty=False):
        self.should_fail = should_fail
        self.return_empty = return_empty
        self.update_count = 0
        self._all_data = None
        
        # Generate consistent dataset
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self._all_data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100), 
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def update(self, current_date=None):
        self.update_count += 1
        
        if self.should_fail:
            raise ValidationError("Mock data component failure")
        
        if self.return_empty:
            return pd.DataFrame()
        
        return self._all_data.copy()
    
    def get_all_data(self):
        """Get complete dataset."""
        if self.return_empty:
            return pd.DataFrame()
        return self._all_data.copy()
    
    def plot(self, **kwargs):
        """Generate visualization for data component."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self._all_data is not None and not self._all_data.empty:
            ax.plot(self._all_data.index, self._all_data['close'], label='Close Price')
            ax.set_title('Mock Data Component - Price Chart')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        return fig


class MockObservationComponent(ObservationComponent):
    """Mock observation component for testing."""
    
    def __init__(self, should_fail=False, return_empty=False):
        self.should_fail = should_fail
        self.return_empty = return_empty
        self.update_count = 0
    
    def update(self, data):
        self.update_count += 1
        
        if self.should_fail:
            raise ValidationError("Mock observation component failure")
            
        if self.return_empty:
            return pd.DataFrame()
        
        # Generate log returns from close prices
        if 'close' in data.columns:
            log_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            return pd.DataFrame({'log_return': log_returns}, index=log_returns.index)
        else:
            return pd.DataFrame({'log_return': np.random.normal(0, 0.02, len(data))}, 
                              index=data.index)
    
    def plot(self, **kwargs):
        """Generate visualization for observation component."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Mock Observation Component\nGenerates log returns', 
                ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


class MockModelComponent(ModelComponent):
    """Mock model component for testing."""
    
    def __init__(self, should_fail=False, return_empty=False):
        self.should_fail = should_fail
        self.return_empty = return_empty
        self.update_count = 0
        self.fit_count = 0
        self.predict_count = 0
        self.is_trained = False
    
    def fit(self, observations):
        self.fit_count += 1
        self.is_trained = True
        
        if self.should_fail:
            raise ValidationError("Mock model fit failure")
    
    def predict(self, observations):
        self.predict_count += 1
        
        if self.should_fail:
            raise ValidationError("Mock model predict failure")
            
        if self.return_empty:
            return pd.DataFrame()
        
        # Generate mock predictions
        n_obs = len(observations)
        np.random.seed(42)
        
        return pd.DataFrame({
            'predicted_state': np.random.choice([0, 1, 2], n_obs),
            'confidence': np.random.uniform(0.6, 0.95, n_obs)
        }, index=observations.index)
    
    def update(self, observations):
        self.update_count += 1
        
        if self.should_fail:
            raise ValidationError("Mock model component failure")
            
        if self.return_empty:
            return pd.DataFrame()
        
        # Fit if not trained, then predict
        if not self.is_trained:
            self.fit(observations)
        
        return self.predict(observations)
    
    def plot(self, **kwargs):
        """Generate visualization for model component."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Mock Model Component\nTrained: {self.is_trained}\nUpdates: {self.update_count}', 
                ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


class MockAnalysisComponent(AnalysisComponent):
    """Mock analysis component for testing."""
    
    def __init__(self, should_fail=False, return_empty=False, raw_data_support=True):
        self.should_fail = should_fail
        self.return_empty = return_empty
        self.raw_data_support = raw_data_support
        self.update_count = 0
        self.received_raw_data = None
    
    def update(self, model_output, raw_data=None):
        self.update_count += 1
        
        if not self.raw_data_support and raw_data is not None:
            # Simulate old analysis component that doesn't accept raw_data
            raise TypeError("update() got an unexpected keyword argument 'raw_data'")
        
        # Only store raw_data if method doesn't fail
        if raw_data is not None:
            self.received_raw_data = raw_data.copy()
        
        if self.should_fail:
            raise ValidationError("Mock analysis component failure")
            
        if self.return_empty:
            return pd.DataFrame()
        
        # Add analysis features to model output
        analysis = model_output.copy()
        analysis['regime_name'] = analysis['predicted_state'].map({
            0: 'Bear', 1: 'Sideways', 2: 'Bull'
        })
        analysis['days_in_regime'] = np.random.randint(1, 20, len(model_output))
        analysis['expected_return'] = analysis['predicted_state'].map({
            0: -0.002, 1: 0.0001, 2: 0.001
        })
        
        return analysis
    
    def plot(self, **kwargs):
        """Generate visualization for analysis component."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Mock Analysis Component\nUpdates: {self.update_count}\nRaw data support: {self.raw_data_support}', 
                ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


class MockReportComponent(ReportComponent):
    """Mock report component for testing."""
    
    def __init__(self, should_fail=False, return_empty=False):
        self.should_fail = should_fail  
        self.return_empty = return_empty
        self.update_count = 0
    
    def update(self, data, observations, model_output, analysis):
        self.update_count += 1
        
        if self.should_fail:
            raise ValidationError("Mock report component failure")
            
        if self.return_empty:
            return ""
        
        # Generate mock markdown report
        current_regime = analysis['regime_name'].iloc[-1] if not analysis.empty else 'Unknown'
        current_confidence = analysis['confidence'].iloc[-1] if not analysis.empty else 0.0
        
        return f"""# Market Regime Analysis Report

## Current Status
- **Regime**: {current_regime}
- **Confidence**: {current_confidence:.1%}
- **Data Points**: {len(data)}
- **Observations**: {len(observations)}

## Analysis Summary
Market analysis completed successfully with {len(analysis)} analysis points.
"""


class TestPipelineCore:
    """Test cases for Pipeline core functionality."""
    
    def test_pipeline_initialization(self):
        """Test basic pipeline initialization."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        assert pipeline.data is data_comp
        assert pipeline.observation is obs_comp
        assert pipeline.model is model_comp
        assert pipeline.analysis is analysis_comp
        assert pipeline.report is None
        
        assert pipeline.last_update is None
        assert pipeline.update_count == 0
        assert isinstance(pipeline.component_outputs, dict)
        assert len(pipeline.component_outputs) == 0
    
    def test_pipeline_initialization_with_report(self):
        """Test pipeline initialization with optional report component."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        report_comp = MockReportComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp,
            report=report_comp
        )
        
        assert pipeline.report is report_comp
    
    def test_successful_pipeline_update_without_report(self):
        """Test complete pipeline update without report component."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        result = pipeline.update()
        
        # Verify components were called
        assert data_comp.update_count == 1
        assert obs_comp.update_count == 1
        assert model_comp.update_count == 1
        assert analysis_comp.update_count == 1
        
        # Verify pipeline state
        assert pipeline.update_count == 1
        assert pipeline.last_update is not None
        
        # Verify component outputs stored
        assert 'data' in pipeline.component_outputs
        assert 'observations' in pipeline.component_outputs
        assert 'model' in pipeline.component_outputs
        assert 'analysis' in pipeline.component_outputs
        
        # Verify data flow
        assert not pipeline.component_outputs['data'].empty
        assert not pipeline.component_outputs['observations'].empty
        assert not pipeline.component_outputs['model'].empty
        assert not pipeline.component_outputs['analysis'].empty
        
        # Verify result is formatted analysis output
        assert isinstance(result, str)
        assert "Analysis Results" in result
        assert "Current Date" in result
        assert "Current Regime" in result
    
    def test_successful_pipeline_update_with_report(self):
        """Test complete pipeline update with report component."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        report_comp = MockReportComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp,
            report=report_comp
        )
        
        result = pipeline.update()
        
        # Verify report component was called
        assert report_comp.update_count == 1
        
        # Verify result is from report component
        assert isinstance(result, str)
        assert "Market Regime Analysis Report" in result
        assert "Current Status" in result
        assert "Analysis Summary" in result
        
        # Verify report output stored
        assert 'report' in pipeline.component_outputs
    
    def test_pipeline_raw_data_passing_with_support(self):
        """Test raw data is passed to analysis component when supported."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent(raw_data_support=True)
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        pipeline.update()
        
        # Verify analysis component received raw data
        assert analysis_comp.received_raw_data is not None
        assert not analysis_comp.received_raw_data.empty
        assert 'close' in analysis_comp.received_raw_data.columns
    
    def test_pipeline_raw_data_fallback_without_support(self):
        """Test fallback when analysis component doesn't support raw_data parameter."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent(raw_data_support=False)
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # Should complete successfully using fallback
        result = pipeline.update()
        
        assert analysis_comp.update_count == 2  # Called twice: once with raw_data (fails), once without
        assert analysis_comp.received_raw_data is None  # Didn't receive raw data on successful call
        assert isinstance(result, str)
        assert "Analysis Results" in result
    
    def test_pipeline_update_with_current_date(self):
        """Test pipeline update with current_date parameter."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        test_date = "2024-03-15"
        result = pipeline.update(current_date=test_date)
        
        assert isinstance(result, str)
        assert pipeline.update_count == 1
    
    def test_pipeline_error_handling_data_failure(self):
        """Test error handling when data component fails."""
        data_comp = MockDataComponent(should_fail=True)
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        with pytest.raises(ValidationError, match="Mock data component failure"):
            pipeline.update()
        
        # Verify subsequent components were not called
        assert data_comp.update_count == 1
        assert obs_comp.update_count == 0
        assert model_comp.update_count == 0
        assert analysis_comp.update_count == 0
    
    def test_pipeline_error_handling_empty_data(self):
        """Test error handling when data component returns empty data."""
        data_comp = MockDataComponent(return_empty=True)
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        with pytest.raises(ValueError, match="No data available from data component"):
            pipeline.update()
    
    def test_pipeline_error_handling_empty_observations(self):
        """Test error handling when observation component returns empty data."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent(return_empty=True)
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        with pytest.raises(ValueError, match="No observations generated from data"):
            pipeline.update()
    
    def test_pipeline_error_handling_empty_model_output(self):
        """Test error handling when model component returns empty output."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent(return_empty=True)
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        with pytest.raises(ValueError, match="No predictions generated from model"):
            pipeline.update()
    
    def test_get_component_output(self):
        """Test getting component outputs."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # Before update
        assert pipeline.get_component_output('data') is None
        assert pipeline.get_component_output('model') is None
        assert pipeline.get_component_output('nonexistent') is None
        
        # After update
        pipeline.update()
        
        data_output = pipeline.get_component_output('data')
        model_output = pipeline.get_component_output('model')
        
        assert data_output is not None
        assert model_output is not None
        assert isinstance(data_output, pd.DataFrame)
        assert isinstance(model_output, pd.DataFrame)
        assert 'close' in data_output.columns
        assert 'predicted_state' in model_output.columns
    
    def test_get_summary_stats(self):
        """Test pipeline summary statistics."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # Before update
        stats = pipeline.get_summary_stats()
        
        assert stats['update_count'] == 0
        assert stats['last_update'] is None
        assert 'components' in stats
        assert stats['components']['data'] == 'MockDataComponent'
        assert stats['components']['model'] == 'MockModelComponent'
        assert stats['components']['report'] is None
        
        # After update
        pipeline.update()
        
        stats = pipeline.get_summary_stats()
        
        assert stats['update_count'] == 1
        assert stats['last_update'] is not None
        assert 'data_shape' in stats
        assert 'model_output_shape' in stats
    
    def test_pipeline_state_persistence(self):
        """Test pipeline state tracking across updates."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # First update
        result1 = pipeline.update()
        first_update_time = pipeline.last_update
        
        assert pipeline.update_count == 1
        assert first_update_time is not None
        
        # Second update
        result2 = pipeline.update()
        second_update_time = pipeline.last_update
        
        assert pipeline.update_count == 2
        assert second_update_time > first_update_time
        
        # Verify components called again
        assert data_comp.update_count == 2
        assert model_comp.update_count == 2
    
    def test_pipeline_string_representation(self):
        """Test pipeline string representation."""
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        report_comp = MockReportComponent()
        
        # Without report
        pipeline_no_report = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        repr_str = repr(pipeline_no_report)
        
        assert "Pipeline(" in repr_str
        assert "Data: MockDataComponent" in repr_str
        assert "Model: MockModelComponent" in repr_str
        assert "Analysis: MockAnalysisComponent" in repr_str
        assert "updates=0" in repr_str
        
        # With report
        pipeline_with_report = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp,
            report=report_comp
        )
        
        repr_str = repr(pipeline_with_report)
        assert "Report: MockReportComponent" in repr_str
    
    def test_pipeline_serialization_support(self):
        """Test pipeline serialization/deserialization support."""
        import pickle
        
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # Run pipeline to populate state
        pipeline.update()
        original_update_count = pipeline.update_count
        
        # Serialize and deserialize
        serialized = pickle.dumps(pipeline)
        restored_pipeline = pickle.loads(serialized)
        
        # Verify state preserved
        assert restored_pipeline.update_count == original_update_count
        assert restored_pipeline.last_update is not None
        assert len(restored_pipeline.component_outputs) > 0
        
        # Verify logger was recreated
        assert hasattr(restored_pipeline, 'logger')
        assert restored_pipeline.logger is not None
    
    def test_multiple_updates_performance(self):
        """Test pipeline performance with multiple updates."""
        import time
        
        data_comp = MockDataComponent()
        obs_comp = MockObservationComponent()
        model_comp = MockModelComponent()
        analysis_comp = MockAnalysisComponent()
        
        pipeline = Pipeline(
            data=data_comp,
            observation=obs_comp,
            model=model_comp,
            analysis=analysis_comp
        )
        
        # Time multiple updates
        start_time = time.time()
        
        for i in range(5):
            pipeline.update()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly (mock components)
        assert total_time < 5.0  # Less than 5 seconds for 5 updates
        assert pipeline.update_count == 5
        
        # Verify all components called correct number of times
        assert data_comp.update_count == 5
        assert obs_comp.update_count == 5
        assert model_comp.update_count == 5
        assert analysis_comp.update_count == 5


if __name__ == "__main__":
    pytest.main([__file__])