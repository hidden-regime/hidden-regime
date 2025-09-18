"""
Unit tests for Pipeline interfaces.

Tests the abstract interfaces that define the contracts for pipeline components
including DataComponent, ObservationComponent, ModelComponent, etc.
"""

import pytest
import pandas as pd
import numpy as np
from abc import ABC
from typing import Dict, Any

from hidden_regime.pipeline.interfaces import (
    DataComponent,
    ObservationComponent,
    ModelComponent,
    AnalysisComponent,
    ReportComponent
)
from hidden_regime.utils.exceptions import ValidationError


class TestDataComponent:
    """Test DataComponent interface."""
    
    def test_data_component_is_abstract(self):
        """Test that DataComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataComponent()
    
    def test_data_component_interface_definition(self):
        """Test DataComponent interface definition."""
        assert issubclass(DataComponent, ABC)
        assert hasattr(DataComponent, 'update')
        
        # Check method signature through abstract method
        import inspect
        sig = inspect.signature(DataComponent.update)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'current_date' in params
    
    def test_concrete_data_component_implementation(self):
        """Test concrete implementation of DataComponent."""
        
        class ConcreteDataComponent(DataComponent):
            def update(self, current_date=None):
                # Return sample data
                dates = pd.date_range('2024-01-01', periods=10, freq='D')
                return pd.DataFrame({
                    'open': np.random.uniform(95, 105, 10),
                    'high': np.random.uniform(100, 110, 10),
                    'low': np.random.uniform(90, 100, 10),
                    'close': np.random.uniform(95, 105, 10),
                    'volume': np.random.randint(1000000, 5000000, 10)
                }, index=dates)
            
            def get_all_data(self):
                return self.update()
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = ConcreteDataComponent()
        result = component.update()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 10
        assert 'close' in result.columns
    
    def test_data_component_current_date_parameter(self):
        """Test DataComponent with current_date parameter."""
        
        class DateAwareDataComponent(DataComponent):
            def update(self, current_date=None):
                self.last_current_date = current_date
                return pd.DataFrame({'close': [100.0]}, index=[pd.Timestamp('2024-01-01')])
            
            def get_all_data(self):
                return self.update()
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = DateAwareDataComponent()
        
        # Test without date
        component.update()
        assert component.last_current_date is None
        
        # Test with date
        test_date = "2024-03-15"
        component.update(current_date=test_date)
        assert component.last_current_date == test_date


class TestObservationComponent:
    """Test ObservationComponent interface."""
    
    def test_observation_component_is_abstract(self):
        """Test that ObservationComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ObservationComponent()
    
    def test_observation_component_interface_definition(self):
        """Test ObservationComponent interface definition."""
        assert issubclass(ObservationComponent, ABC)
        assert hasattr(ObservationComponent, 'update')
        
        import inspect
        sig = inspect.signature(ObservationComponent.update)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'data' in params
    
    def test_concrete_observation_component_implementation(self):
        """Test concrete implementation of ObservationComponent."""
        
        class ConcreteObservationComponent(ObservationComponent):
            def update(self, data):
                if 'close' in data.columns:
                    log_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
                    return pd.DataFrame({'log_return': log_returns}, index=log_returns.index)
                else:
                    return pd.DataFrame()
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = ConcreteObservationComponent()
        
        # Test with valid data
        input_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        result = component.update(input_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'log_return' in result.columns
        assert len(result) == 4  # One less due to shift
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'volume': [1000, 2000]})
        result = component.update(invalid_data)
        assert result.empty


class TestModelComponent:
    """Test ModelComponent interface."""
    
    def test_model_component_is_abstract(self):
        """Test that ModelComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelComponent()
    
    def test_model_component_interface_definition(self):
        """Test ModelComponent interface definition."""
        assert issubclass(ModelComponent, ABC)
        assert hasattr(ModelComponent, 'update')
        
        import inspect
        sig = inspect.signature(ModelComponent.update)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'observations' in params
    
    def test_concrete_model_component_implementation(self):
        """Test concrete implementation of ModelComponent."""
        
        class ConcreteModelComponent(ModelComponent):
            def __init__(self):
                self.is_trained = False
            
            def fit(self, observations):
                self.is_trained = True
            
            def predict(self, observations):
                n_obs = len(observations)
                np.random.seed(42)
                return pd.DataFrame({
                    'predicted_state': np.random.choice([0, 1, 2], n_obs),
                    'confidence': np.random.uniform(0.6, 0.95, n_obs)
                }, index=observations.index)
            
            def update(self, observations):
                if not self.is_trained:
                    self.fit(observations)
                return self.predict(observations)
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = ConcreteModelComponent()
        
        # Test training and prediction
        observations = pd.DataFrame({
            'log_return': np.random.normal(0, 0.02, 20)
        }, index=pd.date_range('2024-01-01', periods=20, freq='D'))
        
        assert not component.is_trained
        
        result = component.update(observations)
        
        assert component.is_trained
        assert isinstance(result, pd.DataFrame)
        assert 'predicted_state' in result.columns
        assert 'confidence' in result.columns
        assert len(result) == len(observations)
        
        # Verify states are valid
        assert result['predicted_state'].isin([0, 1, 2]).all()
        assert (result['confidence'] >= 0).all()
        assert (result['confidence'] <= 1).all()


class TestAnalysisComponent:
    """Test AnalysisComponent interface."""
    
    def test_analysis_component_is_abstract(self):
        """Test that AnalysisComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AnalysisComponent()
    
    def test_analysis_component_interface_definition(self):
        """Test AnalysisComponent interface definition."""
        assert issubclass(AnalysisComponent, ABC)
        assert hasattr(AnalysisComponent, 'update')
        
        import inspect
        sig = inspect.signature(AnalysisComponent.update)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'model_output' in params
        
        # Check if raw_data parameter is optional (for Phase 3 enhancement)
        param_details = {name: param for name, param in sig.parameters.items()}
        if 'raw_data' in param_details:
            assert param_details['raw_data'].default is not param_details['raw_data'].empty
    
    def test_concrete_analysis_component_basic(self):
        """Test basic concrete implementation of AnalysisComponent."""
        
        class ConcreteAnalysisComponent(AnalysisComponent):
            def update(self, model_output, raw_data=None):
                # Add basic analysis features
                analysis = model_output.copy()
                analysis['regime_name'] = analysis['predicted_state'].map({
                    0: 'Bear', 1: 'Sideways', 2: 'Bull'
                })
                analysis['signal_strength'] = analysis['confidence'] * 2 - 1
                
                return analysis
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = ConcreteAnalysisComponent()
        
        # Test with model output
        model_output = pd.DataFrame({
            'predicted_state': [0, 1, 2, 1, 0],
            'confidence': [0.8, 0.7, 0.9, 0.6, 0.85]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        result = component.update(model_output)
        
        assert isinstance(result, pd.DataFrame)
        assert 'regime_name' in result.columns
        assert 'signal_strength' in result.columns
        assert len(result) == len(model_output)
        
        # Verify regime names
        expected_names = ['Bear', 'Sideways', 'Bull', 'Sideways', 'Bear']
        assert result['regime_name'].tolist() == expected_names
    
    def test_concrete_analysis_component_with_raw_data(self):
        """Test analysis component that uses raw_data parameter."""
        
        class RawDataAnalysisComponent(AnalysisComponent):
            def update(self, model_output, raw_data=None):
                analysis = model_output.copy()
                
                if raw_data is not None and 'close' in raw_data.columns:
                    # Calculate price-based features
                    common_index = model_output.index.intersection(raw_data.index)
                    aligned_prices = raw_data.loc[common_index, 'close']
                    
                    analysis['price'] = aligned_prices
                    analysis['price_change'] = aligned_prices.pct_change()
                
                return analysis
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig
        
        component = RawDataAnalysisComponent()
        
        # Test without raw data
        model_output = pd.DataFrame({
            'predicted_state': [0, 1, 2],
            'confidence': [0.8, 0.7, 0.9]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        result = component.update(model_output)
        assert 'price' not in result.columns
        
        # Test with raw data
        raw_data = pd.DataFrame({
            'close': [100, 101, 99]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        result = component.update(model_output, raw_data=raw_data)
        assert 'price' in result.columns
        assert 'price_change' in result.columns
        assert result['price'].tolist() == [100, 101, 99]


class TestReportComponent:
    """Test ReportComponent interface."""
    
    def test_report_component_is_abstract(self):
        """Test that ReportComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ReportComponent()
    
    def test_report_component_interface_definition(self):
        """Test ReportComponent interface definition."""
        assert issubclass(ReportComponent, ABC)
        assert hasattr(ReportComponent, 'update')
        
        import inspect
        sig = inspect.signature(ReportComponent.update)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'kwargs' in params
    
    def test_concrete_report_component_implementation(self):
        """Test concrete implementation of ReportComponent."""
        
        class ConcreteReportComponent(ReportComponent):
            def update(self, data, observations, model_output, analysis):
                current_regime = analysis['regime_name'].iloc[-1] if not analysis.empty else 'Unknown'
                current_confidence = analysis['confidence'].iloc[-1] if not analysis.empty else 0.0
                
                return f"""# Market Analysis Report

## Current Status
- **Regime**: {current_regime}
- **Confidence**: {current_confidence:.1%}

## Data Summary
- **Data Points**: {len(data)}
- **Observations**: {len(observations)}
- **Model Predictions**: {len(model_output)}
- **Analysis Features**: {len(analysis.columns) if not analysis.empty else 0}

Generated at: {pd.Timestamp.now()}
"""
        
        component = ConcreteReportComponent()
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        
        data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98]
        }, index=dates)
        
        observations = pd.DataFrame({
            'log_return': np.log(data['close'] / data['close'].shift(1)).dropna()
        })
        
        model_output = pd.DataFrame({
            'predicted_state': [0, 1, 2, 1],
            'confidence': [0.8, 0.7, 0.9, 0.6]
        }, index=observations.index)
        
        analysis = model_output.copy()
        analysis['regime_name'] = analysis['predicted_state'].map({
            0: 'Bear', 1: 'Sideways', 2: 'Bull'
        })
        
        result = component.update(data, observations, model_output, analysis)
        
        assert isinstance(result, str)
        assert "Market Analysis Report" in result
        assert "Current Status" in result
        assert "Regime**: Sideways" in result  # Last regime
        assert "Data Points**: 5" in result
        assert "Observations**: 4" in result


class TestInterfaceCompliance:
    """Test interface compliance and contract validation."""
    
    def test_all_interfaces_are_abstract(self):
        """Test that all interface classes are abstract."""
        interfaces = [
            DataComponent,
            ObservationComponent,
            ModelComponent,
            AnalysisComponent,
            ReportComponent
        ]
        
        for interface in interfaces:
            assert issubclass(interface, ABC)
            with pytest.raises(TypeError):
                interface()
    
    def test_interface_method_signatures(self):
        """Test that interface method signatures are properly defined."""
        import inspect
        
        # DataComponent.update(self, current_date=None)
        sig = inspect.signature(DataComponent.update)
        params = list(sig.parameters.keys())
        assert params == ['self', 'current_date']
        assert sig.parameters['current_date'].default is None
        
        # ObservationComponent.update(self, data)
        sig = inspect.signature(ObservationComponent.update)
        params = list(sig.parameters.keys())
        assert params == ['self', 'data']
        
        # ModelComponent.update(self, observations)
        sig = inspect.signature(ModelComponent.update)
        params = list(sig.parameters.keys())
        assert params == ['self', 'observations']
        
        # AnalysisComponent.update(self, model_output, raw_data=None)
        sig = inspect.signature(AnalysisComponent.update)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'model_output' in params
        # raw_data parameter may be optional based on implementation
        
        # ReportComponent.update(self, **kwargs)
        sig = inspect.signature(ReportComponent.update)
        params = list(sig.parameters.keys())
        expected = ['self', 'kwargs']
        assert params == expected
    
    def test_interface_inheritance_hierarchy(self):
        """Test interface inheritance hierarchy."""
        interfaces = [
            DataComponent,
            ObservationComponent,
            ModelComponent,
            AnalysisComponent,
            ReportComponent
        ]
        
        for interface in interfaces:
            # All should inherit from ABC
            assert issubclass(interface, ABC)
            
            # Should have abstractmethod decorator on update
            assert hasattr(interface.update, '__isabstractmethod__')
            assert interface.update.__isabstractmethod__
    
    def test_concrete_implementation_requirements(self):
        """Test that concrete implementations must implement all abstract methods."""
        
        # Incomplete implementation should fail
        class IncompleteDataComponent(DataComponent):
            pass  # Missing update method
        
        with pytest.raises(TypeError):
            IncompleteDataComponent()
        
        # Complete implementation should succeed
        class CompleteDataComponent(DataComponent):
            def update(self, current_date=None):
                return pd.DataFrame()
            
            def get_all_data(self):
                return pd.DataFrame()
            
            def plot(self, **kwargs):
                fig, ax = plt.subplots()
                return fig
        
        component = CompleteDataComponent()
        assert isinstance(component, DataComponent)
        assert isinstance(component, ABC)


class TestInterfaceDocumentation:
    """Test interface documentation and metadata."""
    
    def test_interface_docstrings(self):
        """Test that interfaces have proper docstrings."""
        interfaces = [
            DataComponent,
            ObservationComponent,
            ModelComponent,
            AnalysisComponent,
            ReportComponent
        ]
        
        for interface in interfaces:
            assert interface.__doc__ is not None
            assert len(interface.__doc__.strip()) > 0
            
            # Update method should have docstring
            assert interface.update.__doc__ is not None
            assert len(interface.update.__doc__.strip()) > 0
    
    def test_interface_module_imports(self):
        """Test that interfaces can be imported correctly."""
        # Test individual imports
        from hidden_regime.pipeline.interfaces import DataComponent
        from hidden_regime.pipeline.interfaces import ObservationComponent
        from hidden_regime.pipeline.interfaces import ModelComponent
        from hidden_regime.pipeline.interfaces import AnalysisComponent
        from hidden_regime.pipeline.interfaces import ReportComponent
        
        # Test bulk import
        from hidden_regime.pipeline.interfaces import (
            DataComponent as DC,
            ObservationComponent as OC,
            ModelComponent as MC,
            AnalysisComponent as AC,
            ReportComponent as RC
        )
        
        assert issubclass(DC, ABC)
        assert issubclass(OC, ABC)
        assert issubclass(MC, ABC)
        assert issubclass(AC, ABC)
        assert issubclass(RC, ABC)


if __name__ == "__main__":
    pytest.main([__file__])