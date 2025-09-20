"""
Integration tests for configuration validation and management.

Tests configuration validation, serialization, and error handling
across all configuration classes in the system.
"""

import pytest
import tempfile
import os
import json

import hidden_regime as hr
from hidden_regime.config.model import HMMConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.observation import FinancialObservationConfig
from hidden_regime.config.analysis import FinancialAnalysisConfig
from hidden_regime.config.report import ReportConfig
from hidden_regime.utils.exceptions import ConfigurationError


class TestConfigurationValidation:
    """Test configuration validation for all config classes."""
    
    def test_hmm_config_validation_valid(self):
        """Test valid HMM configuration passes validation."""
        config = HMMConfig(
            n_states=3,
            max_iterations=100,
            tolerance=1e-6,
            regularization=1e-6,
            forgetting_factor=0.98,
            adaptation_rate=0.05
        )
        
        # Should not raise exception
        config.validate()
        
        # Verify properties
        assert config.n_states == 3
        assert config.max_iterations == 100
        assert config.tolerance == 1e-6
    
    def test_hmm_config_validation_invalid_n_states(self):
        """Test invalid n_states raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="n_states must be at least 2"):
            config = HMMConfig(n_states=1)
            config.validate()
        
        with pytest.raises(ConfigurationError, match="n_states should not exceed 10"):
            config = HMMConfig(n_states=15)
            config.validate()
    
    def test_hmm_config_validation_invalid_tolerance(self):
        """Test invalid tolerance raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="tolerance must be positive"):
            config = HMMConfig(tolerance=-1.0)
            config.validate()
        
        with pytest.raises(ConfigurationError, match="tolerance must be positive"):
            config = HMMConfig(tolerance=0.0)
            config.validate()
    
    def test_hmm_config_validation_invalid_forgetting_factor(self):
        """Test invalid forgetting factor raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="forgetting_factor must be between 0.8 and 1.0"):
            config = HMMConfig(forgetting_factor=0.5)
            config.validate()
        
        with pytest.raises(ConfigurationError, match="forgetting_factor must be between 0.8 and 1.0"):
            config = HMMConfig(forgetting_factor=1.5)
            config.validate()
    
    def test_hmm_config_validation_invalid_adaptation_rate(self):
        """Test invalid adaptation rate raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="adaptation_rate must be between 0.001 and 0.5"):
            config = HMMConfig(adaptation_rate=0.0005)
            config.validate()
        
        with pytest.raises(ConfigurationError, match="adaptation_rate must be between 0.001 and 0.5"):
            config = HMMConfig(adaptation_rate=0.8)
            config.validate()
    
    def test_financial_data_config_validation_valid(self):
        """Test valid financial data configuration."""
        config = FinancialDataConfig(
            ticker='AAPL',
            source='yfinance',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Should not raise exception
        config.validate()
        
        # Verify properties
        assert config.ticker == 'AAPL'
        assert config.source == 'yfinance'
        assert config.start_date == '2023-01-01'
        assert config.end_date == '2023-12-31'
    
    def test_financial_data_config_validation_empty_ticker(self):
        """Test empty ticker raises validation error."""
        with pytest.raises((ConfigurationError, ValueError), match="ticker|empty"):
            config = FinancialDataConfig(ticker='')
            config.validate()
    
    def test_financial_observation_config_validation(self):
        """Test financial observation configuration validation."""
        config = FinancialObservationConfig.create_default_financial()
        
        # Should not raise exception
        config.validate()
        
        # Verify basic properties exist (based on actual config output)
        assert hasattr(config, 'generators')
        assert hasattr(config, 'price_column')
        assert hasattr(config, 'volume_column')
    
    def test_financial_analysis_config_validation(self):
        """Test financial analysis configuration validation."""
        config = FinancialAnalysisConfig.create_comprehensive_financial()
        
        # Should not raise exception
        config.validate()
        
        # Verify n_states is set correctly
        assert hasattr(config, 'n_states')
    
    def test_report_config_validation(self):
        """Test report configuration validation."""
        config = ReportConfig.create_comprehensive()
        
        # Should not raise exception
        config.validate()
        
        # Verify basic properties (based on actual config output)
        assert hasattr(config, 'output_format')
        assert hasattr(config, 'include_summary')
        assert hasattr(config, 'template_style')


class TestConfigurationPresets:
    """Test configuration preset creation and properties."""
    
    def test_hmm_config_conservative_preset(self):
        """Test conservative HMM preset has expected properties."""
        config = HMMConfig.create_conservative()
        
        # Conservative should have stricter tolerances and more stability
        assert config.tolerance == 1e-8
        assert config.smoothing_weight == 0.9
        assert config.forgetting_factor == 0.99
        assert config.adaptation_rate == 0.01
        assert config.enable_change_detection == False
        assert config.max_iterations == 200
    
    def test_hmm_config_aggressive_preset(self):
        """Test aggressive HMM preset has expected properties."""
        config = HMMConfig.create_aggressive()
        
        # Aggressive should adapt faster with lower tolerances
        assert config.tolerance == 1e-4
        assert config.smoothing_weight == 0.6
        assert config.forgetting_factor == 0.95
        assert config.adaptation_rate == 0.1
        assert config.enable_change_detection == True
        assert config.change_detection_threshold == 2.0
        assert config.max_iterations == 50
    
    def test_hmm_config_balanced_preset(self):
        """Test balanced HMM preset has expected properties."""
        config = HMMConfig.create_balanced()
        
        # Balanced should be middle ground
        assert config.tolerance == 1e-6
        assert config.smoothing_weight == 0.8
        assert config.forgetting_factor == 0.98
        assert config.adaptation_rate == 0.05
        assert config.enable_change_detection == True
        assert config.max_iterations == 100
    
    def test_preset_configurations_all_valid(self):
        """Test all preset configurations pass validation."""
        presets = [
            HMMConfig.create_conservative(),
            HMMConfig.create_aggressive(),
            HMMConfig.create_balanced()
        ]
        
        for preset in presets:
            # Should not raise exception
            preset.validate()
    
    def test_financial_observation_config_presets(self):
        """Test financial observation configuration presets."""
        config = FinancialObservationConfig.create_default_financial()
        
        # Should not raise exception
        config.validate()
        
        # Should have reasonable defaults
        assert hasattr(config, 'generators')
        assert hasattr(config, 'price_column')
        assert config.price_column == 'close'
    
    def test_financial_analysis_config_presets(self):
        """Test financial analysis configuration presets."""
        config = FinancialAnalysisConfig.create_comprehensive_financial()
        
        # Should not raise exception
        config.validate()
        
        # Should have n_states property
        assert hasattr(config, 'n_states')


class TestConfigurationSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_hmm_config_to_dict_and_back(self):
        """Test HMM config serialization to dict and back."""
        original_config = HMMConfig.create_aggressive()
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['n_states'] == original_config.n_states
        assert config_dict['tolerance'] == original_config.tolerance
        
        # Deserialize back
        restored_config = HMMConfig.from_dict(config_dict)
        
        # Should be equivalent
        assert restored_config.n_states == original_config.n_states
        assert restored_config.tolerance == original_config.tolerance
        assert restored_config.adaptation_rate == original_config.adaptation_rate
    
    def test_hmm_config_json_serialization(self):
        """Test HMM config JSON serialization."""
        original_config = HMMConfig.create_balanced()
        
        # Serialize to JSON string
        json_str = original_config.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        json_data = json.loads(json_str)
        assert isinstance(json_data, dict)
        assert json_data['n_states'] == original_config.n_states
        
        # Deserialize back
        restored_config = HMMConfig.from_json(json_str)
        
        # Should be equivalent
        assert restored_config.n_states == original_config.n_states
        assert restored_config.tolerance == original_config.tolerance
    
    def test_financial_data_config_serialization(self):
        """Test financial data config serialization."""
        original_config = FinancialDataConfig(
            ticker='NVDA',
            source='yfinance',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Serialize and deserialize
        config_dict = original_config.to_dict()
        restored_config = FinancialDataConfig.from_dict(config_dict)
        
        # Should be equivalent
        assert restored_config.ticker == original_config.ticker
        assert restored_config.source == original_config.source
        assert restored_config.start_date == original_config.start_date
        assert restored_config.end_date == original_config.end_date
    
    def test_config_file_save_and_load(self):
        """Test saving and loading config to/from file."""
        original_config = HMMConfig.create_conservative()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'test_config.json')
            
            # Save to file
            original_config.save(config_file)
            assert os.path.exists(config_file)
            
            # Load from file
            restored_config = HMMConfig.load(config_file)
            
            # Should be equivalent
            assert restored_config.n_states == original_config.n_states
            assert restored_config.tolerance == original_config.tolerance
            assert restored_config.max_iterations == original_config.max_iterations


class TestConfigurationIntegration:
    """Test configuration integration with pipeline creation."""
    
    def test_pipeline_creation_with_custom_configs(self):
        """Test pipeline creation using custom configuration objects."""
        # Create custom configurations
        data_config = FinancialDataConfig(
            ticker='MSFT',
            source='yfinance',
            start_date='2023-06-01',
            end_date='2023-12-31'
        )
        
        model_config = HMMConfig.create_aggressive().copy(n_states=4)

        observation_config = FinancialObservationConfig.create_default_financial()
        analysis_config = FinancialAnalysisConfig.create_comprehensive_financial().copy(n_states=4)
        
        # Create pipeline using custom configs
        pipeline = hr.create_pipeline(
            data_config=data_config,
            observation_config=observation_config,
            model_config=model_config,
            analysis_config=analysis_config
        )
        
        # Verify configurations are applied
        assert pipeline.data.config.ticker == 'MSFT'
        assert pipeline.data.config.start_date == '2023-06-01'
        assert pipeline.model.config.n_states == 4
        assert pipeline.model.config.tolerance == 1e-4  # aggressive preset
    
    def test_configuration_error_propagation_in_pipeline(self):
        """Test that configuration errors are caught during config creation."""
        # Test that invalid config creation raises error immediately
        with pytest.raises(ConfigurationError, match="n_states must be at least 2"):
            invalid_config = HMMConfig(n_states=1)  # Invalid: too few states

        # Test that pipeline creation with valid configs works
        valid_config = HMMConfig(n_states=3)
        pipeline = hr.create_pipeline(
            data_config=FinancialDataConfig(ticker='TEST'),
            observation_config=FinancialObservationConfig.create_default_financial(),
            model_config=valid_config,
            analysis_config=FinancialAnalysisConfig.create_comprehensive_financial()
        )
        assert pipeline is not None
    
    def test_factory_function_validation(self):
        """Test that factory functions validate parameters."""
        # Test invalid n_states
        with pytest.raises((ConfigurationError, ValueError)):
            hr.create_financial_pipeline('TEST', n_states=1)
        
        # Test invalid model config overrides
        with pytest.raises((ConfigurationError, ValueError)):
            hr.create_financial_pipeline(
                'TEST',
                n_states=3,
                model_config_overrides={'tolerance': -1.0}
            )


class TestConfigurationCaching:
    """Test configuration caching and cache key generation."""
    
    def test_hmm_config_cache_key_generation(self):
        """Test HMM config generates consistent cache keys."""
        config1 = HMMConfig.create_balanced()
        config2 = HMMConfig.create_balanced()
        
        # Same configurations should have same cache key
        key1 = config1.get_cache_key()
        key2 = config2.get_cache_key()
        assert key1 == key2
        
        # Different configurations should have different cache keys
        config3 = HMMConfig.create_aggressive()
        key3 = config3.get_cache_key()
        assert key1 != key3
    
    def test_cache_key_includes_relevant_parameters(self):
        """Test cache key changes when relevant parameters change."""
        config = HMMConfig.create_balanced()
        original_key = config.get_cache_key()

        # Changing n_states should change cache key
        config_modified = HMMConfig(
            n_states=4,  # Changed from original
            observed_signal=config.observed_signal,
            initialization_method=config.initialization_method
        )
        new_key = config_modified.get_cache_key()
        assert original_key != new_key

        # Changing observed_signal should change cache key
        config_modified2 = HMMConfig(
            n_states=config_modified.n_states,
            observed_signal='close_price',  # Changed from original
            initialization_method=config.initialization_method
        )
        newer_key = config_modified2.get_cache_key()
        assert new_key != newer_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])