#!/usr/bin/env python3
"""
Basic pipeline functionality test.

Tests the new pipeline architecture with a simple example to ensure
all components work together correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("Testing new Pipeline architecture...")
    
    try:
        # Import the new pipeline components
        print("\n1. Testing imports...")
        import hidden_regime as hr
        print("✅ Main module imported successfully")
        
        # Test configuration creation
        print("\n2. Testing configuration creation...")
        data_config = hr.FinancialDataConfig(
            ticker='SPY',
            start_date='2023-01-01',
            end_date='2023-03-01',
            source='yfinance'
        )
        print("✅ FinancialDataConfig created")
        
        obs_config = hr.FinancialObservationConfig.create_default_financial()
        print("✅ FinancialObservationConfig created")
        
        model_config = hr.HMMConfig.create_balanced()
        model_config.n_states = 3
        print("✅ HMMConfig created")
        
        analysis_config = hr.FinancialAnalysisConfig.create_minimal_financial()
        analysis_config.n_states = 3
        print("✅ FinancialAnalysisConfig created")
        
        report_config = hr.ReportConfig.create_minimal()
        print("✅ ReportConfig created")
        
        # Test component creation
        print("\n3. Testing component creation...")
        
        # Test data component
        data_component = hr.component_factory.create_data_component(data_config)
        print("✅ Data component created")
        
        # Test observation component  
        obs_component = hr.component_factory.create_observation_component(obs_config)
        print("✅ Observation component created")
        
        # Test model component
        model_component = hr.component_factory.create_model_component(model_config)
        print("✅ Model component created")
        
        # Test analysis component
        analysis_component = hr.component_factory.create_analysis_component(analysis_config)
        print("✅ Analysis component created")
        
        # Test report component
        report_component = hr.component_factory.create_report_component(report_config)
        print("✅ Report component created")
        
        # Test pipeline creation
        print("\n4. Testing pipeline creation...")
        pipeline = hr.Pipeline(
            data=data_component,
            observation=obs_component,
            model=model_component,
            analysis=analysis_component,
            report=report_component
        )
        print("✅ Pipeline created successfully")
        
        # Test convenience functions
        print("\n5. Testing convenience functions...")
        simple_pipeline = hr.create_simple_regime_pipeline('SPY', n_states=3)
        print("✅ Simple pipeline created via convenience function")
        
        trading_pipeline = hr.create_trading_pipeline('AAPL', n_states=3)
        print("✅ Trading pipeline created via convenience function")
        
        print("\n🎉 All tests passed! Pipeline architecture is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_data_flow():
    """Test the data flow through pipeline components."""
    print("\n" + "="*60)
    print("Testing Pipeline Data Flow...")
    print("="*60)
    
    try:
        import hidden_regime as hr
        import numpy as np
        import pandas as pd
        
        # Create a simple pipeline with mock data
        print("\n1. Creating simple pipeline...")
        pipeline = hr.create_simple_regime_pipeline('SPY', n_states=3)
        
        # Get sample data (this will try to load real data)
        print("\n2. Loading sample data...")
        try:
            sample_data = pipeline.data.get_all_data()
            print(f"✅ Data loaded: {sample_data.shape}")
            print(f"   Columns: {list(sample_data.columns)}")
            print(f"   Date range: {sample_data.index.min()} to {sample_data.index.max()}")
        except Exception as e:
            print(f"⚠️  Real data loading failed (expected): {e}")
            print("   Creating mock data for testing...")
            
            # Create mock data for testing
            dates = pd.date_range('2023-01-01', '2023-02-01', freq='D')
            np.random.seed(42)
            prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.01, len(dates))))
            
            mock_data = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            # Inject mock data into pipeline
            pipeline.data._last_data = mock_data
            sample_data = mock_data
            print(f"✅ Mock data created: {sample_data.shape}")
        
        # Test observation generation
        print("\n3. Testing observation generation...")
        observations = pipeline.observation.update(sample_data)
        print(f"✅ Observations generated: {observations.shape}")
        print(f"   Observation columns: {list(observations.columns)}")
        
        # Test model training and prediction
        print("\n4. Testing model processing...")
        model_output = pipeline.model.update(observations)
        print(f"✅ Model output generated: {model_output.shape}")
        print(f"   Model output columns: {list(model_output.columns)}")
        
        # Test analysis
        print("\n5. Testing analysis...")
        analysis_output = pipeline.analysis.update(model_output)
        print(f"✅ Analysis completed: {analysis_output.shape}")
        print(f"   Analysis columns: {list(analysis_output.columns)}")
        
        # Show current regime
        if len(analysis_output) > 0:
            current = analysis_output.iloc[-1]
            regime_name = current.get('regime_name', 'Unknown')
            confidence = current.get('confidence', 0) * 100
            print(f"\n📊 Current Analysis Results:")
            print(f"   Regime: {regime_name}")
            print(f"   Confidence: {confidence:.1f}%")
        
        print("\n🎉 Pipeline data flow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("HIDDEN REGIME PIPELINE ARCHITECTURE TEST")
    print("="*60)
    
    # Run basic tests
    basic_success = test_basic_pipeline()
    
    # Run data flow tests
    flow_success = test_pipeline_data_flow()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic Pipeline Test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Data Flow Test: {'✅ PASS' if flow_success else '❌ FAIL'}")
    
    if basic_success and flow_success:
        print("\n🚀 All tests passed! The new pipeline architecture is ready for use.")
        print("\nExample usage:")
        print("  import hidden_regime as hr")
        print("  pipeline = hr.create_simple_regime_pipeline('AAPL')")
        print("  result = pipeline.update()")
        print("  print(result)")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    sys.exit(0 if (basic_success and flow_success) else 1)