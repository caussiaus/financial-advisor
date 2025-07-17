#!/usr/bin/env python3
"""
Test script to verify that all import fixes are working correctly
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports from the main package"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test main package import
        import __init__
        print("✅ Main package import successful")
        
        # Test project info
        info = __init__.get_project_info()
        print(f"✅ Project info: {info['version']} by {info['author']}")
        
        # Test import check
        success = __init__.check_imports()
        if success:
            print("✅ Import check passed")
        else:
            print("❌ Import check failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        return False
    
    return True

def test_training_imports():
    """Test training module imports"""
    print("\n🧪 Testing training module imports...")
    
    try:
        from src.training.portfolio_training_engine import PortfolioTrainingEngine, PortfolioWeights
        print("✅ Portfolio training engine import successful")
        
        from src.training.mesh_training_engine import MeshTrainingEngine, TrainingScenario
        print("✅ Mesh training engine import successful")
        
        from src.training.training_controller import TrainingController
        print("✅ Training controller import successful")
        
    except Exception as e:
        print(f"❌ Training import test failed: {e}")
        return False
    
    return True

def test_core_imports():
    """Test core module imports"""
    print("\n🧪 Testing core module imports...")
    
    try:
        from src.core.stochastic_mesh_engine import StochasticMeshEngine
        print("✅ Stochastic mesh engine import successful")
        
        from src.core.state_space_mesh_engine import EnhancedMeshEngine
        print("✅ State space mesh engine import successful")
        
        from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine
        print("✅ Time uncertainty mesh engine import successful")
        
    except Exception as e:
        print(f"❌ Core import test failed: {e}")
        return False
    
    return True

def test_business_logic_imports():
    """Test business logic module imports"""
    print("\n🧪 Testing business logic imports...")
    
    try:
        from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine
        print("✅ Synthetic lifestyle engine import successful")
        
        from src.commutator_decision_engine import CommutatorDecisionEngine
        print("✅ Commutator decision engine import successful")
        
        from src.unified_cash_flow_model import UnifiedCashFlowModel
        print("✅ Unified cash flow model import successful")
        
        from src.enhanced_accounting_logger import EnhancedAccountingLogger
        print("✅ Enhanced accounting logger import successful")
        
    except Exception as e:
        print(f"❌ Business logic import test failed: {e}")
        return False
    
    return True

def test_demo_imports():
    """Test demo imports"""
    print("\n🧪 Testing demo imports...")
    
    try:
        # Test that the demo can import its dependencies
        import demos.demo_portfolio_training
        print("✅ Portfolio training demo import successful")
        
    except Exception as e:
        print(f"❌ Demo import test failed: {e}")
        return False
    
    return True

def main():
    """Run all import tests"""
    print("=" * 80)
    print("IMPORT FIX VERIFICATION")
    print("=" * 80)
    
    tests = [
        test_basic_imports,
        test_training_imports,
        test_core_imports,
        test_business_logic_imports,
        test_demo_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All import tests passed! The import fixes are working correctly.")
        return True
    else:
        print("\n⚠️ Some import tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 