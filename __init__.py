"""
Financial Advisor Mesh System

A comprehensive financial analysis and optimization system that uses mesh networks,
stochastic processes, and machine learning to provide personalized financial advice.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add src directory to path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import key modules
try:
    # Core modules
    from src.core.stochastic_mesh_engine import StochasticMeshEngine
    from src.core.state_space_mesh_engine import EnhancedMeshEngine
    from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine
    
    # Training modules
    from src.training.mesh_training_engine import MeshTrainingEngine
    from src.training.portfolio_training_engine import PortfolioTrainingEngine
    from src.training.training_controller import TrainingController
    
    # Analysis modules
    from src.analysis.mesh_congruence_engine import MeshCongruenceEngine
    from src.analysis.mesh_vector_database import MeshVectorDatabase
    
    # Integration modules
    from src.integration.mesh_engine_layer import MeshEngineLayer
    from src.integration.time_uncertainty_integration import TimeUncertaintyIntegration
    from src.integration.market_mesh_integration import MeshMarketIntegration
    from src.integration.state_space_integration import StateSpaceIntegration
    
    # Utility modules
    from src.utilities.mesh_memory_manager import MeshMemoryManager
    from src.utilities.adaptive_mesh_generator import AdaptiveMeshGenerator
    
    # Core business logic
    from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine
    from src.commutator_decision_engine import CommutatorDecisionEngine
    from src.unified_cash_flow_model import UnifiedCashFlowModel
    from src.vectorized_accounting import VectorizedAccountingEngine
    from src.accounting_reconciliation import AccountingReconciliationEngine
    from src.enhanced_accounting_logger import EnhancedAccountingLogger
    
    # Visualization
    from src.visualization.mesh_3d_visualizer import Mesh3DVisualizer
    
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
    
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)
    print(f"⚠️ Import error in main __init__.py: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Financial Advisor Team"
__description__ = "Advanced financial mesh system with stochastic optimization"

# Export main classes
__all__ = [
    # Core mesh engines
    'StochasticMeshEngine',
    'EnhancedMeshEngine', 
    'TimeUncertaintyMeshEngine',
    
    # Training system
    'MeshTrainingEngine',
    'PortfolioTrainingEngine',
    'TrainingController',
    
    # Analysis
    'MeshCongruenceEngine',
    'MeshVectorDatabase',
    
    # Integration
    'MeshEngineLayer',
    'TimeUncertaintyIntegration',
    'MeshMarketIntegration',
    'StateSpaceIntegration',
    
    # Utilities
    'MeshMemoryManager',
    'AdaptiveMeshGenerator',
    
    # Core business logic
    'SyntheticLifestyleEngine',
    'CommutatorDecisionEngine',
    'UnifiedCashFlowModel',
    'VectorizedAccountingEngine',
    'AccountingReconciliationEngine',
    'EnhancedAccountingLogger',
    
    # Visualization
    'Mesh3DVisualizer',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]

def get_project_info():
    """Get project information and status"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'import_success': IMPORT_SUCCESS,
        'import_error': IMPORT_ERROR,
        'project_root': str(project_root),
        'src_path': str(src_path)
    }

def check_imports():
    """Check if all critical imports are working"""
    if not IMPORT_SUCCESS:
        print(f"❌ Import check failed: {IMPORT_ERROR}")
        return False
    
    print("✅ All critical imports successful")
    return True 