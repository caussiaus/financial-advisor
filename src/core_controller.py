"""
Core Controller for Omega Mesh Financial System

This central controller manages all imports and dependencies to avoid circular import issues.
It provides a clean interface for the dashboard and other components.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all core components
try:
    from .core.stochastic_mesh_engine import StochasticMeshEngine, OmegaNode
    from .core.time_uncertainty_mesh import TimeUncertaintyMeshEngine, SeedEvent, EventVector
    from .core.state_space_mesh_engine import EnhancedMeshEngine as StateSpaceMeshEngine
    
    from .enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone
    from .accounting_reconciliation import AccountingReconciliationEngine
    from .financial_recommendation_engine import FinancialRecommendationEngine
    from .synthetic_lifestyle_engine import SyntheticLifestyleEngine
    
    from .analysis.mesh_vector_database import MeshVectorDatabase, MeshEmbedding, SimilarityMatch
    from .analysis.mesh_congruence_engine import MeshCongruenceEngine, MeshCongruenceResult
    from .analysis.mesh_backtesting_framework import MeshBacktestingFramework, BacktestReport
    
    from .integration.market_mesh_integration import MeshMarketIntegration
    from .integration.state_space_integration import EnhancedMeshIntegration
    from .integration.time_uncertainty_integration import TimeUncertaintyIntegration
    from .integration.mesh_engine_layer import MeshEngineLayer, MeshConfig, MeshNode
    
    from .utilities.mesh_memory_manager import MeshMemoryManager, CompressedNode
    from .utilities.adaptive_mesh_generator import AdaptiveMeshGenerator
    from .utilities.mesh_testing_framework import ComprehensiveMeshTesting, TestResult
    
    from .visualization.flexibility_comfort_mesh import FlexibilityComfortMeshEngine, PaymentType
    
    from .market_tracking_backtest import (
        MarketDataTracker, PersonalFinanceInvestmentMapper, BacktestEngine,
        BacktestAnalyzer, PersonalFinanceAction, InvestmentDecision, BacktestResult
    )
    
    from .json_to_vector_converter import ClientVectorProfile, LifeStage, EventCategory
    
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
    
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)
    print(f"⚠️ Import error in core controller: {e}")


class CoreController:
    """
    Central controller for the Omega Mesh Financial System.
    Manages all components and provides a clean interface.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        if not IMPORT_SUCCESS:
            self.logger.error(f"Failed to import core components: {IMPORT_ERROR}")
            raise ImportError(f"Core components import failed: {IMPORT_ERROR}")
        
        self.logger.info("✅ Core Controller initialized successfully")
        
        # Initialize core components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the core controller"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all core components"""
        self.logger.info("🔄 Initializing core components...")
        
        # Core engines
        self.stochastic_mesh_engine = None
        self.time_uncertainty_engine = None
        self.state_space_engine = None
        
        # Processing components
        self.pdf_processor = None
        self.accounting_engine = None
        self.recommendation_engine = None
        self.synthetic_engine = None
        
        # Analysis components
        self.vector_database = None
        self.congruence_engine = None
        self.backtesting_framework = None
        
        # Integration components
        self.market_integration = None
        self.state_space_integration = None
        self.time_uncertainty_integration = None
        self.mesh_engine_layer = None
        
        # Utility components
        self.memory_manager = None
        self.adaptive_mesh_generator = None
        self.testing_framework = None
        
        # Visualization components
        self.flexibility_comfort_mesh = None
        
        # Market tracking components
        self.market_tracker = None
        self.investment_mapper = None
        self.backtest_engine = None
        self.backtest_analyzer = None
        
        self.logger.info("✅ Core components initialized")
    
    def initialize_mesh_system(self, initial_financial_state: Dict[str, float]) -> bool:
        """Initialize the complete mesh system"""
        try:
            self.logger.info("🚀 Initializing complete mesh system...")
            
            # Initialize core engines
            self.stochastic_mesh_engine = StochasticMeshEngine(current_financial_state=initial_financial_state)
            self.time_uncertainty_engine = TimeUncertaintyMeshEngine(use_gpu=True)
            self.state_space_engine = StateSpaceMeshEngine()
            
            # Initialize processing components
            self.pdf_processor = EnhancedPDFProcessor()
            self.accounting_engine = AccountingReconciliationEngine()
            self.recommendation_engine = FinancialRecommendationEngine(
                self.stochastic_mesh_engine, self.accounting_engine
            )
            self.synthetic_engine = SyntheticLifestyleEngine()
            
            # Initialize analysis components
            self.vector_database = MeshVectorDatabase()
            self.congruence_engine = MeshCongruenceEngine()
            self.backtesting_framework = MeshBacktestingFramework()
            
            # Initialize integration components
            self.market_integration = MeshMarketIntegration()
            self.state_space_integration = EnhancedMeshIntegration()
            self.time_uncertainty_integration = TimeUncertaintyIntegration(initial_financial_state)
            self.mesh_engine_layer = MeshEngineLayer()
            
            # Initialize utility components
            self.memory_manager = MeshMemoryManager(max_nodes=10000)
            self.adaptive_mesh_generator = AdaptiveMeshGenerator(
                initial_state=initial_financial_state, 
                memory_manager=self.memory_manager
            )
            self.testing_framework = ComprehensiveMeshTesting()
            
            # Initialize visualization components
            self.flexibility_comfort_mesh = FlexibilityComfortMeshEngine()
            
            # Initialize market tracking components
            self.market_tracker = MarketDataTracker()
            self.investment_mapper = PersonalFinanceInvestmentMapper()
            self.backtest_engine = BacktestEngine(self.market_tracker, self.investment_mapper)
            self.backtest_analyzer = BacktestAnalyzer()
            
            self.logger.info("✅ Complete mesh system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize mesh system: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of all system components"""
        return {
            'import_success': IMPORT_SUCCESS,
            'import_error': IMPORT_ERROR,
            'components_initialized': {
                'stochastic_mesh_engine': self.stochastic_mesh_engine is not None,
                'time_uncertainty_engine': self.time_uncertainty_engine is not None,
                'state_space_engine': self.state_space_engine is not None,
                'pdf_processor': self.pdf_processor is not None,
                'accounting_engine': self.accounting_engine is not None,
                'recommendation_engine': self.recommendation_engine is not None,
                'synthetic_engine': self.synthetic_engine is not None,
                'vector_database': self.vector_database is not None,
                'congruence_engine': self.congruence_engine is not None,
                'backtesting_framework': self.backtesting_framework is not None,
                'market_integration': self.market_integration is not None,
                'state_space_integration': self.state_space_integration is not None,
                'time_uncertainty_integration': self.time_uncertainty_integration is not None,
                'mesh_engine_layer': self.mesh_engine_layer is not None,
                'memory_manager': self.memory_manager is not None,
                'adaptive_mesh_generator': self.adaptive_mesh_generator is not None,
                'testing_framework': self.testing_framework is not None,
                'flexibility_comfort_mesh': self.flexibility_comfort_mesh is not None,
                'market_tracker': self.market_tracker is not None,
                'investment_mapper': self.investment_mapper is not None,
                'backtest_engine': self.backtest_engine is not None,
                'backtest_analyzer': self.backtest_analyzer is not None
            }
        }
    
    def get_available_components(self) -> Dict[str, Any]:
        """Get all available components for external use"""
        return {
            'StochasticMeshEngine': StochasticMeshEngine,
            'TimeUncertaintyMeshEngine': TimeUncertaintyMeshEngine,
            'StateSpaceMeshEngine': StateSpaceMeshEngine,
            'EnhancedPDFProcessor': EnhancedPDFProcessor,
            'AccountingReconciliationEngine': AccountingReconciliationEngine,
            'FinancialRecommendationEngine': FinancialRecommendationEngine,
            'SyntheticLifestyleEngine': SyntheticLifestyleEngine,
            'MeshVectorDatabase': MeshVectorDatabase,
            'MeshCongruenceEngine': MeshCongruenceEngine,
            'MeshBacktestingFramework': MeshBacktestingFramework,
            'MeshMarketIntegration': MeshMarketIntegration,
            'EnhancedMeshIntegration': EnhancedMeshIntegration,
            'TimeUncertaintyIntegration': TimeUncertaintyIntegration,
            'MeshEngineLayer': MeshEngineLayer,
            'MeshMemoryManager': MeshMemoryManager,
            'AdaptiveMeshGenerator': AdaptiveMeshGenerator,
            'ComprehensiveMeshTesting': ComprehensiveMeshTesting,
            'FlexibilityComfortMeshEngine': FlexibilityComfortMeshEngine,
            'MarketDataTracker': MarketDataTracker,
            'PersonalFinanceInvestmentMapper': PersonalFinanceInvestmentMapper,
            'BacktestEngine': BacktestEngine,
            'BacktestAnalyzer': BacktestAnalyzer,
            'ClientVectorProfile': ClientVectorProfile,
            'FinancialMilestone': FinancialMilestone,
            'OmegaNode': OmegaNode,
            'SeedEvent': SeedEvent,
            'EventVector': EventVector
        }


# Global instance for easy access
_core_controller = None

def get_core_controller() -> CoreController:
    """Get the global core controller instance"""
    global _core_controller
    if _core_controller is None:
        _core_controller = CoreController()
    return _core_controller

def initialize_system(initial_financial_state: Dict[str, float]) -> bool:
    """Initialize the complete system"""
    controller = get_core_controller()
    return controller.initialize_mesh_system(initial_financial_state)

def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    controller = get_core_controller()
    return controller.get_system_status()

def get_components() -> Dict[str, Any]:
    """Get all available components"""
    controller = get_core_controller()
    return controller.get_available_components() 