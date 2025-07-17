"""
Training Controller for Mesh Training Engine

This controller manages all training-related imports and dependencies to avoid circular import issues.
It provides a clean interface for the training system and follows the same pattern as the core controller.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all training components
try:
    from .mesh_training_engine import MeshTrainingEngine, TrainingScenario, CommutatorRoute, TrainingResult
    
    # Import dependencies from core controller pattern
    from ..core.stochastic_mesh_engine import StochasticMeshEngine
    from ..core.state_space_mesh_engine import EnhancedMeshEngine
    from ..commutator_decision_engine import CommutatorDecisionEngine, FinancialState, CommutatorOperation
    from ..synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
    from ..unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent
    from ..layers.financial_space_mapper import FinancialSpaceMapper
    
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
    
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)
    print(f"⚠️ Import error in training controller: {e}")


class TrainingController:
    """
    Central controller for the Mesh Training Engine.
    Manages all training components and provides a clean interface.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        if not IMPORT_SUCCESS:
            self.logger.error(f"Failed to import training components: {IMPORT_ERROR}")
            raise ImportError(f"Training components import failed: {IMPORT_ERROR}")
        
        self.logger.info("✅ Training Controller initialized successfully")
        
        # Initialize training components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the training controller"""
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
        """Initialize all training components"""
        self.logger.info("🔄 Initializing training components...")
        
        # Training engine
        self.training_engine = None
        
        # Core dependencies
        self.synthetic_engine = None
        self.financial_mapper = None
        
        # Mesh engines
        self.stochastic_mesh_engine = None
        self.enhanced_mesh_engine = None
        
        # Commutator engine
        self.commutator_engine = None
        
        # Cash flow model
        self.unified_cash_flow_model = None
        
        self.logger.info("✅ Training components initialized")
    
    def initialize_training_system(self, config: Optional[Dict] = None) -> bool:
        """Initialize the complete training system"""
        try:
            self.logger.info("🚀 Initializing complete training system...")
            
            # Initialize training engine
            self.training_engine = MeshTrainingEngine(config)
            
            # Initialize core dependencies
            self.synthetic_engine = SyntheticLifestyleEngine()
            self.financial_mapper = FinancialSpaceMapper()
            
            # Initialize mesh engines (will be created per scenario)
            self.stochastic_mesh_engine = None  # Created per scenario
            self.enhanced_mesh_engine = None    # Created per scenario
            
            # Initialize commutator engine (will be created per scenario)
            self.commutator_engine = None       # Created per scenario
            
            # Initialize cash flow model
            self.unified_cash_flow_model = None  # Created per scenario
            
            self.logger.info("✅ Complete training system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training system: {e}")
            return False
    
    def run_training_session(self, num_scenarios: int = 100, 
                           age_distribution: Optional[Dict[int, float]] = None) -> TrainingResult:
        """
        Run a complete training session
        
        Args:
            num_scenarios: Number of training scenarios to generate
            age_distribution: Optional age distribution for synthetic people
            
        Returns:
            TrainingResult with performance metrics
        """
        if not self.training_engine:
            self.initialize_training_system()
        
        self.logger.info(f"🎓 Starting training session with {num_scenarios} scenarios...")
        
        # Generate training scenarios
        scenarios = self.training_engine.generate_training_scenarios(
            num_scenarios=num_scenarios,
            age_distribution=age_distribution
        )
        
        # Run training session
        result = self.training_engine.run_training_session(scenarios)
        
        # Save results
        self.training_engine.save_training_results()
        
        self.logger.info(f"✅ Training session completed: {result.successful_recoveries}/{result.num_scenarios} successful recoveries")
        return result
    
    def generate_scenarios_only(self, num_scenarios: int = 100, 
                              age_distribution: Optional[Dict[int, float]] = None) -> List[TrainingScenario]:
        """
        Generate training scenarios without running the full training session
        
        Args:
            num_scenarios: Number of scenarios to generate
            age_distribution: Optional age distribution for synthetic people
            
        Returns:
            List of training scenarios
        """
        if not self.training_engine:
            self.initialize_training_system()
        
        self.logger.info(f"📊 Generating {num_scenarios} training scenarios...")
        
        scenarios = self.training_engine.generate_training_scenarios(
            num_scenarios=num_scenarios,
            age_distribution=age_distribution
        )
        
        self.logger.info(f"✅ Generated {len(scenarios)} training scenarios")
        return scenarios
    
    def analyze_training_results(self) -> Dict[str, Any]:
        """
        Analyze existing training results
        
        Returns:
            Dictionary with analysis results
        """
        if not self.training_engine:
            self.initialize_training_system()
        
        # Load existing results
        self.training_engine.load_training_results()
        
        # Analyze results
        analysis = {
            'total_routes': len(self.training_engine.successful_routes) + len(self.training_engine.failed_routes),
            'successful_routes': len(self.training_engine.successful_routes),
            'failed_routes': len(self.training_engine.failed_routes),
            'success_rate': len(self.training_engine.successful_routes) / max(1, len(self.training_engine.successful_routes) + len(self.training_engine.failed_routes)),
            'training_history': len(self.training_engine.training_history)
        }
        
        if self.training_engine.successful_routes:
            # Calculate average metrics
            success_scores = [route.success_score for route in self.training_engine.successful_routes]
            recovery_times = [route.recovery_time for route in self.training_engine.successful_routes]
            
            analysis.update({
                'average_success_score': sum(success_scores) / len(success_scores),
                'average_recovery_time': sum(recovery_times) / len(recovery_times),
                'best_success_score': max(success_scores),
                'worst_success_score': min(success_scores)
            })
        
        return analysis
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get the status of all training components"""
        return {
            'import_success': IMPORT_SUCCESS,
            'import_error': IMPORT_ERROR,
            'components_initialized': {
                'training_engine': self.training_engine is not None,
                'synthetic_engine': self.synthetic_engine is not None,
                'financial_mapper': self.financial_mapper is not None,
                'stochastic_mesh_engine': self.stochastic_mesh_engine is not None,
                'enhanced_mesh_engine': self.enhanced_mesh_engine is not None,
                'commutator_engine': self.commutator_engine is not None,
                'unified_cash_flow_model': self.unified_cash_flow_model is not None
            },
            'training_metrics': {
                'scenarios_generated': len(self.training_engine.training_scenarios) if self.training_engine else 0,
                'successful_routes': len(self.training_engine.successful_routes) if self.training_engine else 0,
                'failed_routes': len(self.training_engine.failed_routes) if self.training_engine else 0,
                'training_sessions': len(self.training_engine.training_history) if self.training_engine else 0
            }
        }
    
    def get_available_components(self) -> Dict[str, Any]:
        """Get available training components"""
        return {
            'training_engine': self.training_engine,
            'synthetic_engine': self.synthetic_engine,
            'financial_mapper': self.financial_mapper,
            'stochastic_mesh_engine': self.stochastic_mesh_engine,
            'enhanced_mesh_engine': self.enhanced_mesh_engine,
            'commutator_engine': self.commutator_engine,
            'unified_cash_flow_model': self.unified_cash_flow_model
        }


def get_training_controller() -> TrainingController:
    """Get the training controller instance"""
    return TrainingController()


def initialize_training_system(config: Optional[Dict] = None) -> bool:
    """Initialize the training system"""
    controller = get_training_controller()
    return controller.initialize_training_system(config)


def run_training_session(num_scenarios: int = 100, 
                        age_distribution: Optional[Dict[int, float]] = None) -> TrainingResult:
    """Run a training session"""
    controller = get_training_controller()
    return controller.run_training_session(num_scenarios, age_distribution)


def get_training_status() -> Dict[str, Any]:
    """Get training system status"""
    controller = get_training_controller()
    return controller.get_training_status()


def get_training_components() -> Dict[str, Any]:
    """Get available training components"""
    controller = get_training_controller()
    return controller.get_available_components() 