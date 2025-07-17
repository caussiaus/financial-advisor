"""
Training module for financial mesh systems

This module contains the training engine for generating synthetic people,
applying financial shocks, and learning optimal commutator routes.
"""

# Import the main training engine classes
from .mesh_training_engine import (
    MeshTrainingEngine,
    TrainingScenario,
    CommutatorRoute,
    TrainingResult
)

# Import portfolio training engine
from .portfolio_training_engine import (
    PortfolioTrainingEngine,
    PortfolioWeights,
    TrainingScenario as PortfolioTrainingScenario
)

__all__ = [
    # Training engine classes
    'MeshTrainingEngine',
    'TrainingScenario',
    'CommutatorRoute', 
    'TrainingResult',
    # Portfolio training engine classes
    'PortfolioTrainingEngine',
    'PortfolioWeights',
    'PortfolioTrainingScenario'
] 