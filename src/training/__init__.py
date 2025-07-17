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

__all__ = [
    # Training engine classes
    'MeshTrainingEngine',
    'TrainingScenario',
    'CommutatorRoute', 
    'TrainingResult'
] 