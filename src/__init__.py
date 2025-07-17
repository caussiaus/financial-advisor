"""
Omega Mesh Financial Engine - Module-Based Architecture

Quantitative Finance Mesh System with clear separation of concerns:

CORE ENGINES:
- Stochastic mesh engine (GBM for portfolio evolution)
- Time uncertainty mesh (vectorized GBM for event uncertainty)
- State-space mesh engine (complete financial histories)

ANALYSIS TOOLS:
- Mesh congruence engine (Delaunay triangulation & CVT)
- Mesh vector database (similarity matching & clustering)
- Mesh backtesting framework (historical performance)

INTEGRATION LAYERS:
- Mesh engine layer (modular API)
- State-space integration (path-dependent analysis)
- Market mesh integration (real-time market data)
- Time uncertainty integration (uncertainty modeling)

UTILITIES:
- Adaptive mesh generator (similarity matching)
- Mesh memory manager (efficient storage)
- Mesh testing framework (comprehensive validation)

VISUALIZATION:
- Flexibility comfort mesh (2D financial state visualization)
"""

# Core mesh engines
from .core import (
    StochasticMeshEngine,
    TimeUncertaintyMeshEngine,
    SeedEvent,
    EventVector,
    EnhancedMeshEngine,
    EnhancedMeshNode,
    CashFlowSeries,
    FinancialState
)

# Analysis tools
from .analysis import (
    MeshCongruenceEngine,
    MeshCongruenceResult,
    MeshVectorDatabase,
    MeshEmbedding,
    SimilarityMatch,
    MeshBacktestingFramework,
    BacktestReport
)

# Integration layers
from .integration import (
    MeshEngineLayer,
    MeshConfig,
    MeshNode,
    EnhancedMeshIntegration,
    MeshMarketIntegration,
    TimeUncertaintyIntegration
)

# Utilities
from .utilities import (
    AdaptiveMeshGenerator,
    MeshMemoryManager,
    CompressedNode,
    ComprehensiveMeshTesting,
    TestResult
)

# Visualization
from .visualization import (
    FlexibilityComfortMesh,
    PaymentType
)

# Legacy imports for backward compatibility
from .enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone

__version__ = "2.0.0"

__all__ = [
    # Core engines
    'StochasticMeshEngine',
    'TimeUncertaintyMeshEngine', 
    'SeedEvent',
    'EventVector',
    'EnhancedMeshEngine',
    'EnhancedMeshNode',
    'CashFlowSeries',
    'FinancialState',
    
    # Analysis tools
    'MeshCongruenceEngine',
    'MeshCongruenceResult',
    'MeshVectorDatabase',
    'MeshEmbedding',
    'SimilarityMatch',
    'MeshBacktestingFramework',
    'BacktestReport',
    
    # Integration layers
    'MeshEngineLayer',
    'MeshConfig',
    'MeshNode',
    'EnhancedMeshIntegration',
    'MeshMarketIntegration',
    'TimeUncertaintyIntegration',
    
    # Utilities
    'AdaptiveMeshGenerator',
    'MeshMemoryManager',
    'CompressedNode',
    'ComprehensiveMeshTesting',
    'TestResult',
    
    # Visualization
    'FlexibilityComfortMesh',
    'PaymentType',
    
    # Legacy
    'EnhancedPDFProcessor',
    'FinancialMilestone'
] 