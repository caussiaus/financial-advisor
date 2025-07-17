"""
Modular Financial Engine Architecture

Five-layer architecture:
1. PDF Processor Layer - Document processing and milestone extraction
2. Mesh Engine Layer - Stochastic mesh core with GBM paths  
3. Accounting Layer - Financial state tracking and reconciliation
4. Recommendation Engine Layer - Financial advice generation
5. UI Layer - Web interface and visualization
"""

from .pdf_processor import PDFProcessorLayer
from .mesh_engine import MeshEngineLayer
from .accounting import AccountingLayer
from .recommendation_engine import RecommendationEngineLayer
from .ui import UILayer

__all__ = [
    'PDFProcessorLayer',
    'MeshEngineLayer', 
    'AccountingLayer',
    'RecommendationEngineLayer',
    'UILayer'
] 