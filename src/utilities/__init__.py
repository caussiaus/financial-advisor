"""
Mesh Utilities & Support

Support utilities and frameworks:
- Adaptive mesh generation
- Memory management for large mesh structures
- Comprehensive testing framework
"""

from .adaptive_mesh_generator import AdaptiveMeshGenerator
from .mesh_memory_manager import MeshMemoryManager, CompressedNode
from .mesh_testing_framework import ComprehensiveMeshTesting, TestResult

__all__ = [
    'AdaptiveMeshGenerator',
    'MeshMemoryManager',
    'CompressedNode',
    'ComprehensiveMeshTesting',
    'TestResult'
] 