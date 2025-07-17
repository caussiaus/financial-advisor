"""
Minimal Mesh Engine Layer for Legacy Compatibility
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import networkx as nx

@dataclass
class MeshNode:
    node_id: str
    timestamp: datetime
    financial_state: Dict[str, float]
    probability: float
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    event_triggers: List[str] = field(default_factory=list)
    payment_opportunities: Dict[str, Dict] = field(default_factory=dict)
    visibility_radius: float = 1.0
    is_solidified: bool = False

class MeshEngineLayer:
    """
    Minimal Mesh Engine Layer for legacy mesh training compatibility.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.mesh = nx.DiGraph()
        self.current_position: Optional[str] = None
        self.performance_metrics: Dict = {}
        self.use_acceleration = False

    def initialize_mesh(self, initial_state: Dict[str, float], milestones: List, time_horizon_years: float = None) -> Dict:
        # Minimal stub for legacy compatibility
        return {'status': 'initialized', 'nodes': 1, 'edges': 0}

    def get_mesh_status(self) -> Dict:
        return {'total_nodes': len(self.mesh.nodes), 'total_edges': len(self.mesh.edges)} 