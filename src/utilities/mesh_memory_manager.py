"""
Memory management system for efficient node storage and retrieval.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict
from datetime import datetime


class LRUCache:
    """Least Recently Used Cache implementation"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Dict]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Dict):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def batch_get(self, keys: List[str]) -> List[Optional[Dict]]:
        return [self.get(key) for key in keys]


class StateCompressor:
    """Compresses financial state data for efficient storage"""
    
    def __init__(self):
        self.base_states = {}
        self.compression_threshold = 0.001
    
    def compress(self, state: Dict) -> Dict:
        """Compress a financial state by storing only significant deltas"""
        compressed = {}
        for key, value in state.items():
            if abs(value) > self.compression_threshold:
                compressed[key] = value
        return compressed
    
    def decompress(self, compressed_state: Dict, base_state: Dict) -> Dict:
        """Reconstruct full state from compressed data"""
        full_state = base_state.copy()
        full_state.update(compressed_state)
        return full_state


@dataclass
class CompressedNode:
    """Memory-efficient node representation"""
    node_id: str
    timestamp: np.datetime64
    state_delta: Dict[str, float]  # Only store changes from parent
    probability: float
    parent_id: Optional[str]
    aggregation_key: str  # For state space aggregation


class MeshMemoryManager:
    """Main memory management system"""
    
    def __init__(self, max_nodes: int = 10000):
        self.node_cache = LRUCache(max_size=max_nodes)
        self.state_compressor = StateCompressor()
        self.base_states = {}
        self.node_clusters = {}
    
    def store_node(self, node: 'OmegaNode'):
        """Efficiently store node with compression"""
        compressed_state = self.state_compressor.compress(node.financial_state)
        compressed_node = CompressedNode(
            node_id=node.node_id,
            timestamp=np.datetime64(node.timestamp),
            state_delta=compressed_state,
            probability=node.probability,
            parent_id=node.parent_nodes[0] if node.parent_nodes else None,
            aggregation_key=self._generate_aggregation_key(node)
        )
        self.node_cache.put(node.node_id, compressed_node)
    
    def batch_retrieve(self, node_ids: List[str]) -> List['OmegaNode']:
        """Batch retrieval of nodes"""
        compressed_nodes = self.node_cache.batch_get(node_ids)
        return [self._decompress_node(cn) if cn else None for cn in compressed_nodes]
    
    def _generate_aggregation_key(self, node: 'OmegaNode') -> str:
        """Generate key for state space aggregation"""
        # Round financial values to reduce state space
        rounded_state = {
            k: round(v, 2) 
            for k, v in node.financial_state.items()
        }
        timestamp_key = node.timestamp.strftime('%Y%m%d')
        return f"{timestamp_key}_{hash(frozenset(rounded_state.items()))}"
    
    def _decompress_node(self, compressed_node: CompressedNode) -> 'OmegaNode':
        """Reconstruct full node from compressed data"""
        if not compressed_node:
            return None
            
        # Get base state from parent if available
        base_state = {}
        if compressed_node.parent_id:
            parent = self.node_cache.get(compressed_node.parent_id)
            if parent:
                base_state = self.state_compressor.decompress(
                    parent.state_delta,
                    self.base_states.get(parent.parent_id, {})
                )
        
        # Reconstruct full state
        full_state = self.state_compressor.decompress(
            compressed_node.state_delta,
            base_state
        )
        
        # Import here to avoid circular imports
        from src.stochastic_mesh_engine import OmegaNode
        
        return OmegaNode(
            node_id=compressed_node.node_id,
            timestamp=compressed_node.timestamp.astype(datetime),
            financial_state=full_state,
            probability=compressed_node.probability,
            parent_nodes=[compressed_node.parent_id] if compressed_node.parent_id else []
        ) 