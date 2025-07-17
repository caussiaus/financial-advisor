"""
Mesh Engine Layer

Responsible for:
- Stochastic mesh generation with GBM paths
- Dynamic pruning and visibility updates
- Path optimization and memory management
- Performance benchmarks and acceleration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set, Protocol
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
import json
import math
import platform
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Try importing acceleration libraries
try:
    import torch
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        METAL_AVAILABLE = torch.backends.mps.is_available()
        if METAL_AVAILABLE:
            device = torch.device("mps")
            torch.set_default_dtype(torch.float32)
        else:
            device = torch.device("cpu")
        CUDA_AVAILABLE = False
    else:
        CUDA_AVAILABLE = torch.cuda.is_available()
        METAL_AVAILABLE = False
        if CUDA_AVAILABLE:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    ACCELERATION_AVAILABLE = CUDA_AVAILABLE or METAL_AVAILABLE
except ImportError:
    ACCELERATION_AVAILABLE = False
    CUDA_AVAILABLE = False
    METAL_AVAILABLE = False
    device = None


@dataclass
class MeshNode:
    """Represents a node in the stochastic mesh"""
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


@dataclass
class MeshConfig:
    """Configuration for mesh generation"""
    time_horizon_years: float = 10.0
    num_paths: int = 1000
    num_steps: int = 120  # Monthly steps for 10 years
    drift: float = 0.05
    volatility: float = 0.15
    dt: float = 1/12  # Monthly time step
    use_acceleration: bool = True
    visibility_radius: float = 365 * 5  # 5 years visibility


class PathGenerator(Protocol):
    """Protocol for path generation capabilities"""
    
    def generate_paths(self, config: MeshConfig, initial_value: float) -> np.ndarray:
        """Generate stochastic paths"""
        ...


class MeshOptimizer(Protocol):
    """Protocol for mesh optimization capabilities"""
    
    def optimize_mesh(self, mesh: nx.DiGraph, config: MeshConfig) -> nx.DiGraph:
        """Optimize mesh structure"""
        ...


class MeshEngineLayer:
    """
    Mesh Engine Layer - Clean API for stochastic mesh operations
    
    Responsibilities:
    - Stochastic mesh generation with GBM paths
    - Dynamic pruning and visibility updates
    - Path optimization and memory management
    - Performance benchmarks and acceleration
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or MeshConfig()
        self.mesh = nx.DiGraph()
        self.current_position: Optional[str] = None
        self.performance_metrics: Dict = {}
        
        # Initialize acceleration
        self.use_acceleration = self.config.use_acceleration and ACCELERATION_AVAILABLE
        if self.use_acceleration:
            print(f"🚀 Using {'Metal' if METAL_AVAILABLE else 'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
    
    def initialize_mesh(self, initial_state: Dict[str, float], 
                       milestones: List, time_horizon_years: float = None) -> Dict:
        """
        Initialize the stochastic mesh
        
        Args:
            initial_state: Initial financial state
            milestones: List of financial milestones
            time_horizon_years: Time horizon for mesh
            
        Returns:
            Mesh status dictionary
        """
        if time_horizon_years:
            self.config.time_horizon_years = time_horizon_years
        
        print(f"Initializing mesh with {len(milestones)} milestones...")
        
        # Create initial node
        initial_node = MeshNode(
            node_id="mesh_0_0",
            timestamp=datetime.now(),
            financial_state=initial_state.copy(),
            probability=1.0,
            visibility_radius=self.config.visibility_radius
        )
        
        self.mesh.add_node(initial_node.node_id, **initial_node.__dict__)
        self.current_position = initial_node.node_id
        
        # Generate optimized mesh structure
        self._generate_optimized_mesh(milestones)
        
        print(f"Mesh initialized with {len(self.mesh)} nodes")
        return self.get_mesh_status()
    
    def _generate_optimized_mesh(self, milestones: List):
        """Generate optimized mesh structure"""
        # Generate stochastic paths
        initial_value = sum(self.mesh.nodes[self.current_position]['financial_state'].values())
        paths = self._generate_paths(initial_value)
        
        # Convert paths to mesh structure
        self._paths_to_mesh(paths, milestones)
        
        # Optimize mesh
        self._optimize_mesh_structure()
    
    def _generate_paths(self, initial_value: float) -> np.ndarray:
        """Generate stochastic paths using GBM"""
        if self.use_acceleration:
            return self._generate_paths_accelerated(initial_value)
        else:
            return self._generate_paths_cpu(initial_value)
    
    def _generate_paths_accelerated(self, initial_value: float) -> np.ndarray:
        """Generate paths using GPU acceleration"""
        paths = np.zeros((self.config.num_paths, self.config.num_steps + 1), dtype=np.float32)
        paths[:, 0] = initial_value
        random_shocks = np.random.normal(0, 1, (self.config.num_paths, self.config.num_steps)).astype(np.float32)
        
        with torch.no_grad():
            paths_tensor = torch.tensor(paths, device=device, dtype=torch.float32)
            random_shocks_tensor = torch.tensor(random_shocks, device=device, dtype=torch.float32)
            
            drift = torch.tensor(self.config.drift, device=device, dtype=torch.float32)
            volatility = torch.tensor(self.config.volatility, device=device, dtype=torch.float32)
            dt = torch.tensor(self.config.dt, device=device, dtype=torch.float32)
            
            for t in range(1, paths.shape[1]):
                paths_tensor[:, t] = paths_tensor[:, t-1] * torch.exp(
                    (drift - 0.5 * volatility**2) * dt + 
                    volatility * torch.sqrt(dt) * random_shocks_tensor[:, t-1]
                )
            
            return paths_tensor.cpu().numpy().astype(np.float32)
    
    def _generate_paths_cpu(self, initial_value: float) -> np.ndarray:
        """Generate paths using CPU"""
        paths = np.zeros((self.config.num_paths, self.config.num_steps + 1), dtype=np.float32)
        paths[:, 0] = initial_value
        random_shocks = np.random.normal(0, 1, (self.config.num_paths, self.config.num_steps)).astype(np.float32)
        
        for t in range(1, self.config.num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.config.drift - 0.5 * self.config.volatility**2) * self.config.dt + 
                self.config.volatility * np.sqrt(self.config.dt) * random_shocks[:, t-1]
            )
        
        return paths
    
    def _paths_to_mesh(self, paths: np.ndarray, milestones: List):
        """Convert stochastic paths to mesh structure"""
        current_time = datetime.now()
        
        for path_idx in range(self.config.num_paths):
            for step_idx in range(self.config.num_steps + 1):
                # Create node for this path and time step
                node_id = f"mesh_{path_idx}_{step_idx}"
                timestamp = current_time + timedelta(days=step_idx * 30)  # Monthly steps
                
                # Calculate financial state based on path value
                path_value = paths[path_idx, step_idx]
                financial_state = self._calculate_financial_state(path_value, step_idx)
                
                # Create mesh node
                node = MeshNode(
                    node_id=node_id,
                    timestamp=timestamp,
                    financial_state=financial_state,
                    probability=1.0 / self.config.num_paths,  # Equal probability for now
                    visibility_radius=self.config.visibility_radius
                )
                
                self.mesh.add_node(node_id, **node.__dict__)
                
                # Connect to previous node in same path
                if step_idx > 0:
                    prev_node_id = f"mesh_{path_idx}_{step_idx - 1}"
                    if prev_node_id in self.mesh:
                        self.mesh.add_edge(prev_node_id, node_id)
                        self.mesh.nodes[node_id]['parent_nodes'].append(prev_node_id)
                        self.mesh.nodes[prev_node_id]['child_nodes'].append(node_id)
    
    def _calculate_financial_state(self, path_value: float, step_idx: int) -> Dict[str, float]:
        """Calculate financial state based on path value and time step"""
        # Simple allocation model
        total_wealth = path_value
        cash_ratio = max(0.05, 0.2 - step_idx * 0.01)  # Decreasing cash over time
        investment_ratio = 1.0 - cash_ratio
        
        return {
            'total_wealth': total_wealth,
            'cash': total_wealth * cash_ratio,
            'investments': total_wealth * investment_ratio,
            'debt': 0.0,  # Simplified for now
            'income': 150000 / 12,  # Monthly income
            'expenses': 60000 / 12   # Monthly expenses
        }
    
    def _optimize_mesh_structure(self):
        """Optimize mesh structure for performance"""
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.mesh))
        self.mesh.remove_nodes_from(isolated_nodes)
        
        # Prune low-probability paths
        self._prune_low_probability_paths()
        
        # Update visibility
        self._update_visibility()
    
    def _prune_low_probability_paths(self, threshold: float = 0.01):
        """Prune paths with low probability"""
        nodes_to_remove = []
        
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            if node_data['probability'] < threshold and len(node_data['child_nodes']) == 0:
                nodes_to_remove.append(node_id)
        
        self.mesh.remove_nodes_from(nodes_to_remove)
    
    def _update_visibility(self):
        """Update visibility radius for all nodes"""
        current_time = datetime.now()
        
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            time_diff = (node_data['timestamp'] - current_time).days
            
            if time_diff > self.config.visibility_radius:
                node_data['visibility_radius'] = max(0, self.config.visibility_radius - time_diff)
    
    def execute_payment(self, milestone_id: str, amount: float, 
                       payment_date: datetime = None) -> bool:
        """
        Execute a payment and update mesh accordingly
        
        Args:
            milestone_id: ID of the milestone to pay
            amount: Payment amount
            payment_date: Date of payment
            
        Returns:
            Success status
        """
        if not payment_date:
            payment_date = datetime.now()
        
        # Find nodes that need to be updated
        nodes_to_update = self._find_nodes_for_payment(milestone_id, payment_date)
        
        if not nodes_to_update:
            return False
        
        # Update financial states
        for node_id in nodes_to_update:
            self._update_node_financial_state(node_id, amount)
        
        # Update mesh structure
        self._update_mesh_after_payment(milestone_id, amount, payment_date)
        
        return True
    
    def _find_nodes_for_payment(self, milestone_id: str, payment_date: datetime) -> List[str]:
        """Find nodes that need to be updated for a payment"""
        nodes_to_update = []
        
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            if node_data['timestamp'] >= payment_date:
                nodes_to_update.append(node_id)
        
        return nodes_to_update
    
    def _update_node_financial_state(self, node_id: str, amount: float):
        """Update financial state of a node after payment"""
        node_data = self.mesh.nodes[node_id]
        financial_state = node_data['financial_state']
        
        # Reduce cash by payment amount
        financial_state['cash'] = max(0, financial_state['cash'] - amount)
        financial_state['total_wealth'] = financial_state['cash'] + financial_state['investments']
    
    def _update_mesh_after_payment(self, milestone_id: str, amount: float, payment_date: datetime):
        """Update mesh structure after payment"""
        # Solidify current path
        self._solidify_current_path()
        
        # Prune past paths
        self._prune_past_paths()
        
        # Update future probabilities
        self._update_future_probabilities(milestone_id, amount, payment_date)
        
        # Update visibility
        self._update_visibility_after_payment()
    
    def _solidify_current_path(self):
        """Mark current path as solidified"""
        if self.current_position:
            self.mesh.nodes[self.current_position]['is_solidified'] = True
    
    def _prune_past_paths(self):
        """Remove solidified paths that are no longer needed"""
        nodes_to_remove = []
        
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            if (node_data['is_solidified'] and 
                len(node_data['child_nodes']) == 0 and
                node_id != self.current_position):
                nodes_to_remove.append(node_id)
        
        self.mesh.remove_nodes_from(nodes_to_remove)
    
    def _update_future_probabilities(self, milestone_id: str, amount: float, payment_date: datetime):
        """Update probabilities of future nodes after payment"""
        # Simple probability update - could be more sophisticated
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            if node_data['timestamp'] > payment_date:
                # Adjust probability based on payment impact
                node_data['probability'] *= 0.95  # Slight reduction due to payment
    
    def _update_visibility_after_payment(self):
        """Update visibility after payment"""
        # Increase visibility for nodes near payment date
        payment_time = datetime.now()
        
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            time_diff = abs((node_data['timestamp'] - payment_time).days)
            
            if time_diff < 365:  # Within 1 year
                node_data['visibility_radius'] = min(
                    self.config.visibility_radius,
                    node_data['visibility_radius'] * 1.2
                )
    
    def get_payment_options(self, milestone_id: str = None) -> Dict[str, List[Dict]]:
        """Get available payment options"""
        if not self.current_position:
            return {}
        
        current_node = self.mesh.nodes[self.current_position]
        financial_state = current_node['financial_state']
        
        available_cash = financial_state['cash']
        
        options = {
            'immediate_payment': {
                'amount': min(available_cash, 10000),  # Example amount
                'description': 'Pay immediately with available cash',
                'impact': 'Reduces cash reserves'
            },
            'partial_payment': {
                'amount': available_cash * 0.5,
                'description': 'Pay 50% of available cash',
                'impact': 'Moderate cash reduction'
            },
            'deferred_payment': {
                'amount': 0,
                'description': 'Defer payment to future',
                'impact': 'No immediate cash impact'
            }
        }
        
        return {'options': list(options.values())}
    
    def advance_time(self, new_timestamp: datetime):
        """Advance mesh time to new timestamp"""
        # Update current position based on time
        for node_id in self.mesh.nodes():
            node_data = self.mesh.nodes[node_id]
            if abs((node_data['timestamp'] - new_timestamp).days) < 30:  # Within 1 month
                self.current_position = node_id
                break
        
        # Update visibility for all nodes
        self._update_visibility()
    
    def get_mesh_status(self) -> Dict:
        """Get current mesh status"""
        if not self.current_position:
            return {'status': 'not_initialized'}
        
        current_node = self.mesh.nodes[self.current_position]
        
        return {
            'status': 'active',
            'total_nodes': len(self.mesh),
            'current_position': self.current_position,
            'current_timestamp': current_node['timestamp'].isoformat(),
            'financial_state': current_node['financial_state'],
            'probability': current_node['probability'],
            'visibility_radius': current_node['visibility_radius'],
            'is_solidified': current_node['is_solidified'],
            'performance_metrics': self.performance_metrics
        }
    
    def export_mesh_state(self, filepath: str):
        """Export mesh state to file"""
        mesh_data = {
            'config': self.config.__dict__,
            'nodes': dict(self.mesh.nodes(data=True)),
            'edges': list(self.mesh.edges()),
            'current_position': self.current_position,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(mesh_data, f, default=str, indent=2)
    
    def benchmark_performance(self) -> Dict:
        """Run performance benchmarks"""
        import time
        
        start_time = time.time()
        
        # Benchmark path generation
        initial_value = 100000
        paths = self._generate_paths(initial_value)
        
        path_time = time.time() - start_time
        
        # Benchmark mesh operations
        start_time = time.time()
        self._optimize_mesh_structure()
        optimization_time = time.time() - start_time
        
        self.performance_metrics = {
            'path_generation_time': path_time,
            'optimization_time': optimization_time,
            'total_nodes': len(self.mesh),
            'acceleration_used': self.use_acceleration
        }
        
        return self.performance_metrics 