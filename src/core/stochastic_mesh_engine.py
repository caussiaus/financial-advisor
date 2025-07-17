"""
Core Omega mesh engine with optimized path generation and state tracking.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
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
        # Use Metal for M1/M2 Macs
        METAL_AVAILABLE = torch.backends.mps.is_available()
        if METAL_AVAILABLE:
            device = torch.device("mps")
            # Set default tensor type to float32 for Metal
            torch.set_default_dtype(torch.float32)
        else:
            device = torch.device("cpu")
        CUDA_AVAILABLE = False
    else:
        # Try CUDA for other systems
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
    print("GPU acceleration not available. Installing PyTorch may improve performance.")

from .mesh_memory_manager import MeshMemoryManager, CompressedNode
from .adaptive_mesh_generator import AdaptiveMeshGenerator
from .vectorized_accounting import VectorizedAccountingEngine, AccountingState

def accelerated_process_paths(paths, drift, volatility, dt, random_shocks):
    """Process paths using available acceleration (Metal/CUDA/CPU)"""
    if not ACCELERATION_AVAILABLE:
        return paths
        
    with torch.no_grad():
        # Convert to torch tensors with float32
        paths_tensor = torch.tensor(paths, device=device, dtype=torch.float32)
        random_shocks_tensor = torch.tensor(random_shocks, device=device, dtype=torch.float32)
        
        # Convert scalar parameters to float32
        drift = torch.tensor(drift, device=device, dtype=torch.float32)
        volatility = torch.tensor(volatility, device=device, dtype=torch.float32)
        dt = torch.tensor(dt, device=device, dtype=torch.float32)
        
        # Calculate in parallel
        for t in range(1, paths.shape[1]):
            paths_tensor[:, t] = paths_tensor[:, t-1] * torch.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * torch.sqrt(dt) * random_shocks_tensor[:, t-1]
            )
        
        # Move back to CPU and convert to numpy
        return paths_tensor.cpu().numpy().astype(np.float32)

@dataclass
class OmegaNode:
    """
    Represents a node in the Omega mesh - a potential financial state at a given time
    """
    node_id: str
    timestamp: datetime
    financial_state: Dict[str, float]  # Cash, investments, debts, etc.
    probability: float
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    event_triggers: List[str] = field(default_factory=list)
    payment_opportunities: Dict[str, Dict] = field(default_factory=dict)
    visibility_radius: float = 1.0  # How far into future this node can "see"
    is_solidified: bool = False  # Whether this path has been actualized

class StochasticMeshEngine:
    """
    The core Omega mesh engine that creates and manages the continuous stochastic process
    for financial milestones and payment structures
    """
    
    def __init__(self, current_financial_state: Dict[str, float]):
        self.current_state = {k: float(v) for k, v in current_financial_state.items()}  # Ensure float32
        self.omega_mesh = nx.DiGraph()
        self.memory_manager = MeshMemoryManager()
        self.adaptive_generator = AdaptiveMeshGenerator(
            self.current_state,
            self.memory_manager
        )
        self.accounting_engine = VectorizedAccountingEngine()
        self.current_position = None
        self.use_acceleration = ACCELERATION_AVAILABLE
        self.current_visibility_radius = 365 * 5  # 5 years visibility
        
        if self.use_acceleration:
            print(f"🚀 Using {'Metal' if METAL_AVAILABLE else 'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
            
    def _generate_paths_accelerated(self, num_paths: int, num_steps: int, 
                                  initial_value: float, drift: float, 
                                  volatility: float, dt: float) -> np.ndarray:
        """Generate paths using available acceleration"""
        if not self.use_acceleration:
            return self._generate_paths_cpu(num_paths, num_steps, initial_value, 
                                          drift, volatility, dt)
        
        # Initialize arrays with float32
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        paths[:, 0] = initial_value
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps)).astype(np.float32)
        
        # Process using acceleration
        return accelerated_process_paths(paths, drift, volatility, dt, random_shocks)
        
    def _generate_paths_cpu(self, num_paths: int, num_steps: int,
                          initial_value: float, drift: float,
                          volatility: float, dt: float) -> np.ndarray:
        """Fallback CPU implementation for path generation"""
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        paths[:, 0] = initial_value
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps)).astype(np.float32)
        
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        return paths

    def _process_states_parallel(self, states: List[Dict[str, float]], 
                               operations: List[Dict]) -> List[Dict[str, float]]:
        """Process multiple states in parallel using available hardware"""
        if self.use_acceleration:
            # Convert to tensors with float32
            state_list = [[float(v) for v in state.values()] for state in states]
            state_tensor = torch.tensor(state_list, device=device, dtype=torch.float32)
            
            # Process operations
            with torch.no_grad():
                for op in operations:
                    if op['type'] == 'investment_growth':
                        state_tensor[:, 2] *= (1 + float(op['return']))  # Assuming index 2 is investments
                
                # Convert back to CPU and restructure
                result_array = state_tensor.cpu().numpy()
                return [dict(zip(states[0].keys(), row)) for row in result_array]
        else:
            # Use CPU parallel processing
            with ThreadPoolExecutor() as executor:
                return list(executor.map(
                    lambda s: self.accounting_engine.process_single_state(s, operations),
                    states
                ))
    
    def initialize_mesh(self, milestones: List, time_horizon_years: float = 10):
        """
        Initialize the Omega mesh with initial conditions and milestones
        """
        print(f"Initializing Omega mesh with {len(milestones)} milestones...")
        
        # Create initial node
        initial_node = OmegaNode(
            node_id="omega_0_0",
            timestamp=datetime.now(),
            financial_state=self.current_state.copy(),  # Make sure to copy
            probability=1.0,
            visibility_radius=self.current_visibility_radius
        )
        
        # Store initial node
        self.memory_manager.store_node(initial_node)
        self.omega_mesh.add_node(initial_node.node_id)
        self.current_position = initial_node.node_id
        
        # Generate optimized mesh structure
        self._generate_optimized_mesh(milestones, time_horizon_years)
        
        print(f"Omega mesh initialized with {len(self.omega_mesh)} nodes")
        return self.get_mesh_status()
    
    def _generate_optimized_mesh(self, milestones: List, time_horizon_years: float):
        """
        Generate optimized mesh structure using adaptive techniques
        """
        # Get critical dates and cluster milestones
        milestone_clusters = self.adaptive_generator._get_critical_dates(milestones)
        
        # Generate important paths
        paths = self.adaptive_generator._generate_important_paths(
            milestone_clusters,
            num_paths=500  # Reduced from 1000
        )
        
        # Refine high-value paths
        refined_paths = self.adaptive_generator._refine_high_value_paths(
            paths, milestone_clusters
        )
        
        # Aggregate similar states
        unique_states = self.adaptive_generator._aggregate_similar_states(refined_paths)
        
        # Convert paths to mesh
        self._paths_to_mesh(refined_paths, milestones)
    
    def _paths_to_mesh(self, paths: List[np.ndarray], milestones: List):
        """
        Convert optimized paths to mesh structure
        """
        dt = 1.0 / 12.0  # Monthly steps instead of daily
        current_time = datetime.now()
        
        # Check if paths exist
        if not paths or len(paths) == 0:
            print("⚠️ No paths generated for mesh. Creating default mesh structure.")
            self._create_default_mesh(milestones)
            return
        
        # Create nodes for each unique state
        for month in range(int(len(paths[0]) * dt)):  # Convert to months
            current_time = datetime.now() + timedelta(days=30 * month)
            
            # Get unique states for this time point
            month_idx = int(month / dt)
            if month_idx >= len(paths[0]):
                break
                
            states_at_t = np.unique(
                [path[month_idx] for path in paths],
                return_counts=True
            )
            
            # Limit number of states per time point
            max_states = 10  # Reduced from default
            if len(states_at_t[0]) > max_states:
                # Keep most common states
                indices = np.argsort(states_at_t[1])[-max_states:]
                states_at_t = (states_at_t[0][indices], states_at_t[1][indices])
            
            # Create nodes for unique states
            for state_idx, (state_value, count) in enumerate(zip(*states_at_t)):
                node_id = f"omega_{month}_{state_idx}"
                
                # Calculate probability based on frequency
                probability = count / len(paths)
                
                # Create financial state
                total_wealth = state_value
                financial_state = {
                    'total_wealth': total_wealth,
                    'cash': total_wealth * 0.2,  # Example allocation
                    'investments': total_wealth * 0.8
                }
                
                # Create node
                node = OmegaNode(
                    node_id=node_id,
                    timestamp=current_time,
                    financial_state=financial_state,
                    probability=probability,
                    visibility_radius=max(0, 365 * 5 - month * 30)  # Decreasing visibility
                )
                
                # Store node efficiently
                self.memory_manager.store_node(node)
                self.omega_mesh.add_node(node_id)
                
                # Connect to previous layer
                if month > 0:
                    self._connect_to_previous_layer(node_id, month-1, state_value)
    
    def _connect_to_previous_layer(self, node_id: str, prev_month: int, state_value: float):
        """
        Connect node to appropriate nodes in previous layer
        """
        # Find potential parent nodes
        prev_nodes = [
            n for n in self.omega_mesh.nodes()
            if n.startswith(f"omega_{prev_month}_")
        ]
        
        for prev_node_id in prev_nodes:
            prev_node = self.memory_manager.batch_retrieve([prev_node_id])[0]
            if prev_node:
                # Check if connection is reasonable
                prev_value = prev_node.financial_state['total_wealth']
                if abs(state_value - prev_value) / prev_value < 0.2:  # Increased threshold
                    self.omega_mesh.add_edge(prev_node_id, node_id)
                    
                    # Update node lists
                    node = self.memory_manager.batch_retrieve([node_id])[0]
                    if node:
                        node.parent_nodes.append(prev_node_id)
                        prev_node.child_nodes.append(node_id)
                        
                        # Update stored nodes
                        self.memory_manager.store_node(node)
                        self.memory_manager.store_node(prev_node)
    
    def execute_payment(self, milestone_id: str, amount: float, payment_date: datetime = None) -> bool:
        """
        Execute a flexible payment and update the mesh
        """
        if payment_date is None:
            payment_date = datetime.now()
        
        print(f"Executing payment: ${amount} for {milestone_id} on {payment_date}")
        
        # Get current node
        current_node = self.memory_manager.batch_retrieve([self.current_position])[0]
        if not current_node:
            return False
        
        # Check if payment is possible
        available_funds = current_node.financial_state.get('total_wealth', 0)
        if amount > available_funds:
            print(f"Insufficient funds: ${available_funds} available, ${amount} requested")
            return False
        
        # Create accounting state
        state = AccountingState(
            cash=amount,
            investments=current_node.financial_state.get('investments', {}),
            debts=current_node.financial_state.get('debts', {}),
            timestamp=np.datetime64(payment_date)
        )
        
        # Process payment through accounting engine
        operation = {
            'type': 'custom_transfer',
            'from_account': 'cash',
            'to_account': milestone_id,
            'amount': amount
        }
        
        # Process state
        processed_state = self.accounting_engine.batch_process_states(
            [state],
            [operation],
            time_horizon=1.0
        )[0]
        
        # Update node state
        current_node.financial_state.update(processed_state.__dict__)
        current_node.is_solidified = True
        
        # Store updated node
        self.memory_manager.store_node(current_node)
        
        # Update mesh
        self._update_mesh_after_payment(milestone_id, amount, payment_date)
        
        print(f"Payment executed successfully. New wealth: ${current_node.financial_state['total_wealth']}")
        return True
    
    def _update_mesh_after_payment(self, milestone_id: str, amount: float, payment_date: datetime):
        """
        Update mesh structure after payment execution
        """
        # Solidify current path
        self._solidify_current_path()
        
        # Prune impossible past paths
        self._prune_past_paths()
        
        # Update future probabilities
        self._update_future_probabilities(milestone_id, amount, payment_date)
        
        # Update visibility
        self._update_visibility_after_payment()
    
    def _solidify_current_path(self):
        """Mark the current path as actualized and remove alternative pasts"""
        current_node = self.memory_manager.batch_retrieve([self.current_position])[0]
        
        # Trace back to solidify the path
        path_to_solidify = []
        queue = [self.current_position]
        
        while queue:
            node_id = queue.pop(0)
            node = self.memory_manager.batch_retrieve([node_id])[0]
            if node and not node.is_solidified:
                node.is_solidified = True
                path_to_solidify.append(node_id)
                queue.extend(node.parent_nodes)
    
    def _prune_past_paths(self):
        """Remove past alternative paths that are no longer possible"""
        nodes_to_remove = []
        
        for node_id, node in self.memory_manager.batch_retrieve(list(self.omega_mesh.nodes())):
            if node and node.timestamp < datetime.now() and not node.is_solidified:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self._remove_node_and_connections(node_id)
    
    def _update_future_probabilities(self, milestone_id: str, amount: float, payment_date: datetime):
        """Update probabilities of future nodes based on the payment made"""
        for node_id, node in self.memory_manager.batch_retrieve(list(self.omega_mesh.nodes())):
            if node and node.timestamp > datetime.now():
                # Recalculate probability based on the payment
                if milestone_id in node.payment_opportunities:
                    remaining = node.payment_opportunities[milestone_id]['remaining_amount']
                    if remaining <= 0:
                        # Milestone fully paid, increase probability of success paths
                        node.probability *= 1.2
                    else:
                        # Partial payment, moderate increase
                        payment_ratio = amount / node.payment_opportunities[milestone_id]['target_amount']
                        node.probability *= (1 + payment_ratio * 0.5)
    
    def _update_visibility_after_payment(self):
        """Update visibility radius after a payment solidifies the position"""
        current_node = self.memory_manager.batch_retrieve([self.current_position])[0]
        
        # Increase visibility radius due to reduced uncertainty
        current_node.visibility_radius *= 1.1
        
        # Update visibility for connected nodes
        for child_id in current_node.child_nodes:
            child_node = self.memory_manager.batch_retrieve([child_id])[0]
            if child_node:
                days_ahead = (child_node.timestamp - datetime.now()).days
                
                # Nodes further away become less visible
                visibility_reduction = np.exp(-days_ahead / current_node.visibility_radius)
                
                if visibility_reduction < 0.01:  # Remove nodes that are barely visible
                    self._remove_node_and_connections(child_id)
    
    def _create_default_mesh(self, milestones: List):
        """Creates a default mesh structure when no paths are generated"""
        print("🔄 Creating default mesh structure...")
        
        # Create a simple linear mesh with basic financial states
        current_time = datetime.now()
        initial_wealth = self.current_state.get('total_wealth', 1000000)
        
        for month in range(12):  # 12efault
            current_time = datetime.now() + timedelta(days=30 * month)
            
            # Create a simple node with basic financial state
            node_id = f"default_omega_{month}"
            
            # Simple wealth growth model
            growth_rate =0.05#5growth
            monthly_growth = (1 + growth_rate) ** (month / 12)
            wealth = initial_wealth * monthly_growth
            
            financial_state = {
                'total_wealth': wealth,
                'cash': wealth * 0.2,
                'investments': wealth * 0.8
            }
            
            # Create node
            node = OmegaNode(
                node_id=node_id,
                timestamp=current_time,
                financial_state=financial_state,
                probability=1.0,  # Default probability
                visibility_radius=365 * 5 - month *30
            )
            
            # Store node
            self.memory_manager.store_node(node)
            self.omega_mesh.add_node(node_id)
            
            # Connect to previous node
            if month > 0:
                prev_node_id = f"default_omega_{month-1}"
                self.omega_mesh.add_edge(prev_node_id, node_id)
        
        print(f"✅ Created default mesh with {12} nodes")

    def _remove_node_and_connections(self, node_id: str):
        """Safely remove a node and all its connections"""
        node = self.memory_manager.batch_retrieve([node_id])[0]
        if node:
            
            # Remove connections
            for parent_id in node.parent_nodes:
                parent_node = self.memory_manager.batch_retrieve([parent_id])[0]
                if parent_node:
                    parent_node.child_nodes.remove(node_id)
            
            for child_id in node.child_nodes:
                child_node = self.memory_manager.batch_retrieve([child_id])[0]
                if child_node:
                    child_node.parent_nodes.remove(node_id)
            
            # Remove from graph and nodes dict
            if self.omega_mesh.has_node(node_id):
                self.omega_mesh.remove_node(node_id)
            
            del self.memory_manager.nodes[node_id] # Use memory_manager to remove
    
    def get_payment_options(self, milestone_id: str = None) -> Dict[str, List[Dict]]:
        """
        Get all available payment options from the current position
        Supports ultra-flexible payment structures as requested
        """
        current_node = self.memory_manager.batch_retrieve([self.current_position])[0]
        if not current_node:
            return {}
        
        available_cash = current_node.financial_state.get('total_wealth', 0)
        
        options = {}
        
        opportunities = current_node.payment_opportunities
        if milestone_id:
            opportunities = {milestone_id: opportunities.get(milestone_id, {})}
        
        for m_id, opportunity in opportunities.items():
            if opportunity.get('remaining_amount', 0) > 0:
                options[m_id] = []
                
                # Ultra-flexible payment options
                remaining = opportunity['remaining_amount']
                
                # 1% today option
                options[m_id].append({
                    'type': 'percentage_immediate',
                    'amount': remaining * 0.01,
                    'date': datetime.now(),
                    'description': '1% payment today'
                })
                
                # 11% next Tuesday option
                next_tuesday = self._get_next_tuesday()
                options[m_id].append({
                    'type': 'percentage_scheduled',
                    'amount': remaining * 0.11,
                    'date': next_tuesday,
                    'description': '11% payment next Tuesday'
                })
                
                # Grandmother's birthday option (placeholder date)
                grandma_birthday = datetime(datetime.now().year, 6, 15)  # Example date
                if grandma_birthday < datetime.now():
                    grandma_birthday = datetime(datetime.now().year + 1, 6, 15)
                
                remaining_after_partial = remaining * (1 - 0.01 - 0.11)
                options[m_id].append({
                    'type': 'custom_date',
                    'amount': remaining_after_partial,
                    'date': grandma_birthday,
                    'description': "Remaining balance on grandmother's birthday"
                })
                
                # Custom amount on any date
                options[m_id].append({
                    'type': 'fully_custom',
                    'amount': 'user_defined',
                    'date': 'user_defined',
                    'description': 'Any amount on any date'
                })
                
                # Milestone-triggered payment
                options[m_id].append({
                    'type': 'milestone_triggered',
                    'amount': remaining,
                    'trigger': 'achievement_condition',
                    'description': 'Pay when specific milestone is achieved'
                })
        
        return options
    
    def _get_next_tuesday(self) -> datetime:
        """Get the date of next Tuesday"""
        today = datetime.now()
        days_ahead = 1 - today.weekday()  # Tuesday is 1
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def get_mesh_status(self) -> Dict:
        """Get current status of the mesh"""
        total_nodes = len(self.omega_mesh)
        solidified_nodes = len([n for n in self.omega_mesh.nodes if self.memory_manager.batch_retrieve([n])[0].is_solidified])
        visible_nodes = len([n for n in self.omega_mesh.nodes if not self.memory_manager.batch_retrieve([n])[0].is_solidified])
        
        return {
            'total_nodes': total_nodes,
            'solidified_nodes': solidified_nodes,
            'visible_nodes': visible_nodes,
            'current_position': self.current_position,
            'current_visibility_radius': self.current_visibility_radius,
            'current_wealth': self.current_state.get('total_wealth', 0),
            'acceleration_type': 'Metal' if METAL_AVAILABLE else 'CUDA' if CUDA_AVAILABLE else 'CPU',
            'is_accelerated': self.use_acceleration
        }
    
    def advance_time(self, new_timestamp: datetime):
        """
        Advance the current position in time
        This triggers the mesh update where past omega disappears
        """
        # Find the closest node to the new timestamp
        best_node = None
        min_time_diff = float('inf')
        
        for node_id, node in self.memory_manager.batch_retrieve(list(self.omega_mesh.nodes())):
            if node and node.is_solidified and node.timestamp <= new_timestamp:
                time_diff = abs((node.timestamp - new_timestamp).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_node = node_id
        
        if best_node:
            self.current_position = best_node
            # Trigger mesh cleanup
            self._prune_past_paths()
            self._update_visibility_after_payment()
    
    def export_mesh_state(self, filepath: str):
        """Export the current mesh state to JSON"""
        export_data = {
            'current_position': self.current_position,
            'nodes': {},
            'edges': list(self.omega_mesh.edges(data=True)),
            'mesh_statistics': self.get_mesh_status(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Export node data
        for node_id, node in self.memory_manager.batch_retrieve(list(self.omega_mesh.nodes())):
            if node:
                export_data['nodes'][node_id] = {
                    'timestamp': node.timestamp.isoformat(),
                    'financial_state': node.financial_state,
                    'probability': node.probability,
                    'visibility_radius': node.visibility_radius,
                    'is_solidified': node.is_solidified,
                    'payment_opportunities': node.payment_opportunities
                }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    initial_state = {
        'total_wealth': 100000,
        'cash': 50000,
        'investments': 50000,
        'debts': 0
    }
    
    engine = StochasticMeshEngine(initial_state)
    
    # This would be called with actual milestones from the PDF processor
    engine.initialize_mesh([], time_horizon_years=5)
    
    print("Omega mesh initialized!")
    print(f"Mesh status: {engine.get_mesh_status()}")
    
    # Example payment execution
    payment_options = engine.get_payment_options()
    print(f"Available payment options: {len(payment_options)}")