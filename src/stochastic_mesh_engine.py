import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import networkx as nx
from scipy.stats import norm
import json
import math
from concurrent.futures import ThreadPoolExecutor
import asyncio


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


@dataclass
class StochasticPath:
    """
    Represents a potential path through the Omega mesh
    """
    path_id: str
    nodes: List[str]
    cumulative_probability: float
    total_value: float
    milestones_achieved: List[str]
    payment_schedule: List[Dict]
    path_efficiency: float


class GeometricBrownianMotionEngine:
    """
    Implements geometric Brownian motion for financial modeling
    """
    
    def __init__(self, initial_value: float, drift: float = 0.07, volatility: float = 0.15):
        self.initial_value = initial_value
        self.drift = drift  # Expected annual return
        self.volatility = volatility  # Annual volatility
    
    def simulate_path(self, time_horizon: float, num_steps: int, num_paths: int = 1000) -> np.ndarray:
        """
        Simulate GBM paths
        
        Args:
            time_horizon: Time in years
            num_steps: Number of time steps
            num_paths: Number of simulation paths
        
        Returns:
            Array of shape (num_paths, num_steps + 1) with simulated values
        """
        dt = time_horizon / num_steps
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps))
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.initial_value
        
        # Generate paths using GBM formula
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt + 
                self.volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        return paths
    
    def get_confidence_intervals(self, paths: np.ndarray, confidence_levels: List[float]) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals for the simulated paths"""
        intervals = {}
        for level in confidence_levels:
            lower_percentile = (1 - level) / 2 * 100
            upper_percentile = (1 + level) / 2 * 100
            
            lower_bound = np.percentile(paths, lower_percentile, axis=0)
            upper_bound = np.percentile(paths, upper_percentile, axis=0)
            
            intervals[level] = (lower_bound, upper_bound)
        
        return intervals


class StochasticMeshEngine:
    """
    The core Omega mesh engine that creates and manages the continuous stochastic process
    for financial milestones and payment structures
    """
    
    def __init__(self, current_financial_state: Dict[str, float]):
        self.current_state = current_financial_state
        self.omega_mesh = nx.DiGraph()  # Directed graph representing the mesh
        self.nodes = {}  # node_id -> OmegaNode mapping
        self.current_position = None  # Current node in the mesh
        self.gbm_engine = GeometricBrownianMotionEngine(
            initial_value=current_financial_state.get('total_wealth', 100000)
        )
        self.visibility_decay_rate = 0.1  # How quickly future visibility decays
        self.time_step_hours = 1  # Granularity of time steps
        self.max_lookahead_days = 365 * 5  # Maximum future visibility
        
    def initialize_mesh(self, milestones: List, time_horizon_years: float = 10):
        """
        Initialize the Omega mesh with initial conditions and milestones
        """
        print(f"Initializing Omega mesh with {len(milestones)} milestones...")
        
        # Create initial node
        initial_node = OmegaNode(
            node_id="omega_0_0",
            timestamp=datetime.now(),
            financial_state=self.current_state.copy(),
            probability=1.0,
            visibility_radius=self.max_lookahead_days
        )
        
        # Add payment opportunities to initial node
        initial_node.payment_opportunities = self._generate_payment_opportunities(
            datetime.now(), self.current_state, milestones
        )
        
        self.nodes[initial_node.node_id] = initial_node
        self.omega_mesh.add_node(initial_node.node_id, node_data=initial_node)
        self.current_position = initial_node.node_id
        
        # Generate the mesh structure
        self._generate_mesh_structure(milestones, time_horizon_years)
        
        print(f"Omega mesh initialized with {len(self.nodes)} nodes")
        
        # Return mesh status
        return self.get_mesh_status()
    
    def _generate_mesh_structure(self, milestones: List, time_horizon_years: float):
        """
        Generate the mesh structure with branching paths for different scenarios
        """
        num_time_steps = int(time_horizon_years * 365 * 24 / self.time_step_hours)
        num_scenario_branches = 50  # Number of scenario branches at each major decision point
        
        # Create time-based layers
        current_time = datetime.now()
        milestone_times = sorted([m.timestamp for m in milestones])
        
        # Generate GBM paths for wealth evolution
        gbm_paths = self.gbm_engine.simulate_path(
            time_horizon=time_horizon_years,
            num_steps=365 * int(time_horizon_years),  # Daily steps
            num_paths=num_scenario_branches
        )
        
        # Create nodes for each significant time point
        layer_nodes = [self.current_position]
        
        for day in range(1, min(365 * int(time_horizon_years), self.max_lookahead_days)):
            current_time += timedelta(days=1)
            
            # Create multiple scenario nodes for this time point
            next_layer_nodes = []
            
            for scenario_idx in range(min(num_scenario_branches, len(layer_nodes) * 3)):
                node_id = f"omega_{day}_{scenario_idx}"
                
                # Calculate financial state based on GBM and milestones
                financial_state = self._calculate_financial_state(
                    current_time, gbm_paths[scenario_idx % len(gbm_paths), day] if day < len(gbm_paths[0]) else gbm_paths[scenario_idx % len(gbm_paths), -1],
                    milestones
                )
                
                # Calculate probability based on path likelihood
                probability = self._calculate_node_probability(
                    current_time, scenario_idx, milestones
                )
                
                # Calculate visibility radius (decreases with time and uncertainty)
                visibility_radius = self._calculate_visibility_radius(current_time)
                
                # Create payment opportunities for this node
                payment_opportunities = self._generate_payment_opportunities(
                    current_time, financial_state, milestones
                )
                
                node = OmegaNode(
                    node_id=node_id,
                    timestamp=current_time,
                    financial_state=financial_state,
                    probability=probability,
                    visibility_radius=visibility_radius,
                    payment_opportunities=payment_opportunities
                )
                
                self.nodes[node_id] = node
                self.omega_mesh.add_node(node_id, node_data=node)
                next_layer_nodes.append(node_id)
                
                # Connect to previous layer nodes
                self._connect_to_previous_layer(node_id, layer_nodes, probability)
            
            # Prune low-probability nodes to keep mesh manageable
            next_layer_nodes = self._prune_low_probability_nodes(next_layer_nodes)
            layer_nodes = next_layer_nodes
            
            # Check if we should stop early due to computational limits
            if len(self.nodes) > 10000:  # Computational limit
                break
    
    def _calculate_financial_state(self, timestamp: datetime, wealth_value: float, milestones: List) -> Dict[str, float]:
        """Calculate financial state at a given time point"""
        state = self.current_state.copy()
        state['total_wealth'] = wealth_value
        
        # Apply milestone impacts
        for milestone in milestones:
            if milestone.timestamp <= timestamp and milestone.financial_impact:
                # Apply milestone impact with some random variation
                impact = milestone.financial_impact * (1 + np.random.normal(0, 0.1))
                if milestone.event_type in ['investment', 'income']:
                    state['total_wealth'] += impact
                else:
                    state['total_wealth'] -= impact
        
        # Ensure non-negative wealth
        state['total_wealth'] = max(0, state['total_wealth'])
        
        return state
    
    def _calculate_node_probability(self, timestamp: datetime, scenario_idx: int, milestones: List) -> float:
        """Calculate the probability of reaching this node"""
        base_probability = 1.0 / (scenario_idx + 1)  # Higher scenarios less likely
        
        # Adjust based on milestone probabilities
        for milestone in milestones:
            if abs((milestone.timestamp - timestamp).days) < 30:
                base_probability *= milestone.probability
        
        # Add time decay
        days_from_now = (timestamp - datetime.now()).days
        time_decay = np.exp(-self.visibility_decay_rate * days_from_now / 365)
        
        return base_probability * time_decay
    
    def _calculate_visibility_radius(self, timestamp: datetime) -> float:
        """Calculate how far this node can see into the future"""
        days_from_now = (timestamp - datetime.now()).days
        return max(1.0, self.max_lookahead_days - days_from_now) * np.exp(-days_from_now / 365)
    
    def _generate_payment_opportunities(self, timestamp: datetime, financial_state: Dict, milestones: List) -> Dict[str, Dict]:
        """Generate flexible payment opportunities for this node"""
        opportunities = {}
        
        available_cash = financial_state.get('total_wealth', 0)
        
        for milestone in milestones:
            if milestone.financial_impact and milestone.timestamp >= timestamp:
                # Generate flexible payment options
                milestone_id = f"{milestone.event_type}_{milestone.timestamp.year}"
                
                opportunities[milestone_id] = {
                    'target_amount': milestone.financial_impact,
                    'remaining_amount': milestone.financial_impact,
                    'min_payment': milestone.financial_impact * 0.01,  # 1% minimum
                    'max_payment': min(available_cash, milestone.financial_impact),
                    'flexible_dates': True,
                    'payment_methods': [
                        'percentage_based',
                        'custom_amount',
                        'milestone_triggered',
                        'date_specific'
                    ],
                    'deadline': milestone.timestamp,
                    'priority': milestone.probability
                }
        
        return opportunities
    
    def _connect_to_previous_layer(self, current_node_id: str, previous_layer: List[str], transition_probability: float):
        """Connect current node to appropriate nodes in the previous layer"""
        # Use probabilistic connection based on similarity and transition likelihood
        for prev_node_id in previous_layer:
            if np.random.random() < transition_probability:
                self.omega_mesh.add_edge(prev_node_id, current_node_id, weight=transition_probability)
                self.nodes[current_node_id].parent_nodes.append(prev_node_id)
                self.nodes[prev_node_id].child_nodes.append(current_node_id)
    
    def _prune_low_probability_nodes(self, nodes: List[str], probability_threshold: float = 0.001) -> List[str]:
        """Remove nodes with very low probability to keep mesh manageable"""
        return [node_id for node_id in nodes if self.nodes[node_id].probability > probability_threshold]
    
    def execute_payment(self, milestone_id: str, amount: float, payment_date: datetime = None) -> bool:
        """
        Execute a flexible payment and update the mesh accordingly
        
        This is where the magic happens - any payment structure is supported
        """
        if payment_date is None:
            payment_date = datetime.now()
        
        print(f"Executing payment: ${amount} for {milestone_id} on {payment_date}")
        
        # Find the current node and update its state
        current_node = self.nodes[self.current_position]
        
        # Check if payment is possible
        available_funds = current_node.financial_state.get('total_wealth', 0)
        if amount > available_funds:
            print(f"Insufficient funds: ${available_funds} available, ${amount} requested")
            return False
        
        # Execute the payment
        current_node.financial_state['total_wealth'] -= amount
        
        # Update payment opportunities
        if milestone_id in current_node.payment_opportunities:
            current_node.payment_opportunities[milestone_id]['remaining_amount'] -= amount
            current_node.payment_opportunities[milestone_id]['paid_amount'] = current_node.payment_opportunities[milestone_id].get('paid_amount', 0) + amount
        
        # Solidify this path - mark as actualized
        current_node.is_solidified = True
        
        # Prune impossible future paths and update probabilities
        self._update_mesh_after_payment(milestone_id, amount, payment_date)
        
        print(f"Payment executed successfully. New wealth: ${current_node.financial_state['total_wealth']}")
        return True
    
    def _update_mesh_after_payment(self, milestone_id: str, amount: float, payment_date: datetime):
        """
        Update the entire mesh after a payment is made
        This implements the core logic where past omega disappears and future visibility changes
        """
        # 1. Solidify the current path
        self._solidify_current_path()
        
        # 2. Prune impossible past paths
        self._prune_past_paths()
        
        # 3. Update future probabilities based on the payment
        self._update_future_probabilities(milestone_id, amount, payment_date)
        
        # 4. Reduce visibility for distant nodes
        self._update_visibility_after_payment()
    
    def _solidify_current_path(self):
        """Mark the current path as actualized and remove alternative pasts"""
        current_node = self.nodes[self.current_position]
        
        # Trace back to solidify the path
        path_to_solidify = []
        queue = [self.current_position]
        
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]
            if not node.is_solidified:
                node.is_solidified = True
                path_to_solidify.append(node_id)
                queue.extend(node.parent_nodes)
    
    def _prune_past_paths(self):
        """Remove past alternative paths that are no longer possible"""
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            # Remove past nodes that weren't solidified
            if node.timestamp < datetime.now() and not node.is_solidified:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self._remove_node_and_connections(node_id)
    
    def _update_future_probabilities(self, milestone_id: str, amount: float, payment_date: datetime):
        """Update probabilities of future nodes based on the payment made"""
        for node_id, node in self.nodes.items():
            if node.timestamp > datetime.now():
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
        current_node = self.nodes[self.current_position]
        
        # Increase visibility radius due to reduced uncertainty
        current_node.visibility_radius *= 1.1
        
        # Update visibility for connected nodes
        for child_id in current_node.child_nodes:
            child_node = self.nodes[child_id]
            days_ahead = (child_node.timestamp - datetime.now()).days
            
            # Nodes further away become less visible
            visibility_reduction = np.exp(-days_ahead / current_node.visibility_radius)
            
            if visibility_reduction < 0.01:  # Remove nodes that are barely visible
                self._remove_node_and_connections(child_id)
    
    def _remove_node_and_connections(self, node_id: str):
        """Safely remove a node and all its connections"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove connections
            for parent_id in node.parent_nodes:
                if parent_id in self.nodes:
                    self.nodes[parent_id].child_nodes.remove(node_id)
            
            for child_id in node.child_nodes:
                if child_id in self.nodes:
                    self.nodes[child_id].parent_nodes.remove(node_id)
            
            # Remove from graph and nodes dict
            if self.omega_mesh.has_node(node_id):
                self.omega_mesh.remove_node(node_id)
            
            del self.nodes[node_id]
    
    def get_payment_options(self, milestone_id: str = None) -> Dict[str, List[Dict]]:
        """
        Get all available payment options from the current position
        Supports ultra-flexible payment structures as requested
        """
        current_node = self.nodes[self.current_position]
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
        """Get current status of the Omega mesh"""
        total_nodes = len(self.nodes)
        solidified_nodes = sum(1 for node in self.nodes.values() if node.is_solidified)
        visible_nodes = sum(1 for node in self.nodes.values() if node.timestamp >= datetime.now())
        
        current_node = self.nodes[self.current_position]
        
        return {
            'total_nodes': total_nodes,
            'solidified_nodes': solidified_nodes,
            'visible_future_nodes': visible_nodes,
            'current_position': self.current_position,
            'current_wealth': current_node.financial_state.get('total_wealth', 0),
            'current_visibility_radius': current_node.visibility_radius,
            'available_opportunities': len(current_node.payment_opportunities),
            'mesh_connectivity': len(self.omega_mesh.edges())
        }
    
    def advance_time(self, new_timestamp: datetime):
        """
        Advance the current position in time
        This triggers the mesh update where past omega disappears
        """
        # Find the closest node to the new timestamp
        best_node = None
        min_time_diff = float('inf')
        
        for node_id, node in self.nodes.items():
            if node.is_solidified and node.timestamp <= new_timestamp:
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
        for node_id, node in self.nodes.items():
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