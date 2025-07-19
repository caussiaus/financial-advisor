#!/usr/bin/env python3
"""
Stochastic Mesh Engine
Core mesh engine for financial state space exploration and optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

@dataclass
class MeshNode:
    """Represents a node in the financial mesh"""
    node_id: str
    financial_state: Dict[str, float]
    timestamp: datetime
    probability: float = 0.0
    value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MeshEdge:
    """Represents an edge between mesh nodes"""
    edge_id: str
    from_node: str
    to_node: str
    action: Dict[str, float]
    probability: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class StochasticMeshEngine:
    """Stochastic mesh engine for financial state space exploration"""
    
    def __init__(self, current_financial_state: Optional[Dict[str, float]] = None):
        """Initialize the stochastic mesh engine"""
        self.current_financial_state = current_financial_state or {}
        self.nodes = {}
        self.edges = {}
        self.node_counter = 0
        self.edge_counter = 0
        
        # Mesh configuration
        self.max_nodes = 10000
        self.max_edges = 50000
        self.time_horizon_years = 10
        self.monthly_steps = self.time_horizon_years * 12
        
        # Initialize with current state
        if self.current_financial_state:
            self._add_initial_node()
        
        logger.info("✅ Stochastic Mesh Engine initialized")
    
    def _add_initial_node(self):
        """Add initial node based on current financial state"""
        initial_node = MeshNode(
            node_id=f"node_{self.node_counter:06d}",
            financial_state=self.current_financial_state.copy(),
            timestamp=datetime.now(),
            probability=1.0,
            value=self._calculate_node_value(self.current_financial_state)
        )
        
        self.nodes[initial_node.node_id] = initial_node
        self.node_counter += 1
        
        logger.info(f"✅ Added initial node: {initial_node.node_id}")
    
    def _calculate_node_value(self, financial_state: Dict[str, float]) -> float:
        """Calculate the value of a financial state"""
        # Simple value calculation based on net worth
        assets = sum(v for k, v in financial_state.items() if 'asset' in k.lower() or k in ['cash', 'investments'])
        liabilities = sum(v for k, v in financial_state.items() if 'debt' in k.lower() or k in ['mortgage', 'loans'])
        
        return assets - liabilities
    
    def initialize_mesh(self, milestones: List[Dict], time_horizon_years: int = 10) -> Dict[str, Any]:
        """Initialize the mesh with milestones and time horizon"""
        self.time_horizon_years = time_horizon_years
        self.monthly_steps = time_horizon_years * 12
        
        # Process milestones
        processed_milestones = self._process_milestones(milestones)
        
        # Generate mesh nodes
        self._generate_mesh_nodes(processed_milestones)
        
        # Generate mesh edges
        self._generate_mesh_edges()
        
        # Calculate probabilities
        self._calculate_probabilities()
        
        mesh_status = {
            "status": "active",
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "time_horizon_years": time_horizon_years,
            "milestones_processed": len(processed_milestones),
            "initialized_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Mesh initialized: {mesh_status['total_nodes']} nodes, {mesh_status['total_edges']} edges")
        return mesh_status
    
    def _process_milestones(self, milestones: List[Dict]) -> List[Dict]:
        """Process and validate milestones"""
        processed_milestones = []
        
        for milestone in milestones:
            # Ensure milestone has required fields
            processed_milestone = {
                "id": milestone.get("id", f"milestone_{len(processed_milestones)}"),
                "type": milestone.get("type", "unknown"),
                "amount": milestone.get("amount", 0.0),
                "probability": milestone.get("probability", 0.5),
                "year": milestone.get("year", 1),
                "month": milestone.get("month", 1),
                "description": milestone.get("description", ""),
                "impact": milestone.get("impact", {})
            }
            
            processed_milestones.append(processed_milestone)
        
        return processed_milestones
    
    def _generate_mesh_nodes(self, milestones: List[Dict]):
        """Generate mesh nodes based on milestones and time horizon"""
        current_state = self.current_financial_state.copy()
        
        for month in range(self.monthly_steps):
            # Apply monthly changes
            current_state = self._apply_monthly_changes(current_state, month)
            
            # Apply milestone effects
            for milestone in milestones:
                if self._should_apply_milestone(milestone, month):
                    current_state = self._apply_milestone_effect(current_state, milestone)
            
            # Create node for this month
            node = MeshNode(
                node_id=f"node_{self.node_counter:06d}",
                financial_state=current_state.copy(),
                timestamp=datetime.now() + timedelta(days=month * 30),
                probability=self._calculate_node_probability(current_state, month),
                value=self._calculate_node_value(current_state)
            )
            
            self.nodes[node.node_id] = node
            self.node_counter += 1
            
            # Check if we've reached the node limit
            if len(self.nodes) >= self.max_nodes:
                logger.warning(f"⚠️ Reached maximum node limit: {self.max_nodes}")
                break
    
    def _apply_monthly_changes(self, state: Dict[str, float], month: int) -> Dict[str, float]:
        """Apply monthly changes to financial state"""
        new_state = state.copy()
        
        # Apply investment returns
        if "investments" in new_state:
            monthly_return = np.random.normal(0.005, 0.02)  # 0.5% monthly return with volatility
            new_state["investments"] *= (1 + monthly_return)
        
        # Apply income
        if "income" in new_state:
            monthly_income = new_state["income"] / 12
            new_state["cash"] = new_state.get("cash", 0) + monthly_income
        
        # Apply expenses
        if "expenses" in new_state:
            monthly_expenses = new_state["expenses"] / 12
            new_state["cash"] = max(0, new_state.get("cash", 0) - monthly_expenses)
        
        # Apply debt payments
        if "mortgage" in new_state:
            monthly_payment = new_state["mortgage"] * 0.005  # 0.5% monthly payment
            new_state["mortgage"] = max(0, new_state["mortgage"] - monthly_payment)
            new_state["cash"] = max(0, new_state.get("cash", 0) - monthly_payment)
        
        return new_state
    
    def _should_apply_milestone(self, milestone: Dict, month: int) -> bool:
        """Check if milestone should be applied in given month"""
        milestone_month = (milestone["year"] - 1) * 12 + milestone["month"] - 1
        return month == milestone_month and random.random() < milestone["probability"]
    
    def _apply_milestone_effect(self, state: Dict[str, float], milestone: Dict) -> Dict[str, float]:
        """Apply milestone effect to financial state"""
        new_state = state.copy()
        impact = milestone["impact"]
        
        # Apply impact to relevant state variables
        for key, value in impact.items():
            if key in new_state:
                new_state[key] += value
            else:
                new_state[key] = value
        
        return new_state
    
    def _calculate_node_probability(self, state: Dict[str, float], month: int) -> float:
        """Calculate probability of reaching this node"""
        # Base probability decreases with time
        base_probability = 1.0 - (month / self.monthly_steps) * 0.3
        
        # Adjust based on financial health
        net_worth = self._calculate_node_value(state)
        if net_worth < 0:
            base_probability *= 0.8  # Lower probability for negative net worth
        
        return max(0.01, base_probability)
    
    def _generate_mesh_edges(self):
        """Generate edges between mesh nodes"""
        node_ids = list(self.nodes.keys())
        
        for i, from_node_id in enumerate(node_ids[:-1]):
            from_node = self.nodes[from_node_id]
            to_node_id = node_ids[i + 1]
            to_node = self.nodes[to_node_id]
            
            # Create edge
            edge = MeshEdge(
                edge_id=f"edge_{self.edge_counter:06d}",
                from_node=from_node_id,
                to_node=to_node_id,
                action=self._calculate_transition_action(from_node, to_node),
                probability=self._calculate_edge_probability(from_node, to_node),
                cost=self._calculate_edge_cost(from_node, to_node)
            )
            
            self.edges[edge.edge_id] = edge
            self.edge_counter += 1
            
            # Check if we've reached the edge limit
            if len(self.edges) >= self.max_edges:
                logger.warning(f"⚠️ Reached maximum edge limit: {self.max_edges}")
                break
    
    def _calculate_transition_action(self, from_node: MeshNode, to_node: MeshNode) -> Dict[str, float]:
        """Calculate the action that leads from one node to another"""
        from_state = from_node.financial_state
        to_state = to_node.financial_state
        
        action = {
            "invest": 0.0,
            "save": 0.0,
            "spend": 0.0,
            "borrow": 0.0,
            "repay": 0.0
        }
        
        # Calculate changes in state
        cash_change = to_state.get("cash", 0) - from_state.get("cash", 0)
        investment_change = to_state.get("investments", 0) - from_state.get("investments", 0)
        debt_change = to_state.get("mortgage", 0) - from_state.get("mortgage", 0)
        
        # Map changes to actions
        if investment_change > 0:
            action["invest"] = investment_change
        if cash_change > 0:
            action["save"] = cash_change
        if cash_change < 0:
            action["spend"] = abs(cash_change)
        if debt_change > 0:
            action["borrow"] = debt_change
        if debt_change < 0:
            action["repay"] = abs(debt_change)
        
        return action
    
    def _calculate_edge_probability(self, from_node: MeshNode, to_node: MeshNode) -> float:
        """Calculate probability of transition between nodes"""
        # Base probability
        base_probability = 0.8
        
        # Adjust based on financial health
        from_value = from_node.value
        to_value = to_node.value
        
        if to_value < from_value:
            base_probability *= 0.9  # Lower probability for declining value
        
        return base_probability
    
    def _calculate_edge_cost(self, from_node: MeshNode, to_node: MeshNode) -> float:
        """Calculate cost of transition between nodes"""
        # Simple cost based on value change
        value_change = to_node.value - from_node.value
        
        if value_change < 0:
            return abs(value_change) * 0.1  # Cost for value decline
        else:
            return 0.0  # No cost for value increase
    
    def _calculate_probabilities(self):
        """Calculate and normalize probabilities across the mesh"""
        # Normalize node probabilities
        total_probability = sum(node.probability for node in self.nodes.values())
        if total_probability > 0:
            for node in self.nodes.values():
                node.probability /= total_probability
        
        # Normalize edge probabilities
        for edge in self.edges.values():
            # Edge probability is based on connected nodes
            from_node = self.nodes[edge.from_node]
            to_node = self.nodes[edge.to_node]
            edge.probability = from_node.probability * to_node.probability
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get current mesh status"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "visible_future_nodes": len([n for n in self.nodes.values() if n.probability > 0.01]),
            "average_node_value": np.mean([n.value for n in self.nodes.values()]),
            "max_node_value": max([n.value for n in self.nodes.values()]),
            "min_node_value": min([n.value for n in self.nodes.values()]),
            "time_horizon_years": self.time_horizon_years
        }
    
    def get_payment_options(self) -> List[Dict[str, Any]]:
        """Get available payment options from current state"""
        if not self.nodes:
            return []
        
        # Get current node (first node)
        current_node = list(self.nodes.values())[0]
        current_state = current_node.financial_state
        
        options = []
        
        # Investment options
        if current_state.get("cash", 0) > 1000:
            options.append({
                "type": "invest",
                "amount": current_state["cash"] * 0.2,
                "description": "Invest 20% of cash",
                "expected_return": 0.08
            })
        
        # Debt repayment options
        if current_state.get("mortgage", 0) > 0:
            options.append({
                "type": "repay",
                "amount": current_state["mortgage"] * 0.05,
                "description": "Extra mortgage payment",
                "expected_return": 0.04  # Interest savings
            })
        
        # Savings options
        if current_state.get("income", 0) > 0:
            options.append({
                "type": "save",
                "amount": current_state["income"] * 0.1,
                "description": "Save 10% of income",
                "expected_return": 0.02
            })
        
        return options
    
    def find_optimal_path(self, target_value: float) -> List[MeshNode]:
        """Find optimal path to target value"""
        if not self.nodes:
            return []
        
        # Simple path finding - find nodes closest to target value
        target_nodes = []
        for node in self.nodes.values():
            if abs(node.value - target_value) < target_value * 0.1:  # Within 10%
                target_nodes.append(node)
        
        if not target_nodes:
            return []
        
        # Return path to highest probability target node
        best_target = max(target_nodes, key=lambda n: n.probability)
        
        # Simple path: current -> best target
        current_node = list(self.nodes.values())[0]
        return [current_node, best_target]
    
    def export_mesh_data(self, filepath: str):
        """Export mesh data to file"""
        mesh_data = {
            "nodes": {node_id: {
                "financial_state": node.financial_state,
                "timestamp": node.timestamp.isoformat(),
                "probability": node.probability,
                "value": node.value,
                "metadata": node.metadata
            } for node_id, node in self.nodes.items()},
            "edges": {edge_id: {
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "action": edge.action,
                "probability": edge.probability,
                "cost": edge.cost,
                "metadata": edge.metadata
            } for edge_id, edge in self.edges.items()},
            "status": self.get_mesh_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(mesh_data, f, indent=2, default=str)
        
        logger.info(f"✅ Mesh data exported to {filepath}")

def main():
    """Main function for testing"""
    # Initialize with sample financial state
    initial_state = {
        "cash": 50000,
        "investments": 100000,
        "income": 80000,
        "expenses": 60000,
        "mortgage": 300000
    }
    
    # Create mesh engine
    mesh_engine = StochasticMeshEngine(initial_state)
    
    # Sample milestones
    milestones = [
        {
            "id": "milestone_1",
            "type": "investment",
            "amount": 10000,
            "probability": 0.8,
            "year": 2,
            "month": 6,
            "description": "Additional investment",
            "impact": {"investments": 10000, "cash": -10000}
        },
        {
            "id": "milestone_2",
            "type": "expense",
            "amount": 5000,
            "probability": 0.6,
            "year": 3,
            "month": 3,
            "description": "Home renovation",
            "impact": {"cash": -5000}
        }
    ]
    
    # Initialize mesh
    status = mesh_engine.initialize_mesh(milestones, time_horizon_years=5)
    print(f"Mesh status: {json.dumps(status, indent=2, default=str)}")
    
    # Get payment options
    options = mesh_engine.get_payment_options()
    print(f"Payment options: {json.dumps(options, indent=2, default=str)}")
    
    # Export mesh data
    mesh_engine.export_mesh_data("mesh_data.json")

if __name__ == "__main__":
    main() 