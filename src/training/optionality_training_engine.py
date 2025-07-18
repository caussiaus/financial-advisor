"""
Optionality-Based Training Engine for Financial Path Optimization

This module implements the optionality algorithm for training optimal path switching
and stress minimization under various market conditions. The algorithm treats
"how many ways you can get from here to a safe (good) zone" as a first-class
metric of optionality or flexibility in financial state.

Key Components:
1. State space formalization (S) - financial state snapshots
2. Action space (A_s) - allowable moves/actions
3. Transition function - market + action effects
4. Good region (G) - financially safe states
5. Optionality measurement Ω(s) - count of feasible actions leading to G
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
import json
import logging
import random
from pathlib import Path
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing components
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.commutator_decision_engine import CommutatorDecisionEngine, FinancialState, CommutatorOperation
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.core.state_space_mesh_engine import EnhancedMeshEngine, EnhancedMeshNode
from src.unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent
from src.layers.financial_space_mapper import FinancialSpaceMapper


@dataclass
class FinancialStateNode:
    """Represents a node in the financial state space"""
    state_id: str
    timestamp: datetime
    financial_state: Dict[str, float]  # cash, debt, invested, etc.
    optionality_score: float = 0.0
    stress_level: float = 0.0
    is_good_region: bool = False
    parent_states: List[str] = field(default_factory=list)
    child_states: List[str] = field(default_factory=list)
    feasible_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDefinition:
    """Represents an action in the action space"""
    action_id: str
    action_type: str  # 'spend', 'rebalance', 'debt_paydown', 'draw'
    parameters: Dict[str, float]  # e.g., {'alpha': 0.1} for spend fraction
    capital_required: float = 0.0
    risk_change: float = 0.0
    stress_impact: float = 0.0
    optionality_impact: float = 0.0


@dataclass
class OptionalityPath:
    """Represents a path from current state to good region"""
    path_id: str
    start_state: str
    end_state: str
    actions: List[str]
    total_stress: float
    optionality_gain: float
    probability: float
    time_horizon: int  # months


@dataclass
class MarketCondition:
    """Represents market conditions affecting transitions"""
    condition_id: str
    timestamp: datetime
    market_stress: float  # 0-1 scale
    volatility: float
    interest_rate: float
    inflation_rate: float
    growth_rate: float
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.eye(5))


@dataclass
class OptionalityTrainingResult:
    """Results from optionality-based training"""
    session_id: str
    num_states_explored: int
    num_paths_found: int
    average_optionality: float
    stress_minimization_success: float
    optimal_paths: List[OptionalityPath]
    state_optionality_map: Dict[str, float]
    market_condition_performance: Dict[str, float]
    training_insights: Dict[str, Any]


class OptionalityTrainingEngine:
    """
    Training engine implementing the optionality algorithm for financial path optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.synthetic_engine = SyntheticLifestyleEngine()
        self.financial_mapper = FinancialSpaceMapper()
        
        # State space components
        self.state_space: Dict[str, FinancialStateNode] = {}
        self.action_space: Dict[str, ActionDefinition] = {}
        self.good_region_states: Set[str] = set()
        self.good_region_criteria: Dict[str, Any] = {}
        
        # Training components
        self.market_conditions: List[MarketCondition] = []
        self.optionality_paths: List[OptionalityPath] = []
        self.training_history: List[OptionalityTrainingResult] = []
        
        # Algorithm parameters
        self.horizon_months = self.config.get('horizon_months', 60)
        self.sampling_paths = self.config.get('sampling_paths', 1000)
        self.optionality_threshold = self.config.get('optionality_threshold', 0.1)
        self.stress_tolerance = self.config.get('stress_tolerance', 0.3)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize action space
        self._initialize_action_space()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training operations"""
        logger = logging.getLogger('optionality_training_engine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_action_space(self):
        """Initialize the action space with discretized parameters"""
        actions = []
        
        # Spending actions (fraction of assets)
        spend_fractions = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for i, alpha in enumerate(spend_fractions):
            actions.append(ActionDefinition(
                action_id=f"spend_{alpha}",
                action_type="spend",
                parameters={'alpha': alpha},
                capital_required=0.0,
                risk_change=0.0,
                stress_impact=-0.1 * alpha,  # Spending reduces stress
                optionality_impact=-0.05 * alpha  # But reduces optionality
            ))
        
        # Rebalancing actions (portfolio weights)
        rebalance_weights = [
            {'cash': 0.2, 'stocks': 0.4, 'bonds': 0.4},
            {'cash': 0.3, 'stocks': 0.3, 'bonds': 0.4},
            {'cash': 0.1, 'stocks': 0.6, 'bonds': 0.3},
            {'cash': 0.4, 'stocks': 0.2, 'bonds': 0.4},
            {'cash': 0.1, 'stocks': 0.7, 'bonds': 0.2},
        ]
        
        for i, weights in enumerate(rebalance_weights):
            actions.append(ActionDefinition(
                action_id=f"rebalance_{i}",
                action_type="rebalance",
                parameters={'weights': weights},
                capital_required=0.0,
                risk_change=self._calculate_risk_change(weights),
                stress_impact=self._calculate_stress_impact(weights),
                optionality_impact=self._calculate_optionality_impact(weights)
            ))
        
        # Debt paydown actions
        debt_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        for i, fraction in enumerate(debt_fractions):
            actions.append(ActionDefinition(
                action_id=f"debt_paydown_{fraction}",
                action_type="debt_paydown",
                parameters={'fraction': fraction},
                capital_required=0.0,
                risk_change=-0.1 * fraction,  # Reduces risk
                stress_impact=-0.2 * fraction,  # Reduces stress
                optionality_impact=0.1 * fraction  # Increases optionality
            ))
        
        # Draw actions (taking on debt)
        draw_fractions = [0.05, 0.1, 0.15, 0.2, 0.25]
        for i, fraction in enumerate(draw_fractions):
            actions.append(ActionDefinition(
                action_id=f"draw_{fraction}",
                action_type="draw",
                parameters={'fraction': fraction},
                capital_required=0.0,
                risk_change=0.15 * fraction,  # Increases risk
                stress_impact=0.1 * fraction,  # Increases stress
                optionality_impact=0.05 * fraction  # Slight optionality increase
            ))
        
        # Store actions
        for action in actions:
            self.action_space[action.action_id] = action
    
    def _calculate_risk_change(self, weights: Dict[str, float]) -> float:
        """Calculate risk change for rebalancing action"""
        # Simplified risk calculation
        stock_weight = weights.get('stocks', 0.0)
        bond_weight = weights.get('bonds', 0.0)
        cash_weight = weights.get('cash', 0.0)
        
        # Risk scores: stocks=0.8, bonds=0.3, cash=0.1
        total_risk = stock_weight * 0.8 + bond_weight * 0.3 + cash_weight * 0.1
        return total_risk - 0.5  # Relative to baseline
    
    def _calculate_stress_impact(self, weights: Dict[str, float]) -> float:
        """Calculate stress impact for rebalancing action"""
        # More cash = less stress
        cash_weight = weights.get('cash', 0.0)
        return -0.2 * (cash_weight - 0.2)  # Relative to 20% cash baseline
    
    def _calculate_optionality_impact(self, weights: Dict[str, float]) -> float:
        """Calculate optionality impact for rebalancing action"""
        # More cash = more optionality
        cash_weight = weights.get('cash', 0.0)
        return 0.3 * (cash_weight - 0.2)  # Relative to 20% cash baseline
    
    def define_good_region(self, criteria: Dict[str, Any]):
        """
        Define the good region (G) - financially safe states
        
        Args:
            criteria: Dictionary defining good region criteria
        """
        self.good_region_criteria = criteria
        
        # Mark existing states as good region
        for state_id, state in self.state_space.items():
            if self._is_good_region_state(state.financial_state, criteria):
                state.is_good_region = True
                self.good_region_states.add(state_id)
    
    def _is_good_region_state(self, financial_state: Dict[str, float], 
                             criteria: Dict[str, Any]) -> bool:
        """Check if a state is in the good region"""
        total_wealth = financial_state.get('total_wealth', 0)
        cash = financial_state.get('cash', 0)
        debt = financial_state.get('debt', 0)
        
        # Wealth threshold
        if total_wealth < criteria.get('min_wealth', 100000):
            return False
        
        # Cash ratio threshold
        if total_wealth > 0:
            cash_ratio = cash / total_wealth
            if cash_ratio < criteria.get('min_cash_ratio', 0.1):
                return False
        
        # Debt ratio threshold
        if total_wealth > 0:
            debt_ratio = debt / total_wealth
            if debt_ratio > criteria.get('max_debt_ratio', 0.4):
                return False
        
        # Stress level threshold
        stress = self._calculate_stress_level(financial_state)
        if stress > criteria.get('max_stress', 0.5):
            return False
        
        return True
    
    def _calculate_stress_level(self, financial_state: Dict[str, float]) -> float:
        """Calculate stress level for a financial state"""
        total_wealth = financial_state.get('total_wealth', 0)
        cash = financial_state.get('cash', 0)
        debt = financial_state.get('debt', 0)
        income = financial_state.get('income', 0)
        expenses = financial_state.get('expenses', 0)
        
        stress = 0.0
        
        # Debt stress
        if total_wealth > 0:
            debt_ratio = debt / total_wealth
            stress += min(1.0, debt_ratio * 2)
        
        # Liquidity stress
        if total_wealth > 0:
            cash_ratio = cash / total_wealth
            if cash_ratio < 0.1:
                stress += 0.3
            elif cash_ratio < 0.2:
                stress += 0.1
        
        # Income-expense stress
        if expenses > 0:
            income_ratio = income / expenses
            if income_ratio < 1.2:
                stress += 0.2
            elif income_ratio < 1.5:
                stress += 0.1
        
        return min(1.0, stress)
    
    def calculate_optionality(self, state_id: str, market_condition: MarketCondition) -> float:
        """
        Calculate optionality Ω(s) for a given state
        
        Ω(s) = |{a ∈ A_s : ∃ path s' → ... → g ∈ G}|
        
        Args:
            state_id: ID of the state to calculate optionality for
            market_condition: Current market condition
            
        Returns:
            Optionality score (number of feasible actions leading to good region)
        """
        if state_id not in self.state_space:
            return 0.0
        
        state = self.state_space[state_id]
        feasible_actions = []
        
        # Check each action in the action space
        for action_id, action in self.action_space.items():
            if self._is_action_feasible(state, action):
                # Check if this action can lead to good region
                if self._can_reach_good_region(state, action, market_condition):
                    feasible_actions.append(action_id)
        
        optionality_score = len(feasible_actions) / len(self.action_space)
        
        # Update state
        state.optionality_score = optionality_score
        state.feasible_actions = feasible_actions
        
        return optionality_score
    
    def _is_action_feasible(self, state: FinancialStateNode, action: ActionDefinition) -> bool:
        """Check if an action is feasible for a given state"""
        financial_state = state.financial_state
        total_wealth = financial_state.get('total_wealth', 0)
        cash = financial_state.get('cash', 0)
        
        # Check capital requirements
        if action.capital_required > cash:
            return False
        
        # Check risk constraints
        current_stress = self._calculate_stress_level(financial_state)
        new_stress = current_stress + action.stress_impact
        if new_stress > self.stress_tolerance:
            return False
        
        # Check action-specific constraints
        if action.action_type == "spend":
            spend_amount = total_wealth * action.parameters['alpha']
            if spend_amount > cash:
                return False
        
        elif action.action_type == "debt_paydown":
            debt = financial_state.get('debt', 0)
            paydown_amount = debt * action.parameters['fraction']
            if paydown_amount > cash:
                return False
        
        return True
    
    def _can_reach_good_region(self, state: FinancialStateNode, action: ActionDefinition,
                               market_condition: MarketCondition) -> bool:
        """
        Check if an action can lead to good region using sampling approach
        
        Args:
            state: Current state
            action: Action to evaluate
            market_condition: Current market condition
            
        Returns:
            True if action can lead to good region
        """
        # Sample multiple paths from this action
        successful_paths = 0
        
        for _ in range(self.sampling_paths // 10):  # Reduce sampling for efficiency
            # Apply action to get new state
            new_state = self._apply_action_to_state(state, action)
            
            # Simulate market evolution
            evolved_state = self._simulate_market_evolution(new_state, market_condition)
            
            # Check if evolved state is in good region
            if self._is_good_region_state(evolved_state, self.good_region_criteria):
                successful_paths += 1
        
        # If at least one path reaches good region, action is feasible
        return successful_paths > 0
    
    def _apply_action_to_state(self, state: FinancialStateNode, action: ActionDefinition) -> Dict[str, float]:
        """Apply an action to a state to get new state"""
        new_state = state.financial_state.copy()
        
        if action.action_type == "spend":
            alpha = action.parameters['alpha']
            spend_amount = new_state.get('total_wealth', 0) * alpha
            new_state['cash'] = max(0, new_state.get('cash', 0) - spend_amount)
            new_state['total_wealth'] = max(0, new_state.get('total_wealth', 0) - spend_amount)
        
        elif action.action_type == "rebalance":
            weights = action.parameters['weights']
            total_wealth = new_state.get('total_wealth', 0)
            new_state['cash'] = total_wealth * weights.get('cash', 0.2)
            new_state['investments'] = total_wealth * (weights.get('stocks', 0.4) + weights.get('bonds', 0.4))
        
        elif action.action_type == "debt_paydown":
            fraction = action.parameters['fraction']
            debt = new_state.get('debt', 0)
            paydown_amount = debt * fraction
            new_state['debt'] = max(0, debt - paydown_amount)
            new_state['cash'] = max(0, new_state.get('cash', 0) - paydown_amount)
        
        elif action.action_type == "draw":
            fraction = action.parameters['fraction']
            total_wealth = new_state.get('total_wealth', 0)
            draw_amount = total_wealth * fraction
            new_state['debt'] = new_state.get('debt', 0) + draw_amount
            new_state['cash'] = new_state.get('cash', 0) + draw_amount
            new_state['total_wealth'] = total_wealth + draw_amount
        
        return new_state
    
    def _simulate_market_evolution(self, state: Dict[str, float], 
                                  market_condition: MarketCondition) -> Dict[str, float]:
        """Simulate market evolution of a state"""
        evolved_state = state.copy()
        
        # Apply market stress
        stress_factor = 1.0 - market_condition.market_stress * 0.2
        
        # Apply volatility
        volatility_shock = np.random.normal(0, market_condition.volatility)
        
        # Apply growth
        growth_factor = 1.0 + market_condition.growth_rate * 0.01
        
        # Update investments
        investments = evolved_state.get('investments', 0)
        evolved_state['investments'] = investments * stress_factor * growth_factor * (1 + volatility_shock)
        
        # Update total wealth
        evolved_state['total_wealth'] = (
            evolved_state.get('cash', 0) + 
            evolved_state.get('investments', 0) - 
            evolved_state.get('debt', 0)
        )
        
        return evolved_state
    
    def find_optimal_paths(self, start_state_id: str, target_optionality: float = 0.5) -> List[OptionalityPath]:
        """
        Find optimal paths that maximize optionality while minimizing stress
        
        Args:
            start_state_id: Starting state ID
            target_optionality: Target optionality level
            
        Returns:
            List of optimal paths
        """
        if start_state_id not in self.state_space:
            return []
        
        start_state = self.state_space[start_state_id]
        optimal_paths = []
        
        # Use dynamic programming to find optimal paths
        paths = self._find_all_paths_to_good_region(start_state_id)
        
        for path in paths:
            # Calculate path metrics
            total_stress = self._calculate_path_stress(path)
            optionality_gain = self._calculate_optionality_gain(path)
            
            # Check if path meets criteria
            if (optionality_gain >= target_optionality and 
                total_stress <= self.stress_tolerance):
                
                optimal_path = OptionalityPath(
                    path_id=f"path_{len(optimal_paths)}",
                    start_state=start_state_id,
                    end_state=path[-1],
                    actions=path[1:-1],  # Exclude start and end states
                    total_stress=total_stress,
                    optionality_gain=optionality_gain,
                    probability=self._calculate_path_probability(path),
                    time_horizon=len(path) * 3  # Assume 3 months per step
                )
                optimal_paths.append(optimal_path)
        
        # Sort by optimality score (optionality gain / stress)
        optimal_paths.sort(key=lambda p: p.optionality_gain / (p.total_stress + 0.01), reverse=True)
        
        return optimal_paths[:10]  # Return top 10 paths
    
    def _find_all_paths_to_good_region(self, start_state_id: str) -> List[List[str]]:
        """Find all paths from start state to good region using BFS"""
        if start_state_id not in self.state_space:
            return []
        
        paths = []
        queue = [(start_state_id, [start_state_id])]
        visited = set()
        
        while queue:
            current_state_id, path = queue.pop(0)
            
            if current_state_id in visited:
                continue
            
            visited.add(current_state_id)
            
            # Check if we reached good region
            if current_state_id in self.good_region_states:
                paths.append(path)
                continue
            
            # Explore neighbors
            current_state = self.state_space[current_state_id]
            for action_id in current_state.feasible_actions:
                # Find next state after applying action
                next_state_id = self._get_next_state_id(current_state_id, action_id)
                if next_state_id and next_state_id not in visited:
                    queue.append((next_state_id, path + [next_state_id]))
        
        return paths
    
    def _get_next_state_id(self, state_id: str, action_id: str) -> Optional[str]:
        """Get next state ID after applying action"""
        if state_id not in self.state_space or action_id not in self.action_space:
            return None
        
        state = self.state_space[state_id]
        action = self.action_space[action_id]
        
        # Apply action to get new state
        new_state_dict = self._apply_action_to_state(state, action)
        
        # Find or create state node
        new_state_id = self._find_or_create_state(new_state_dict)
        
        return new_state_id
    
    def _find_or_create_state(self, financial_state: Dict[str, float]) -> str:
        """Find existing state or create new one"""
        # Simple hash-based state ID
        state_hash = hash(tuple(sorted(financial_state.items())))
        state_id = f"state_{state_hash}"
        
        if state_id not in self.state_space:
            # Create new state node
            new_state = FinancialStateNode(
                state_id=state_id,
                timestamp=datetime.now(),
                financial_state=financial_state,
                stress_level=self._calculate_stress_level(financial_state),
                is_good_region=self._is_good_region_state(financial_state, self.good_region_criteria)
            )
            self.state_space[state_id] = new_state
            
            if new_state.is_good_region:
                self.good_region_states.add(state_id)
        
        return state_id
    
    def _calculate_path_stress(self, path: List[str]) -> float:
        """Calculate total stress along a path"""
        total_stress = 0.0
        
        for state_id in path:
            if state_id in self.state_space:
                state = self.state_space[state_id]
                total_stress += state.stress_level
        
        return total_stress / len(path) if path else 0.0
    
    def _calculate_optionality_gain(self, path: List[str]) -> float:
        """Calculate optionality gain along a path"""
        if not path:
            return 0.0
        
        start_optionality = self.state_space[path[0]].optionality_score
        end_optionality = self.state_space[path[-1]].optionality_score
        
        return end_optionality - start_optionality
    
    def _calculate_path_probability(self, path: List[str]) -> float:
        """Calculate probability of a path occurring"""
        if len(path) < 2:
            return 1.0
        
        probability = 1.0
        
        for i in range(len(path) - 1):
            current_state = self.state_space[path[i]]
            next_state = self.state_space[path[i + 1]]
            
            # Calculate transition probability based on state similarity
            similarity = self._calculate_state_similarity(
                current_state.financial_state, 
                next_state.financial_state
            )
            
            probability *= similarity
        
        return probability
    
    def _calculate_state_similarity(self, state1: Dict[str, float], 
                                  state2: Dict[str, float]) -> float:
        """Calculate similarity between two financial states"""
        # Convert to vectors for cosine similarity
        keys = set(state1.keys()) | set(state2.keys())
        vec1 = [state1.get(k, 0.0) for k in keys]
        vec2 = [state2.get(k, 0.0) for k in keys]
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1 = [x / norm1 for x in vec1]
        vec2 = [x / norm2 for x in vec2]
        
        # Calculate cosine similarity
        similarity = sum(a * b for a, b in zip(vec1, vec2))
        
        return max(0.0, similarity)
    
    def train_optionality_model(self, num_scenarios: int = 100) -> OptionalityTrainingResult:
        """
        Train the optionality model using synthetic scenarios
        
        Args:
            num_scenarios: Number of training scenarios
            
        Returns:
            Training results
        """
        self.logger.info(f"Starting optionality training with {num_scenarios} scenarios...")
        
        # Generate synthetic scenarios
        scenarios = self._generate_training_scenarios(num_scenarios)
        
        # Define good region criteria
        good_region_criteria = {
            'min_wealth': 200000,
            'min_cash_ratio': 0.15,
            'max_debt_ratio': 0.3,
            'max_stress': 0.4
        }
        self.define_good_region(good_region_criteria)
        
        # Generate market conditions
        self._generate_market_conditions()
        
        # Calculate optionality for all states
        total_optionality = 0.0
        num_states = 0
        
        for scenario in scenarios:
            for state_id in scenario['states']:
                if state_id in self.state_space:
                    state = self.state_space[state_id]
                    
                    # Use average market condition for training
                    avg_market_condition = self._get_average_market_condition()
                    optionality = self.calculate_optionality(state_id, avg_market_condition)
                    
                    total_optionality += optionality
                    num_states += 1
        
        average_optionality = total_optionality / num_states if num_states > 0 else 0.0
        
        # Find optimal paths
        optimal_paths = []
        for scenario in scenarios:
            if scenario['states']:
                start_state = scenario['states'][0]
                paths = self.find_optimal_paths(start_state, target_optionality=0.3)
                optimal_paths.extend(paths)
        
        # Calculate training metrics
        stress_minimization_success = self._calculate_stress_minimization_success(optimal_paths)
        market_performance = self._calculate_market_condition_performance()
        
        # Create training result
        result = OptionalityTrainingResult(
            session_id=f"optionality_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_states_explored=len(self.state_space),
            num_paths_found=len(optimal_paths),
            average_optionality=average_optionality,
            stress_minimization_success=stress_minimization_success,
            optimal_paths=optimal_paths,
            state_optionality_map={state_id: state.optionality_score 
                                 for state_id, state in self.state_space.items()},
            market_condition_performance=market_performance,
            training_insights=self._generate_training_insights()
        )
        
        self.training_history.append(result)
        
        self.logger.info(f"Training completed. Average optionality: {average_optionality:.3f}")
        
        return result
    
    def _generate_training_scenarios(self, num_scenarios: int) -> List[Dict[str, Any]]:
        """Generate training scenarios with synthetic people"""
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate synthetic person
            person = self.synthetic_engine.generate_synthetic_client()
            
            # Create initial financial state
            initial_state = self._create_initial_financial_state(person)
            state_id = self._find_or_create_state(initial_state)
            
            # Generate additional states through random actions
            states = [state_id]
            for _ in range(5):  # Generate 5 additional states per scenario
                if states:
                    current_state_id = random.choice(states)
                    action_id = random.choice(list(self.action_space.keys()))
                    next_state_id = self._get_next_state_id(current_state_id, action_id)
                    if next_state_id:
                        states.append(next_state_id)
            
            scenarios.append({
                'scenario_id': f"scenario_{i}",
                'person': person,
                'states': states
            })
        
        return scenarios
    
    def _create_initial_financial_state(self, person: SyntheticClientData) -> Dict[str, float]:
        """Create initial financial state from synthetic person"""
        # Extract financial data from person
        income = person.income if hasattr(person, 'income') else 75000
        age = person.age if hasattr(person, 'age') else 35
        
        # Create realistic financial state
        total_wealth = income * (age - 20) * 0.1  # Simplified wealth accumulation
        cash_ratio = 0.2
        debt_ratio = 0.3
        
        return {
            'total_wealth': total_wealth,
            'cash': total_wealth * cash_ratio,
            'investments': total_wealth * (1 - cash_ratio - debt_ratio),
            'debt': total_wealth * debt_ratio,
            'income': income / 12,  # Monthly income
            'expenses': income * 0.6 / 12  # Monthly expenses
        }
    
    def _generate_market_conditions(self):
        """Generate various market conditions for training"""
        conditions = []
        
        # Normal market condition
        conditions.append(MarketCondition(
            condition_id="normal",
            timestamp=datetime.now(),
            market_stress=0.3,
            volatility=0.15,
            interest_rate=0.04,
            inflation_rate=0.02,
            growth_rate=0.03
        ))
        
        # High stress market condition
        conditions.append(MarketCondition(
            condition_id="high_stress",
            timestamp=datetime.now(),
            market_stress=0.8,
            volatility=0.25,
            interest_rate=0.06,
            inflation_rate=0.05,
            growth_rate=-0.02
        ))
        
        # Low stress market condition
        conditions.append(MarketCondition(
            condition_id="low_stress",
            timestamp=datetime.now(),
            market_stress=0.1,
            volatility=0.10,
            interest_rate=0.02,
            inflation_rate=0.01,
            growth_rate=0.05
        ))
        
        self.market_conditions = conditions
    
    def _get_average_market_condition(self) -> MarketCondition:
        """Get average market condition for training"""
        if not self.market_conditions:
            return MarketCondition(
                condition_id="average",
                timestamp=datetime.now(),
                market_stress=0.4,
                volatility=0.17,
                interest_rate=0.04,
                inflation_rate=0.025,
                growth_rate=0.025
            )
        
        # Calculate averages
        avg_stress = np.mean([c.market_stress for c in self.market_conditions])
        avg_volatility = np.mean([c.volatility for c in self.market_conditions])
        avg_interest = np.mean([c.interest_rate for c in self.market_conditions])
        avg_inflation = np.mean([c.inflation_rate for c in self.market_conditions])
        avg_growth = np.mean([c.growth_rate for c in self.market_conditions])
        
        return MarketCondition(
            condition_id="average",
            timestamp=datetime.now(),
            market_stress=avg_stress,
            volatility=avg_volatility,
            interest_rate=avg_interest,
            inflation_rate=avg_inflation,
            growth_rate=avg_growth
        )
    
    def _calculate_stress_minimization_success(self, paths: List[OptionalityPath]) -> float:
        """Calculate success rate of stress minimization"""
        if not paths:
            return 0.0
        
        successful_paths = [p for p in paths if p.total_stress <= self.stress_tolerance]
        return len(successful_paths) / len(paths)
    
    def _calculate_market_condition_performance(self) -> Dict[str, float]:
        """Calculate performance under different market conditions"""
        performance = {}
        
        for condition in self.market_conditions:
            # Calculate average optionality under this condition
            total_optionality = 0.0
            num_states = 0
            
            for state_id, state in self.state_space.items():
                optionality = self.calculate_optionality(state_id, condition)
                total_optionality += optionality
                num_states += 1
            
            avg_optionality = total_optionality / num_states if num_states > 0 else 0.0
            performance[condition.condition_id] = avg_optionality
        
        return performance
    
    def _generate_training_insights(self) -> Dict[str, Any]:
        """Generate insights from training"""
        insights = {
            'total_states': len(self.state_space),
            'good_region_states': len(self.good_region_states),
            'action_space_size': len(self.action_space),
            'average_stress_level': np.mean([state.stress_level for state in self.state_space.values()]),
            'optionality_distribution': {
                'high': len([s for s in self.state_space.values() if s.optionality_score > 0.7]),
                'medium': len([s for s in self.state_space.values() if 0.3 <= s.optionality_score <= 0.7]),
                'low': len([s for s in self.state_space.values() if s.optionality_score < 0.3])
            }
        }
        
        return insights
    
    def save_training_results(self, result: OptionalityTrainingResult, 
                            output_dir: str = "data/outputs/optionality_training"):
        """Save training results to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_file = Path(output_dir) / f"{result.session_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'session_id': result.session_id,
                'num_states_explored': result.num_states_explored,
                'num_paths_found': result.num_paths_found,
                'average_optionality': result.average_optionality,
                'stress_minimization_success': result.stress_minimization_success,
                'market_condition_performance': result.market_condition_performance,
                'training_insights': result.training_insights
            }, f, indent=2, default=str)
        
        # Save optimal paths
        paths_file = Path(output_dir) / f"{result.session_id}_paths.json"
        with open(paths_file, 'w') as f:
            paths_data = []
            for path in result.optimal_paths:
                paths_data.append({
                    'path_id': path.path_id,
                    'start_state': path.start_state,
                    'end_state': path.end_state,
                    'actions': path.actions,
                    'total_stress': path.total_stress,
                    'optionality_gain': path.optionality_gain,
                    'probability': path.probability,
                    'time_horizon': path.time_horizon
                })
            json.dump(paths_data, f, indent=2)
        
        # Save state optionality map
        optionality_file = Path(output_dir) / f"{result.session_id}_optionality.json"
        with open(optionality_file, 'w') as f:
            json.dump(result.state_optionality_map, f, indent=2)
        
        self.logger.info(f"Training results saved to {output_dir}")


def run_optionality_training(num_scenarios: int = 100) -> OptionalityTrainingResult:
    """
    Run optionality-based training session
    
    Args:
        num_scenarios: Number of training scenarios
        
    Returns:
        Training results
    """
    engine = OptionalityTrainingEngine()
    result = engine.train_optionality_model(num_scenarios)
    
    # Save results
    engine.save_training_results(result)
    
    return result


if __name__ == "__main__":
    # Run training
    result = run_optionality_training(num_scenarios=50)
    print(f"Training completed with {result.num_paths_found} optimal paths found")
    print(f"Average optionality: {result.average_optionality:.3f}")
    print(f"Stress minimization success: {result.stress_minimization_success:.3f}") 