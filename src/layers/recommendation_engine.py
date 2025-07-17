"""
Recommendation Engine Layer

Responsible for:
- Commutator-based portfolio reallocation algorithms
- Optimal path finding from current to target states
- Set theoretic state space exploration
- Recursive commutator generation for financial moves
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Protocol
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import json
from enum import Enum
from itertools import combinations, permutations
import math


class MoveType(Enum):
    """Types of financial moves (like Rubik's cube moves)"""
    REALLOCATE = "reallocate"
    REBALANCE = "rebalance"
    LIQUIDATE = "liquidate"
    PURCHASE = "purchase"
    SELL = "sell"
    TRANSFER = "transfer"
    CONSOLIDATE = "consolidate"
    DIVERSIFY = "diversify"


@dataclass
class FinancialMove:
    """Represents a financial move (like a Rubik's cube move)"""
    move_id: str
    move_type: MoveType
    from_asset: Optional[str] = None
    to_asset: Optional[str] = None
    amount: float = 0.0
    priority: int = 1
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Commutator:
    """Represents a commutator sequence: [A, B] = A B A' B'"""
    sequence_id: str
    move_a: FinancialMove
    move_b: FinancialMove
    inverse_a: FinancialMove
    inverse_b: FinancialMove
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class TargetState:
    """Represents a target financial state (solved state range)"""
    state_id: str
    name: str
    asset_allocation: Dict[str, float]  # Target percentages
    risk_tolerance: float  # 0-1 scale
    liquidity_requirement: float  # Minimum cash ratio
    time_horizon: int  # Years
    constraints: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Recommendation:
    """Represents a financial recommendation"""
    recommendation_id: str
    timestamp: datetime
    current_state: Dict[str, float]
    target_state: TargetState
    commutator_sequence: List[Commutator]
    expected_duration: int  # Days
    confidence: float  # 0-1
    risk_score: float  # 0-1
    description: str = ""
    metadata: Dict = field(default_factory=dict)


class StateAnalyzer(Protocol):
    """Protocol for state analysis capabilities"""
    
    def analyze_state(self, financial_state: Dict[str, float]) -> Dict:
        """Analyze current financial state"""
        ...
    
    def calculate_distance_to_target(self, current_state: Dict[str, float], 
                                   target_state: TargetState) -> float:
        """Calculate distance to target state"""
        ...


class CommutatorGenerator(Protocol):
    """Protocol for commutator generation capabilities"""
    
    def generate_commutators(self, current_state: Dict[str, float], 
                           target_state: TargetState) -> List[Commutator]:
        """Generate commutator sequences"""
        ...


class RecommendationEngineLayer:
    """
    Recommendation Engine Layer - Commutator-based portfolio optimization
    
    Responsibilities:
    - Commutator-based portfolio reallocation algorithms
    - Optimal path finding from current to target states
    - Set theoretic state space exploration
    - Recursive commutator generation for financial moves
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.asset_classes = self._initialize_asset_classes()
        self.move_templates = self._initialize_move_templates()
        self.target_states = self._initialize_target_states()
        self.commutator_cache = {}
        
    def _initialize_asset_classes(self) -> Dict[str, Dict]:
        """Initialize asset class definitions"""
        return {
            'cash': {'risk': 0.0, 'liquidity': 1.0, 'expected_return': 0.02},
            'bonds': {'risk': 0.2, 'liquidity': 0.8, 'expected_return': 0.04},
            'stocks': {'risk': 0.6, 'liquidity': 0.9, 'expected_return': 0.08},
            'real_estate': {'risk': 0.4, 'liquidity': 0.3, 'expected_return': 0.06},
            'commodities': {'risk': 0.7, 'liquidity': 0.7, 'expected_return': 0.05},
            'crypto': {'risk': 0.9, 'liquidity': 0.9, 'expected_return': 0.15}
        }
    
    def _initialize_move_templates(self) -> Dict[str, FinancialMove]:
        """Initialize move templates for commutator generation"""
        templates = {}
        
        # Reallocation moves
        templates['reallocate_stocks_to_bonds'] = FinancialMove(
            move_id="reallocate_stocks_to_bonds",
            move_type=MoveType.REALLOCATE,
            from_asset="stocks",
            to_asset="bonds",
            description="Reallocate from stocks to bonds"
        )
        
        templates['reallocate_bonds_to_stocks'] = FinancialMove(
            move_id="reallocate_bonds_to_stocks",
            move_type=MoveType.REALLOCATE,
            from_asset="bonds",
            to_asset="stocks",
            description="Reallocate from bonds to stocks"
        )
        
        templates['increase_cash'] = FinancialMove(
            move_id="increase_cash",
            move_type=MoveType.REALLOCATE,
            to_asset="cash",
            description="Increase cash position"
        )
        
        templates['decrease_cash'] = FinancialMove(
            move_id="decrease_cash",
            move_type=MoveType.REALLOCATE,
            from_asset="cash",
            description="Decrease cash position"
        )
        
        return templates
    
    def _initialize_target_states(self) -> Dict[str, TargetState]:
        """Initialize target state definitions"""
        states = {}
        
        # Conservative target state
        states['conservative'] = TargetState(
            state_id="conservative",
            name="Conservative Portfolio",
            asset_allocation={
                'cash': 0.20,
                'bonds': 0.50,
                'stocks': 0.25,
                'real_estate': 0.05
            },
            risk_tolerance=0.2,
            liquidity_requirement=0.15,
            time_horizon=5
        )
        
        # Moderate target state
        states['moderate'] = TargetState(
            state_id="moderate",
            name="Moderate Portfolio",
            asset_allocation={
                'cash': 0.10,
                'bonds': 0.30,
                'stocks': 0.50,
                'real_estate': 0.10
            },
            risk_tolerance=0.5,
            liquidity_requirement=0.10,
            time_horizon=10
        )
        
        # Aggressive target state
        states['aggressive'] = TargetState(
            state_id="aggressive",
            name="Aggressive Portfolio",
            asset_allocation={
                'cash': 0.05,
                'bonds': 0.15,
                'stocks': 0.70,
                'real_estate': 0.10
            },
            risk_tolerance=0.8,
            liquidity_requirement=0.05,
            time_horizon=15
        )
        
        return states
    
    def analyze_current_state(self, financial_state: Dict[str, float]) -> Dict:
        """
        Analyze current financial state
        
        Args:
            financial_state: Current financial state
            
        Returns:
            State analysis
        """
        total_assets = sum(financial_state.values())
        
        if total_assets == 0:
            return {'error': 'No assets found'}
        
        # Calculate current allocation
        current_allocation = {
            asset: value / total_assets 
            for asset, value in financial_state.items() 
            if value > 0
        }
        
        # Calculate risk metrics
        portfolio_risk = self._calculate_portfolio_risk(current_allocation)
        liquidity_ratio = current_allocation.get('cash', 0)
        
        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(current_allocation)
        
        return {
            'current_allocation': current_allocation,
            'total_assets': total_assets,
            'portfolio_risk': portfolio_risk,
            'liquidity_ratio': liquidity_ratio,
            'diversification_score': diversification_score,
            'risk_analysis': self._analyze_risk_profile(current_allocation)
        }
    
    def _calculate_portfolio_risk(self, allocation: Dict[str, float]) -> float:
        """Calculate portfolio risk based on allocation"""
        portfolio_risk = 0.0
        
        for asset, weight in allocation.items():
            if asset in self.asset_classes:
                asset_risk = self.asset_classes[asset]['risk']
                portfolio_risk += weight * asset_risk
        
        return portfolio_risk
    
    def _calculate_diversification_score(self, allocation: Dict[str, float]) -> float:
        """Calculate diversification score (0-1)"""
        if not allocation:
            return 0.0
        
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(weight ** 2 for weight in allocation.values())
        
        # Convert to diversification score (1 - normalized HHI)
        n = len(allocation)
        max_hhi = 1.0  # Maximum concentration
        min_hhi = 1.0 / n  # Minimum concentration (equal weights)
        
        if max_hhi == min_hhi:
            return 1.0
        
        normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
        return 1.0 - normalized_hhi
    
    def _analyze_risk_profile(self, allocation: Dict[str, float]) -> Dict:
        """Analyze risk profile of current allocation"""
        risk_metrics = {
            'overall_risk': self._calculate_portfolio_risk(allocation),
            'cash_risk': 1.0 - allocation.get('cash', 0),
            'bond_risk': allocation.get('bonds', 0) * 0.2,
            'stock_risk': allocation.get('stocks', 0) * 0.6,
            'real_estate_risk': allocation.get('real_estate', 0) * 0.4
        }
        
        return risk_metrics
    
    def find_optimal_target_state(self, current_state: Dict[str, float], 
                                risk_preference: str = 'moderate') -> TargetState:
        """
        Find optimal target state based on current state and preferences
        
        Args:
            current_state: Current financial state
            risk_preference: Risk preference ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Optimal target state
        """
        current_analysis = self.analyze_current_state(current_state)
        current_risk = current_analysis['portfolio_risk']
        
        # Find closest target state
        best_target = None
        min_distance = float('inf')
        
        for target_id, target_state in self.target_states.items():
            distance = self._calculate_state_distance(current_analysis, target_state)
            
            # Apply risk preference filter
            if risk_preference == 'conservative' and target_state.risk_tolerance > 0.4:
                continue
            elif risk_preference == 'aggressive' and target_state.risk_tolerance < 0.6:
                continue
            
            if distance < min_distance:
                min_distance = distance
                best_target = target_state
        
        return best_target or self.target_states['moderate']
    
    def _calculate_state_distance(self, current_analysis: Dict, 
                                target_state: TargetState) -> float:
        """Calculate distance between current and target states"""
        current_allocation = current_analysis['current_allocation']
        target_allocation = target_state.asset_allocation
        
        # Calculate allocation distance
        allocation_distance = 0.0
        all_assets = set(current_allocation.keys()) | set(target_allocation.keys())
        
        for asset in all_assets:
            current_weight = current_allocation.get(asset, 0)
            target_weight = target_allocation.get(asset, 0)
            allocation_distance += (current_weight - target_weight) ** 2
        
        # Calculate risk distance
        current_risk = current_analysis['portfolio_risk']
        risk_distance = abs(current_risk - target_state.risk_tolerance)
        
        # Calculate liquidity distance
        current_liquidity = current_analysis['liquidity_ratio']
        liquidity_distance = abs(current_liquidity - target_state.liquidity_requirement)
        
        # Weighted combination
        total_distance = (
            0.6 * allocation_distance +
            0.3 * risk_distance +
            0.1 * liquidity_distance
        )
        
        return total_distance
    
    def generate_commutator_sequence(self, current_state: Dict[str, float], 
                                   target_state: TargetState) -> List[Commutator]:
        """
        Generate commutator sequence to transform current state to target state
        
        Args:
            current_state: Current financial state
            target_state: Target financial state
            
        Returns:
            List of commutators to execute
        """
        current_analysis = self.analyze_current_state(current_state)
        current_allocation = current_analysis['current_allocation']
        target_allocation = target_state.asset_allocation
        
        # Generate basic moves to reach target
        basic_moves = self._generate_basic_moves(current_allocation, target_allocation)
        
        # Convert basic moves to commutators
        commutators = self._convert_moves_to_commutators(basic_moves)
        
        # Optimize commutator sequence
        optimized_commutators = self._optimize_commutator_sequence(commutators)
        
        return optimized_commutators
    
    def _generate_basic_moves(self, current_allocation: Dict[str, float], 
                             target_allocation: Dict[str, float]) -> List[FinancialMove]:
        """Generate basic moves to transform current allocation to target"""
        moves = []
        
        all_assets = set(current_allocation.keys()) | set(target_allocation.keys())
        
        for asset in all_assets:
            current_weight = current_allocation.get(asset, 0)
            target_weight = target_allocation.get(asset, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # Significant difference
                if weight_diff > 0:
                    # Need to increase this asset
                    move = FinancialMove(
                        move_id=f"increase_{asset}",
                        move_type=MoveType.REALLOCATE,
                        to_asset=asset,
                        amount=abs(weight_diff),
                        description=f"Increase {asset} allocation by {abs(weight_diff):.1%}"
                    )
                else:
                    # Need to decrease this asset
                    move = FinancialMove(
                        move_id=f"decrease_{asset}",
                        move_type=MoveType.REALLOCATE,
                        from_asset=asset,
                        amount=abs(weight_diff),
                        description=f"Decrease {asset} allocation by {abs(weight_diff):.1%}"
                    )
                
                moves.append(move)
        
        return moves
    
    def _convert_moves_to_commutators(self, moves: List[FinancialMove]) -> List[Commutator]:
        """Convert basic moves to commutator sequences"""
        commutators = []
        
        # Group moves by type
        reallocation_moves = [m for m in moves if m.move_type == MoveType.REALLOCATE]
        
        # Create commutators for reallocation moves
        for i in range(0, len(reallocation_moves), 2):
            if i + 1 < len(reallocation_moves):
                move_a = reallocation_moves[i]
                move_b = reallocation_moves[i + 1]
                
                # Create inverse moves
                inverse_a = self._create_inverse_move(move_a)
                inverse_b = self._create_inverse_move(move_b)
                
                commutator = Commutator(
                    sequence_id=f"commutator_{len(commutators)}",
                    move_a=move_a,
                    move_b=move_b,
                    inverse_a=inverse_a,
                    inverse_b=inverse_b,
                    description=f"Commutator: {move_a.description} then {move_b.description}"
                )
                
                commutators.append(commutator)
        
        return commutators
    
    def _create_inverse_move(self, move: FinancialMove) -> FinancialMove:
        """Create inverse of a move"""
        if move.move_type == MoveType.REALLOCATE:
            return FinancialMove(
                move_id=f"inverse_{move.move_id}",
                move_type=move.move_type,
                from_asset=move.to_asset,
                to_asset=move.from_asset,
                amount=move.amount,
                description=f"Inverse: {move.description}"
            )
        
        return move
    
    def _optimize_commutator_sequence(self, commutators: List[Commutator]) -> List[Commutator]:
        """Optimize commutator sequence for efficiency"""
        if len(commutators) <= 1:
            return commutators
        
        # Simple optimization: combine related commutators
        optimized = []
        i = 0
        
        while i < len(commutators):
            current = commutators[i]
            
            # Look for next commutator that can be combined
            if i + 1 < len(commutators):
                next_comm = commutators[i + 1]
                
                # Check if commutators can be combined
                if self._can_combine_commutators(current, next_comm):
                    combined = self._combine_commutators(current, next_comm)
                    optimized.append(combined)
                    i += 2
                else:
                    optimized.append(current)
                    i += 1
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def _can_combine_commutators(self, comm1: Commutator, comm2: Commutator) -> bool:
        """Check if two commutators can be combined"""
        # Check if they operate on related assets
        assets1 = {comm1.move_a.from_asset, comm1.move_a.to_asset, 
                  comm1.move_b.from_asset, comm1.move_b.to_asset}
        assets2 = {comm2.move_a.from_asset, comm2.move_a.to_asset, 
                  comm2.move_b.from_asset, comm2.move_b.to_asset}
        
        return bool(assets1 & assets2)  # Check for intersection
    
    def _combine_commutators(self, comm1: Commutator, comm2: Commutator) -> Commutator:
        """Combine two commutators into one"""
        return Commutator(
            sequence_id=f"combined_{comm1.sequence_id}_{comm2.sequence_id}",
            move_a=comm1.move_a,
            move_b=comm2.move_a,
            inverse_a=comm1.inverse_a,
            inverse_b=comm2.inverse_a,
            description=f"Combined: {comm1.description} + {comm2.description}"
        )
    
    def generate_recommendation(self, current_state: Dict[str, float], 
                              risk_preference: str = 'moderate') -> Recommendation:
        """
        Generate comprehensive recommendation using commutator algorithms
        
        Args:
            current_state: Current financial state
            risk_preference: Risk preference
            
        Returns:
            Financial recommendation
        """
        # Find optimal target state
        target_state = self.find_optimal_target_state(current_state, risk_preference)
        
        # Generate commutator sequence
        commutator_sequence = self.generate_commutator_sequence(current_state, target_state)
        
        # Calculate recommendation metrics
        confidence = self._calculate_confidence(current_state, target_state, commutator_sequence)
        risk_score = self._calculate_risk_score(commutator_sequence)
        expected_duration = self._calculate_expected_duration(commutator_sequence)
        
        # Create recommendation
        recommendation = Recommendation(
            recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            current_state=current_state,
            target_state=target_state,
            commutator_sequence=commutator_sequence,
            expected_duration=expected_duration,
            confidence=confidence,
            risk_score=risk_score,
            description=f"Transform portfolio to {target_state.name} using {len(commutator_sequence)} commutators"
        )
        
        return recommendation
    
    def _calculate_confidence(self, current_state: Dict[str, float], 
                            target_state: TargetState, 
                            commutator_sequence: List[Commutator]) -> float:
        """Calculate confidence in recommendation"""
        # Base confidence on distance to target
        current_analysis = self.analyze_current_state(current_state)
        distance = self._calculate_state_distance(current_analysis, target_state)
        
        # Normalize distance to 0-1 scale
        max_distance = 2.0  # Maximum possible distance
        distance_score = 1.0 - min(1.0, distance / max_distance)
        
        # Factor in commutator complexity
        complexity_score = 1.0 / (1.0 + len(commutator_sequence) * 0.1)
        
        # Combine scores
        confidence = (distance_score + complexity_score) / 2.0
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_risk_score(self, commutator_sequence: List[Commutator]) -> float:
        """Calculate risk score of commutator sequence"""
        risk_score = 0.0
        
        for commutator in commutator_sequence:
            # Risk based on move types
            if commutator.move_a.move_type == MoveType.REALLOCATE:
                risk_score += 0.1
            elif commutator.move_a.move_type == MoveType.LIQUIDATE:
                risk_score += 0.3
            elif commutator.move_a.move_type == MoveType.PURCHASE:
                risk_score += 0.2
            
            # Risk based on asset classes involved
            assets_involved = {
                commutator.move_a.from_asset, commutator.move_a.to_asset,
                commutator.move_b.from_asset, commutator.move_b.to_asset
            }
            
            for asset in assets_involved:
                if asset and asset in self.asset_classes:
                    risk_score += self.asset_classes[asset]['risk'] * 0.1
        
        return min(1.0, risk_score)
    
    def _calculate_expected_duration(self, commutator_sequence: List[Commutator]) -> int:
        """Calculate expected duration to execute commutator sequence"""
        # Base duration per commutator
        base_duration_per_commutator = 30  # days
        
        # Complexity factor
        complexity_factor = 1.0 + len(commutator_sequence) * 0.2
        
        total_duration = len(commutator_sequence) * base_duration_per_commutator * complexity_factor
        
        return int(total_duration)
    
    def execute_commutator(self, commutator: Commutator, 
                          current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Execute a single commutator on current state
        
        Args:
            commutator: Commutator to execute
            current_state: Current financial state
            
        Returns:
            Updated financial state
        """
        new_state = current_state.copy()
        
        # Execute move A
        new_state = self._execute_move(commutator.move_a, new_state)
        
        # Execute move B
        new_state = self._execute_move(commutator.move_b, new_state)
        
        # Execute inverse A
        new_state = self._execute_move(commutator.inverse_a, new_state)
        
        # Execute inverse B
        new_state = self._execute_move(commutator.inverse_b, new_state)
        
        return new_state
    
    def _execute_move(self, move: FinancialMove, state: Dict[str, float]) -> Dict[str, float]:
        """Execute a single move on financial state"""
        new_state = state.copy()
        
        if move.move_type == MoveType.REALLOCATE:
            if move.from_asset and move.to_asset:
                # Transfer between assets
                amount = move.amount * sum(new_state.values())  # Convert percentage to amount
                
                if move.from_asset in new_state and new_state[move.from_asset] >= amount:
                    new_state[move.from_asset] -= amount
                    new_state[move.to_asset] = new_state.get(move.to_asset, 0) + amount
            elif move.to_asset:
                # Increase asset (from cash)
                amount = move.amount * sum(new_state.values())
                new_state[move.to_asset] = new_state.get(move.to_asset, 0) + amount
                new_state['cash'] = max(0, new_state.get('cash', 0) - amount)
            elif move.from_asset:
                # Decrease asset (to cash)
                amount = move.amount * sum(new_state.values())
                if move.from_asset in new_state:
                    new_state[move.from_asset] = max(0, new_state[move.from_asset] - amount)
                    new_state['cash'] = new_state.get('cash', 0) + amount
        
        return new_state
    
    def generate_recursive_commutators(self, depth: int = 3) -> List[Commutator]:
        """
        Generate recursive commutators for complex transformations
        
        Args:
            depth: Recursion depth
            
        Returns:
            List of recursive commutators
        """
        if depth <= 0:
            return []
        
        # Generate basic commutators
        basic_commutators = []
        for template_id, template in self.move_templates.items():
            for other_template_id, other_template in self.move_templates.items():
                if template_id != other_template_id:
                    commutator = Commutator(
                        sequence_id=f"recursive_{template_id}_{other_template_id}",
                        move_a=template,
                        move_b=other_template,
                        inverse_a=self._create_inverse_move(template),
                        inverse_b=self._create_inverse_move(other_template),
                        description=f"Recursive: {template.description} + {other_template.description}"
                    )
                    basic_commutators.append(commutator)
        
        if depth == 1:
            return basic_commutators
        
        # Recursively combine commutators
        recursive_commutators = []
        for i, comm1 in enumerate(basic_commutators):
            for j, comm2 in enumerate(basic_commutators):
                if i != j:
                    combined = self._combine_commutators(comm1, comm2)
                    combined.sequence_id = f"recursive_depth_{depth}_{i}_{j}"
                    recursive_commutators.append(combined)
        
        return recursive_commutators + self.generate_recursive_commutators(depth - 1) 