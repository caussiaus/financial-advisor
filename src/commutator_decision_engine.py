"""
Commutator Decision Engine for Financial State Switching
Implements algorithms to transform suboptimal financial states into positive ones
through strategic position redistribution while maintaining capital constraints.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial.distance import cosine
import json
import math
from src.dl_friendly_storage import save_analysis_results


@dataclass
class FinancialState:
    """Represents a financial state with positions and constraints"""
    cash: float
    investments: Dict[str, float]  # asset_type -> amount
    debts: Dict[str, float]  # debt_type -> amount
    income_streams: Dict[str, float]  # income_type -> monthly_amount
    constraints: Dict[str, float]  # constraint_type -> limit
    timestamp: datetime
    state_id: str
    
    def total_wealth(self) -> float:
        """Calculate total net worth"""
        total = self.cash
        total += sum(self.investments.values())
        total -= sum(self.debts.values())
        return total
    
    def available_capital(self) -> float:
        """Calculate available capital for redistribution"""
        return self.cash - self.constraints.get('min_cash_reserve', 0)
    
    def risk_score(self) -> float:
        """Calculate current risk score (0-1)"""
        total_assets = self.cash + sum(self.investments.values())
        if total_assets == 0:
            return 0.5
        
        # Risk based on debt-to-asset ratio and investment concentration
        debt_ratio = sum(self.debts.values()) / total_assets if total_assets > 0 else 0
        concentration_risk = self._calculate_concentration_risk()
        
        return min(1.0, debt_ratio + concentration_risk)
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate risk from investment concentration"""
        total_investments = sum(self.investments.values())
        if total_investments == 0:
            return 0
        
        # Herfindahl index for concentration
        weights = [amount / total_investments for amount in self.investments.values()]
        concentration = sum(w * w for w in weights)
        return concentration


@dataclass
class CommutatorOperation:
    """Represents a commutator operation for state transformation"""
    operation_type: str  # 'rebalance', 'debt_restructure', 'income_optimization'
    source_positions: Dict[str, float]
    target_positions: Dict[str, float]
    expected_impact: Dict[str, float]
    risk_change: float
    capital_required: float
    execution_time: timedelta
    success_probability: float
    
    def commutator_score(self) -> float:
        """Calculate the effectiveness score of this commutator"""
        # Weighted score based on impact, risk, and probability
        impact_score = sum(self.expected_impact.values()) / len(self.expected_impact) if self.expected_impact else 0
        risk_adjusted_score = impact_score * (1 - abs(self.risk_change))
        return risk_adjusted_score * self.success_probability


class CommutatorDecisionEngine:
    """
    Engine for implementing commutator algorithms in financial decision making.
    Transforms suboptimal states into positive ones through strategic operations.
    """
    
    def __init__(self, initial_state: FinancialState):
        self.current_state = initial_state
        self.state_history: List[FinancialState] = [initial_state]
        self.operation_history: List[CommutatorOperation] = []
        self.optimal_states: Dict[str, FinancialState] = {}
        self.constraint_graph = nx.DiGraph()
        self._initialize_constraint_graph()
    
    def _initialize_constraint_graph(self):
        """Initialize the constraint dependency graph"""
        # Add nodes for different constraint types
        constraint_types = ['capital_preservation', 'risk_limits', 'liquidity_requirements', 
                          'income_stability', 'debt_service', 'investment_diversification']
        
        for constraint in constraint_types:
            self.constraint_graph.add_node(constraint)
        
        # Add edges showing constraint dependencies
        self.constraint_graph.add_edge('capital_preservation', 'risk_limits')
        self.constraint_graph.add_edge('risk_limits', 'investment_diversification')
        self.constraint_graph.add_edge('liquidity_requirements', 'capital_preservation')
        self.constraint_graph.add_edge('income_stability', 'debt_service')
        self.constraint_graph.add_edge('debt_service', 'capital_preservation')
    
    def identify_suboptimal_aspects(self) -> Dict[str, float]:
        """Identify suboptimal aspects of current state"""
        suboptimal_indicators = {}
        
        # Risk assessment
        current_risk = self.current_state.risk_score()
        optimal_risk = 0.3  # Target risk level
        suboptimal_indicators['risk_deviation'] = abs(current_risk - optimal_risk)
        
        # Capital efficiency
        available_capital = self.current_state.available_capital()
        total_wealth = self.current_state.total_wealth()
        capital_efficiency = available_capital / total_wealth if total_wealth > 0 else 0
        suboptimal_indicators['capital_efficiency'] = 1 - capital_efficiency
        
        # Debt burden
        total_debt = sum(self.current_state.debts.values())
        debt_ratio = total_debt / total_wealth if total_wealth > 0 else 0
        suboptimal_indicators['debt_burden'] = max(0, debt_ratio - 0.4)  # Target 40% max
        
        # Investment concentration
        concentration_risk = self.current_state._calculate_concentration_risk()
        suboptimal_indicators['concentration_risk'] = concentration_risk
        
        # Income stability
        income_diversity = len(self.current_state.income_streams)
        suboptimal_indicators['income_diversity'] = max(0, 3 - income_diversity) / 3
        
        return suboptimal_indicators
    
    def generate_commutator_operations(self) -> List[CommutatorOperation]:
        """Generate possible commutator operations to improve state"""
        operations = []
        suboptimal_aspects = self.identify_suboptimal_aspects()
        
        # Rebalancing operations
        if suboptimal_aspects.get('concentration_risk', 0) > 0.1:
            operations.extend(self._generate_rebalancing_operations())
        
        # Debt restructuring operations
        if suboptimal_aspects.get('debt_burden', 0) > 0.1:
            operations.extend(self._generate_debt_restructuring_operations())
        
        # Income optimization operations
        if suboptimal_aspects.get('income_diversity', 0) > 0.1:
            operations.extend(self._generate_income_optimization_operations())
        
        # Capital efficiency operations
        if suboptimal_aspects.get('capital_efficiency', 0) > 0.1:
            operations.extend(self._generate_capital_efficiency_operations())
        
        return operations
    
    def _generate_rebalancing_operations(self) -> List[CommutatorOperation]:
        """Generate portfolio rebalancing operations"""
        operations = []
        current_investments = self.current_state.investments.copy()
        
        # Calculate optimal allocation based on risk tolerance
        total_investments = sum(current_investments.values())
        if total_investments == 0:
            return operations
        
        # Define target allocations
        target_allocations = {
            'equities': 0.4,
            'bonds': 0.3,
            'real_estate': 0.2,
            'cash_equivalents': 0.1
        }
        
        # Calculate rebalancing needs
        for asset_type, target_weight in target_allocations.items():
            current_amount = current_investments.get(asset_type, 0)
            target_amount = total_investments * target_weight
            difference = target_amount - current_amount
            
            if abs(difference) > total_investments * 0.05:  # 5% threshold
                operation = CommutatorOperation(
                    operation_type='rebalance',
                    source_positions={asset_type: current_amount},
                    target_positions={asset_type: target_amount},
                    expected_impact={'risk_reduction': 0.1, 'return_optimization': 0.05},
                    risk_change=-0.05,
                    capital_required=abs(difference),
                    execution_time=timedelta(days=1),
                    success_probability=0.9
                )
                operations.append(operation)
        
        return operations
    
    def _generate_debt_restructuring_operations(self) -> List[CommutatorOperation]:
        """Generate debt restructuring operations"""
        operations = []
        current_debts = self.current_state.debts.copy()
        
        # Identify high-cost debt for restructuring
        high_cost_debts = {debt_type: amount for debt_type, amount in current_debts.items() 
                          if self._get_debt_cost(debt_type) > 0.08}  # 8% threshold
        
        for debt_type, amount in high_cost_debts.items():
            # Calculate potential savings from refinancing
            current_cost = self._get_debt_cost(debt_type)
            potential_cost = current_cost * 0.7  # Assume 30% reduction
            annual_savings = amount * (current_cost - potential_cost)
            
            operation = CommutatorOperation(
                operation_type='debt_restructure',
                source_positions={debt_type: amount},
                target_positions={f'refined_{debt_type}': amount},
                expected_impact={'interest_savings': annual_savings, 'cash_flow_improvement': annual_savings/12},
                risk_change=-0.02,
                capital_required=amount * 0.02,  # 2% refinancing cost
                execution_time=timedelta(days=30),
                success_probability=0.85
            )
            operations.append(operation)
        
        return operations
    
    def _generate_income_optimization_operations(self) -> List[CommutatorOperation]:
        """Generate income optimization operations"""
        operations = []
        current_income = self.current_state.income_streams.copy()
        
        # Identify opportunities for income diversification
        if len(current_income) < 3:
            # Suggest new income streams
            potential_streams = ['investment_income', 'side_business', 'rental_income']
            
            for stream in potential_streams:
                if stream not in current_income:
                    estimated_income = self.current_state.total_wealth() * 0.04 / 12  # 4% annual return
                    
                    operation = CommutatorOperation(
                        operation_type='income_optimization',
                        source_positions={},
                        target_positions={stream: estimated_income},
                        expected_impact={'income_diversification': 0.2, 'stability_improvement': 0.15},
                        risk_change=0.01,
                        capital_required=estimated_income * 12 * 0.1,  # 10% of annual income
                        execution_time=timedelta(days=90),
                        success_probability=0.7
                    )
                    operations.append(operation)
        
        return operations
    
    def _generate_capital_efficiency_operations(self) -> List[CommutatorOperation]:
        """Generate capital efficiency operations"""
        operations = []
        available_capital = self.current_state.available_capital()
        
        if available_capital > self.current_state.total_wealth() * 0.1:  # More than 10% idle
            # Suggest investment opportunities
            investment_opportunities = [
                ('high_yield_savings', 0.04, 0.01),
                ('bond_ladder', 0.05, 0.02),
                ('dividend_stocks', 0.07, 0.15)
            ]
            
            for opportunity, expected_return, risk in investment_opportunities:
                operation = CommutatorOperation(
                    operation_type='capital_efficiency',
                    source_positions={'cash': available_capital * 0.3},
                    target_positions={opportunity: available_capital * 0.3},
                    expected_impact={'return_improvement': expected_return, 'capital_utilization': 0.3},
                    risk_change=risk,
                    capital_required=available_capital * 0.3,
                    execution_time=timedelta(days=7),
                    success_probability=0.95
                )
                operations.append(operation)
        
        return operations
    
    def _get_debt_cost(self, debt_type: str) -> float:
        """Get the cost rate for a debt type"""
        debt_costs = {
            'credit_card': 0.18,
            'personal_loan': 0.12,
            'mortgage': 0.06,
            'student_loan': 0.08,
            'business_loan': 0.10
        }
        return debt_costs.get(debt_type, 0.10)
    
    def evaluate_operation_feasibility(self, operation: CommutatorOperation) -> bool:
        """Evaluate if an operation is feasible given current constraints"""
        # Check capital availability
        if operation.capital_required > self.current_state.available_capital():
            return False
        
        # Check risk constraints
        current_risk = self.current_state.risk_score()
        new_risk = current_risk + operation.risk_change
        if new_risk > self.current_state.constraints.get('max_risk', 0.7):
            return False
        
        # Check liquidity constraints
        if operation.operation_type == 'rebalance':
            required_liquidity = sum(operation.source_positions.values())
            if required_liquidity > self.current_state.cash * 0.8:
                return False
        
        return True
    
    def select_optimal_commutator_sequence(self, operations: List[CommutatorOperation]) -> List[CommutatorOperation]:
        """Select the optimal sequence of commutator operations"""
        feasible_operations = [op for op in operations if self.evaluate_operation_feasibility(op)]
        
        if not feasible_operations:
            return []
        
        # Sort by commutator score (effectiveness)
        feasible_operations.sort(key=lambda op: op.commutator_score(), reverse=True)
        
        # Select operations that can be executed within capital constraints
        selected_operations = []
        remaining_capital = self.current_state.available_capital()
        
        for operation in feasible_operations:
            if operation.capital_required <= remaining_capital:
                selected_operations.append(operation)
                remaining_capital -= operation.capital_required
        
        return selected_operations
    
    def execute_commutator_sequence(self, operations: List[CommutatorOperation]) -> bool:
        """Execute a sequence of commutator operations"""
        if not operations:
            return False
        
        # Create new state based on operations
        new_state = self._apply_operations_to_state(operations)
        
        # Validate the new state
        if self._validate_state_transition(new_state):
            self.current_state = new_state
            self.state_history.append(new_state)
            self.operation_history.extend(operations)
            return True
        
        return False
    
    def execute_commutator_sequence_by_type(self, operation_types: List[str] = None) -> Dict[str, any]:
        """
        Execute a sequence of commutator operations by type
        """
        # Generate all possible operations
        all_operations = self.generate_commutator_operations()
        
        # Filter by operation types if specified
        if operation_types:
            filtered_operations = [op for op in all_operations if op.operation_type in operation_types]
        else:
            filtered_operations = all_operations
        
        # Select optimal sequence
        optimal_sequence = self.select_optimal_commutator_sequence(filtered_operations)
        
        if not optimal_sequence:
            return {'success': False, 'error': 'No feasible operations found'}
        
        # Execute sequence
        success = self.execute_commutator_sequence(optimal_sequence)
        
        return {
            'success': success,
            'operations_executed': len(optimal_sequence),
            'operation_types': [op.operation_type for op in optimal_sequence],
            'total_impact': sum(sum(op.expected_impact.values()) for op in optimal_sequence),
            'risk_change': sum(op.risk_change for op in optimal_sequence)
        }
    
    def _apply_operations_to_state(self, operations: List[CommutatorOperation]) -> FinancialState:
        """Apply operations to create a new state"""
        new_investments = self.current_state.investments.copy()
        new_debts = self.current_state.debts.copy()
        new_income = self.current_state.income_streams.copy()
        new_cash = self.current_state.cash
        
        for operation in operations:
            # Apply source position changes
            for position, amount in operation.source_positions.items():
                if position in new_investments:
                    new_investments[position] -= amount
                elif position in new_debts:
                    new_debts[position] -= amount
                elif position == 'cash':
                    new_cash -= amount
            
            # Apply target position changes
            for position, amount in operation.target_positions.items():
                if position.startswith('refined_'):
                    # Debt restructuring
                    original_debt = position.replace('refined_', '')
                    if original_debt in new_debts:
                        new_debts[original_debt] -= amount
                        new_debts[position] = amount
                elif position in ['investment_income', 'side_business', 'rental_income']:
                    # Income optimization
                    new_income[position] = amount
                else:
                    # Investment rebalancing or capital efficiency
                    new_investments[position] = amount
            
            # Apply capital requirements
            new_cash -= operation.capital_required
        
        return FinancialState(
            cash=new_cash,
            investments=new_investments,
            debts=new_debts,
            income_streams=new_income,
            constraints=self.current_state.constraints,
            timestamp=datetime.now(),
            state_id=f"state_{len(self.state_history) + 1}"
        )
    
    def _validate_state_transition(self, new_state: FinancialState) -> bool:
        """Validate that a state transition is acceptable"""
        # Check that total wealth hasn't decreased significantly
        wealth_change = (new_state.total_wealth() - self.current_state.total_wealth()) / self.current_state.total_wealth()
        if wealth_change < -0.05:  # 5% threshold
            return False
        
        # Check that risk hasn't increased beyond acceptable levels
        risk_change = new_state.risk_score() - self.current_state.risk_score()
        if risk_change > 0.1:  # 10% threshold
            return False
        
        # Check that cash reserves are maintained
        min_cash = self.current_state.constraints.get('min_cash_reserve', 0)
        if new_state.cash < min_cash:
            return False
        
        return True
    
    def optimize_state(self, target_metrics: Dict[str, float]) -> bool:
        """Optimize current state towards target metrics"""
        # Generate possible operations
        operations = self.generate_commutator_operations()
        
        # Select optimal sequence
        optimal_sequence = self.select_optimal_commutator_sequence(operations)
        
        # Execute the sequence
        success = self.execute_commutator_sequence(optimal_sequence)
        
        return success
    
    def get_state_analysis(self) -> Dict[str, any]:
        """Get comprehensive analysis of current state"""
        return {
            'current_state': {
                'total_wealth': self.current_state.total_wealth(),
                'risk_score': self.current_state.risk_score(),
                'available_capital': self.current_state.available_capital(),
                'debt_ratio': sum(self.current_state.debts.values()) / self.current_state.total_wealth(),
                'income_diversity': len(self.current_state.income_streams)
            },
            'suboptimal_aspects': self.identify_suboptimal_aspects(),
            'recommended_operations': [
                {
                    'type': op.operation_type,
                    'impact': op.expected_impact,
                    'risk_change': op.risk_change,
                    'capital_required': op.capital_required,
                    'success_probability': op.success_probability
                }
                for op in self.generate_commutator_operations()
            ],
            'state_history': len(self.state_history),
            'operations_executed': len(self.operation_history)
        }
    
    def export_state_data(self, filepath: str):
        """Export current state and analysis data"""
        export_data = {
            'current_state': {
                'cash': self.current_state.cash,
                'investments': self.current_state.investments,
                'debts': self.current_state.debts,
                'income_streams': self.current_state.income_streams,
                'constraints': self.current_state.constraints,
                'timestamp': self.current_state.timestamp.isoformat(),
                'state_id': self.current_state.state_id
            },
            'analysis': self.get_state_analysis(),
            'operation_history': [
                {
                    'type': op.operation_type,
                    'source_positions': op.source_positions,
                    'target_positions': op.target_positions,
                    'expected_impact': op.expected_impact,
                    'risk_change': op.risk_change,
                    'capital_required': op.capital_required,
                    'success_probability': op.success_probability
                }
                for op in self.operation_history
            ]
        }
        
        save_analysis_results(export_data, filepath)


# Convenience functions for integration
def create_commutator_engine(initial_state_dict: Dict[str, any]) -> CommutatorDecisionEngine:
    """Create a commutator engine from a state dictionary"""
    state = FinancialState(
        cash=initial_state_dict.get('cash', 0),
        investments=initial_state_dict.get('investments', {}),
        debts=initial_state_dict.get('debts', {}),
        income_streams=initial_state_dict.get('income_streams', {}),
        constraints=initial_state_dict.get('constraints', {}),
        timestamp=datetime.now(),
        state_id='initial_state'
    )
    
    return CommutatorDecisionEngine(state)


def optimize_financial_state(state_dict: Dict[str, any], target_metrics: Dict[str, float]) -> Dict[str, any]:
    """Optimize a financial state using commutator algorithms"""
    engine = create_commutator_engine(state_dict)
    success = engine.optimize_state(target_metrics)
    
    return {
        'success': success,
        'analysis': engine.get_state_analysis(),
        'optimized_state': {
            'cash': engine.current_state.cash,
            'investments': engine.current_state.investments,
            'debts': engine.current_state.debts,
            'income_streams': engine.current_state.income_streams
        }
    } 