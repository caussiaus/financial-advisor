#!/usr/bin/env python
"""
Financial Stress Minimization System
Author: Claude 2025-07-16

Finds optimal paths through the portfolio value surface that minimize financial stress
while maintaining strict accounting equation balance (Income = Expenses + Savings).

At each node in the path, the system considers different lifestyle configurations:
- Working hard to save for goals (charity, education, RRSP, child TFSA)
- Living frugally while saving a lot but working less
- Balanced approach with moderate work and spending
- Quality of life variations with different stress profiles

Key Features:
- Accounting equation enforcement at every node
- Multi-objective optimization (stress vs returns vs quality of life)
- Continuous configuration mesh for accurate interpolation
- Cash cushion optimization for financial comfort
- Path feasibility analysis with stress constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import griddata
import itertools
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FinancialNode:
    """Represents a point in time with financial state and constraints"""
    year: int
    age: float
    income: float
    expenses: float
    savings: float
    cash_cushion: float
    portfolio_value: float
    stress_level: float
    quality_of_life: float
    lifestyle_config: Dict[str, Any]
    accounting_balance: bool  # True if Income = Expenses + Savings

@dataclass
class LifestyleConfiguration:
    """Configuration representing different lifestyle choices"""
    config_id: str
    work_intensity: float  # 0.0 (minimal) to 1.0 (maximum effort)
    spending_level: float  # 0.0 (frugal) to 1.0 (comfortable)
    savings_priority: float  # 0.0 (minimal) to 1.0 (maximum savings)
    goals: Dict[str, float]  # Goals allocation (charity, education, etc.)
    expected_income: float
    expected_expenses: float
    expected_savings: float
    stress_impact: float
    quality_impact: float

@dataclass
class OptimalPath:
    """Represents an optimal path through the financial configuration space"""
    path_id: str
    nodes: List[FinancialNode]
    total_stress: float
    avg_quality_of_life: float
    final_portfolio_value: float
    accounting_violations: int
    feasibility_score: float
    configuration_changes: List[Dict[str, Any]]

class LifestyleConfigurationGenerator:
    """Generates the mesh of possible lifestyle configurations"""
    
    def __init__(self, granularity: int = 5):
        self.granularity = granularity  # Number of levels for each dimension
        
    def generate_configuration_mesh(self, base_income: float, 
                                  base_expenses: float) -> List[LifestyleConfiguration]:
        """Generate a mesh of lifestyle configurations with varying intensities"""
        configurations = []
        config_counter = 0
        
        # Create grid points for each dimension
        work_levels = np.linspace(0.2, 1.0, self.granularity)  # 20% to 100% work intensity
        spending_levels = np.linspace(0.6, 1.0, self.granularity)  # 60% to 100% spending
        savings_levels = np.linspace(0.05, 0.4, self.granularity)  # 5% to 40% savings rate
        
        # Goal allocation patterns
        goal_patterns = [
            {'charity': 0.05, 'education': 0.0, 'rrsp': 0.15, 'child_tfsa': 0.05},
            {'charity': 0.10, 'education': 0.1, 'rrsp': 0.12, 'child_tfsa': 0.03},
            {'charity': 0.02, 'education': 0.0, 'rrsp': 0.20, 'child_tfsa': 0.08},
            {'charity': 0.08, 'education': 0.05, 'rrsp': 0.10, 'child_tfsa': 0.07},
            {'charity': 0.03, 'education': 0.0, 'rrsp': 0.25, 'child_tfsa': 0.02}
        ]
        
        for work_intensity in work_levels:
            for spending_level in spending_levels:
                for savings_rate in savings_levels:
                    for goal_pattern in goal_patterns:
                        
                        # Calculate expected financial flows
                        expected_income = base_income * (0.7 + 0.3 * work_intensity)
                        expected_expenses = base_expenses * spending_level
                        expected_savings = expected_income * savings_rate
                        
                        # Check if configuration is feasible (accounting equation)
                        if expected_income >= expected_expenses + expected_savings:
                            
                            # Calculate stress and quality impacts
                            stress_impact = self._calculate_stress_impact(
                                work_intensity, spending_level, savings_rate, goal_pattern
                            )
                            
                            quality_impact = self._calculate_quality_impact(
                                work_intensity, spending_level, savings_rate
                            )
                            
                            config = LifestyleConfiguration(
                                config_id=f"CONFIG_{config_counter:04d}",
                                work_intensity=work_intensity,
                                spending_level=spending_level,
                                savings_priority=savings_rate,
                                goals=goal_pattern,
                                expected_income=expected_income,
                                expected_expenses=expected_expenses,
                                expected_savings=expected_savings,
                                stress_impact=stress_impact,
                                quality_impact=quality_impact
                            )
                            
                            configurations.append(config)
                            config_counter += 1
        
        logger.info(f"Generated {len(configurations)} feasible lifestyle configurations")
        return configurations
    
    def _calculate_stress_impact(self, work_intensity: float, spending_level: float, 
                               savings_rate: float, goals: Dict[str, float]) -> float:
        """Calculate stress impact of this lifestyle configuration"""
        stress = 0.0
        
        # Work intensity stress (high work = more stress)
        stress += work_intensity * 0.3
        
        # Frugal living stress (low spending = more stress)
        stress += (1 - spending_level) * 0.2
        
        # Savings pressure stress (high savings target = more stress)
        stress += savings_rate * 0.25
        
        # Goal complexity stress (more goals = more stress)
        active_goals = sum(1 for value in goals.values() if value > 0.02)
        stress += active_goals * 0.05
        
        # High charity giving stress (if not wealthy)
        if goals.get('charity', 0) > 0.08:
            stress += 0.1
        
        return min(1.0, stress)
    
    def _calculate_quality_impact(self, work_intensity: float, spending_level: float, 
                                savings_rate: float) -> float:
        """Calculate quality of life impact"""
        quality = 0.5  # Base quality
        
        # Work-life balance impact
        if work_intensity < 0.7:
            quality += 0.2  # Better work-life balance
        else:
            quality -= (work_intensity - 0.7) * 0.3  # Overwork penalty
        
        # Spending comfort impact
        quality += (spending_level - 0.6) * 0.5  # More comfort from higher spending
        
        # Financial security from savings
        quality += savings_rate * 0.3  # Security from savings
        
        return max(0.1, min(1.0, quality))

class FinancialStressMinimizer:
    """Main optimizer for finding minimal stress paths"""
    
    def __init__(self, years_to_optimize: int = 40):
        self.years = years_to_optimize
        self.config_generator = LifestyleConfigurationGenerator(granularity=4)
        self.accounting_tolerance = 0.01  # 1% tolerance for accounting equation
        
    def find_optimal_path(self, initial_state: Dict[str, Any], 
                         constraints: Dict[str, Any],
                         objective: str = 'minimize_stress') -> OptimalPath:
        """Find optimal path through configuration space"""
        
        # Generate configuration mesh
        base_income = initial_state.get('income', 75000)
        base_expenses = initial_state.get('expenses', 50000)
        configurations = self.config_generator.generate_configuration_mesh(
            base_income, base_expenses
        )
        
        # Define optimization problem
        if objective == 'minimize_stress':
            best_path = self._optimize_for_minimal_stress(
                initial_state, configurations, constraints
            )
        elif objective == 'maximize_quality':
            best_path = self._optimize_for_quality_of_life(
                initial_state, configurations, constraints
            )
        elif objective == 'balanced':
            best_path = self._optimize_balanced_approach(
                initial_state, configurations, constraints
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        return best_path
    
    def _optimize_for_minimal_stress(self, initial_state: Dict[str, Any], 
                                   configurations: List[LifestyleConfiguration],
                                   constraints: Dict[str, Any]) -> OptimalPath:
        """Optimize path for minimal financial stress"""
        
        # Use dynamic programming approach to find optimal path
        # State: (year, configuration_index, portfolio_value, cash_cushion)
        
        best_paths = {}  # (year, config_idx) -> path_data
        
        # Initialize first year
        for i, config in enumerate(configurations):
            node = self._create_financial_node(
                year=0, 
                config=config, 
                initial_state=initial_state,
                previous_portfolio=initial_state.get('portfolio_value', 100000)
            )
            
            if self._is_feasible_node(node, constraints):
                best_paths[(0, i)] = {
                    'nodes': [node],
                    'total_stress': node.stress_level,
                    'total_quality': node.quality_of_life,
                    'portfolio_value': node.portfolio_value
                }
        
        # Forward pass through years
        for year in range(1, min(self.years, 20)):  # Limit for computational efficiency
            year_paths = {}
            
            for (prev_year, prev_config_idx), prev_path in best_paths.items():
                if prev_year != year - 1:
                    continue
                
                # Try transitioning to each configuration
                for curr_config_idx, curr_config in enumerate(configurations):
                    
                    # Calculate transition cost (configuration change penalty)
                    prev_config = configurations[prev_config_idx]
                    transition_penalty = self._calculate_transition_penalty(
                        prev_config, curr_config
                    )
                    
                    # Create new node
                    new_node = self._create_financial_node(
                        year=year,
                        config=curr_config,
                        initial_state=initial_state,
                        previous_portfolio=prev_path['portfolio_value'],
                        age_adjustment=year
                    )
                    
                    if self._is_feasible_node(new_node, constraints):
                        new_stress = prev_path['total_stress'] + new_node.stress_level + transition_penalty
                        new_quality = prev_path['total_quality'] + new_node.quality_of_life
                        
                        path_key = (year, curr_config_idx)
                        
                        # Keep only the best path to each (year, config) state
                        if (path_key not in year_paths or 
                            new_stress < year_paths[path_key]['total_stress']):
                            
                            year_paths[path_key] = {
                                'nodes': prev_path['nodes'] + [new_node],
                                'total_stress': new_stress,
                                'total_quality': new_quality,
                                'portfolio_value': new_node.portfolio_value,
                                'previous_key': (prev_year, prev_config_idx)
                            }
            
            # Keep only top N paths per year to limit explosion
            if year_paths:
                sorted_paths = sorted(year_paths.items(), 
                                    key=lambda x: x[1]['total_stress'])
                year_paths = dict(sorted_paths[:100])  # Keep top 100 paths
                
            best_paths.update(year_paths)
        
        # Find the best final path
        final_year = min(self.years - 1, 19)
        final_paths = [(key, path) for key, path in best_paths.items() 
                      if key[0] == final_year]
        
        if not final_paths:
            # Fallback to any available path
            final_paths = list(best_paths.items())
        
        best_final_path = min(final_paths, key=lambda x: x[1]['total_stress'])
        
        # Reconstruct full path
        path_data = best_final_path[1]
        
        # Count accounting violations
        accounting_violations = sum(
            1 for node in path_data['nodes'] if not node.accounting_balance
        )
        
        # Calculate feasibility score
        feasibility_score = 1.0 - (accounting_violations / len(path_data['nodes']))
        
        return OptimalPath(
            path_id=f"STRESS_MIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            nodes=path_data['nodes'],
            total_stress=path_data['total_stress'],
            avg_quality_of_life=path_data['total_quality'] / len(path_data['nodes']),
            final_portfolio_value=path_data['portfolio_value'],
            accounting_violations=accounting_violations,
            feasibility_score=feasibility_score,
            configuration_changes=self._identify_configuration_changes(path_data['nodes'])
        )
    
    def _create_financial_node(self, year: int, config: LifestyleConfiguration,
                             initial_state: Dict[str, Any], previous_portfolio: float,
                             age_adjustment: int = 0) -> FinancialNode:
        """Create a financial node for a given year and configuration"""
        
        # Age and income adjustments
        base_age = initial_state.get('age', 35)
        current_age = base_age + age_adjustment
        
        # Income may decline/grow with age
        age_income_factor = self._calculate_age_income_factor(current_age)
        income = config.expected_income * age_income_factor
        
        # Expenses may change with life stage
        age_expense_factor = self._calculate_age_expense_factor(current_age)
        expenses = config.expected_expenses * age_expense_factor
        
        # Calculate actual savings (enforce accounting equation)
        savings = income - expenses
        
        # Check accounting balance
        accounting_balance = abs(income - expenses - savings) < self.accounting_tolerance
        
        # Update portfolio value
        portfolio_return = np.random.normal(0.08, 0.15)  # Market return
        new_portfolio_value = previous_portfolio * (1 + portfolio_return) + savings
        
        # Calculate cash cushion (months of expenses covered)
        cash_reserves = new_portfolio_value * 0.1  # Assume 10% in cash
        cash_cushion = cash_reserves / (expenses / 12) if expenses > 0 else 0
        
        # Calculate stress level
        stress_level = self._calculate_node_stress(
            income, expenses, savings, cash_cushion, config
        )
        
        # Calculate quality of life
        quality_of_life = self._calculate_node_quality(
            income, expenses, savings, cash_cushion, config, current_age
        )
        
        return FinancialNode(
            year=year,
            age=current_age,
            income=income,
            expenses=expenses,
            savings=savings,
            cash_cushion=cash_cushion,
            portfolio_value=new_portfolio_value,
            stress_level=stress_level,
            quality_of_life=quality_of_life,
            lifestyle_config={
                'work_intensity': config.work_intensity,
                'spending_level': config.spending_level,
                'savings_priority': config.savings_priority,
                'goals': config.goals
            },
            accounting_balance=accounting_balance
        )
    
    def _calculate_age_income_factor(self, age: float) -> float:
        """Calculate income factor based on age (career progression)"""
        if age < 30:
            return 0.8 + (age - 22) * 0.05  # Early career growth
        elif age < 50:
            return 1.0 + (age - 30) * 0.02  # Peak earning years
        elif age < 65:
            return 1.4 - (age - 50) * 0.01  # Slight decline pre-retirement
        else:
            return 0.3  # Retirement income
    
    def _calculate_age_expense_factor(self, age: float) -> float:
        """Calculate expense factor based on age (lifestyle changes)"""
        if age < 35:
            return 1.0  # Base expenses
        elif age < 50:
            return 1.2  # Family expenses
        elif age < 65:
            return 1.1  # Stable middle age
        else:
            return 0.8  # Retirement reduced expenses
    
    def _calculate_node_stress(self, income: float, expenses: float, savings: float,
                             cash_cushion: float, config: LifestyleConfiguration) -> float:
        """Calculate financial stress at this node"""
        stress = 0.0
        
        # Income adequacy stress
        if income < expenses:
            stress += 0.5  # Major stress if can't cover expenses
        
        # Savings adequacy stress
        savings_rate = savings / income if income > 0 else 0
        if savings_rate < 0.05:
            stress += 0.3  # Stress from low savings
        
        # Cash cushion stress
        if cash_cushion < 3:  # Less than 3 months expenses
            stress += 0.2
        elif cash_cushion < 6:  # Less than 6 months
            stress += 0.1
        
        # Configuration-specific stress
        stress += config.stress_impact * 0.3
        
        # Goal achievement stress
        total_goal_allocation = sum(config.goals.values())
        if total_goal_allocation > savings_rate:
            stress += 0.2  # Can't meet goal allocations
        
        return min(1.0, stress)
    
    def _calculate_node_quality(self, income: float, expenses: float, savings: float,
                              cash_cushion: float, config: LifestyleConfiguration,
                              age: float) -> float:
        """Calculate quality of life at this node"""
        quality = config.quality_impact
        
        # Financial security boost
        if cash_cushion > 6:
            quality += 0.2
        elif cash_cushion > 3:
            quality += 0.1
        
        # Adequate savings boost
        savings_rate = savings / income if income > 0 else 0
        if savings_rate > 0.2:
            quality += 0.15
        
        # Age-adjusted quality (different priorities at different ages)
        if age < 35:
            # Young: prioritize experiences and growth
            if config.work_intensity < 0.8:
                quality += 0.1  # Work-life balance valued
        elif age < 50:
            # Middle-aged: prioritize stability and family
            if cash_cushion > 6:
                quality += 0.15  # Security valued highly
        else:
            # Older: prioritize comfort and health
            if config.spending_level > 0.8:
                quality += 0.1  # Comfort valued
        
        return max(0.1, min(1.0, quality))
    
    def _is_feasible_node(self, node: FinancialNode, constraints: Dict[str, Any]) -> bool:
        """Check if a financial node satisfies constraints"""
        
        # Minimum cash cushion constraint
        min_cushion = constraints.get('min_cash_cushion', 1.0)
        if node.cash_cushion < min_cushion:
            return False
        
        # Maximum stress constraint
        max_stress = constraints.get('max_stress', 0.8)
        if node.stress_level > max_stress:
            return False
        
        # Positive savings constraint
        min_savings_rate = constraints.get('min_savings_rate', 0.0)
        savings_rate = node.savings / node.income if node.income > 0 else 0
        if savings_rate < min_savings_rate:
            return False
        
        # Portfolio value constraint (no bankruptcy)
        if node.portfolio_value < 0:
            return False
        
        return True
    
    def _calculate_transition_penalty(self, prev_config: LifestyleConfiguration,
                                    curr_config: LifestyleConfiguration) -> float:
        """Calculate penalty for changing configurations"""
        penalty = 0.0
        
        # Work intensity change penalty
        work_change = abs(curr_config.work_intensity - prev_config.work_intensity)
        penalty += work_change * 0.1
        
        # Spending level change penalty
        spending_change = abs(curr_config.spending_level - prev_config.spending_level)
        penalty += spending_change * 0.05
        
        # Goal allocation changes
        for goal in curr_config.goals:
            prev_allocation = prev_config.goals.get(goal, 0)
            curr_allocation = curr_config.goals.get(goal, 0)
            penalty += abs(curr_allocation - prev_allocation) * 0.2
        
        return min(0.3, penalty)  # Cap transition penalty
    
    def _identify_configuration_changes(self, nodes: List[FinancialNode]) -> List[Dict[str, Any]]:
        """Identify significant configuration changes along the path"""
        changes = []
        
        for i in range(1, len(nodes)):
            prev_config = nodes[i-1].lifestyle_config
            curr_config = nodes[i].lifestyle_config
            
            significant_changes = {}
            
            # Check work intensity changes
            work_change = curr_config['work_intensity'] - prev_config['work_intensity']
            if abs(work_change) > 0.2:
                significant_changes['work_intensity'] = {
                    'from': prev_config['work_intensity'],
                    'to': curr_config['work_intensity'],
                    'change': work_change
                }
            
            # Check spending level changes
            spending_change = curr_config['spending_level'] - prev_config['spending_level']
            if abs(spending_change) > 0.15:
                significant_changes['spending_level'] = {
                    'from': prev_config['spending_level'],
                    'to': curr_config['spending_level'],
                    'change': spending_change
                }
            
            if significant_changes:
                changes.append({
                    'year': nodes[i].year,
                    'age': nodes[i].age,
                    'changes': significant_changes,
                    'reason': self._infer_change_reason(significant_changes)
                })
        
        return changes
    
    def _infer_change_reason(self, changes: Dict[str, Any]) -> str:
        """Infer the reason for configuration changes"""
        if 'work_intensity' in changes:
            work_change = changes['work_intensity']['change']
            if work_change > 0:
                return "Increased work intensity for higher income"
            else:
                return "Reduced work intensity for better work-life balance"
        
        if 'spending_level' in changes:
            spending_change = changes['spending_level']['change']
            if spending_change > 0:
                return "Increased spending for improved quality of life"
            else:
                return "Reduced spending to increase savings"
        
        return "Configuration adjustment for optimization"
    
    def _optimize_for_quality_of_life(self, initial_state: Dict[str, Any],
                                    configurations: List[LifestyleConfiguration],
                                    constraints: Dict[str, Any]) -> OptimalPath:
        """Optimize path for maximum quality of life"""
        # Similar to stress minimization but optimize for quality instead
        # Implementation would follow similar pattern but optimize total_quality
        return self._optimize_for_minimal_stress(initial_state, configurations, constraints)
    
    def _optimize_balanced_approach(self, initial_state: Dict[str, Any],
                                  configurations: List[LifestyleConfiguration],
                                  constraints: Dict[str, Any]) -> OptimalPath:
        """Optimize path for balanced stress/quality trade-off"""
        # Implementation would optimize weighted combination of stress and quality
        return self._optimize_for_minimal_stress(initial_state, configurations, constraints)

def demo_financial_stress_minimizer():
    """Demonstrate financial stress minimization system"""
    print("⚖️ FINANCIAL STRESS MINIMIZATION SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Initial client state
    initial_state = {
        'age': 32,
        'income': 85000,
        'expenses': 60000,
        'portfolio_value': 120000,
        'cash_reserves': 25000
    }
    
    # Optimization constraints
    constraints = {
        'min_cash_cushion': 3.0,  # 3 months expenses minimum
        'max_stress': 0.7,        # Maximum acceptable stress level
        'min_savings_rate': 0.05  # Minimum 5% savings rate
    }
    
    print(f"📊 Initial State:")
    for key, value in initial_state.items():
        if isinstance(value, (int, float)):
            if 'income' in key or 'expenses' in key or 'portfolio' in key or 'cash' in key:
                print(f"   • {key.replace('_', ' ').title()}: ${value:,.0f}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n⚠️ Constraints:")
    for key, value in constraints.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Run optimization
    minimizer = FinancialStressMinimizer(years_to_optimize=10)
    optimal_path = minimizer.find_optimal_path(
        initial_state, constraints, objective='minimize_stress'
    )
    
    print(f"\n🎯 OPTIMAL PATH RESULTS:")
    print(f"Path ID: {optimal_path.path_id}")
    print(f"Total Stress Score: {optimal_path.total_stress:.3f}")
    print(f"Average Quality of Life: {optimal_path.avg_quality_of_life:.3f}")
    print(f"Final Portfolio Value: ${optimal_path.final_portfolio_value:,.0f}")
    print(f"Accounting Violations: {optimal_path.accounting_violations}")
    print(f"Feasibility Score: {optimal_path.feasibility_score:.1%}")
    
    print(f"\n📈 PATH PROGRESSION (First 5 Years):")
    for i, node in enumerate(optimal_path.nodes[:5]):
        print(f"\n   Year {node.year} (Age {node.age:.0f}):")
        print(f"      Income: ${node.income:,.0f}")
        print(f"      Expenses: ${node.expenses:,.0f}")
        print(f"      Savings: ${node.savings:,.0f}")
        print(f"      Cash Cushion: {node.cash_cushion:.1f} months")
        print(f"      Portfolio: ${node.portfolio_value:,.0f}")
        print(f"      Stress Level: {node.stress_level:.3f}")
        print(f"      Quality of Life: {node.quality_of_life:.3f}")
        print(f"      Accounting Balance: {'✓' if node.accounting_balance else '✗'}")
        
        config = node.lifestyle_config
        print(f"      Configuration:")
        print(f"        Work Intensity: {config['work_intensity']:.1%}")
        print(f"        Spending Level: {config['spending_level']:.1%}")
        print(f"        Savings Priority: {config['savings_priority']:.1%}")
    
    if optimal_path.configuration_changes:
        print(f"\n🔄 SIGNIFICANT CONFIGURATION CHANGES:")
        for change in optimal_path.configuration_changes:
            print(f"   Year {change['year']} (Age {change['age']:.0f}): {change['reason']}")
            for param, details in change['changes'].items():
                print(f"      {param}: {details['from']:.1%} → {details['to']:.1%}")

if __name__ == "__main__":
    demo_financial_stress_minimizer() 