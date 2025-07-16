#!/usr/bin/env python
"""
Fuzzy Sets Qualitative Comparative Analysis (fsQCA) Optimizer
Author: Claude 2025-07-16

Implements proper fsQCA methodology with continuous membership functions to identify
set combinations that consistently lead to financial stability and low stress.

Based on fsQCA methodology for finding:
- Necessary conditions for financial stability
- Sufficient combinations of conditions for low stress
- Consistency and coverage metrics for causal relationships
- Continuous fuzzy membership scales (0.0 to 1.0)

Key Features:
- Proper fuzzy membership function calibration
- Truth table construction with continuous values
- Consistency and coverage analysis
- Complex, parsimonious, and intermediate solutions
- Path optimization for minimal financial stress
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import itertools
from scipy.optimize import minimize
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class FuzzyCondition:
    """Represents a fuzzy condition with membership function"""
    name: str
    raw_values: List[float]
    membership_values: List[float]
    calibration_anchors: Tuple[float, float, float]  # full_non_membership, crossover, full_membership
    description: str

@dataclass
class fsQCAResult:
    """Results from fsQCA analysis"""
    conditions: Dict[str, FuzzyCondition]
    outcome: FuzzyCondition
    necessity_analysis: Dict[str, Dict[str, float]]
    sufficiency_analysis: Dict[str, Any]
    optimal_paths: List[Dict[str, Any]]
    consistency_threshold: float
    coverage_threshold: float

class FuzzyMembershipCalibrator:
    """Calibrates raw values into fuzzy membership functions"""
    
    @staticmethod
    def direct_method(values: List[float], full_non: float, crossover: float, full_mem: float) -> List[float]:
        """Direct method calibration using three anchor points"""
        memberships = []
        
        for value in values:
            if value <= full_non:
                membership = 0.0
            elif value >= full_mem:
                membership = 1.0
            elif value == crossover:
                membership = 0.5
            elif value < crossover:
                # Between full_non and crossover
                log_odds = np.log(0.5 / (1 - 0.5)) - np.log((crossover - full_non) / (value - full_non))
                membership = np.exp(log_odds) / (1 + np.exp(log_odds))
            else:
                # Between crossover and full_mem
                log_odds = np.log(0.5 / (1 - 0.5)) + np.log((value - crossover) / (full_mem - crossover))
                membership = np.exp(log_odds) / (1 + np.exp(log_odds))
            
            # Ensure membership is in [0, 1] and avoid exact 0.5
            membership = max(0.001, min(0.999, membership))
            if abs(membership - 0.5) < 0.001:
                membership = 0.501 if membership > 0.5 else 0.499
            
            memberships.append(membership)
        
        return memberships
    
    @staticmethod
    def percentile_method(values: List[float], percentiles: Tuple[float, float, float] = (10, 50, 90)) -> List[float]:
        """Percentile-based calibration"""
        sorted_values = sorted(values)
        anchors = [np.percentile(sorted_values, p) for p in percentiles]
        return FuzzyMembershipCalibrator.direct_method(values, anchors[0], anchors[1], anchors[2])

class fsQCAAnalyzer:
    """Main fsQCA analyzer with proper methodology"""
    
    def __init__(self, consistency_threshold: float = 0.8, coverage_threshold: float = 0.6):
        self.consistency_threshold = consistency_threshold
        self.coverage_threshold = coverage_threshold
        self.calibrator = FuzzyMembershipCalibrator()
    
    def prepare_conditions(self, data: Dict[str, List[float]], 
                         calibration_spec: Dict[str, Dict[str, Any]]) -> Dict[str, FuzzyCondition]:
        """Prepare fuzzy conditions from raw data"""
        conditions = {}
        
        for condition_name, raw_values in data.items():
            if condition_name in calibration_spec:
                spec = calibration_spec[condition_name]
                
                if spec['method'] == 'direct':
                    anchors = spec['anchors']
                    membership_values = self.calibrator.direct_method(
                        raw_values, anchors[0], anchors[1], anchors[2]
                    )
                elif spec['method'] == 'percentile':
                    percentiles = spec.get('percentiles', (10, 50, 90))
                    membership_values = self.calibrator.percentile_method(raw_values, percentiles)
                    anchors = (np.percentile(raw_values, percentiles[0]),
                             np.percentile(raw_values, percentiles[1]),
                             np.percentile(raw_values, percentiles[2]))
                else:
                    raise ValueError(f"Unknown calibration method: {spec['method']}")
                
                conditions[condition_name] = FuzzyCondition(
                    name=condition_name,
                    raw_values=raw_values,
                    membership_values=membership_values,
                    calibration_anchors=anchors,
                    description=spec.get('description', condition_name)
                )
        
        return conditions
    
    def analyze_necessity(self, conditions: Dict[str, FuzzyCondition], 
                         outcome: FuzzyCondition) -> Dict[str, Dict[str, float]]:
        """Analyze necessary conditions for the outcome"""
        necessity_results = {}
        
        outcome_memberships = outcome.membership_values
        
        for condition_name, condition in conditions.items():
            condition_memberships = condition.membership_values
            
            # Calculate consistency of necessity (coverage of outcome)
            # Consistency = sum(min(Xi, Yi)) / sum(Yi)
            min_values = [min(xi, yi) for xi, yi in zip(condition_memberships, outcome_memberships)]
            necessity_consistency = sum(min_values) / sum(outcome_memberships)
            
            # Calculate coverage of necessity (coverage of condition)
            # Coverage = sum(min(Xi, Yi)) / sum(Xi)
            necessity_coverage = sum(min_values) / sum(condition_memberships)
            
            # Relevance of necessity (RoN)
            # RoN = Consistency - Coverage_of_~condition_for_outcome
            not_condition = [1 - x for x in condition_memberships]
            min_not_values = [min(xi, yi) for xi, yi in zip(not_condition, outcome_memberships)]
            coverage_not_condition = sum(min_not_values) / sum(outcome_memberships)
            relevance = necessity_consistency - coverage_not_condition
            
            necessity_results[condition_name] = {
                'consistency': necessity_consistency,
                'coverage': necessity_coverage,
                'relevance': relevance,
                'necessary': necessity_consistency >= 0.9  # Standard threshold for necessity
            }
        
        return necessity_results
    
    def construct_truth_table(self, conditions: Dict[str, FuzzyCondition], 
                            outcome: FuzzyCondition) -> pd.DataFrame:
        """Construct fuzzy truth table with continuous values"""
        # Create all possible combinations of conditions (using membership values)
        condition_names = list(conditions.keys())
        n_cases = len(list(conditions.values())[0].membership_values)
        
        truth_table_data = []
        
        for i in range(n_cases):
            row = {}
            
            # Add condition memberships
            for name in condition_names:
                row[name] = conditions[name].membership_values[i]
            
            # Add outcome membership
            row['outcome'] = outcome.membership_values[i]
            
            # Calculate consistency for this configuration
            # Consistency = min(condition_memberships) -> outcome_membership
            min_condition_membership = min(conditions[name].membership_values[i] for name in condition_names)
            row['min_condition'] = min_condition_membership
            row['implication'] = min(1.0, 1 - min_condition_membership + outcome.membership_values[i])
            
            truth_table_data.append(row)
        
        truth_table = pd.DataFrame(truth_table_data)
        return truth_table
    
    def analyze_sufficiency(self, conditions: Dict[str, FuzzyCondition], 
                           outcome: FuzzyCondition) -> Dict[str, Any]:
        """Analyze sufficient conditions using truth table analysis"""
        
        # Generate all possible logical combinations
        condition_names = list(conditions.keys())
        n_conditions = len(condition_names)
        
        # Create combinations (including negations)
        combinations = []
        for r in range(1, n_conditions + 1):
            for combination in itertools.combinations(condition_names, r):
                # For each combination, consider all possible negation patterns
                for negation_pattern in itertools.product([False, True], repeat=len(combination)):
                    combo_dict = {}
                    formula_parts = []
                    
                    for i, condition in enumerate(combination):
                        is_negated = negation_pattern[i]
                        combo_dict[condition] = not is_negated
                        if is_negated:
                            formula_parts.append(f"~{condition}")
                        else:
                            formula_parts.append(condition)
                    
                    combinations.append({
                        'conditions': combo_dict,
                        'formula': "*".join(formula_parts),
                        'complexity': len(combination)
                    })
        
        # Analyze each combination
        results = []
        
        for combo in combinations:
            consistency, coverage = self._calculate_consistency_coverage(
                combo['conditions'], conditions, outcome
            )
            
            if consistency >= self.consistency_threshold:
                results.append({
                    'formula': combo['formula'],
                    'conditions': combo['conditions'],
                    'consistency': consistency,
                    'coverage': coverage,
                    'complexity': combo['complexity']
                })
        
        # Sort by consistency (descending) then coverage (descending)
        results.sort(key=lambda x: (-x['consistency'], -x['coverage']))
        
        # Find minimal solutions (Boolean minimization)
        minimal_solutions = self._find_minimal_solutions(results)
        
        return {
            'all_sufficient': results,
            'minimal_solutions': minimal_solutions,
            'complex_solution': self._create_complex_solution(minimal_solutions),
            'parsimonious_solution': self._create_parsimonious_solution(minimal_solutions),
            'intermediate_solution': self._create_intermediate_solution(minimal_solutions)
        }
    
    def _calculate_consistency_coverage(self, combination: Dict[str, bool], 
                                      conditions: Dict[str, FuzzyCondition], 
                                      outcome: FuzzyCondition) -> Tuple[float, float]:
        """Calculate consistency and coverage for a combination"""
        n_cases = len(outcome.membership_values)
        
        combination_memberships = []
        
        for i in range(n_cases):
            # Calculate membership in this combination (intersection)
            memberships = []
            
            for condition_name, is_positive in combination.items():
                if is_positive:
                    memberships.append(conditions[condition_name].membership_values[i])
                else:
                    memberships.append(1 - conditions[condition_name].membership_values[i])
            
            # Intersection (minimum)
            combination_membership = min(memberships) if memberships else 0.0
            combination_memberships.append(combination_membership)
        
        # Consistency: sum(min(Xi, Yi)) / sum(Xi)
        min_values = [min(xi, yi) for xi, yi in zip(combination_memberships, outcome.membership_values)]
        consistency = sum(min_values) / sum(combination_memberships) if sum(combination_memberships) > 0 else 0.0
        
        # Coverage: sum(min(Xi, Yi)) / sum(Yi)
        coverage = sum(min_values) / sum(outcome.membership_values) if sum(outcome.membership_values) > 0 else 0.0
        
        return consistency, coverage
    
    def _find_minimal_solutions(self, sufficient_combinations: List[Dict]) -> List[Dict]:
        """Find logically minimal sufficient combinations"""
        # Simple implementation - in practice would use Quine-McCluskey or similar
        minimal = []
        
        for combo in sufficient_combinations:
            is_minimal = True
            
            # Check if any other combination is a subset and equally good
            for other in sufficient_combinations:
                if (other != combo and 
                    other['complexity'] < combo['complexity'] and
                    other['consistency'] >= combo['consistency'] * 0.95):  # Allow small tolerance
                    
                    # Check if other is subset of combo
                    is_subset = True
                    for condition, value in other['conditions'].items():
                        if condition not in combo['conditions'] or combo['conditions'][condition] != value:
                            is_subset = False
                            break
                    
                    if is_subset:
                        is_minimal = False
                        break
            
            if is_minimal:
                minimal.append(combo)
        
        return minimal[:10]  # Limit to top 10 for practicality
    
    def _create_complex_solution(self, minimal_solutions: List[Dict]) -> str:
        """Create complex solution (most detailed)"""
        if not minimal_solutions:
            return ""
        
        # Use all minimal solutions
        formulas = [sol['formula'] for sol in minimal_solutions]
        return " + ".join(formulas)
    
    def _create_parsimonious_solution(self, minimal_solutions: List[Dict]) -> str:
        """Create parsimonious solution (most simplified)"""
        if not minimal_solutions:
            return ""
        
        # Use only the simplest solutions
        min_complexity = min(sol['complexity'] for sol in minimal_solutions)
        simple_solutions = [sol for sol in minimal_solutions if sol['complexity'] == min_complexity]
        
        formulas = [sol['formula'] for sol in simple_solutions[:3]]  # Top 3
        return " + ".join(formulas)
    
    def _create_intermediate_solution(self, minimal_solutions: List[Dict]) -> str:
        """Create intermediate solution (balanced complexity)"""
        if not minimal_solutions:
            return ""
        
        # Use solutions with moderate complexity and high consistency
        good_solutions = [sol for sol in minimal_solutions if sol['consistency'] >= 0.85]
        
        if len(good_solutions) <= 3:
            formulas = [sol['formula'] for sol in good_solutions]
        else:
            # Select diverse complexity levels
            good_solutions.sort(key=lambda x: x['complexity'])
            selected = good_solutions[:3]
            formulas = [sol['formula'] for sol in selected]
        
        return " + ".join(formulas)

class FinancialStabilityOptimizer:
    """Optimizer specifically for financial stability and stress minimization"""
    
    def __init__(self):
        self.fsqca = fsQCAAnalyzer()
    
    def analyze_financial_stability_paths(self, client_data: List[Dict[str, Any]]) -> fsQCAResult:
        """Analyze paths to financial stability using fsQCA"""
        
        # Extract financial metrics
        raw_data = self._extract_financial_metrics(client_data)
        
        # Define calibration specifications for financial conditions
        calibration_spec = {
            'income_stability': {
                'method': 'percentile',
                'percentiles': (25, 50, 85),
                'description': 'Stable and sufficient income'
            },
            'expense_control': {
                'method': 'percentile', 
                'percentiles': (15, 50, 85),
                'description': 'Controlled and predictable expenses'
            },
            'savings_rate': {
                'method': 'direct',
                'anchors': (0.05, 0.15, 0.30),  # 5%, 15%, 30% savings rate
                'description': 'Adequate savings rate'
            },
            'debt_management': {
                'method': 'percentile',
                'percentiles': (15, 50, 85),
                'description': 'Manageable debt levels'
            },
            'emergency_fund': {
                'method': 'direct',
                'anchors': (1, 3, 6),  # Months of expenses
                'description': 'Adequate emergency fund'
            },
            'investment_growth': {
                'method': 'percentile',
                'percentiles': (25, 50, 80),
                'description': 'Positive investment performance'
            }
        }
        
        # Prepare fuzzy conditions
        conditions = self.fsqca.prepare_conditions(raw_data, calibration_spec)
        
        # Define outcome: Financial Stability & Low Stress
        financial_stress_scores = [data.get('financial_stress', 0.3) for data in client_data]
        # Invert stress to get stability (low stress = high stability)
        stability_scores = [1 - stress for stress in financial_stress_scores]
        
        outcome = FuzzyCondition(
            name='financial_stability_low_stress',
            raw_values=stability_scores,
            membership_values=self.fsqca.calibrator.direct_method(
                stability_scores, 0.4, 0.7, 0.9
            ),
            calibration_anchors=(0.4, 0.7, 0.9),
            description='Financial stability with low stress'
        )
        
        # Perform fsQCA analysis
        necessity_analysis = self.fsqca.analyze_necessity(conditions, outcome)
        sufficiency_analysis = self.fsqca.analyze_sufficiency(conditions, outcome)
        
        # Identify optimal paths for stress minimization
        optimal_paths = self._identify_optimal_paths(sufficiency_analysis, necessity_analysis)
        
        return fsQCAResult(
            conditions=conditions,
            outcome=outcome,
            necessity_analysis=necessity_analysis,
            sufficiency_analysis=sufficiency_analysis,
            optimal_paths=optimal_paths,
            consistency_threshold=self.fsqca.consistency_threshold,
            coverage_threshold=self.fsqca.coverage_threshold
        )
    
    def _extract_financial_metrics(self, client_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract and calculate financial metrics from client data"""
        metrics = {
            'income_stability': [],
            'expense_control': [],
            'savings_rate': [],
            'debt_management': [],
            'emergency_fund': [],
            'investment_growth': []
        }
        
        for data in client_data:
            # Income stability (CV of income over time)
            income_history = data.get('income_history', [data.get('income', 75000)])
            income_cv = np.std(income_history) / np.mean(income_history) if len(income_history) > 1 else 0.1
            metrics['income_stability'].append(1 - min(1.0, income_cv))  # Lower CV = higher stability
            
            # Expense control (expenses vs income ratio)
            total_income = data.get('income', 75000)
            total_expenses = data.get('total_expenses', total_income * 0.7)
            expense_ratio = total_expenses / total_income
            metrics['expense_control'].append(1 - min(1.0, expense_ratio))  # Lower ratio = better control
            
            # Savings rate
            savings = total_income - total_expenses
            savings_rate = max(0, savings / total_income)
            metrics['savings_rate'].append(savings_rate)
            
            # Debt management (debt-to-income ratio)
            total_debt = data.get('total_debt', 0)
            debt_ratio = total_debt / total_income
            metrics['debt_management'].append(1 - min(1.0, debt_ratio))  # Lower debt = better management
            
            # Emergency fund (months of expenses covered)
            emergency_fund = data.get('emergency_fund', total_expenses * 2)
            monthly_expenses = total_expenses / 12
            months_covered = emergency_fund / monthly_expenses if monthly_expenses > 0 else 0
            metrics['emergency_fund'].append(months_covered)
            
            # Investment growth (portfolio return)
            portfolio_return = data.get('portfolio_return', 0.08)
            metrics['investment_growth'].append(max(0, portfolio_return))
        
        return metrics
    
    def _identify_optimal_paths(self, sufficiency_analysis: Dict[str, Any], 
                              necessity_analysis: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify optimal paths that minimize financial stress"""
        optimal_paths = []
        
        # Start with minimal solutions from sufficiency analysis
        minimal_solutions = sufficiency_analysis.get('minimal_solutions', [])
        
        for solution in minimal_solutions[:5]:  # Top 5 solutions
            path = {
                'path_id': f"path_{len(optimal_paths) + 1}",
                'formula': solution['formula'],
                'conditions_required': solution['conditions'],
                'consistency': solution['consistency'],
                'coverage': solution['coverage'],
                'complexity': solution['complexity'],
                'stress_reduction_potential': self._calculate_stress_reduction_potential(solution),
                'implementation_difficulty': self._assess_implementation_difficulty(solution),
                'recommendations': self._generate_path_recommendations(solution, necessity_analysis)
            }
            optimal_paths.append(path)
        
        # Sort by effectiveness (combination of consistency, coverage, and stress reduction)
        optimal_paths.sort(key=lambda x: (
            x['consistency'] * 0.4 + 
            x['coverage'] * 0.3 + 
            x['stress_reduction_potential'] * 0.3
        ), reverse=True)
        
        return optimal_paths
    
    def _calculate_stress_reduction_potential(self, solution: Dict[str, Any]) -> float:
        """Calculate potential stress reduction from following this path"""
        # Higher consistency and coverage generally mean better stress reduction
        base_potential = (solution['consistency'] * 0.6 + solution['coverage'] * 0.4)
        
        # Adjust for complexity (simpler paths often more effective)
        complexity_penalty = min(0.2, solution['complexity'] * 0.05)
        
        return max(0, base_potential - complexity_penalty)
    
    def _assess_implementation_difficulty(self, solution: Dict[str, Any]) -> str:
        """Assess how difficult it would be to implement this path"""
        complexity = solution['complexity']
        
        if complexity <= 2:
            return 'easy'
        elif complexity <= 4:
            return 'moderate'
        else:
            return 'difficult'
    
    def _generate_path_recommendations(self, solution: Dict[str, Any], 
                                     necessity_analysis: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate specific recommendations for following this path"""
        recommendations = []
        
        conditions = solution['conditions']
        
        for condition, is_positive in conditions.items():
            if is_positive:
                # This condition should be present
                if condition == 'income_stability':
                    recommendations.append("Focus on stable income sources and diversification")
                elif condition == 'expense_control':
                    recommendations.append("Implement strict budgeting and expense tracking")
                elif condition == 'savings_rate':
                    recommendations.append("Increase savings rate through automated transfers")
                elif condition == 'debt_management':
                    recommendations.append("Aggressively pay down high-interest debt")
                elif condition == 'emergency_fund':
                    recommendations.append("Build emergency fund to 6+ months expenses")
                elif condition == 'investment_growth':
                    recommendations.append("Optimize investment portfolio for consistent growth")
            else:
                # This condition should be absent (negated)
                recommendations.append(f"Minimize reliance on {condition.replace('_', ' ')}")
        
        # Add necessity-based recommendations
        necessary_conditions = [name for name, metrics in necessity_analysis.items() 
                              if metrics['necessary']]
        
        for necessary in necessary_conditions:
            if necessary not in [c for c in conditions.keys()]:
                recommendations.append(f"Essential: Ensure {necessary.replace('_', ' ')} (necessary condition)")
        
        return recommendations

# Demo and testing
def demo_fuzzy_sets_optimizer():
    """Demonstrate fuzzy sets optimizer for financial stability"""
    print("🔬 FUZZY SETS OPTIMIZER - fsQCA DEMONSTRATION")
    print("=" * 70)
    
    # Create sample client data
    np.random.seed(42)
    
    client_data = []
    for i in range(50):  # 50 sample clients
        income = np.random.normal(85000, 25000)
        income = max(40000, income)
        
        # Correlated financial metrics
        savings_rate = max(0, np.random.normal(0.15, 0.08))
        expense_ratio = np.random.uniform(0.6, 0.95)
        debt_ratio = np.random.uniform(0, 0.4)
        
        # Calculate stress based on financial situation
        stress_factors = [
            expense_ratio - 0.7,  # High expenses increase stress
            max(0, debt_ratio - 0.2),  # High debt increases stress
            max(0, 0.1 - savings_rate),  # Low savings increase stress
        ]
        financial_stress = min(1.0, sum(max(0, factor) for factor in stress_factors))
        
        client_data.append({
            'client_id': f'CLIENT_{i:03d}',
            'income': income,
            'income_history': [income * (1 + np.random.normal(0, 0.1)) for _ in range(5)],
            'total_expenses': income * expense_ratio,
            'total_debt': income * debt_ratio,
            'emergency_fund': income * np.random.uniform(0.1, 0.8),
            'portfolio_return': np.random.normal(0.08, 0.15),
            'financial_stress': financial_stress
        })
    
    print(f"📊 Analyzing {len(client_data)} client cases for financial stability patterns...")
    
    # Run fsQCA analysis
    optimizer = FinancialStabilityOptimizer()
    results = optimizer.analyze_financial_stability_paths(client_data)
    
    print(f"\n🎯 NECESSITY ANALYSIS:")
    print("Conditions necessary for financial stability:")
    
    for condition, metrics in results.necessity_analysis.items():
        necessity_status = "✓ NECESSARY" if metrics['necessary'] else "○ Not necessary"
        print(f"   • {condition.replace('_', ' ').title()}: {necessity_status}")
        print(f"     Consistency: {metrics['consistency']:.3f}, Coverage: {metrics['coverage']:.3f}")
    
    print(f"\n🔄 SUFFICIENCY ANALYSIS:")
    sufficient_solutions = results.sufficiency_analysis['minimal_solutions']
    print(f"Found {len(sufficient_solutions)} sufficient combinations:")
    
    for i, solution in enumerate(sufficient_solutions[:5]):
        print(f"\n   Solution {i+1}: {solution['formula']}")
        print(f"   Consistency: {solution['consistency']:.3f}, Coverage: {solution['coverage']:.3f}")
    
    print(f"\n🎯 OPTIMAL PATHS FOR STRESS MINIMIZATION:")
    for i, path in enumerate(results.optimal_paths[:3]):
        print(f"\n📍 Path {i+1}: {path['formula']}")
        print(f"   Effectiveness Score: {path['consistency'] * 0.4 + path['coverage'] * 0.3 + path['stress_reduction_potential'] * 0.3:.3f}")
        print(f"   Implementation: {path['implementation_difficulty'].title()}")
        print(f"   Key Recommendations:")
        for rec in path['recommendations'][:3]:
            print(f"      • {rec}")
    
    print(f"\n📋 SOLUTION FORMULAS:")
    print(f"Complex Solution: {results.sufficiency_analysis['complex_solution']}")
    print(f"Parsimonious Solution: {results.sufficiency_analysis['parsimonious_solution']}")
    print(f"Intermediate Solution: {results.sufficiency_analysis['intermediate_solution']}")
    
    # Show membership function calibration
    print(f"\n📏 MEMBERSHIP FUNCTION CALIBRATION:")
    for name, condition in results.conditions.items():
        anchors = condition.calibration_anchors
        print(f"   • {name.replace('_', ' ').title()}:")
        print(f"     Full Non-membership: {anchors[0]:.3f}")
        print(f"     Crossover Point: {anchors[1]:.3f}")
        print(f"     Full Membership: {anchors[2]:.3f}")

if __name__ == "__main__":
    demo_fuzzy_sets_optimizer() 