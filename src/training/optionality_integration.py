"""
Optionality Integration Module

This module integrates the optionality-based training algorithm with the existing
mesh training infrastructure to provide comprehensive financial path optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

# Import existing components
from src.training.mesh_training_engine import MeshTrainingEngine, TrainingScenario, CommutatorRoute
from src.training.optionality_training_engine import OptionalityTrainingEngine, OptionalityPath, FinancialStateNode
from src.commutator_decision_engine import CommutatorDecisionEngine, FinancialState, CommutatorOperation
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine


@dataclass
class IntegratedTrainingResult:
    """Combined results from mesh and optionality training"""
    session_id: str
    mesh_results: Any  # TrainingResult from mesh training
    optionality_results: Any  # OptionalityTrainingResult
    combined_insights: Dict[str, Any]
    optimal_strategies: List[Dict[str, Any]]


class OptionalityIntegrationEngine:
    """
    Integration engine that combines mesh training with optionality optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.mesh_engine = MeshTrainingEngine(config)
        self.optionality_engine = OptionalityTrainingEngine(config)
        self.synthetic_engine = SyntheticLifestyleEngine()
        
        # Integration parameters
        self.optionality_weight = self.config.get('optionality_weight', 0.3)
        self.stress_weight = self.config.get('stress_weight', 0.4)
        self.commutator_weight = self.config.get('commutator_weight', 0.3)
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for integration operations"""
        logger = logging.getLogger('optionality_integration_engine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_integrated_training(self, num_scenarios: int = 100) -> IntegratedTrainingResult:
        """
        Run integrated training combining mesh and optionality approaches
        
        Args:
            num_scenarios: Number of training scenarios
            
        Returns:
            Integrated training results
        """
        self.logger.info(f"Starting integrated training with {num_scenarios} scenarios...")
        
        # Step 1: Run mesh training
        self.logger.info("Running mesh training...")
        mesh_scenarios = self.mesh_engine.generate_training_scenarios(num_scenarios)
        mesh_results = self.mesh_engine.run_training_session(mesh_scenarios)
        
        # Step 2: Run optionality training
        self.logger.info("Running optionality training...")
        optionality_results = self.optionality_engine.train_optionality_model(num_scenarios)
        
        # Step 3: Integrate results
        self.logger.info("Integrating training results...")
        combined_insights = self._combine_training_insights(mesh_results, optionality_results)
        optimal_strategies = self._generate_optimal_strategies(mesh_results, optionality_results)
        
        # Create integrated result
        result = IntegratedTrainingResult(
            session_id=f"integrated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mesh_results=mesh_results,
            optionality_results=optionality_results,
            combined_insights=combined_insights,
            optimal_strategies=optimal_strategies
        )
        
        self.logger.info("Integrated training completed successfully")
        
        return result
    
    def _combine_training_insights(self, mesh_results: Any, 
                                  optionality_results: Any) -> Dict[str, Any]:
        """Combine insights from both training approaches"""
        insights = {
            'mesh_insights': {
                'successful_recoveries': mesh_results.successful_recoveries,
                'failed_recoveries': mesh_results.failed_recoveries,
                'average_recovery_time': mesh_results.average_recovery_time,
                'shock_type_performance': mesh_results.shock_type_performance
            },
            'optionality_insights': {
                'average_optionality': optionality_results.average_optionality,
                'stress_minimization_success': optionality_results.stress_minimization_success,
                'market_condition_performance': optionality_results.market_condition_performance,
                'optimal_paths_found': len(optionality_results.optimal_paths)
            },
            'combined_metrics': {
                'overall_success_rate': self._calculate_overall_success_rate(mesh_results, optionality_results),
                'flexibility_score': self._calculate_flexibility_score(mesh_results, optionality_results),
                'stress_resilience': self._calculate_stress_resilience(mesh_results, optionality_results)
            }
        }
        
        return insights
    
    def _calculate_overall_success_rate(self, mesh_results: Any, 
                                       optionality_results: Any) -> float:
        """Calculate overall success rate combining both approaches"""
        mesh_success = mesh_results.successful_recoveries / (mesh_results.successful_recoveries + mesh_results.failed_recoveries)
        optionality_success = optionality_results.stress_minimization_success
        
        # Weighted combination
        return (mesh_success * 0.6) + (optionality_success * 0.4)
    
    def _calculate_flexibility_score(self, mesh_results: Any, 
                                   optionality_results: Any) -> float:
        """Calculate overall flexibility score"""
        # Mesh flexibility based on successful recovery routes
        mesh_flexibility = len(mesh_results.best_commutator_routes) / 10  # Normalized
        
        # Optionality flexibility based on average optionality
        optionality_flexibility = optionality_results.average_optionality
        
        # Combined flexibility
        return (mesh_flexibility * 0.4) + (optionality_flexibility * 0.6)
    
    def _calculate_stress_resilience(self, mesh_results: Any, 
                                   optionality_results: Any) -> float:
        """Calculate stress resilience score"""
        # Mesh resilience based on recovery time
        max_recovery_time = 60  # months
        mesh_resilience = 1.0 - (mesh_results.average_recovery_time / max_recovery_time)
        
        # Optionality resilience based on stress minimization
        optionality_resilience = optionality_results.stress_minimization_success
        
        # Combined resilience
        return (mesh_resilience * 0.5) + (optionality_resilience * 0.5)
    
    def _generate_optimal_strategies(self, mesh_results: Any, 
                                   optionality_results: Any) -> List[Dict[str, Any]]:
        """Generate optimal strategies combining both approaches"""
        strategies = []
        
        # Strategy 1: High Optionality + Low Stress
        if optionality_results.optimal_paths:
            best_optionality_path = max(optionality_results.optimal_paths, 
                                      key=lambda p: p.optionality_gain / (p.total_stress + 0.01))
            
            strategies.append({
                'strategy_id': 'high_optionality_low_stress',
                'name': 'High Optionality, Low Stress Strategy',
                'description': 'Maximize optionality while minimizing stress',
                'approach': 'optionality_focused',
                'actions': best_optionality_path.actions,
                'expected_optionality_gain': best_optionality_path.optionality_gain,
                'expected_stress': best_optionality_path.total_stress,
                'time_horizon': best_optionality_path.time_horizon,
                'confidence': best_optionality_path.probability
            })
        
        # Strategy 2: Commutator-Based Recovery
        if mesh_results.best_commutator_routes:
            best_commutator_route = max(mesh_results.best_commutator_routes, 
                                      key=lambda r: r.success_score)
            
            strategies.append({
                'strategy_id': 'commutator_recovery',
                'name': 'Commutator-Based Recovery Strategy',
                'description': 'Use commutator operations for financial recovery',
                'approach': 'commutator_focused',
                'operations': [op.operation_type for op in best_commutator_route.commutator_sequence],
                'expected_recovery_time': best_commutator_route.recovery_time,
                'success_score': best_commutator_route.success_score,
                'shock_type': best_commutator_route.shock_type,
                'confidence': best_commutator_route.success_score
            })
        
        # Strategy 3: Balanced Approach
        strategies.append({
            'strategy_id': 'balanced_approach',
            'name': 'Balanced Optionality and Recovery Strategy',
            'description': 'Combine optionality optimization with commutator recovery',
            'approach': 'integrated',
            'optionality_weight': self.optionality_weight,
            'stress_weight': self.stress_weight,
            'commutator_weight': self.commutator_weight,
            'expected_flexibility': self._calculate_flexibility_score(mesh_results, optionality_results),
            'expected_resilience': self._calculate_stress_resilience(mesh_results, optionality_results),
            'confidence': self._calculate_overall_success_rate(mesh_results, optionality_results)
        })
        
        return strategies
    
    def apply_optimal_strategy(self, strategy_id: str, current_financial_state: Dict[str, float],
                             market_condition: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply an optimal strategy to a current financial state
        
        Args:
            strategy_id: ID of the strategy to apply
            current_financial_state: Current financial state
            market_condition: Current market condition
            
        Returns:
            Strategy application results
        """
        if strategy_id == 'high_optionality_low_stress':
            return self._apply_optionality_strategy(current_financial_state, market_condition)
        elif strategy_id == 'commutator_recovery':
            return self._apply_commutator_strategy(current_financial_state, market_condition)
        elif strategy_id == 'balanced_approach':
            return self._apply_balanced_strategy(current_financial_state, market_condition)
        else:
            raise ValueError(f"Unknown strategy ID: {strategy_id}")
    
    def _apply_optionality_strategy(self, current_financial_state: Dict[str, float],
                                  market_condition: Dict[str, float]) -> Dict[str, Any]:
        """Apply optionality-focused strategy"""
        # Find best optionality path for current state
        state_id = self.optionality_engine._find_or_create_state(current_financial_state)
        
        # Calculate optionality for current state
        optionality = self.optionality_engine.calculate_optionality(state_id, 
                                                                  self._dict_to_market_condition(market_condition))
        
        # Find optimal paths
        optimal_paths = self.optionality_engine.find_optimal_paths(state_id, target_optionality=0.5)
        
        if not optimal_paths:
            return {
                'success': False,
                'message': 'No optimal paths found for current state',
                'current_optionality': optionality
            }
        
        best_path = optimal_paths[0]
        
        return {
            'success': True,
            'strategy': 'high_optionality_low_stress',
            'recommended_actions': best_path.actions,
            'expected_optionality_gain': best_path.optionality_gain,
            'expected_stress': best_path.total_stress,
            'time_horizon': best_path.time_horizon,
            'confidence': best_path.probability,
            'current_optionality': optionality
        }
    
    def _apply_commutator_strategy(self, current_financial_state: Dict[str, float],
                                 market_condition: Dict[str, float]) -> Dict[str, Any]:
        """Apply commutator-based recovery strategy"""
        # Convert to FinancialState
        financial_state = self._dict_to_financial_state(current_financial_state)
        
        # Create commutator engine
        commutator_engine = CommutatorDecisionEngine(financial_state)
        
        # Generate commutator operations
        operations = commutator_engine.generate_commutator_operations()
        
        # Select optimal sequence
        optimal_sequence = commutator_engine.select_optimal_commutator_sequence(operations)
        
        if not optimal_sequence:
            return {
                'success': False,
                'message': 'No feasible commutator operations found',
                'current_risk': financial_state.risk_score()
            }
        
        # Execute sequence
        success = commutator_engine.execute_commutator_sequence(optimal_sequence)
        
        return {
            'success': success,
            'strategy': 'commutator_recovery',
            'recommended_operations': [op.operation_type for op in optimal_sequence],
            'expected_risk_change': sum(op.risk_change for op in optimal_sequence),
            'capital_required': sum(op.capital_required for op in optimal_sequence),
            'confidence': np.mean([op.commutator_score() for op in optimal_sequence])
        }
    
    def _apply_balanced_strategy(self, current_financial_state: Dict[str, float],
                               market_condition: Dict[str, float]) -> Dict[str, Any]:
        """Apply balanced strategy combining both approaches"""
        # Get optionality strategy
        optionality_result = self._apply_optionality_strategy(current_financial_state, market_condition)
        
        # Get commutator strategy
        commutator_result = self._apply_commutator_strategy(current_financial_state, market_condition)
        
        # Combine results
        combined_score = 0.0
        if optionality_result['success']:
            combined_score += optionality_result.get('expected_optionality_gain', 0) * self.optionality_weight
        
        if commutator_result['success']:
            combined_score += commutator_result.get('confidence', 0) * self.commutator_weight
        
        # Calculate stress impact
        stress_impact = 0.0
        if optionality_result['success']:
            stress_impact += optionality_result.get('expected_stress', 0) * self.stress_weight
        
        return {
            'success': optionality_result['success'] or commutator_result['success'],
            'strategy': 'balanced_approach',
            'optionality_component': optionality_result,
            'commutator_component': commutator_result,
            'combined_score': combined_score,
            'stress_impact': stress_impact,
            'recommendation': self._generate_balanced_recommendation(optionality_result, commutator_result)
        }
    
    def _generate_balanced_recommendation(self, optionality_result: Dict[str, Any],
                                        commutator_result: Dict[str, Any]) -> str:
        """Generate balanced recommendation based on both approaches"""
        if not optionality_result['success'] and not commutator_result['success']:
            return "No optimal strategies found. Consider conservative approach."
        
        if optionality_result['success'] and not commutator_result['success']:
            return "Focus on optionality optimization to improve financial flexibility."
        
        if not optionality_result['success'] and commutator_result['success']:
            return "Focus on commutator operations to improve financial position."
        
        # Both successful - provide balanced recommendation
        optionality_gain = optionality_result.get('expected_optionality_gain', 0)
        commutator_confidence = commutator_result.get('confidence', 0)
        
        if optionality_gain > 0.3 and commutator_confidence > 0.7:
            return "Excellent position. Combine optionality optimization with strategic commutator operations."
        elif optionality_gain > 0.2:
            return "Good optionality. Focus on maintaining flexibility while considering commutator operations."
        else:
            return "Limited optionality. Prioritize commutator operations to improve financial position."
    
    def _dict_to_market_condition(self, market_dict: Dict[str, float]) -> Any:
        """Convert dictionary to MarketCondition object"""
        from src.training.optionality_training_engine import MarketCondition
        
        return MarketCondition(
            condition_id="current",
            timestamp=datetime.now(),
            market_stress=market_dict.get('market_stress', 0.4),
            volatility=market_dict.get('volatility', 0.15),
            interest_rate=market_dict.get('interest_rate', 0.04),
            inflation_rate=market_dict.get('inflation_rate', 0.025),
            growth_rate=market_dict.get('growth_rate', 0.03)
        )
    
    def _dict_to_financial_state(self, state_dict: Dict[str, float]) -> FinancialState:
        """Convert dictionary to FinancialState object"""
        return FinancialState(
            cash=state_dict.get('cash', 0),
            investments={'investments': state_dict.get('investments', 0)},
            debts={'debt': state_dict.get('debt', 0)},
            income_streams={'income': state_dict.get('income', 0)},
            constraints={'min_cash_reserve': state_dict.get('cash', 0) * 0.1},
            timestamp=datetime.now(),
            state_id="current_state"
        )
    
    def save_integrated_results(self, result: IntegratedTrainingResult,
                              output_dir: str = "data/outputs/integrated_training"):
        """Save integrated training results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_file = Path(output_dir) / f"{result.session_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'session_id': result.session_id,
                'combined_insights': result.combined_insights,
                'optimal_strategies': result.optimal_strategies
            }, f, indent=2, default=str)
        
        # Save detailed results
        mesh_file = Path(output_dir) / f"{result.session_id}_mesh.json"
        with open(mesh_file, 'w') as f:
            json.dump({
                'successful_recoveries': result.mesh_results.successful_recoveries,
                'failed_recoveries': result.mesh_results.failed_recoveries,
                'average_recovery_time': result.mesh_results.average_recovery_time,
                'shock_type_performance': result.mesh_results.shock_type_performance
            }, f, indent=2, default=str)
        
        optionality_file = Path(output_dir) / f"{result.session_id}_optionality.json"
        with open(optionality_file, 'w') as f:
            json.dump({
                'average_optionality': result.optionality_results.average_optionality,
                'stress_minimization_success': result.optionality_results.stress_minimization_success,
                'market_condition_performance': result.optionality_results.market_condition_performance,
                'optimal_paths_found': len(result.optionality_results.optimal_paths)
            }, f, indent=2, default=str)
        
        self.logger.info(f"Integrated results saved to {output_dir}")


def run_integrated_training(num_scenarios: int = 100) -> IntegratedTrainingResult:
    """
    Run integrated training session
    
    Args:
        num_scenarios: Number of training scenarios
        
    Returns:
        Integrated training results
    """
    engine = OptionalityIntegrationEngine()
    result = engine.run_integrated_training(num_scenarios)
    
    # Save results
    engine.save_integrated_results(result)
    
    return result


if __name__ == "__main__":
    # Run integrated training
    result = run_integrated_training(num_scenarios=50)
    print(f"Integrated training completed")
    print(f"Overall success rate: {result.combined_insights['combined_metrics']['overall_success_rate']:.3f}")
    print(f"Flexibility score: {result.combined_insights['combined_metrics']['flexibility_score']:.3f}")
    print(f"Stress resilience: {result.combined_insights['combined_metrics']['stress_resilience']:.3f}")
    print(f"Optimal strategies found: {len(result.optimal_strategies)}") 