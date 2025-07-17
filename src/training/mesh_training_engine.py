"""
Mesh Training Engine for Financial Commutator Optimization

This module implements a comprehensive training system that:
1. Generates synthetic people with realistic financial profiles
2. Applies financial shocks to test resilience
3. Runs cash flows through mesh systems
4. Tracks commutator routes and edge paths
5. Determines optimal node changes and routing strategies
6. Learns from successful recovery patterns

The training phase creates a database of successful commutator sequences
that can be applied to real financial situations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import json
import logging
import random
from pathlib import Path
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance

# Import existing components
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.commutator_decision_engine import CommutatorDecisionEngine, FinancialState, CommutatorOperation
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.core.state_space_mesh_engine import EnhancedMeshEngine, EnhancedMeshNode
from src.unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent
from src.layers.financial_space_mapper import FinancialSpaceMapper


@dataclass
class TrainingScenario:
    """Represents a training scenario with synthetic person and shocks"""
    scenario_id: str
    person: SyntheticClientData
    initial_state: Dict[str, float]
    shocks: List[Dict[str, Any]]
    mesh_engine: Any
    commutator_engine: CommutatorDecisionEngine
    success_metrics: Dict[str, float] = field(default_factory=dict)
    recovery_path: List[str] = field(default_factory=list)
    commutator_sequence: List[CommutatorOperation] = field(default_factory=list)


@dataclass
class CommutatorRoute:
    """Represents a successful commutator route"""
    route_id: str
    initial_state: Dict[str, float]
    final_state: Dict[str, float]
    shock_type: str
    shock_magnitude: float
    commutator_sequence: List[CommutatorOperation]
    edge_path: List[str]
    success_score: float
    recovery_time: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Results from a training session"""
    session_id: str
    num_scenarios: int
    successful_recoveries: int
    failed_recoveries: int
    average_recovery_time: float
    best_commutator_routes: List[CommutatorRoute]
    shock_type_performance: Dict[str, float]
    mesh_optimization_insights: Dict[str, Any]


class MeshTrainingEngine:
    """
    Comprehensive training engine for financial mesh systems
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.synthetic_engine = SyntheticLifestyleEngine()
        self.financial_mapper = FinancialSpaceMapper()
        self.training_scenarios: List[TrainingScenario] = []
        self.successful_routes: List[CommutatorRoute] = []
        self.failed_routes: List[CommutatorRoute] = []
        
        # Shock definitions
        self.shock_types = self._initialize_shock_types()
        
        # Training metrics
        self.training_history: List[TrainingResult] = []
        self.route_database: Dict[str, CommutatorRoute] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training operations"""
        logger = logging.getLogger('mesh_training_engine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_shock_types(self) -> Dict[str, Dict]:
        """Initialize different types of financial shocks"""
        return {
            'market_crash': {
                'probability': 0.15,
                'impact_range': (-0.4, -0.1),  # -40% to -10% portfolio value
                'recovery_time_range': (12, 36),  # months
                'category': 'investment',
                'description': 'Significant market downturn affecting portfolio value'
            },
            'job_loss': {
                'probability': 0.08,
                'impact_range': (-0.6, -0.2),  # -60% to -20% income
                'recovery_time_range': (6, 24),
                'category': 'income',
                'description': 'Loss of primary income source'
            },
            'medical_emergency': {
                'probability': 0.12,
                'impact_range': (0.1, 0.3),  # 10% to 30% of assets
                'recovery_time_range': (3, 18),
                'category': 'expense',
                'description': 'Unexpected medical expenses'
            },
            'divorce': {
                'probability': 0.05,
                'impact_range': (-0.5, -0.2),  # -50% to -20% net worth
                'recovery_time_range': (24, 60),
                'category': 'legal',
                'description': 'Divorce settlement and asset division'
            },
            'natural_disaster': {
                'probability': 0.03,
                'impact_range': (0.15, 0.4),  # 15% to 40% of assets
                'recovery_time_range': (6, 36),
                'category': 'property',
                'description': 'Property damage from natural disaster'
            },
            'interest_rate_spike': {
                'probability': 0.20,
                'impact_range': (-0.2, -0.05),  # -20% to -5% bond values
                'recovery_time_range': (6, 24),
                'category': 'interest_rate',
                'description': 'Rapid increase in interest rates'
            },
            'inflation_shock': {
                'probability': 0.25,
                'impact_range': (-0.15, -0.05),  # -15% to -5% purchasing power
                'recovery_time_range': (12, 48),
                'category': 'inflation',
                'description': 'Unexpected inflation reducing real returns'
            }
        }
    
    def generate_training_scenarios(self, num_scenarios: int = 100, 
                                  age_distribution: Optional[Dict[int, float]] = None) -> List[TrainingScenario]:
        """
        Generate training scenarios with synthetic people and financial shocks
        
        Args:
            num_scenarios: Number of scenarios to generate
            age_distribution: Optional age distribution for synthetic people
            
        Returns:
            List of training scenarios
        """
        self.logger.info(f"Generating {num_scenarios} training scenarios...")
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate synthetic person
            if age_distribution:
                ages = list(age_distribution.keys())
                probabilities = list(age_distribution.values())
                target_age = random.choices(ages, weights=probabilities)[0]
                person = self.synthetic_engine.generate_synthetic_client(target_age=target_age)
            else:
                person = self.synthetic_engine.generate_synthetic_client()
            
            # Create initial financial state
            initial_state = self._create_initial_financial_state(person)
            
            # Generate financial shocks
            shocks = self._generate_financial_shocks(person, num_shocks=random.randint(1, 3))
            
            # Initialize mesh engine
            mesh_engine = StochasticMeshEngine(initial_state)
            
            # Initialize commutator engine
            financial_state = self._convert_to_financial_state(initial_state)
            commutator_engine = CommutatorDecisionEngine(financial_state)
            
            scenario = TrainingScenario(
                scenario_id=f"scenario_{i:04d}",
                person=person,
                initial_state=initial_state,
                shocks=shocks,
                mesh_engine=mesh_engine,
                commutator_engine=commutator_engine
            )
            
            scenarios.append(scenario)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i + 1}/{num_scenarios} scenarios")
        
        self.training_scenarios = scenarios
        self.logger.info(f"Successfully generated {len(scenarios)} training scenarios")
        return scenarios
    
    def _create_initial_financial_state(self, person: SyntheticClientData) -> Dict[str, float]:
        """Create initial financial state from synthetic person"""
        profile = person.profile
        metrics = person.financial_metrics
        
        # Calculate total assets
        total_assets = sum(profile.current_assets.values())
        total_debts = sum(profile.debts.values())
        net_worth = total_assets - total_debts
        
        # Create state with realistic allocations
        state = {
            'cash': profile.current_assets.get('checking', 0) + profile.current_assets.get('savings', 0),
            'investments': profile.current_assets.get('investments', 0),
            'bonds': total_assets * 0.3,  # Assume 30% bonds
            'real_estate': profile.current_assets.get('real_estate', 0),
            'retirement': profile.current_assets.get('retirement', 0),
            'income': profile.base_income,
            'expenses': profile.base_income * 0.7,  # Assume 70% expense ratio
            'debts': total_debts,
            'net_worth': net_worth,
            'risk_tolerance': self._encode_risk_tolerance(profile.risk_tolerance)
        }
        
        return state
    
    def _encode_risk_tolerance(self, risk_tolerance: str) -> float:
        """Encode risk tolerance as a float"""
        encoding = {
            'Conservative': 0.2,
            'Moderate': 0.5,
            'Aggressive': 0.8,
            'Very Aggressive': 1.0
        }
        return encoding.get(risk_tolerance, 0.5)
    
    def _generate_financial_shocks(self, person: SyntheticClientData, num_shocks: int) -> List[Dict[str, Any]]:
        """Generate financial shocks for a person"""
        shocks = []
        person_age = person.profile.age
        
        # Select shock types based on age and probability
        available_shocks = []
        for shock_type, shock_data in self.shock_types.items():
            # Adjust probability based on age
            age_factor = self._calculate_age_factor(person_age, shock_type)
            adjusted_probability = shock_data['probability'] * age_factor
            
            if random.random() < adjusted_probability:
                available_shocks.append((shock_type, shock_data))
        
        # Select shocks
        selected_shocks = random.sample(available_shocks, min(num_shocks, len(available_shocks)))
        
        for shock_type, shock_data in selected_shocks:
            # Generate shock magnitude
            magnitude = random.uniform(*shock_data['impact_range'])
            
            # Generate timing (within next 5 years)
            timing_months = random.randint(6, 60)
            shock_date = datetime.now() + timedelta(days=timing_months * 30)
            
            shock = {
                'type': shock_type,
                'magnitude': magnitude,
                'timing': shock_date,
                'category': shock_data['category'],
                'description': shock_data['description'],
                'recovery_time_range': shock_data['recovery_time_range']
            }
            
            shocks.append(shock)
        
        return shocks
    
    def _calculate_age_factor(self, age: int, shock_type: str) -> float:
        """Calculate age factor for shock probability"""
        if shock_type == 'job_loss':
            # Higher probability in early career
            return 1.5 if age < 35 else 0.8 if age > 55 else 1.0
        elif shock_type == 'medical_emergency':
            # Higher probability with age
            return 0.5 if age < 30 else 1.0 if age < 50 else 1.5
        elif shock_type == 'divorce':
            # Peak in middle age
            return 0.3 if age < 25 else 1.2 if 30 <= age <= 50 else 0.6
        else:
            return 1.0
    
    def _convert_to_financial_state(self, state_dict: Dict[str, float]) -> FinancialState:
        """Convert dictionary state to FinancialState object"""
        return FinancialState(
            cash=state_dict.get('cash', 0),
            investments={
                'stocks': state_dict.get('investments', 0),
                'bonds': state_dict.get('bonds', 0),
                'real_estate': state_dict.get('real_estate', 0),
                'retirement': state_dict.get('retirement', 0)
            },
            debts={
                'mortgage': state_dict.get('debts', 0) * 0.7,
                'credit_cards': state_dict.get('debts', 0) * 0.2,
                'other_debt': state_dict.get('debts', 0) * 0.1
            },
            income_streams={
                'salary': state_dict.get('income', 0) / 12,
                'investment_income': state_dict.get('investments', 0) * 0.04 / 12
            },
            constraints={
                'min_cash_reserve': state_dict.get('cash', 0) * 0.1,
                'max_risk': 0.7,
                'min_liquidity': 0.2
            },
            timestamp=datetime.now(),
            state_id='initial_state'
        )
    
    def run_training_session(self, scenarios: List[TrainingScenario]) -> TrainingResult:
        """
        Run a complete training session with all scenarios
        
        Args:
            scenarios: List of training scenarios to process
            
        Returns:
            Training result with performance metrics
        """
        self.logger.info(f"Starting training session with {len(scenarios)} scenarios...")
        
        successful_recoveries = 0
        failed_recoveries = 0
        recovery_times = []
        shock_performance = {shock_type: {'success': 0, 'total': 0} for shock_type in self.shock_types.keys()}
        
        for i, scenario in enumerate(scenarios):
            self.logger.info(f"Processing scenario {i+1}/{len(scenarios)}: {scenario.scenario_id}")
            
            # Run scenario through mesh system
            success, recovery_time, route = self._run_single_scenario(scenario)
            
            if success:
                successful_recoveries += 1
                recovery_times.append(recovery_time)
                self.successful_routes.append(route)
                
                # Track shock type performance
                for shock in scenario.shocks:
                    shock_type = shock['type']
                    shock_performance[shock_type]['success'] += 1
                    shock_performance[shock_type]['total'] += 1
            else:
                failed_recoveries += 1
                self.failed_routes.append(route)
                
                # Track failed shocks
                for shock in scenario.shocks:
                    shock_type = shock['type']
                    shock_performance[shock_type]['total'] += 1
        
        # Calculate metrics
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        shock_success_rates = {
            shock_type: data['success'] / max(data['total'], 1) 
            for shock_type, data in shock_performance.items()
        }
        
        # Get best commutator routes
        best_routes = sorted(self.successful_routes, key=lambda r: r.success_score, reverse=True)[:10]
        
        result = TrainingResult(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_scenarios=len(scenarios),
            successful_recoveries=successful_recoveries,
            failed_recoveries=failed_recoveries,
            average_recovery_time=avg_recovery_time,
            best_commutator_routes=best_routes,
            shock_type_performance=shock_success_rates,
            mesh_optimization_insights=self._extract_mesh_insights()
        )
        
        self.training_history.append(result)
        self.logger.info(f"Training session completed: {successful_recoveries}/{len(scenarios)} successful recoveries")
        
        return result
    
    def _run_single_scenario(self, scenario: TrainingScenario) -> Tuple[bool, int, CommutatorRoute]:
        """
        Run a single training scenario through the mesh system
        
        Returns:
            (success, recovery_time, route)
        """
        # Apply shocks to initial state
        shocked_state = self._apply_shocks_to_state(scenario.initial_state, scenario.shocks)
        
        # Initialize mesh with shocked state
        scenario.mesh_engine.current_state = shocked_state
        
        # Generate commutator operations
        operations = scenario.commutator_engine.generate_commutator_operations()
        
        # Select optimal sequence
        optimal_sequence = scenario.commutator_engine.select_optimal_commutator_sequence(operations)
        
        # Execute sequence and track path
        success, recovery_time, edge_path = self._execute_commutator_sequence(
            scenario.commutator_engine, optimal_sequence, scenario.shocks
        )
        
        # Create route record
        route = CommutatorRoute(
            route_id=f"route_{scenario.scenario_id}",
            initial_state=scenario.initial_state,
            final_state=scenario.commutator_engine.current_state.__dict__,
            shock_type=scenario.shocks[0]['type'] if scenario.shocks else 'none',
            shock_magnitude=scenario.shocks[0]['magnitude'] if scenario.shocks else 0,
            commutator_sequence=optimal_sequence,
            edge_path=edge_path,
            success_score=self._calculate_success_score(scenario.commutator_engine.current_state),
            recovery_time=recovery_time,
            metadata={'scenario_id': scenario.scenario_id}
        )
        
        return success, recovery_time, route
    
    def _apply_shocks_to_state(self, initial_state: Dict[str, float], shocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Apply financial shocks to initial state"""
        shocked_state = initial_state.copy()
        
        for shock in shocks:
            shock_type = shock['type']
            magnitude = shock['magnitude']
            
            if shock_type == 'market_crash':
                # Reduce investment values
                shocked_state['investments'] *= (1 + magnitude)
                shocked_state['bonds'] *= (1 + magnitude * 0.5)
            elif shock_type == 'job_loss':
                # Reduce income
                shocked_state['income'] *= (1 + magnitude)
            elif shock_type == 'medical_emergency':
                # Increase expenses and reduce cash
                expense_increase = abs(magnitude) * shocked_state['net_worth']
                shocked_state['expenses'] += expense_increase
                shocked_state['cash'] = max(0, shocked_state['cash'] - expense_increase)
            elif shock_type == 'divorce':
                # Reduce net worth
                net_worth_reduction = abs(magnitude) * shocked_state['net_worth']
                shocked_state['net_worth'] -= net_worth_reduction
                shocked_state['cash'] = max(0, shocked_state['cash'] - net_worth_reduction * 0.5)
            elif shock_type == 'natural_disaster':
                # Reduce real estate and cash
                property_damage = abs(magnitude) * shocked_state['real_estate']
                shocked_state['real_estate'] -= property_damage
                shocked_state['cash'] = max(0, shocked_state['cash'] - property_damage * 0.3)
            elif shock_type == 'interest_rate_spike':
                # Reduce bond values
                shocked_state['bonds'] *= (1 + magnitude)
            elif shock_type == 'inflation_shock':
                # Reduce real value of all assets
                inflation_factor = 1 + magnitude
                shocked_state['cash'] /= inflation_factor
                shocked_state['investments'] /= inflation_factor
                shocked_state['bonds'] /= inflation_factor
        
        return shocked_state
    
    def _execute_commutator_sequence(self, commutator_engine: CommutatorDecisionEngine, 
                                   sequence: List[CommutatorOperation], 
                                   shocks: List[Dict[str, Any]]) -> Tuple[bool, int, List[str]]:
        """Execute commutator sequence and track the path"""
        edge_path = []
        recovery_time = 0
        
        for i, operation in enumerate(sequence):
            # Execute operation
            success = commutator_engine.execute_commutator_sequence([operation])
            
            if success:
                # Track edge taken
                edge_id = f"edge_{i}_{operation.operation_type}"
                edge_path.append(edge_id)
                recovery_time += operation.execution_time.days
                
                # Check if we've recovered
                if self._is_recovered(commutator_engine.current_state):
                    return True, recovery_time, edge_path
            else:
                # Operation failed
                edge_path.append(f"failed_edge_{i}")
        
        # Check final state
        final_success = self._is_recovered(commutator_engine.current_state)
        return final_success, recovery_time, edge_path
    
    def _is_recovered(self, state: FinancialState) -> bool:
        """Check if financial state has recovered from shocks"""
        # Simple recovery criteria: net worth is positive and cash reserves are adequate
        net_worth = state.total_wealth()
        cash_reserves = state.cash / max(1, state.total_wealth())
        
        return net_worth > 0 and cash_reserves >= 0.1  # At least 10% cash reserves
    
    def _calculate_success_score(self, state: FinancialState) -> float:
        """Calculate success score for a financial state"""
        net_worth = state.total_wealth()
        cash_ratio = state.cash / max(1, net_worth)
        debt_ratio = sum(state.debts.values()) / max(1, net_worth)
        
        # Higher score for higher net worth, adequate cash, and lower debt
        score = min(1.0, net_worth / 1000000) * 0.4  # Net worth component
        score += min(1.0, cash_ratio / 0.2) * 0.3    # Cash ratio component
        score += max(0, 1.0 - debt_ratio) * 0.3       # Debt ratio component
        
        return score
    
    def _extract_mesh_insights(self) -> Dict[str, Any]:
        """Extract insights from successful routes"""
        if not self.successful_routes:
            return {}
        
        # Analyze successful commutator patterns
        operation_frequency = {}
        edge_patterns = {}
        
        for route in self.successful_routes:
            for operation in route.commutator_sequence:
                op_type = operation.operation_type
                operation_frequency[op_type] = operation_frequency.get(op_type, 0) + 1
            
            # Analyze edge patterns
            edge_sequence = ' -> '.join(route.edge_path)
            edge_patterns[edge_sequence] = edge_patterns.get(edge_sequence, 0) + 1
        
        # Find most successful patterns
        most_frequent_operations = sorted(operation_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        most_successful_patterns = sorted(edge_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'most_frequent_operations': most_frequent_operations,
            'most_successful_patterns': most_successful_patterns,
            'average_route_length': np.mean([len(r.commutator_sequence) for r in self.successful_routes]),
            'success_rate_by_shock_type': self._calculate_shock_type_success_rates()
        }
    
    def _calculate_shock_type_success_rates(self) -> Dict[str, float]:
        """Calculate success rates by shock type"""
        shock_success = {}
        
        for route in self.successful_routes:
            shock_type = route.shock_type
            if shock_type not in shock_success:
                shock_success[shock_type] = {'success': 0, 'total': 0}
            
            shock_success[shock_type]['success'] += 1
        
        # Add failed routes
        for route in self.failed_routes:
            shock_type = route.shock_type
            if shock_type not in shock_success:
                shock_success[shock_type] = {'success': 0, 'total': 0}
            
            shock_success[shock_type]['total'] += 1
        
        return {
            shock_type: data['success'] / max(data['total'], 1)
            for shock_type, data in shock_success.items()
        }
    
    def save_training_results(self, output_dir: str = "data/outputs/training"):
        """Save training results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save successful routes
        routes_data = []
        for route in self.successful_routes:
            route_dict = {
                'route_id': route.route_id,
                'shock_type': route.shock_type,
                'shock_magnitude': route.shock_magnitude,
                'success_score': route.success_score,
                'recovery_time': route.recovery_time,
                'edge_path': route.edge_path,
                'commutator_sequence': [
                    {
                        'operation_type': op.operation_type,
                        'capital_required': op.capital_required,
                        'risk_change': op.risk_change,
                        'success_probability': op.success_probability
                    }
                    for op in route.commutator_sequence
                ],
                'metadata': route.metadata
            }
            routes_data.append(route_dict)
        
        with open(output_path / "successful_routes.json", 'w') as f:
            json.dump(routes_data, f, indent=2)
        
        # Save training history
        history_data = []
        for result in self.training_history:
            history_dict = {
                'session_id': result.session_id,
                'num_scenarios': result.num_scenarios,
                'successful_recoveries': result.successful_recoveries,
                'failed_recoveries': result.failed_recoveries,
                'average_recovery_time': result.average_recovery_time,
                'shock_type_performance': result.shock_type_performance,
                'mesh_optimization_insights': result.mesh_optimization_insights
            }
            history_data.append(history_dict)
        
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Training results saved to {output_path}")
    
    def load_training_results(self, input_dir: str = "data/outputs/training"):
        """Load training results from files"""
        input_path = Path(input_dir)
        
        # Load successful routes
        routes_file = input_path / "successful_routes.json"
        if routes_file.exists():
            with open(routes_file, 'r') as f:
                routes_data = json.load(f)
            
            for route_dict in routes_data:
                # Reconstruct CommutatorRoute objects
                # (Simplified reconstruction for now)
                pass
        
        # Load training history
        history_file = input_path / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            for history_dict in history_data:
                # Reconstruct TrainingResult objects
                # (Simplified reconstruction for now)
                pass
        
        self.logger.info(f"Training results loaded from {input_path}")


def run_training_session(num_scenarios: int = 100) -> TrainingResult:
    """Run a complete training session"""
    print("🚀 Starting Mesh Training Session")
    print("=" * 60)
    
    # Initialize training engine
    training_engine = MeshTrainingEngine()
    
    # Generate scenarios
    print(f"📊 Generating {num_scenarios} training scenarios...")
    scenarios = training_engine.generate_training_scenarios(num_scenarios)
    
    # Run training session
    print("🔄 Running training session...")
    result = training_engine.run_training_session(scenarios)
    
    # Save results
    print("💾 Saving training results...")
    training_engine.save_training_results()
    
    # Print summary
    print("\n📈 Training Session Summary:")
    print(f"  Total Scenarios: {result.num_scenarios}")
    print(f"  Successful Recoveries: {result.successful_recoveries}")
    print(f"  Failed Recoveries: {result.failed_recoveries}")
    print(f"  Success Rate: {result.successful_recoveries / result.num_scenarios:.1%}")
    print(f"  Average Recovery Time: {result.average_recovery_time:.1f} days")
    
    print("\n🎯 Best Performing Shock Types:")
    for shock_type, success_rate in sorted(result.shock_type_performance.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {shock_type}: {success_rate:.1%}")
    
    print("\n🔧 Most Successful Commutator Operations:")
    for op_type, count in result.mesh_optimization_insights.get('most_frequent_operations', [])[:5]:
        print(f"  {op_type}: {count} times")
    
    return result


if __name__ == "__main__":
    # Run training session
    result = run_training_session(num_scenarios=50) 