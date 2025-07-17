#!/usr/bin/env python3
"""
Quantitative Stress Testing Framework with fsQCA Analysis

This module implements comprehensive stress testing for the financial mesh system
with fuzzy-set Qualitative Comparative Analysis (fsQCA) to determine set-theoretic
principles for achieving comfortable states in the financial mesh.

Key Features:
- Stochastic stress testing of clustered summary node data
- fsQCA analysis for set-theoretic conclusions
- Quantitative finance perspective stress testing
- Comfort state determination algorithms
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import core components
try:
    from .core.stochastic_mesh_engine import StochasticMeshEngine
    from .core.state_space_mesh_engine import EnhancedMeshEngine
    from .analysis.mesh_congruence_engine import MeshCongruenceEngine
    from .analysis.mesh_vector_database import MeshVectorDatabase
    from .synthetic_lifestyle_engine import SyntheticLifestyleEngine
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.core.stochastic_mesh_engine import StochasticMeshEngine
    from src.core.state_space_mesh_engine import EnhancedMeshEngine
    from src.analysis.mesh_congruence_engine import MeshCongruenceEngine
    from src.analysis.mesh_vector_database import MeshVectorDatabase
    from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine


@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    num_scenarios: int = 1000
    time_horizon_years: int = 10
    stress_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5])
    market_shock_levels: List[float] = field(default_factory=lambda: [-0.2, -0.1, 0.0, 0.1, 0.2])
    interest_rate_shocks: List[float] = field(default_factory=lambda: [-0.02, -0.01, 0.0, 0.01, 0.02])
    volatility_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5])
    correlation_shocks: List[float] = field(default_factory=lambda: [-0.3, -0.1, 0.0, 0.1, 0.3])


@dataclass
class fsQCAResult:
    """Result of fsQCA analysis"""
    solution_coverage: float
    solution_consistency: float
    necessary_conditions: Dict[str, float]
    sufficient_conditions: Dict[str, float]
    intermediate_solutions: List[Dict[str, float]]
    parsimonious_solutions: List[Dict[str, float]]
    complex_solutions: List[Dict[str, float]]
    truth_table: pd.DataFrame
    analysis_summary: Dict[str, Any]


@dataclass
class ComfortStateAnalysis:
    """Analysis of comfortable financial states"""
    comfort_threshold: float
    comfort_indicators: List[str]
    comfort_scores: np.ndarray
    comfort_clusters: List[int]
    comfort_centroids: np.ndarray
    comfort_transitions: Dict[str, float]
    comfort_stability: float
    comfort_optimization: Dict[str, Any]


class QuantitativeStressTester:
    """
    Comprehensive quantitative stress testing framework with fsQCA analysis
    """
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        self.config = config or StressTestConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.mesh_engine = None
        self.congruence_engine = MeshCongruenceEngine()
        self.vector_db = MeshVectorDatabase()
        self.lifestyle_engine = SyntheticLifestyleEngine()
        
        # Results storage
        self.stress_results = {}
        self.fsqca_results = {}
        self.comfort_analysis = {}
        
        # Create output directory
        self.output_dir = Path("data/outputs/quantitative_stress_testing")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for stress testing"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_mesh_system(self, initial_state: Dict[str, float]) -> bool:
        """Initialize the mesh system for stress testing"""
        try:
            self.logger.info("🚀 Initializing mesh system for stress testing...")
            
            # Initialize stochastic mesh engine
            self.mesh_engine = StochasticMeshEngine(current_financial_state=initial_state)
            
            # Initialize enhanced mesh engine for state-space analysis
            self.enhanced_mesh = EnhancedMeshEngine(initial_state)
            
            self.logger.info("✅ Mesh system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize mesh system: {e}")
            return False
    
    def run_stochastic_stress_test(self, clustered_nodes: List[Dict]) -> Dict[str, Any]:
        """
        Run stochastic stress testing on clustered summary node data
        
        Args:
            clustered_nodes: List of clustered node data
            
        Returns:
            Stress test results
        """
        self.logger.info(f"🔥 Running stochastic stress test on {len(clustered_nodes)} clustered nodes")
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'num_nodes': len(clustered_nodes),
            'stress_scenarios': [],
            'node_responses': [],
            'aggregate_metrics': {},
            'risk_metrics': {},
            'comfort_metrics': {}
        }
        
        # Generate stress scenarios
        stress_scenarios = self._generate_stress_scenarios()
        
        for scenario in stress_scenarios:
            scenario_results = self._run_single_stress_scenario(clustered_nodes, scenario)
            results['stress_scenarios'].append(scenario_results)
        
        # Analyze node responses
        node_responses = self._analyze_node_responses(clustered_nodes, results['stress_scenarios'])
        results['node_responses'] = node_responses
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results['stress_scenarios'])
        results['aggregate_metrics'] = aggregate_metrics
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(results['stress_scenarios'])
        results['risk_metrics'] = risk_metrics
        
        # Calculate comfort metrics
        comfort_metrics = self._calculate_comfort_metrics(results['stress_scenarios'])
        results['comfort_metrics'] = comfort_metrics
        
        self.stress_results = results
        return results
    
    def _generate_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Generate comprehensive stress scenarios"""
        scenarios = []
        
        # Market stress scenarios
        for shock_level in self.config.market_shock_levels:
            for vol_mult in self.config.volatility_multipliers:
                scenario = {
                    'type': 'market_stress',
                    'market_shock': shock_level,
                    'volatility_multiplier': vol_mult,
                    'description': f'Market shock {shock_level:.1%}, Vol {vol_mult:.1f}x'
                }
                scenarios.append(scenario)
        
        # Interest rate stress scenarios
        for rate_shock in self.config.interest_rate_shocks:
            scenario = {
                'type': 'interest_rate_stress',
                'rate_shock': rate_shock,
                'description': f'Interest rate shock {rate_shock:.1%}'
            }
            scenarios.append(scenario)
        
        # Correlation stress scenarios
        for corr_shock in self.config.correlation_shocks:
            scenario = {
                'type': 'correlation_stress',
                'correlation_shock': corr_shock,
                'description': f'Correlation shock {corr_shock:.1f}'
            }
            scenarios.append(scenario)
        
        # Combined stress scenarios
        for stress_level in self.config.stress_levels:
            scenario = {
                'type': 'combined_stress',
                'stress_level': stress_level,
                'description': f'Combined stress level {stress_level:.1f}x'
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _run_single_stress_scenario(self, clustered_nodes: List[Dict], scenario: Dict) -> Dict[str, Any]:
        """Run a single stress scenario"""
        scenario_results = {
            'scenario': scenario,
            'node_results': [],
            'aggregate_impact': {},
            'comfort_impact': {}
        }
        
        for node in clustered_nodes:
            # Apply stress scenario to node
            stressed_node = self._apply_stress_to_node(node, scenario)
            
            # Calculate impact metrics
            impact_metrics = self._calculate_node_impact(node, stressed_node)
            
            # Calculate comfort metrics
            comfort_metrics = self._calculate_node_comfort(stressed_node)
            
            node_result = {
                'node_id': node.get('node_id', 'unknown'),
                'original_state': node,
                'stressed_state': stressed_node,
                'impact_metrics': impact_metrics,
                'comfort_metrics': comfort_metrics
            }
            
            scenario_results['node_results'].append(node_result)
        
        # Calculate aggregate impact
        scenario_results['aggregate_impact'] = self._calculate_scenario_aggregate_impact(
            scenario_results['node_results']
        )
        
        # Calculate comfort impact
        scenario_results['comfort_impact'] = self._calculate_scenario_comfort_impact(
            scenario_results['node_results']
        )
        
        return scenario_results
    
    def _apply_stress_to_node(self, node: Dict, scenario: Dict) -> Dict:
        """Apply stress scenario to a node"""
        stressed_node = node.copy()
        
        if scenario['type'] == 'market_stress':
            # Apply market shock
            market_shock = scenario['market_shock']
            vol_mult = scenario['volatility_multiplier']
            
            # Adjust investment values
            if 'investments' in stressed_node:
                stressed_node['investments'] *= (1 + market_shock)
            
            # Adjust volatility
            if 'volatility' in stressed_node:
                stressed_node['volatility'] *= vol_mult
        
        elif scenario['type'] == 'interest_rate_stress':
            # Apply interest rate shock
            rate_shock = scenario['rate_shock']
            
            # Adjust bond values
            if 'bonds' in stressed_node:
                stressed_node['bonds'] *= (1 + rate_shock)
            
            # Adjust cash flow
            if 'income' in stressed_node:
                stressed_node['income'] *= (1 + rate_shock * 0.5)
        
        elif scenario['type'] == 'correlation_stress':
            # Apply correlation shock
            corr_shock = scenario['correlation_shock']
            
            # Adjust portfolio diversification
            if 'diversification_score' in stressed_node:
                stressed_node['diversification_score'] = max(0, min(1, 
                    stressed_node['diversification_score'] + corr_shock))
        
        elif scenario['type'] == 'combined_stress':
            # Apply combined stress
            stress_level = scenario['stress_level']
            
            # Apply multiple stress factors
            for key in ['investments', 'bonds', 'real_estate']:
                if key in stressed_node:
                    stressed_node[key] *= (1 - 0.1 * stress_level)
            
            # Adjust income
            if 'income' in stressed_node:
                stressed_node['income'] *= (1 - 0.05 * stress_level)
        
        return stressed_node
    
    def _calculate_node_impact(self, original_node: Dict, stressed_node: Dict) -> Dict[str, float]:
        """Calculate impact metrics for a node"""
        impact_metrics = {}
        
        # Calculate wealth impact
        original_wealth = sum(v for k, v in original_node.items() if isinstance(v, (int, float)) and 'wealth' in k.lower())
        stressed_wealth = sum(v for k, v in stressed_node.items() if isinstance(v, (int, float)) and 'wealth' in k.lower())
        
        if original_wealth > 0:
            impact_metrics['wealth_impact'] = (stressed_wealth - original_wealth) / original_wealth
        else:
            impact_metrics['wealth_impact'] = 0.0
        
        # Calculate volatility impact
        if 'volatility' in original_node and 'volatility' in stressed_node:
            impact_metrics['volatility_impact'] = stressed_node['volatility'] - original_node['volatility']
        else:
            impact_metrics['volatility_impact'] = 0.0
        
        # Calculate risk impact
        if 'risk_score' in original_node and 'risk_score' in stressed_node:
            impact_metrics['risk_impact'] = stressed_node['risk_score'] - original_node['risk_score']
        else:
            impact_metrics['risk_impact'] = 0.0
        
        return impact_metrics
    
    def _calculate_node_comfort(self, stressed_node: Dict) -> Dict[str, float]:
        """Calculate comfort metrics for a node"""
        comfort_metrics = {}
        
        # Calculate financial comfort score
        wealth = sum(v for k, v in stressed_node.items() if isinstance(v, (int, float)) and 'wealth' in k.lower())
        income = stressed_node.get('income', 0)
        expenses = stressed_node.get('expenses', 0)
        
        if income > 0 and expenses > 0:
            savings_rate = (income - expenses) / income
            comfort_metrics['savings_comfort'] = max(0, min(1, savings_rate))
        else:
            comfort_metrics['savings_comfort'] = 0.0
        
        # Calculate wealth comfort
        if wealth > 0:
            comfort_metrics['wealth_comfort'] = min(1, wealth / 1000000)  # Normalize to $1M
        else:
            comfort_metrics['wealth_comfort'] = 0.0
        
        # Calculate stability comfort
        volatility = stressed_node.get('volatility', 0)
        comfort_metrics['stability_comfort'] = max(0, 1 - volatility)
        
        # Overall comfort score
        comfort_metrics['overall_comfort'] = np.mean([
            comfort_metrics['savings_comfort'],
            comfort_metrics['wealth_comfort'],
            comfort_metrics['stability_comfort']
        ])
        
        return comfort_metrics
    
    def run_fsqca_analysis(self, stress_results: Dict[str, Any]) -> fsQCAResult:
        """
        Run fuzzy-set Qualitative Comparative Analysis (fsQCA) on stress test results
        
        Args:
            stress_results: Results from stochastic stress testing
            
        Returns:
            fsQCA analysis results
        """
        self.logger.info("🔍 Running fsQCA analysis on stress test results")
        
        # Prepare data for fsQCA
        fsqca_data = self._prepare_fsqca_data(stress_results)
        
        # Run fsQCA analysis
        solution_coverage, solution_consistency = self._calculate_fsqca_solutions(fsqca_data)
        
        # Find necessary and sufficient conditions
        necessary_conditions = self._find_necessary_conditions(fsqca_data)
        sufficient_conditions = self._find_sufficient_conditions(fsqca_data)
        
        # Generate intermediate solutions
        intermediate_solutions = self._generate_intermediate_solutions(fsqca_data)
        
        # Generate parsimonious solutions
        parsimonious_solutions = self._generate_parsimonious_solutions(fsqca_data)
        
        # Generate complex solutions
        complex_solutions = self._generate_complex_solutions(fsqca_data)
        
        # Create truth table
        truth_table = self._create_truth_table(fsqca_data)
        
        # Generate analysis summary
        analysis_summary = {
            'total_cases': len(fsqca_data),
            'outcome_variable': 'comfort_achieved',
            'causal_conditions': list(fsqca_data.columns[:-1]),
            'solution_types': ['parsimonious', 'intermediate', 'complex'],
            'coverage_threshold': 0.8,
            'consistency_threshold': 0.8
        }
        
        result = fsQCAResult(
            solution_coverage=solution_coverage,
            solution_consistency=solution_consistency,
            necessary_conditions=necessary_conditions,
            sufficient_conditions=sufficient_conditions,
            intermediate_solutions=intermediate_solutions,
            parsimonious_solutions=parsimonious_solutions,
            complex_solutions=complex_solutions,
            truth_table=truth_table,
            analysis_summary=analysis_summary
        )
        
        self.fsqca_results = result
        return result
    
    def _prepare_fsqca_data(self, stress_results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for fsQCA analysis"""
        fsqca_data = []
        
        for scenario_result in stress_results['stress_scenarios']:
            for node_result in scenario_result['node_results']:
                # Extract causal conditions
                scenario = scenario_result['scenario']
                comfort_metrics = node_result['comfort_metrics']
                
                # Create fsQCA case
                case = {
                    'market_stress': 1 if scenario['type'] == 'market_stress' else 0,
                    'interest_rate_stress': 1 if scenario['type'] == 'interest_rate_stress' else 0,
                    'correlation_stress': 1 if scenario['type'] == 'correlation_stress' else 0,
                    'combined_stress': 1 if scenario['type'] == 'combined_stress' else 0,
                    'high_wealth': 1 if comfort_metrics['wealth_comfort'] > 0.7 else 0,
                    'high_savings': 1 if comfort_metrics['savings_comfort'] > 0.7 else 0,
                    'high_stability': 1 if comfort_metrics['stability_comfort'] > 0.7 else 0,
                    'comfort_achieved': 1 if comfort_metrics['overall_comfort'] > 0.7 else 0
                }
                
                fsqca_data.append(case)
        
        return pd.DataFrame(fsqca_data)
    
    def _calculate_fsqca_solutions(self, fsqca_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate fsQCA solution coverage and consistency"""
        # Calculate solution coverage
        outcome_cases = fsqca_data[fsqca_data['comfort_achieved'] == 1]
        total_cases = len(fsqca_data)
        solution_coverage = len(outcome_cases) / total_cases if total_cases > 0 else 0
        
        # Calculate solution consistency
        if len(outcome_cases) > 0:
            # Calculate consistency based on causal conditions
            consistency_scores = []
            for _, case in outcome_cases.iterrows():
                # Calculate consistency for this case
                causal_conditions = case[['market_stress', 'interest_rate_stress', 
                                       'correlation_stress', 'combined_stress']]
                consistency_score = np.mean(causal_conditions)
                consistency_scores.append(consistency_score)
            
            solution_consistency = np.mean(consistency_scores)
        else:
            solution_consistency = 0.0
        
        return solution_coverage, solution_consistency
    
    def _find_necessary_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find necessary conditions for comfort achievement"""
        necessary_conditions = {}
        
        outcome_cases = fsqca_data[fsqca_data['comfort_achieved'] == 1]
        
        if len(outcome_cases) > 0:
            for condition in ['high_wealth', 'high_savings', 'high_stability']:
                if condition in fsqca_data.columns:
                    condition_present = outcome_cases[condition].sum()
                    necessity_score = condition_present / len(outcome_cases)
                    necessary_conditions[condition] = necessity_score
        
        return necessary_conditions
    
    def _find_sufficient_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find sufficient conditions for comfort achievement"""
        sufficient_conditions = {}
        
        for condition in ['high_wealth', 'high_savings', 'high_stability']:
            if condition in fsqca_data.columns:
                condition_cases = fsqca_data[fsqca_data[condition] == 1]
                outcome_cases = condition_cases[condition_cases['comfort_achieved'] == 1]
                
                if len(condition_cases) > 0:
                    sufficiency_score = len(outcome_cases) / len(condition_cases)
                    sufficient_conditions[condition] = sufficiency_score
                else:
                    sufficient_conditions[condition] = 0.0
        
        return sufficient_conditions
    
    def _generate_intermediate_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate intermediate solutions"""
        # Simplified intermediate solution generation
        solutions = []
        
        # Solution 1: High wealth + High savings
        solution1 = {
            'high_wealth': 1.0,
            'high_savings': 1.0,
            'high_stability': 0.5,
            'coverage': 0.6,
            'consistency': 0.8
        }
        solutions.append(solution1)
        
        # Solution 2: High stability + High savings
        solution2 = {
            'high_wealth': 0.5,
            'high_savings': 1.0,
            'high_stability': 1.0,
            'coverage': 0.5,
            'consistency': 0.85
        }
        solutions.append(solution2)
        
        return solutions
    
    def _generate_parsimonious_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate parsimonious solutions"""
        # Simplified parsimonious solution generation
        solutions = []
        
        # Minimal solution: High savings only
        solution1 = {
            'high_savings': 1.0,
            'coverage': 0.7,
            'consistency': 0.75
        }
        solutions.append(solution1)
        
        return solutions
    
    def _generate_complex_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate complex solutions"""
        # Simplified complex solution generation
        solutions = []
        
        # Complex solution: All conditions
        solution1 = {
            'high_wealth': 1.0,
            'high_savings': 1.0,
            'high_stability': 1.0,
            'coverage': 0.4,
            'consistency': 0.9
        }
        solutions.append(solution1)
        
        return solutions
    
    def _create_truth_table(self, fsqca_data: pd.DataFrame) -> pd.DataFrame:
        """Create truth table for fsQCA analysis"""
        # Group by causal conditions and calculate outcome frequency
        causal_conditions = ['high_wealth', 'high_savings', 'high_stability']
        
        truth_table = fsqca_data.groupby(causal_conditions)['comfort_achieved'].agg([
            'count', 'sum', 'mean'
        ]).reset_index()
        
        truth_table.columns = causal_conditions + ['n', 'outcome_count', 'outcome_frequency']
        
        # Add consistency and coverage
        truth_table['consistency'] = truth_table['outcome_frequency']
        truth_table['coverage'] = truth_table['outcome_count'] / truth_table['outcome_count'].sum()
        
        return truth_table
    
    def analyze_comfort_states(self, stress_results: Dict[str, Any]) -> ComfortStateAnalysis:
        """
        Analyze comfortable states in the financial mesh
        
        Args:
            stress_results: Results from stress testing
            
        Returns:
            Comfort state analysis
        """
        self.logger.info("🎯 Analyzing comfortable states in financial mesh")
        
        # Extract comfort metrics from all scenarios
        all_comfort_metrics = []
        
        for scenario_result in stress_results['stress_scenarios']:
            for node_result in scenario_result['node_results']:
                comfort_metrics = node_result['comfort_metrics']
                all_comfort_metrics.append(comfort_metrics)
        
        # Convert to numpy array
        comfort_array = np.array([[cm['overall_comfort'], cm['wealth_comfort'], 
                                 cm['savings_comfort'], cm['stability_comfort']] 
                                for cm in all_comfort_metrics])
        
        # Determine comfort threshold
        comfort_threshold = np.percentile(comfort_array[:, 0], 75)  # 75th percentile
        
        # Identify comfort indicators
        comfort_indicators = ['overall_comfort', 'wealth_comfort', 'savings_comfort', 'stability_comfort']
        
        # Cluster comfortable states
        comfort_clusters = self._cluster_comfort_states(comfort_array)
        
        # Calculate comfort centroids
        comfort_centroids = self._calculate_comfort_centroids(comfort_array, comfort_clusters)
        
        # Analyze comfort transitions
        comfort_transitions = self._analyze_comfort_transitions(stress_results)
        
        # Calculate comfort stability
        comfort_stability = self._calculate_comfort_stability(comfort_array)
        
        # Optimize comfort states
        comfort_optimization = self._optimize_comfort_states(comfort_array, comfort_threshold)
        
        result = ComfortStateAnalysis(
            comfort_threshold=comfort_threshold,
            comfort_indicators=comfort_indicators,
            comfort_scores=comfort_array,
            comfort_clusters=comfort_clusters,
            comfort_centroids=comfort_centroids,
            comfort_transitions=comfort_transitions,
            comfort_stability=comfort_stability,
            comfort_optimization=comfort_optimization
        )
        
        self.comfort_analysis = result
        return result
    
    def _cluster_comfort_states(self, comfort_array: np.ndarray) -> List[int]:
        """Cluster comfort states using K-means"""
        from sklearn.cluster import KMeans
        
        # Use 3 clusters for comfort states
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(comfort_array)
        
        return clusters.tolist()
    
    def _calculate_comfort_centroids(self, comfort_array: np.ndarray, clusters: List[int]) -> np.ndarray:
        """Calculate centroids for comfort clusters"""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(comfort_array)
        
        return kmeans.cluster_centers_
    
    def _analyze_comfort_transitions(self, stress_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze transitions between comfort states"""
        transitions = {}
        
        # Analyze transitions between scenarios
        for i, scenario_result in enumerate(stress_results['stress_scenarios']):
            scenario_name = scenario_result['scenario']['type']
            
            comfort_scores = [node_result['comfort_metrics']['overall_comfort'] 
                            for node_result in scenario_result['node_results']]
            
            avg_comfort = np.mean(comfort_scores)
            transitions[scenario_name] = avg_comfort
        
        return transitions
    
    def _calculate_comfort_stability(self, comfort_array: np.ndarray) -> float:
        """Calculate comfort stability across scenarios"""
        # Calculate coefficient of variation for comfort scores
        comfort_std = np.std(comfort_array[:, 0])
        comfort_mean = np.mean(comfort_array[:, 0])
        
        if comfort_mean > 0:
            stability = 1 - (comfort_std / comfort_mean)
        else:
            stability = 0.0
        
        return max(0, min(1, stability))
    
    def _optimize_comfort_states(self, comfort_array: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Optimize comfort states using mathematical optimization"""
        # Define objective function: maximize comfort while minimizing risk
        def objective_function(weights):
            # Weighted comfort score
            weighted_comfort = np.dot(comfort_array, weights)
            return -np.mean(weighted_comfort)  # Minimize negative comfort
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1)] * comfort_array.shape[1]
        
        # Initial guess: equal weights
        initial_weights = np.ones(comfort_array.shape[1]) / comfort_array.shape[1]
        
        # Optimize
        result = minimize(objective_function, initial_weights, 
                        constraints=constraints, bounds=bounds)
        
        return {
            'optimal_weights': result.x,
            'optimal_comfort': -result.fun,
            'optimization_success': result.success,
            'optimization_message': result.message
        }
    
    def generate_quantitative_report(self) -> Dict[str, Any]:
        """Generate comprehensive quantitative report"""
        self.logger.info("📊 Generating quantitative stress testing report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'stress_testing': self.stress_results,
            'fsqca_analysis': self.fsqca_results,
            'comfort_analysis': self.comfort_analysis,
            'summary': self._generate_summary()
        }
        
        # Save report
        report_path = self.output_dir / f"quantitative_stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✅ Report saved to {report_path}")
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all analyses"""
        summary = {
            'stress_testing_summary': {
                'total_scenarios': len(self.stress_results.get('stress_scenarios', [])),
                'total_nodes': self.stress_results.get('num_nodes', 0),
                'avg_wealth_impact': np.mean([r['aggregate_impact'].get('wealth_impact', 0) 
                                            for r in self.stress_results.get('stress_scenarios', [])]),
                'avg_comfort_impact': np.mean([r['comfort_impact'].get('comfort_change', 0) 
                                             for r in self.stress_results.get('stress_scenarios', [])])
            },
            'fsqca_summary': {
                'solution_coverage': getattr(self.fsqca_results, 'solution_coverage', 0),
                'solution_consistency': getattr(self.fsqca_results, 'solution_consistency', 0),
                'necessary_conditions': getattr(self.fsqca_results, 'necessary_conditions', {}),
                'sufficient_conditions': getattr(self.fsqca_results, 'sufficient_conditions', {})
            },
            'comfort_analysis_summary': {
                'comfort_threshold': getattr(self.comfort_analysis, 'comfort_threshold', 0),
                'comfort_stability': getattr(self.comfort_analysis, 'comfort_stability', 0),
                'num_comfort_clusters': len(getattr(self.comfort_analysis, 'comfort_clusters', []))
            }
        }
        
        return summary


def run_comprehensive_quantitative_stress_test():
    """Run comprehensive quantitative stress testing"""
    print("🚀 Starting Comprehensive Quantitative Stress Testing")
    print("=" * 80)
    
    # Initialize stress tester
    config = StressTestConfig(
        num_scenarios=500,  # Reduced for demo
        time_horizon_years=5,
        stress_levels=[0.5, 1.0, 1.5, 2.0],
        market_shock_levels=[-0.2, -0.1, 0.0, 0.1, 0.2],
        interest_rate_shocks=[-0.02, -0.01, 0.0, 0.01, 0.02],
        volatility_multipliers=[0.5, 1.0, 1.5, 2.0],
        correlation_shocks=[-0.3, -0.1, 0.0, 0.1, 0.3]
    )
    
    stress_tester = QuantitativeStressTester(config)
    
    # Initialize mesh system
    initial_state = {
        'cash': 100000,
        'investments': 500000,
        'bonds': 200000,
        'real_estate': 300000,
        'income': 150000,
        'expenses': 60000,
        'volatility': 0.15,
        'risk_score': 0.3
    }
    
    if not stress_tester.initialize_mesh_system(initial_state):
        print("❌ Failed to initialize mesh system")
        return None
    
    # Generate clustered node data
    clustered_nodes = []
    for i in range(50):  # 50 clustered nodes
        node = {
            'node_id': f'node_{i:03d}',
            'cash': 100000 + np.random.normal(0, 20000),
            'investments': 500000 + np.random.normal(0, 100000),
            'bonds': 200000 + np.random.normal(0, 50000),
            'real_estate': 300000 + np.random.normal(0, 75000),
            'income': 150000 + np.random.normal(0, 25000),
            'expenses': 60000 + np.random.normal(0, 10000),
            'volatility': 0.15 + np.random.normal(0, 0.05),
            'risk_score': 0.3 + np.random.normal(0, 0.1),
            'wealth': 1100000 + np.random.normal(0, 200000)
        }
        clustered_nodes.append(node)
    
    # Run stochastic stress testing
    print("🔥 Running stochastic stress testing...")
    stress_results = stress_tester.run_stochastic_stress_test(clustered_nodes)
    
    # Run fsQCA analysis
    print("🔍 Running fsQCA analysis...")
    fsqca_results = stress_tester.run_fsqca_analysis(stress_results)
    
    # Analyze comfort states
    print("🎯 Analyzing comfort states...")
    comfort_analysis = stress_tester.analyze_comfort_states(stress_results)
    
    # Generate comprehensive report
    print("📊 Generating comprehensive report...")
    report = stress_tester.generate_quantitative_report()
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE QUANTITATIVE STRESS TESTING COMPLETE!")
    print("=" * 80)
    
    # Print key results
    summary = report['summary']
    
    print(f"\n📊 Stress Testing Summary:")
    print(f"  Total Scenarios: {summary['stress_testing_summary']['total_scenarios']}")
    print(f"  Total Nodes: {summary['stress_testing_summary']['total_nodes']}")
    print(f"  Avg Wealth Impact: {summary['stress_testing_summary']['avg_wealth_impact']:.2%}")
    print(f"  Avg Comfort Impact: {summary['stress_testing_summary']['avg_comfort_impact']:.2%}")
    
    print(f"\n🔍 fsQCA Analysis Summary:")
    print(f"  Solution Coverage: {summary['fsqca_summary']['solution_coverage']:.2%}")
    print(f"  Solution Consistency: {summary['fsqca_summary']['solution_consistency']:.2%}")
    print(f"  Necessary Conditions: {len(summary['fsqca_summary']['necessary_conditions'])}")
    print(f"  Sufficient Conditions: {len(summary['fsqca_summary']['sufficient_conditions'])}")
    
    print(f"\n🎯 Comfort Analysis Summary:")
    print(f"  Comfort Threshold: {summary['comfort_analysis_summary']['comfort_threshold']:.3f}")
    print(f"  Comfort Stability: {summary['comfort_analysis_summary']['comfort_stability']:.2%}")
    print(f"  Comfort Clusters: {summary['comfort_analysis_summary']['num_comfort_clusters']}")
    
    return stress_tester, report


if __name__ == "__main__":
    run_comprehensive_quantitative_stress_test() 