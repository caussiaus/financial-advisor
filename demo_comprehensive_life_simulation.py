#!/usr/bin/env python3
"""
Comprehensive Life Simulation Demo

This demo simulates time for all people under management and compares:
- Smooth ride vs benchmark over sliding windows
- Different birth eras (different market periods)
- Dynamic reallocation vs passive market strategy
- Life shocks impact on both strategies
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core_similarity_matching import CoreSimilarityMatcher, AgeBasedMilestone
from src.stochastic_mesh_engine import StochasticMeshEngine
from src.accounting_reconciliation import AccountingReconciliationEngine
from src.unified_cash_flow_model import UnifiedCashFlowModel
from src.financial_recommendation_engine import FinancialRecommendationEngine
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine

class ComprehensiveLifeSimulator:
    """
    Comprehensive life simulation that tests different strategies across time periods
    """
    
    def __init__(self):
        self.similarity_matcher = CoreSimilarityMatcher()
        self.synthetic_engine = SyntheticLifestyleEngine()
        self.market_periods = self._initialize_market_periods()
        self.life_shock_types = self._initialize_life_shocks()
        
    def _initialize_market_periods(self) -> Dict[str, Dict]:
        """Initialize different market periods for testing different birth eras"""
        return {
            'bull_market_1990s': {
                'start_year': 1990,
                'end_year': 2000,
                'annual_return': 0.15,
                'volatility': 0.12,
                'description': 'Dot-com boom period'
            },
            'bear_market_2000s': {
                'start_year': 2000,
                'end_year': 2010,
                'annual_return': 0.02,
                'volatility': 0.18,
                'description': 'Dot-com bust and financial crisis'
            },
            'recovery_2010s': {
                'start_year': 2010,
                'end_year': 2020,
                'annual_return': 0.12,
                'volatility': 0.14,
                'description': 'Post-crisis recovery'
            },
            'covid_era': {
                'start_year': 2020,
                'end_year': 2024,
                'annual_return': 0.08,
                'volatility': 0.20,
                'description': 'COVID-19 pandemic era'
            }
        }
    
    def _initialize_life_shocks(self) -> Dict[str, Dict]:
        """Initialize different types of life shocks"""
        return {
            'medical_emergency': {
                'probability': 0.15,
                'impact_range': (5000, 50000),
                'recovery_time': 6,
                'category': 'health'
            },
            'job_loss': {
                'probability': 0.10,
                'impact_range': (-30000, -10000),
                'recovery_time': 12,
                'category': 'career'
            },
            'divorce': {
                'probability': 0.08,
                'impact_range': (-100000, -50000),
                'recovery_time': 24,
                'category': 'family'
            },
            'home_repair': {
                'probability': 0.20,
                'impact_range': (5000, 25000),
                'recovery_time': 3,
                'category': 'housing'
            },
            'education_expense': {
                'probability': 0.12,
                'impact_range': (10000, 40000),
                'recovery_time': 18,
                'category': 'education'
            },
            'investment_loss': {
                'probability': 0.25,
                'impact_range': (-20000, -5000),
                'recovery_time': 12,
                'category': 'investment'
            }
        }
    
    def load_sample_data(self) -> List[Dict]:
        """Load the sample data you generated"""
        data_path = Path("data/outputs/analysis_data/fake_clients.json")
        
        if data_path.exists():
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    # Convert to our expected format
                    return self._convert_fake_clients_to_format(data)
            except json.JSONDecodeError:
                print("JSON file corrupted, generating synthetic data...")
                return self._generate_sample_data()
        else:
            print("Sample data not found, generating synthetic data...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample data if not available"""
        clients = []
        
        # Generate diverse client profiles
        profiles = [
            {'age': 25, 'income': 45000, 'life_stage': 'early_career', 'risk_tolerance': 0.7},
            {'age': 32, 'income': 75000, 'life_stage': 'mid_career', 'risk_tolerance': 0.6},
            {'age': 45, 'income': 120000, 'life_stage': 'established', 'risk_tolerance': 0.5},
            {'age': 58, 'income': 95000, 'life_stage': 'pre_retirement', 'risk_tolerance': 0.4},
            {'age': 70, 'income': 65000, 'life_stage': 'retirement', 'risk_tolerance': 0.3}
        ]
        
        for i, profile in enumerate(profiles):
            client = {
                'client_id': f'client_{i:03d}',
                'profile': profile,
                'financial_state': {
                    'cash': profile['income'] * 0.1,
                    'investments': profile['income'] * 0.3,
                    'debts': profile['income'] * 0.2,
                    'total_wealth': profile['income'] * 0.2
                },
                'milestones': self._generate_milestones_for_profile(profile),
                'life_shocks': []
            }
            clients.append(client)
        
        return clients
    
    def _convert_fake_clients_to_format(self, fake_clients_data: List[Dict]) -> List[Dict]:
        """Convert fake clients data to our expected format"""
        converted_clients = []
        
        for i, fake_client in enumerate(fake_clients_data[:10]):  # Use first 10 clients
            # Extract profile information
            profile = {
                'age': fake_client.get('age', 30),
                'income': fake_client.get('income', 60000),
                'life_stage': fake_client.get('life_stage', 'mid_career'),
                'risk_tolerance': fake_client.get('risk_tolerance', 0.5)
            }
            
            # Create financial state
            financial_state = {
                'cash': profile['income'] * 0.1,
                'investments': profile['income'] * 0.3,
                'debts': profile['income'] * 0.2,
                'total_wealth': profile['income'] * 0.2
            }
            
            # Generate milestones
            milestones = self._generate_milestones_for_profile(profile)
            
            client = {
                'client_id': fake_client.get('client_id', f'client_{i:03d}'),
                'profile': profile,
                'financial_state': financial_state,
                'milestones': milestones,
                'life_shocks': []
            }
            
            converted_clients.append(client)
        
        return converted_clients
    
    def _generate_milestones_for_profile(self, profile: Dict) -> List[Dict]:
        """Generate milestones based on profile"""
        milestones = []
        age = profile['age']
        income = profile['income']
        
        if age < 30:
            milestones.extend([
                {'type': 'education', 'description': 'Graduate School', 'age': 26, 'impact': 50000},
                {'type': 'career', 'description': 'Job Change', 'age': 28, 'impact': 5000},
                {'type': 'family', 'description': 'Marriage', 'age': 29, 'impact': 25000}
            ])
        elif age < 45:
            milestones.extend([
                {'type': 'housing', 'description': 'First Home', 'age': 33, 'impact': 50000},
                {'type': 'family', 'description': 'Children', 'age': 31, 'impact': 15000},
                {'type': 'career', 'description': 'Promotion', 'age': 35, 'impact': 25000}
            ])
        elif age < 60:
            milestones.extend([
                {'type': 'investment', 'description': 'Portfolio Diversification', 'age': 47, 'impact': 25000},
                {'type': 'health', 'description': 'Medical Procedure', 'age': 52, 'impact': 20000},
                {'type': 'career', 'description': 'Consulting Startup', 'age': 48, 'impact': 100000}
            ])
        else:
            milestones.extend([
                {'type': 'health', 'description': 'Long-term Care', 'age': 65, 'impact': 50000},
                {'type': 'investment', 'description': 'Estate Planning', 'age': 62, 'impact': 20000},
                {'type': 'housing', 'description': 'Downsizing', 'age': 68, 'impact': -50000}
            ])
        
        return milestones
    
    def simulate_life_over_time(self, client: Dict, market_period: str, 
                               strategy: str = 'dynamic') -> Dict:
        """
        Simulate a client's life over time with different strategies
        
        Args:
            client: Client data
            market_period: Market period to simulate
            strategy: 'dynamic' or 'passive'
            
        Returns:
            Simulation results
        """
        period_data = self.market_periods[market_period]
        years = period_data['end_year'] - period_data['start_year']
        
        # Initialize simulation state
        current_age = client['profile']['age']
        current_wealth = client['financial_state']['total_wealth']
        monthly_income = client['profile']['income'] / 12
        
        # Track performance over time
        wealth_history = []
        income_history = []
        shock_history = []
        reallocation_history = []
        
        for year in range(years):
            current_year = period_data['start_year'] + year
            current_age += 1
            
            # Generate market returns for this year
            market_return = np.random.normal(
                period_data['annual_return'], 
                period_data['volatility']
            )
            
            # Apply strategy-specific logic
            if strategy == 'dynamic':
                wealth_change, reallocation = self._apply_dynamic_strategy(
                    current_wealth, monthly_income, market_return, client, current_age
                )
            else:  # passive
                wealth_change, reallocation = self._apply_passive_strategy(
                    current_wealth, monthly_income, market_return, client, current_age
                )
            
            # Apply life shocks
            shock = self._generate_life_shock(client, current_age, year)
            if shock:
                wealth_change += shock['impact']
                shock_history.append(shock)
            
            # Update wealth
            current_wealth += wealth_change
            
            # Record history
            wealth_history.append({
                'year': current_year,
                'age': current_age,
                'wealth': current_wealth,
                'wealth_change': wealth_change,
                'market_return': market_return,
                'shock': shock
            })
            
            income_history.append(monthly_income * 12)
            reallocation_history.append(reallocation)
            
            # Income growth
            monthly_income *= 1.03  # 3% annual growth
        
        return {
            'client_id': client['client_id'],
            'market_period': market_period,
            'strategy': strategy,
            'final_wealth': current_wealth,
            'total_return': (current_wealth - client['financial_state']['total_wealth']) / client['financial_state']['total_wealth'],
            'wealth_history': wealth_history,
            'income_history': income_history,
            'shock_history': shock_history,
            'reallocation_history': reallocation_history,
            'volatility': np.std([w['wealth_change'] for w in wealth_history]),
            'max_drawdown': self._calculate_max_drawdown(wealth_history)
        }
    
    def _apply_dynamic_strategy(self, wealth: float, income: float, 
                               market_return: float, client: Dict, age: int) -> Tuple[float, Dict]:
        """Apply dynamic reallocation strategy"""
        # Base allocation
        if age < 35:
            equity_allocation = 0.8
            bond_allocation = 0.15
            cash_allocation = 0.05
        elif age < 50:
            equity_allocation = 0.65
            bond_allocation = 0.25
            cash_allocation = 0.10
        elif age < 65:
            equity_allocation = 0.50
            bond_allocation = 0.35
            cash_allocation = 0.15
        else:
            equity_allocation = 0.30
            bond_allocation = 0.50
            cash_allocation = 0.20
        
        # Dynamic adjustments based on market conditions
        if market_return < -0.10:  # Bear market
            equity_allocation *= 0.8
            cash_allocation *= 1.5
        elif market_return > 0.15:  # Bull market
            equity_allocation *= 1.1
            cash_allocation *= 0.8
        
        # Calculate returns
        equity_return = market_return * equity_allocation
        bond_return = 0.04 * bond_allocation  # 4% bond return
        cash_return = 0.02 * cash_allocation  # 2% cash return
        
        total_return = equity_return + bond_return + cash_return
        wealth_change = wealth * total_return + income * 12 * 0.2  # 20% savings rate
        
        reallocation = {
            'equity': equity_allocation,
            'bonds': bond_allocation,
            'cash': cash_allocation,
            'total_return': total_return
        }
        
        return wealth_change, reallocation
    
    def _apply_passive_strategy(self, wealth: float, income: float, 
                               market_return: float, client: Dict, age: int) -> Tuple[float, Dict]:
        """Apply passive market strategy"""
        # Simple age-based allocation
        if age < 35:
            equity_allocation = 0.9
            bond_allocation = 0.1
        elif age < 50:
            equity_allocation = 0.7
            bond_allocation = 0.3
        elif age < 65:
            equity_allocation = 0.5
            bond_allocation = 0.5
        else:
            equity_allocation = 0.3
            bond_allocation = 0.7
        
        # Calculate returns
        equity_return = market_return * equity_allocation
        bond_return = 0.04 * bond_allocation
        
        total_return = equity_return + bond_return
        wealth_change = wealth * total_return + income * 12 * 0.15  # 15% savings rate (lower than dynamic)
        
        reallocation = {
            'equity': equity_allocation,
            'bonds': bond_allocation,
            'cash': 0.0,
            'total_return': total_return
        }
        
        return wealth_change, reallocation
    
    def _generate_life_shock(self, client: Dict, age: int, year: int) -> Optional[Dict]:
        """Generate a life shock based on probabilities"""
        for shock_type, shock_data in self.life_shock_types.items():
            if np.random.random() < shock_data['probability']:
                impact = np.random.uniform(*shock_data['impact_range'])
                return {
                    'type': shock_type,
                    'impact': impact,
                    'age': age,
                    'year': year,
                    'category': shock_data['category'],
                    'recovery_time': shock_data['recovery_time']
                }
        return None
    
    def _calculate_max_drawdown(self, wealth_history: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        wealth_values = [w['wealth'] for w in wealth_history]
        peak = wealth_values[0]
        max_drawdown = 0
        
        for wealth in wealth_values:
            if wealth > peak:
                peak = wealth
            drawdown = (peak - wealth) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def run_comprehensive_simulation(self) -> Dict:
        """Run comprehensive simulation across all clients, periods, and strategies"""
        print("🚀 Starting Comprehensive Life Simulation")
        print("=" * 60)
        
        # Load sample data
        clients = self.load_sample_data()
        print(f"📊 Loaded {len(clients)} clients for simulation")
        
        # Run simulations
        results = {
            'clients': [],
            'market_periods': list(self.market_periods.keys()),
            'strategies': ['dynamic', 'passive'],
            'comparisons': {}
        }
        
        for client in clients:
            client_results = {
                'client_id': client['client_id'],
                'profile': client['profile'],
                'simulations': {}
            }
            
            for market_period in self.market_periods.keys():
                client_results['simulations'][market_period] = {}
                
                for strategy in ['dynamic', 'passive']:
                    print(f"🔄 Simulating {client['client_id']} - {market_period} - {strategy}")
                    
                    simulation = self.simulate_life_over_time(client, market_period, strategy)
                    client_results['simulations'][market_period][strategy] = simulation
            
            results['clients'].append(client_results)
        
        # Calculate comparisons
        results['comparisons'] = self._calculate_comparisons(results)
        
        return results
    
    def _calculate_comparisons(self, results: Dict) -> Dict:
        """Calculate comprehensive comparisons"""
        comparisons = {
            'strategy_performance': {},
            'market_period_analysis': {},
            'smooth_ride_analysis': {},
            'shock_impact_analysis': {}
        }
        
        # Strategy performance comparison
        for market_period in results['market_periods']:
            dynamic_returns = []
            passive_returns = []
            dynamic_volatilities = []
            passive_volatilities = []
            
            for client_result in results['clients']:
                if market_period in client_result['simulations']:
                    dynamic_sim = client_result['simulations'][market_period]['dynamic']
                    passive_sim = client_result['simulations'][market_period]['passive']
                    
                    dynamic_returns.append(dynamic_sim['total_return'])
                    passive_returns.append(passive_sim['total_return'])
                    dynamic_volatilities.append(dynamic_sim['volatility'])
                    passive_volatilities.append(passive_sim['volatility'])
            
            comparisons['strategy_performance'][market_period] = {
                'dynamic_avg_return': np.mean(dynamic_returns),
                'passive_avg_return': np.mean(passive_returns),
                'dynamic_avg_volatility': np.mean(dynamic_volatilities),
                'passive_avg_volatility': np.mean(passive_volatilities),
                'return_advantage': np.mean(dynamic_returns) - np.mean(passive_returns),
                'volatility_reduction': np.mean(passive_volatilities) - np.mean(dynamic_volatilities)
            }
        
        # Smooth ride analysis
        for client_result in results['clients']:
            client_id = client_result['client_id']
            comparisons['smooth_ride_analysis'][client_id] = {}
            
            for market_period in results['market_periods']:
                if market_period in client_result['simulations']:
                    dynamic_sim = client_result['simulations'][market_period]['dynamic']
                    passive_sim = client_result['simulations'][market_period]['passive']
                    
                    # Calculate smoothness metrics
                    dynamic_smoothness = self._calculate_smoothness(dynamic_sim['wealth_history'])
                    passive_smoothness = self._calculate_smoothness(passive_sim['wealth_history'])
                    
                    comparisons['smooth_ride_analysis'][client_id][market_period] = {
                        'dynamic_smoothness': dynamic_smoothness,
                        'passive_smoothness': passive_smoothness,
                        'smoothness_improvement': dynamic_smoothness - passive_smoothness,
                        'dynamic_max_drawdown': dynamic_sim['max_drawdown'],
                        'passive_max_drawdown': passive_sim['max_drawdown'],
                        'drawdown_reduction': passive_sim['max_drawdown'] - dynamic_sim['max_drawdown']
                    }
        
        # Shock impact analysis
        for client_result in results['clients']:
            client_id = client_result['client_id']
            comparisons['shock_impact_analysis'][client_id] = {}
            
            for market_period in results['market_periods']:
                if market_period in client_result['simulations']:
                    dynamic_sim = client_result['simulations'][market_period]['dynamic']
                    passive_sim = client_result['simulations'][market_period]['passive']
                    
                    dynamic_shock_impact = self._calculate_shock_impact(dynamic_sim)
                    passive_shock_impact = self._calculate_shock_impact(passive_sim)
                    
                    comparisons['shock_impact_analysis'][client_id][market_period] = {
                        'dynamic_shock_impact': dynamic_shock_impact,
                        'passive_shock_impact': passive_shock_impact,
                        'shock_mitigation': passive_shock_impact - dynamic_shock_impact
                    }
        
        return comparisons
    
    def _calculate_smoothness(self, wealth_history: List[Dict]) -> float:
        """Calculate smoothness of wealth progression"""
        wealth_changes = [w['wealth_change'] for w in wealth_history]
        return 1.0 / (1.0 + np.std(wealth_changes))  # Higher = smoother
    
    def _calculate_shock_impact(self, simulation: Dict) -> float:
        """Calculate total impact of life shocks"""
        total_impact = 0
        for shock in simulation['shock_history']:
            total_impact += abs(shock['impact'])
        return total_impact
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("# Comprehensive Life Simulation Report")
        report.append("=" * 60)
        report.append("")
        
        # Strategy Performance Summary
        report.append("## Strategy Performance Summary")
        report.append("")
        
        for market_period, performance in results['comparisons']['strategy_performance'].items():
            report.append(f"### {market_period}")
            report.append(f"- Dynamic Strategy: {performance['dynamic_avg_return']:.2%} return, {performance['dynamic_avg_volatility']:.2%} volatility")
            report.append(f"- Passive Strategy: {performance['passive_avg_return']:.2%} return, {performance['passive_avg_volatility']:.2%} volatility")
            report.append(f"- Return Advantage: {performance['return_advantage']:.2%}")
            report.append(f"- Volatility Reduction: {performance['volatility_reduction']:.2%}")
            report.append("")
        
        # Smooth Ride Analysis
        report.append("## Smooth Ride Analysis")
        report.append("")
        
        for client_id, smoothness_data in results['comparisons']['smooth_ride_analysis'].items():
            report.append(f"### Client {client_id}")
            for market_period, metrics in smoothness_data.items():
                report.append(f"**{market_period}:**")
                report.append(f"- Dynamic Smoothness: {metrics['dynamic_smoothness']:.3f}")
                report.append(f"- Passive Smoothness: {metrics['passive_smoothness']:.3f}")
                report.append(f"- Smoothness Improvement: {metrics['smoothness_improvement']:.3f}")
                report.append(f"- Drawdown Reduction: {metrics['drawdown_reduction']:.2%}")
                report.append("")
        
        # Shock Impact Analysis
        report.append("## Life Shock Impact Analysis")
        report.append("")
        
        for client_id, shock_data in results['comparisons']['shock_impact_analysis'].items():
            report.append(f"### Client {client_id}")
            for market_period, metrics in shock_data.items():
                report.append(f"**{market_period}:**")
                report.append(f"- Dynamic Shock Impact: ${metrics['dynamic_shock_impact']:,.0f}")
                report.append(f"- Passive Shock Impact: ${metrics['passive_shock_impact']:,.0f}")
                report.append(f"- Shock Mitigation: ${metrics['shock_mitigation']:,.0f}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, filename: str = "comprehensive_simulation_results.json"):
        """Save results to file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        output_path = Path("data/outputs/analysis_data") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_path}")
        return output_path

def main():
    """Main demo function"""
    print("🌟 Comprehensive Life Simulation Demo")
    print("=" * 60)
    
    # Initialize simulator
    simulator = ComprehensiveLifeSimulator()
    
    # Run comprehensive simulation
    results = simulator.run_comprehensive_simulation()
    
    # Generate report
    report = simulator.generate_comprehensive_report(results)
    
    # Save results
    output_path = simulator.save_results(results)
    
    # Print summary
    print("\n📊 Simulation Summary")
    print("=" * 40)
    print(f"✅ Simulated {len(results['clients'])} clients")
    print(f"✅ Tested {len(results['market_periods'])} market periods")
    print(f"✅ Compared {len(results['strategies'])} strategies")
    
    # Print key findings
    print("\n🎯 Key Findings")
    print("=" * 20)
    
    for market_period, performance in results['comparisons']['strategy_performance'].items():
        print(f"\n{market_period}:")
        print(f"  Dynamic vs Passive Return: {performance['return_advantage']:+.2%}")
        print(f"  Volatility Reduction: {performance['volatility_reduction']:+.2%}")
    
    # Save report
    report_path = Path("data/outputs/analysis_data/comprehensive_simulation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to {report_path}")
    print(f"📊 Results saved to {output_path}")
    print("\n✅ Comprehensive Life Simulation Complete!")

if __name__ == "__main__":
    main() 