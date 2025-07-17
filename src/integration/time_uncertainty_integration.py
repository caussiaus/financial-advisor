"""
Integration module for connecting time uncertainty mesh with existing stochastic mesh engine
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

from ..core.time_uncertainty_mesh import (
    TimeUncertaintyMeshEngine, 
    SeedEvent, 
    EventVector
)
from ..core.stochastic_mesh_engine import StochasticMeshEngine, OmegaNode
from ..enhanced_pdf_processor import FinancialMilestone

class TimeUncertaintyIntegration:
    """
    Integrates time uncertainty mesh with existing stochastic mesh engine
    """
    
    def __init__(self, current_financial_state: Dict[str, float]):
        self.current_financial_state = current_financial_state
        
        # Calculate total wealth for stochastic engine
        total_wealth = (
            current_financial_state.get('cash', 0.0) +
            current_financial_state.get('investments', 0.0) -
            current_financial_state.get('debts', 0.0)
        )
        
        # Create enhanced state with total_wealth
        enhanced_state = current_financial_state.copy()
        enhanced_state['total_wealth'] = total_wealth
        
        self.time_uncertainty_engine = TimeUncertaintyMeshEngine(use_gpu=True)
        self.stochastic_engine = StochasticMeshEngine(enhanced_state)
        self.integrated_mesh_data = None
        self.integrated_risk_analysis = None
        
    def convert_milestones_to_seed_events(self, milestones: List[Dict]) -> List[SeedEvent]:
        """
        Convert existing milestones to seed events for time uncertainty mesh
        
        Args:
            milestones: List of milestone dictionaries from existing system
            
        Returns:
            List of SeedEvent objects
        """
        seed_events = []
        
        for milestone in milestones:
            # Extract timing uncertainty from milestone
            timing_volatility = milestone.get('timing_uncertainty', 0.2)
            amount_volatility = milestone.get('amount_uncertainty', 0.15)
            drift_rate = milestone.get('drift_rate', 0.03)
            probability = milestone.get('probability', 0.8)
            
            # Create seed event
            seed_event = SeedEvent(
                event_id=milestone.get('id', f"milestone_{len(seed_events)}"),
                description=milestone.get('description', 'Unknown milestone'),
                estimated_date=milestone.get('estimated_date', '2025-01-01'),
                amount=milestone.get('amount', 0.0),
                timing_volatility=timing_volatility,
                amount_volatility=amount_volatility,
                drift_rate=drift_rate,
                probability=probability,
                category=milestone.get('category', 'general'),
                dependencies=milestone.get('dependencies', [])
            )
            
            seed_events.append(seed_event)
        
        return seed_events
    
    def _convert_to_financial_milestones(self, milestones: List[Dict]) -> List[FinancialMilestone]:
        """
        Convert milestone dictionaries to FinancialMilestone objects
        
        Args:
            milestones: List of milestone dictionaries
            
        Returns:
            List of FinancialMilestone objects
        """
        financial_milestones = []
        
        for milestone in milestones:
            # Parse estimated date
            estimated_date = pd.to_datetime(milestone.get('estimated_date', '2025-01-01'))
            
            # Create FinancialMilestone object
            financial_milestone = FinancialMilestone(
                timestamp=estimated_date,
                event_type=milestone.get('category', 'general'),
                description=milestone.get('description', 'Unknown milestone'),
                financial_impact=milestone.get('amount', 0.0),
                probability=milestone.get('probability', 0.8),
                entity='client',  # Default entity
                metadata={
                    'timing_uncertainty': milestone.get('timing_uncertainty', 0.2),
                    'amount_uncertainty': milestone.get('amount_uncertainty', 0.15),
                    'drift_rate': milestone.get('drift_rate', 0.03),
                    'original_id': milestone.get('id', 'unknown')
                }
            )
            
            financial_milestones.append(financial_milestone)
        
        return financial_milestones
    
    def initialize_integrated_mesh(self, milestones: List[Dict], 
                                 num_scenarios: int = 10000,
                                 time_horizon_years: float = 10) -> Tuple[Dict, Dict]:
        """
        Initialize integrated mesh combining time uncertainty with stochastic mesh
        
        Args:
            milestones: List of milestone dictionaries
            num_scenarios: Number of Monte Carlo scenarios
            time_horizon_years: Time horizon for mesh
            
        Returns:
            Tuple of (integrated_mesh_data, integrated_risk_analysis)
        """
        print("🔄 Initializing integrated mesh system...")
        
        # Step 1: Convert milestones to seed events
        seed_events = self.convert_milestones_to_seed_events(milestones)
        print(f"📋 Converted {len(milestones)} milestones to {len(seed_events)} seed events")
        
        # Step 2: Initialize time uncertainty mesh
        mesh_data, risk_analysis = self.time_uncertainty_engine.initialize_mesh_with_time_uncertainty(
            seed_events, 
            num_scenarios=num_scenarios,
            time_horizon_years=time_horizon_years
        )
        
        # Step 3: Convert milestones to FinancialMilestone objects for stochastic engine
        financial_milestones = self._convert_to_financial_milestones(milestones)
        
        # Step 4: Initialize stochastic mesh engine
        self.stochastic_engine.initialize_mesh(financial_milestones, time_horizon_years)
        
        # Step 5: Integrate the two systems
        integrated_data = self._integrate_mesh_systems(mesh_data, risk_analysis)
        
        # Store integrated data
        self.integrated_mesh_data = integrated_data
        self.integrated_risk_analysis = risk_analysis
        
        print("✅ Integrated mesh initialization complete")
        
        return integrated_data, risk_analysis
    
    def _integrate_mesh_systems(self, time_mesh_data: Dict, risk_analysis: Dict) -> Dict:
        """
        Integrate time uncertainty mesh with stochastic mesh engine
        
        Args:
            time_mesh_data: Data from time uncertainty mesh
            risk_analysis: Risk analysis from time uncertainty mesh
            
        Returns:
            Integrated mesh data
        """
        print("🔗 Integrating mesh systems...")
        
        # Extract key data
        mesh_states = time_mesh_data['states']
        time_steps = time_mesh_data['time_steps']
        event_vectors = time_mesh_data['event_vectors']
        
        # Create integrated mesh nodes
        integrated_nodes = []
        
        for t, timestamp in enumerate(time_steps):
            for scenario in range(mesh_states.shape[1]):
                # Get financial state for this time/scenario
                financial_state = mesh_states[t, scenario, :]
                
                # Create OmegaNode compatible with existing system
                node = OmegaNode(
                    node_id=f"integrated_{t}_{scenario}",
                    timestamp=timestamp.to_pydatetime(),
                    financial_state={
                        'cash': float(financial_state[0]),
                        'investments': float(financial_state[1]),
                        'debts': float(financial_state[2]),
                        'income': float(financial_state[3]),
                        'expenses': float(financial_state[4])
                    },
                    probability=1.0 / mesh_states.shape[1],  # Equal probability for now
                    visibility_radius=365 * 5
                )
                
                integrated_nodes.append(node)
        
        # Create integrated mesh data structure
        integrated_data = {
            'time_mesh_data': time_mesh_data,
            'stochastic_mesh_status': self.stochastic_engine.get_mesh_status(),
            'integrated_nodes': integrated_nodes,
            'time_steps': time_steps,
            'num_scenarios': mesh_states.shape[1],
            'num_time_steps': len(time_steps),
            'event_vectors': event_vectors
        }
        
        print(f"✅ Integrated {len(integrated_nodes)} nodes across {len(time_steps)} time steps")
        
        return integrated_data
    
    def get_integrated_payment_options(self, milestone_id: str = None) -> Dict[str, List[Dict]]:
        """
        Get payment options from integrated mesh system
        
        Args:
            milestone_id: Specific milestone ID to get options for
            
        Returns:
            Dictionary of payment options
        """
        if self.integrated_mesh_data is None:
            print("⚠️ No integrated mesh data available")
            return {}
        
        # Get payment options from stochastic engine
        stochastic_options = self.stochastic_engine.get_payment_options(milestone_id)
        
        # Enhance with time uncertainty information
        enhanced_options = self._enhance_payment_options_with_uncertainty(
            stochastic_options, 
            self.integrated_mesh_data['event_vectors']
        )
        
        return enhanced_options
    
    def _enhance_payment_options_with_uncertainty(self, 
                                                base_options: Dict[str, List[Dict]], 
                                                event_vectors: EventVector) -> Dict[str, List[Dict]]:
        """
        Enhance payment options with time uncertainty information
        
        Args:
            base_options: Base payment options from stochastic engine
            event_vectors: Event vectors with uncertainty information
            
        Returns:
            Enhanced payment options
        """
        enhanced_options = {}
        
        for milestone_id, options in base_options.items():
            enhanced_options[milestone_id] = []
            
            # Find corresponding event in event vectors
            if milestone_id in event_vectors.event_ids:
                event_idx = event_vectors.event_ids.index(milestone_id)
                
                # Calculate uncertainty metrics
                timing_uncertainty = np.std(event_vectors.timings[event_idx, :])
                amount_uncertainty = np.std(event_vectors.amounts[event_idx, :])
                probability = np.mean(event_vectors.probabilities[event_idx, :])
                
                for option in options:
                    enhanced_option = option.copy()
                    enhanced_option.update({
                        'timing_uncertainty': float(timing_uncertainty),
                        'amount_uncertainty': float(amount_uncertainty),
                        'probability': float(probability),
                        'scenario_count': event_vectors.timings.shape[1]
                    })
                    
                    enhanced_options[milestone_id].append(enhanced_option)
            else:
                # Keep original options if no uncertainty data available
                enhanced_options[milestone_id] = options
        
        return enhanced_options
    
    def execute_integrated_payment(self, milestone_id: str, amount: float, 
                                 payment_date: datetime = None) -> bool:
        """
        Execute payment in integrated mesh system
        
        Args:
            milestone_id: ID of milestone to pay
            amount: Payment amount
            payment_date: Payment date (optional)
            
        Returns:
            Success status
        """
        print(f"💳 Executing integrated payment for {milestone_id}: ${amount:,.2f}")
        
        # Execute in stochastic engine
        stochastic_success = self.stochastic_engine.execute_payment(
            milestone_id, amount, payment_date
        )
        
        # Update time uncertainty mesh if needed
        if stochastic_success:
            self._update_time_uncertainty_after_payment(milestone_id, amount, payment_date)
        
        return stochastic_success
    
    def _update_time_uncertainty_after_payment(self, milestone_id: str, 
                                             amount: float, payment_date: datetime):
        """
        Update time uncertainty mesh after payment execution
        
        Args:
            milestone_id: ID of milestone that was paid
            amount: Payment amount
            payment_date: Payment date
        """
        if self.integrated_mesh_data is None:
            return
        
        # Find corresponding event in event vectors
        event_vectors = self.integrated_mesh_data['event_vectors']
        
        if milestone_id in event_vectors.event_ids:
            event_idx = event_vectors.event_ids.index(milestone_id)
            
            # Update probabilities based on payment
            # This is a simplified update - in practice you might want more sophisticated logic
            if payment_date:
                payment_timestamp = payment_date.timestamp()
                
                # Reduce probability for scenarios where event hasn't happened yet
                future_scenarios = event_vectors.timings[event_idx, :] > payment_timestamp
                event_vectors.probabilities[event_idx, future_scenarios] *= 0.5  # Reduce probability
            
            print(f"🔄 Updated time uncertainty for {milestone_id}")
    
    def get_integrated_risk_analysis(self) -> Dict:
        """
        Get comprehensive risk analysis from integrated system
        
        Returns:
            Dictionary containing integrated risk metrics
        """
        if self.integrated_risk_analysis is None:
            print("⚠️ No integrated risk analysis available")
            return {}
        
        # Get risk analysis from time uncertainty engine
        time_risk = self.integrated_risk_analysis
        
        # Get mesh status from stochastic engine
        stochastic_status = self.stochastic_engine.get_mesh_status()
        
        # Combine risk metrics
        integrated_risk = {
            'time_uncertainty_metrics': time_risk,
            'stochastic_mesh_status': stochastic_status,
            'integrated_summary': {
                'total_nodes': len(self.integrated_mesh_data['integrated_nodes']) if self.integrated_mesh_data else 0,
                'total_scenarios': self.integrated_mesh_data['num_scenarios'] if self.integrated_mesh_data else 0,
                'time_horizon': len(self.integrated_mesh_data['time_steps']) if self.integrated_mesh_data else 0,
                'worst_case_cash': np.min(time_risk.get('min_cash_by_scenario', [0])),
                'best_case_cash': np.max(time_risk.get('min_cash_by_scenario', [0])),
                'avg_drawdown': np.mean(time_risk.get('max_drawdown_by_scenario', [0]))
            }
        }
        
        return integrated_risk
    
    def export_integrated_mesh(self, filepath: str):
        """
        Export integrated mesh data
        
        Args:
            filepath: Path to export file
        """
        if self.integrated_mesh_data is None:
            print("⚠️ No integrated mesh data to export")
            return
        
        export_data = {
            'integrated_mesh_data': self.integrated_mesh_data,
            'risk_analysis': self.integrated_risk_analysis,
            'stochastic_mesh_status': self.stochastic_engine.get_mesh_status(),
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'num_scenarios': self.integrated_mesh_data['num_scenarios'],
                'num_time_steps': self.integrated_mesh_data['num_time_steps'],
                'gpu_accelerated': self.time_uncertainty_engine.use_gpu
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"💾 Integrated mesh exported to {filepath}")
    
    def get_integrated_scenario_summary(self) -> Dict:
        """
        Get summary of integrated scenarios
        
        Returns:
            Dictionary with scenario summary
        """
        if self.integrated_mesh_data is None:
            return {}
        
        time_summary = self.time_uncertainty_engine.get_scenario_summary()
        stochastic_status = self.stochastic_engine.get_mesh_status()
        
        return {
            'time_uncertainty_summary': time_summary,
            'stochastic_mesh_summary': {
                'total_nodes': stochastic_status.get('total_nodes', 0),
                'current_position': stochastic_status.get('current_position', 'unknown'),
                'mesh_status': stochastic_status.get('status', 'unknown')
            },
            'integrated_summary': {
                'total_integrated_nodes': len(self.integrated_mesh_data['integrated_nodes']),
                'time_horizon_years': len(self.integrated_mesh_data['time_steps']) / 12,
                'scenarios_processed': self.integrated_mesh_data['num_scenarios']
            }
        }


# Demo function for integrated system
def demo_integrated_mesh():
    """Demonstrate the integrated mesh system"""
    print("🎯 Integrated Mesh System Demo")
    print("=" * 50)
    
    # Create sample financial state
    initial_state = {
        'cash': 1000000.0,
        'investments': 500000.0,
        'debts': 0.0,
        'income': 150000.0,
        'expenses': 0.0
    }
    
    # Create sample milestones
    milestones = [
        {
            'id': 'college_start',
            'description': 'College starts',
            'estimated_date': '2027-09-01',
            'amount': 80000,
            'timing_uncertainty': 0.2,
            'amount_uncertainty': 0.1,
            'drift_rate': 0.05,
            'probability': 0.9,
            'category': 'education'
        },
        {
            'id': 'wedding',
            'description': 'Wedding expenses',
            'estimated_date': '2028-06-15',
            'amount': 25000,
            'timing_uncertainty': 0.3,
            'amount_uncertainty': 0.15,
            'drift_rate': 0.03,
            'probability': 0.7,
            'category': 'life_event'
        },
        {
            'id': 'house_down_payment',
            'description': 'House down payment',
            'estimated_date': '2030-03-01',
            'amount': 150000,
            'timing_uncertainty': 0.5,
            'amount_uncertainty': 0.2,
            'drift_rate': 0.04,
            'probability': 0.8,
            'category': 'housing'
        }
    ]
    
    # Initialize integrated mesh
    integration = TimeUncertaintyIntegration(initial_state)
    
    # Initialize integrated mesh
    integrated_data, risk_analysis = integration.initialize_integrated_mesh(
        milestones, 
        num_scenarios=5000,  # Reduced for demo
        time_horizon_years=15
    )
    
    # Get payment options
    payment_options = integration.get_integrated_payment_options()
    print(f"\n💳 Payment options available for {len(payment_options)} milestones")
    
    # Get risk analysis
    risk_summary = integration.get_integrated_risk_analysis()
    print(f"\n📊 Risk analysis complete with {len(risk_summary.get('time_uncertainty_metrics', {}))} metrics")
    
    # Get scenario summary
    scenario_summary = integration.get_integrated_scenario_summary()
    print(f"\n📈 Scenario summary:")
    for key, value in scenario_summary.get('integrated_summary', {}).items():
        print(f"   {key}: {value}")
    
    # Export integrated mesh
    integration.export_integrated_mesh("demo_integrated_mesh.json")
    
    print("\n✅ Integrated mesh demo complete!")
    return integration, integrated_data, risk_analysis


if __name__ == "__main__":
    demo_integrated_mesh() 