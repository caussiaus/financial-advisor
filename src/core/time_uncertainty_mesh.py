"""
Vector-Friendly Time Uncertainty Mesh System

This module implements a comprehensive time uncertainty mesh using Geometric Brownian Motion (GBM)
for modeling uncertain event timings and amounts. The system is fully vectorized and GPU-accelerated
for processing thousands of scenarios simultaneously.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import platform
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Try importing acceleration libraries
try:
    import torch
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Use Metal for M1/M2 Macs
        METAL_AVAILABLE = torch.backends.mps.is_available()
        if METAL_AVAILABLE:
            device = torch.device("mps")
            # Set default tensor type to float32 for Metal
            torch.set_default_dtype(torch.float32)
        else:
            device = torch.device("cpu")
        CUDA_AVAILABLE = False
    else:
        # Try CUDA for other systems
        CUDA_AVAILABLE = torch.cuda.is_available()
        METAL_AVAILABLE = False
        if CUDA_AVAILABLE:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    ACCELERATION_AVAILABLE = CUDA_AVAILABLE or METAL_AVAILABLE
except ImportError:
    ACCELERATION_AVAILABLE = False
    CUDA_AVAILABLE = False
    METAL_AVAILABLE = False
    device = None
    print("GPU acceleration not available. Installing PyTorch may improve performance.")

@dataclass
class SeedEvent:
    """Represents a seed event with uncertain timing and amount"""
    event_id: str
    description: str
    estimated_date: str  # ISO format date string
    amount: float
    timing_volatility: float  # Standard deviation of timing uncertainty (years)
    amount_volatility: float  # Standard deviation of amount uncertainty (fraction)
    drift_rate: float  # Annual drift rate for amount
    probability: float  # Probability of event occurring
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)

@dataclass
class EventVector:
    """Vectorized representation of events across scenarios"""
    timings: np.ndarray  # Shape: (n_events, n_scenarios)
    amounts: np.ndarray  # Shape: (n_events, n_scenarios)
    probabilities: np.ndarray  # Shape: (n_events, n_scenarios)
    event_ids: List[str]
    descriptions: List[str]
    categories: List[str]

@dataclass
class MeshNode:
    """Represents a node in the time uncertainty mesh"""
    timestamp: pd.Timestamp
    scenario_id: int
    financial_state: np.ndarray  # [cash, investments, debts, income, expenses]
    probability: float
    event_impacts: Dict[str, float] = field(default_factory=dict)

class TimeUncertaintyMeshEngine:
    """
    Vector-friendly time uncertainty mesh engine using GBM for event timing and amounts
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and ACCELERATION_AVAILABLE
        self.mesh_states = None
        self.risk_metrics = None
        self.event_vectors = None
        self.time_steps = None
        self.scenario_weights = None
        
        if self.use_gpu:
            print(f"🚀 Using {'Metal' if METAL_AVAILABLE else 'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
        else:
            print("📊 Using CPU-only processing")
    
    def convert_events_to_vectors(self, seed_events: List[SeedEvent], 
                                num_scenarios: int = 10000) -> EventVector:
        """
        Convert seed events to vectorized GBM scenarios
        
        Args:
            seed_events: List of seed events with uncertain timing/amounts
            num_scenarios: Number of Monte Carlo scenarios to generate
            
        Returns:
            EventVector containing all scenarios
        """
        print(f"🔄 Converting {len(seed_events)} events to {num_scenarios} scenarios...")
        
        n_events = len(seed_events)
        
        # Pre-allocate arrays for efficiency
        timings = np.zeros((n_events, num_scenarios), dtype=np.float32)
        amounts = np.zeros((n_events, num_scenarios), dtype=np.float32)
        probabilities = np.zeros((n_events, num_scenarios), dtype=np.float32)
        
        # Extract metadata
        event_ids = [event.event_id for event in seed_events]
        descriptions = [event.description for event in seed_events]
        categories = [event.category for event in seed_events]
        
        # Generate scenarios for each event
        for i, event in enumerate(seed_events):
            base_date = pd.to_datetime(event.estimated_date)
            base_amount = event.amount
            
            # Generate GBM scenarios for timing uncertainty
            timing_noise = np.random.normal(0, event.timing_volatility, num_scenarios)
            # Convert years to days and add to base date
            timing_days = timing_noise * 365
            timings[i, :] = (base_date + pd.to_timedelta(timing_days, unit='D')).astype(np.int64) // 10**9  # Convert to Unix timestamp
            
            # Generate GBM scenarios for amount uncertainty
            amount_noise = np.random.normal(0, event.amount_volatility, num_scenarios)
            # Apply drift and volatility
            amounts[i, :] = base_amount * np.exp(
                (event.drift_rate - 0.5 * event.amount_volatility**2) * 1 +  # 1 year horizon
                event.amount_volatility * amount_noise
            )
            
            # Set probabilities (can be adjusted based on timing/amount)
            probabilities[i, :] = event.probability
        
        print(f"✅ Generated {num_scenarios} scenarios for {n_events} events")
        
        return EventVector(
            timings=timings,
            amounts=amounts,
            probabilities=probabilities,
            event_ids=event_ids,
            descriptions=descriptions,
            categories=categories
        )
    
    def generate_mesh_nodes_vectorized(self, event_vectors: EventVector, 
                                     time_horizon_years: float = 10,
                                     monthly_steps: bool = True,
                                     initial_state: Dict[str, float] = None) -> Dict:
        """
        Generate mesh nodes using vectorized operations
        
        Args:
            event_vectors: Vectorized event data
            time_horizon_years: Time horizon for mesh
            monthly_steps: Whether to use monthly or quarterly steps
            initial_state: Initial financial state
            
        Returns:
            Dictionary containing mesh data
        """
        print(f"🌐 Generating mesh nodes for {time_horizon_years} years...")
        
        # Set default initial state if not provided
        if initial_state is None:
            initial_state = {
                'cash': 1000000.0,
                'investments': 500000.0,
                'debts': 0.0,
                'income': 150000.0,
                'expenses': 0.0
            }
        
        # Create time grid
        if monthly_steps:
            time_steps = pd.date_range(
                start=pd.Timestamp.now(),
                periods=int(time_horizon_years * 12),
                freq='ME'
            )
        else:
            time_steps = pd.date_range(
                start=pd.Timestamp.now(),
                periods=int(time_horizon_years * 4),
                freq='QE'
            )
        
        n_timesteps = len(time_steps)
        n_scenarios = event_vectors.timings.shape[1]
        n_financial_features = 5  # cash, investments, debts, income, expenses
        
        # Pre-allocate mesh state matrix
        # Shape: (n_timesteps, n_scenarios, n_financial_features)
        mesh_states = np.zeros((n_timesteps, n_scenarios, n_financial_features), dtype=np.float32)
        
        # Initialize all scenarios with initial state
        for feature_idx, (key, value) in enumerate(initial_state.items()):
            mesh_states[0, :, feature_idx] = value
        
        # Vectorized event processing
        print(f"📊 Processing {n_timesteps} time steps across {n_scenarios} scenarios...")
        
        for t, current_time in enumerate(time_steps):
            if t == 0:
                continue  # Skip initial state
                
            # Copy previous state
            mesh_states[t, :, :] = mesh_states[t-1, :, :]
            
            # For each scenario, check which events have happened by this time
            current_timestamp = current_time.timestamp()
            
            # Vectorized event impact calculation
            for scenario in range(n_scenarios):
                # Find events that have happened in this scenario by current time
                events_happened = event_vectors.timings[:, scenario] <= current_timestamp
                
                # Calculate total impact for this scenario
                if np.any(events_happened):
                    # Sum amounts for events that happened, weighted by probability
                    total_impact = np.sum(
                        event_vectors.amounts[events_happened, scenario] * 
                        event_vectors.probabilities[events_happened, scenario]
                    )
                    
                    # Update financial state
                    mesh_states[t, scenario, 0] -= total_impact  # Reduce cash
                    mesh_states[t, scenario, 4] += total_impact  # Increase expenses
                
                # Apply simple growth/inflation to investments and income
                mesh_states[t, scenario, 1] *= 1.05  # 5% investment growth
                mesh_states[t, scenario, 3] *= 1.02  # 2% income growth
        
        # Calculate scenario weights (equal for now, can be adjusted)
        scenario_weights = np.ones(n_scenarios) / n_scenarios
        
        print(f"✅ Generated mesh with {n_timesteps} time steps and {n_scenarios} scenarios")
        
        return {
            'states': mesh_states,
            'time_steps': time_steps,
            'scenario_weights': scenario_weights,
            'event_vectors': event_vectors
        }
    
    def analyze_mesh_risk_vectorized(self, mesh_data: Dict) -> Dict:
        """
        Vectorized risk analysis across all scenarios
        
        Args:
            mesh_data: Mesh data from generate_mesh_nodes_vectorized
            
        Returns:
            Dictionary containing risk metrics
        """
        print("📈 Performing vectorized risk analysis...")
        
        states = mesh_data['states']  # Shape: (n_timesteps, n_scenarios, n_features)
        scenario_weights = mesh_data['scenario_weights']
        
        # Extract key financial metrics
        cash_positions = states[:, :, 0]  # All cash positions across time and scenarios
        investments = states[:, :, 1]
        debts = states[:, :, 2]
        income = states[:, :, 3]
        expenses = states[:, :, 4]
        
        # Calculate net worth
        net_worth = cash_positions + investments - debts
        
        # Vectorized risk calculations
        min_cash_by_scenario = np.min(cash_positions, axis=0)
        max_drawdown_by_scenario = np.max(net_worth, axis=0) - np.min(net_worth, axis=0)
        
        # Probability of negative cash (vectorized)
        negative_cash_prob = np.mean(cash_positions < 0, axis=0)
        
        # Value at Risk (vectorized)
        var_95 = np.percentile(net_worth, 5, axis=1)  # 95% VaR at each timestep
        var_99 = np.percentile(net_worth, 1, axis=1)  # 99% VaR at each timestep
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.array([
            np.mean(net_worth[t, net_worth[t, :] <= var_95[t]]) 
            for t in range(net_worth.shape[0])
        ])
        
        # Scenario statistics
        worst_case_scenario = np.argmin(np.min(cash_positions, axis=0))
        best_case_scenario = np.argmax(np.max(net_worth, axis=0))
        
        # Portfolio statistics
        total_assets = cash_positions + investments
        debt_to_asset_ratio = np.mean(debts / (total_assets + 1e-8), axis=0)  # Avoid division by zero
        
        print("✅ Risk analysis complete")
        
        return {
            'min_cash_by_scenario': min_cash_by_scenario,
            'max_drawdown_by_scenario': max_drawdown_by_scenario,
            'negative_cash_probability': negative_cash_prob,
            'var_95_timeline': var_95,
            'var_99_timeline': var_99,
            'expected_shortfall_95': es_95,
            'worst_case_scenario': worst_case_scenario,
            'best_case_scenario': best_case_scenario,
            'debt_to_asset_ratio': debt_to_asset_ratio,
            'net_worth_timeline': net_worth,
            'cash_timeline': cash_positions,
            'investment_timeline': investments
        }
    
    def accelerate_mesh_on_gpu(self, mesh_data: Dict) -> Dict:
        """
        Move mesh calculations to GPU for speed
        
        Args:
            mesh_data: Mesh data to accelerate
            
        Returns:
            GPU-accelerated risk analysis
        """
        if not self.use_gpu:
            print("⚠️ GPU acceleration not available, using CPU")
            return self.analyze_mesh_risk_vectorized(mesh_data)
        
        print("🚀 Accelerating mesh calculations on GPU...")
        
        # Convert to GPU tensors
        states_gpu = torch.tensor(mesh_data['states'], device=device, dtype=torch.float32)
        scenario_weights_gpu = torch.tensor(mesh_data['scenario_weights'], device=device, dtype=torch.float32)
        
        # GPU-accelerated risk calculations
        cash_gpu = states_gpu[:, :, 0]
        investments_gpu = states_gpu[:, :, 1]
        debts_gpu = states_gpu[:, :, 2]
        
        # Calculate net worth on GPU
        net_worth_gpu = cash_gpu + investments_gpu - debts_gpu
        
        # GPU-accelerated statistics
        min_cash_gpu = torch.min(cash_gpu, dim=0)[0]
        max_drawdown_gpu = torch.max(net_worth_gpu, dim=0)[0] - torch.min(net_worth_gpu, dim=0)[0]
        
        # Value at Risk on GPU
        var_95_gpu = torch.quantile(net_worth_gpu, 0.05, dim=1)
        var_99_gpu = torch.quantile(net_worth_gpu, 0.01, dim=1)
        
        # Probability calculations on GPU
        negative_cash_prob_gpu = torch.mean((cash_gpu < 0).float(), dim=0)
        
        # Scenario statistics
        worst_case_scenario = torch.argmin(torch.min(cash_gpu, dim=0)[0])
        best_case_scenario = torch.argmax(torch.max(net_worth_gpu, dim=0)[0])
        
        # Portfolio statistics
        total_assets = cash_gpu + investments_gpu
        debt_to_asset_ratio = torch.mean(debts_gpu / (total_assets + 1e-8), dim=0)
        
        # Move results back to CPU
        result = {
            'min_cash_by_scenario': min_cash_gpu.cpu().numpy(),
            'max_drawdown_by_scenario': max_drawdown_gpu.cpu().numpy(),
            'negative_cash_probability': negative_cash_prob_gpu.cpu().numpy(),
            'var_95_timeline': var_95_gpu.cpu().numpy(),
            'var_99_timeline': var_99_gpu.cpu().numpy(),
            'expected_shortfall_95': var_95_gpu.cpu().numpy(),  # Simplified for GPU
            'worst_case_scenario': worst_case_scenario.cpu().numpy().item(),
            'best_case_scenario': best_case_scenario.cpu().numpy().item(),
            'debt_to_asset_ratio': debt_to_asset_ratio.cpu().numpy(),
            'net_worth_timeline': net_worth_gpu.cpu().numpy(),
            'cash_timeline': cash_gpu.cpu().numpy(),
            'investment_timeline': investments_gpu.cpu().numpy(),
            'gpu_accelerated': True
        }
        
        print("✅ GPU acceleration complete")
        return result
    
    def initialize_mesh_with_time_uncertainty(self, seed_events: List[SeedEvent], 
                                           num_scenarios: int = 10000,
                                           time_horizon_years: float = 10) -> Tuple[Dict, Dict]:
        """
        Complete initialization of mesh with time uncertainty
        
        Args:
            seed_events: List of seed events
            num_scenarios: Number of Monte Carlo scenarios
            time_horizon_years: Time horizon for mesh
            
        Returns:
            Tuple of (mesh_data, risk_analysis)
        """
        print(f"🎯 Initializing time uncertainty mesh with {len(seed_events)} events...")
        
        # Step 1: Convert events to vectors
        event_vectors = self.convert_events_to_vectors(seed_events, num_scenarios)
        
        # Step 2: Generate mesh nodes
        mesh_data = self.generate_mesh_nodes_vectorized(
            event_vectors, 
            time_horizon_years=time_horizon_years
        )
        
        # Step 3: Analyze risks
        if self.use_gpu:
            risk_analysis = self.accelerate_mesh_on_gpu(mesh_data)
        else:
            risk_analysis = self.analyze_mesh_risk_vectorized(mesh_data)
        
        # Store for later use
        self.mesh_states = mesh_data['states']
        self.risk_metrics = risk_analysis
        self.event_vectors = event_vectors
        self.time_steps = mesh_data['time_steps']
        self.scenario_weights = mesh_data['scenario_weights']
        
        print(f"✅ Time uncertainty mesh initialized successfully")
        print(f"   - {len(seed_events)} events")
        print(f"   - {num_scenarios} scenarios")
        print(f"   - {len(mesh_data['time_steps'])} time steps")
        print(f"   - {'GPU' if self.use_gpu else 'CPU'} accelerated")
        
        return mesh_data, risk_analysis
    
    def get_scenario_summary(self) -> Dict:
        """Get summary statistics across all scenarios"""
        if self.risk_metrics is None:
            return {}
        
        return {
            'total_scenarios': len(self.scenario_weights),
            'worst_case_cash': np.min(self.risk_metrics['min_cash_by_scenario']),
            'best_case_cash': np.max(self.risk_metrics['min_cash_by_scenario']),
            'avg_max_drawdown': np.mean(self.risk_metrics['max_drawdown_by_scenario']),
            'scenarios_with_negative_cash': np.sum(self.risk_metrics['negative_cash_probability'] > 0),
            'worst_scenario_id': self.risk_metrics['worst_case_scenario'],
            'best_scenario_id': self.risk_metrics['best_case_scenario']
        }
    
    def export_mesh_data(self, filepath: str):
        """Export mesh data to JSON file"""
        if self.mesh_states is None:
            print("⚠️ No mesh data to export")
            return
        
        export_data = {
            'mesh_states': self.mesh_states.tolist(),
            'time_steps': [ts.isoformat() for ts in self.time_steps],
            'scenario_weights': self.scenario_weights.tolist(),
            'risk_metrics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.risk_metrics.items()
            },
            'event_vectors': {
                'timings': self.event_vectors.timings.tolist(),
                'amounts': self.event_vectors.amounts.tolist(),
                'probabilities': self.event_vectors.probabilities.tolist(),
                'event_ids': self.event_vectors.event_ids,
                'descriptions': self.event_vectors.descriptions,
                'categories': self.event_vectors.categories
            },
            'metadata': {
                'num_scenarios': len(self.scenario_weights),
                'num_time_steps': len(self.time_steps),
                'num_events': len(self.event_vectors.event_ids),
                'gpu_accelerated': self.use_gpu,
                'export_timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Mesh data exported to {filepath}")
    
    def load_mesh_data(self, filepath: str):
        """Load mesh data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct mesh data
        self.mesh_states = np.array(data['mesh_states'])
        self.time_steps = pd.to_datetime(data['time_steps'])
        self.scenario_weights = np.array(data['scenario_weights'])
        
        # Reconstruct event vectors
        event_data = data['event_vectors']
        self.event_vectors = EventVector(
            timings=np.array(event_data['timings']),
            amounts=np.array(event_data['amounts']),
            probabilities=np.array(event_data['probabilities']),
            event_ids=event_data['event_ids'],
            descriptions=event_data['descriptions'],
            categories=event_data['categories']
        )
        
        # Reconstruct risk metrics
        self.risk_metrics = {}
        for k, v in data['risk_metrics'].items():
            if isinstance(v, list):
                self.risk_metrics[k] = np.array(v)
            else:
                self.risk_metrics[k] = v
        
        print(f"📂 Mesh data loaded from {filepath}")
        print(f"   - {len(self.scenario_weights)} scenarios")
        print(f"   - {len(self.time_steps)} time steps")
        print(f"   - {len(self.event_vectors.event_ids)} events")


# Example usage and demo functions
def create_sample_events() -> List[SeedEvent]:
    """Create sample events for demonstration"""
    return [
        SeedEvent(
            event_id="college_start",
            description="College starts",
            estimated_date="2027-09-01",
            amount=80000,
            timing_volatility=0.2,
            amount_volatility=0.1,
            drift_rate=0.05,
            probability=0.9,
            category="education"
        ),
        SeedEvent(
            event_id="wedding",
            description="Wedding expenses",
            estimated_date="2028-06-15",
            amount=25000,
            timing_volatility=0.3,
            amount_volatility=0.15,
            drift_rate=0.03,
            probability=0.7,
            category="life_event"
        ),
        SeedEvent(
            event_id="house_down_payment",
            description="House down payment",
            estimated_date="2030-03-01",
            amount=150000,
            timing_volatility=0.5,
            amount_volatility=0.2,
            drift_rate=0.04,
            probability=0.8,
            category="housing"
        ),
        SeedEvent(
            event_id="retirement",
            description="Retirement",
            estimated_date="2045-01-01",
            amount=0,  # Income reduction
            timing_volatility=0.8,
            amount_volatility=0.1,
            drift_rate=-0.02,  # Negative drift for income reduction
            probability=0.95,
            category="retirement"
        )
    ]


def demo_time_uncertainty_mesh():
    """Demonstrate the time uncertainty mesh system"""
    print("🎯 Time Uncertainty Mesh Demo")
    print("=" * 50)
    
    # Create sample events
    events = create_sample_events()
    print(f"📋 Created {len(events)} sample events")
    
    # Initialize mesh engine
    engine = TimeUncertaintyMeshEngine(use_gpu=True)
    
    # Initialize mesh with time uncertainty
    mesh_data, risk_analysis = engine.initialize_mesh_with_time_uncertainty(
        events, 
        num_scenarios=5000,  # Reduced for demo
        time_horizon_years=15
    )
    
    # Get scenario summary
    summary = engine.get_scenario_summary()
    print("\n📊 Scenario Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Export mesh data
    engine.export_mesh_data("demo_mesh_data.json")
    
    print("\n✅ Demo complete!")
    return engine, mesh_data, risk_analysis


if __name__ == "__main__":
    demo_time_uncertainty_mesh() 