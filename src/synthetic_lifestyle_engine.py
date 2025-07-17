"""
Synthetic Lifestyle Engine

This module creates a comprehensive pipeline for generating synthetic client data
with realistic lifestyle events that naturally occur based on age and life stage.
It integrates the JSON-to-vector converter with the existing synthetic data generator
to create a surface of discretionary spending homogeneously sorted.

Key Features:
- Generates synthetic client profiles with realistic demographics
- Creates lifestyle events based on age and life stage probabilities
- Models cash flow impacts and discretionary spending patterns
- Integrates with mesh engines for financial modeling
- Provides analysis of financial standing across different life stages
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import random
import logging
from dataclasses import dataclass, field

# Import existing components
from .json_to_vector_converter import (
    JSONToVectorConverter, 
    ClientVectorProfile, 
    LifeStage, 
    EventCategory,
    LifestyleEvent
)
from .synthetic_data_generator import SyntheticFinancialDataGenerator, PersonProfile
from .time_uncertainty_mesh import TimeUncertaintyMeshEngine, SeedEvent


@dataclass
class SyntheticClientData:
    """Complete synthetic client data with lifestyle events"""
    client_id: str
    profile: PersonProfile
    vector_profile: ClientVectorProfile
    lifestyle_events: List[LifestyleEvent]
    seed_events: List[SeedEvent]
    mesh_data: Optional[Dict] = None
    risk_analysis: Optional[Dict] = None
    financial_metrics: Dict[str, float] = field(default_factory=dict)


class SyntheticLifestyleEngine:
    """
    Comprehensive engine for generating synthetic client data with lifestyle events
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.json_converter = JSONToVectorConverter(use_gpu=use_gpu)
        self.synthetic_generator = SyntheticFinancialDataGenerator()
        self.mesh_engine = TimeUncertaintyMeshEngine(use_gpu=use_gpu)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Configuration for different life stages
        self.life_stage_configs = self._initialize_life_stage_configs()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the engine"""
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
    
    def _initialize_life_stage_configs(self) -> Dict[LifeStage, Dict]:
        """Initialize configurations for different life stages"""
        configs = {
            LifeStage.EARLY_CAREER: {
                'age_range': (22, 30),
                'event_frequency': 0.8,  # Higher event frequency
                'education_weight': 0.4,
                'career_weight': 0.3,
                'family_weight': 0.2,
                'housing_weight': 0.1,
                'avg_events_per_year': 2.5
            },
            LifeStage.MID_CAREER: {
                'age_range': (31, 45),
                'event_frequency': 0.7,
                'education_weight': 0.2,
                'career_weight': 0.4,
                'family_weight': 0.3,
                'housing_weight': 0.1,
                'avg_events_per_year': 2.0
            },
            LifeStage.ESTABLISHED: {
                'age_range': (46, 60),
                'event_frequency': 0.6,
                'education_weight': 0.1,
                'career_weight': 0.3,
                'family_weight': 0.2,
                'housing_weight': 0.3,
                'health_weight': 0.1,
                'avg_events_per_year': 1.8
            },
            LifeStage.PRE_RETIREMENT: {
                'age_range': (61, 67),
                'event_frequency': 0.5,
                'education_weight': 0.05,
                'career_weight': 0.2,
                'family_weight': 0.1,
                'housing_weight': 0.2,
                'health_weight': 0.3,
                'retirement_weight': 0.15,
                'avg_events_per_year': 1.5
            },
            LifeStage.RETIREMENT: {
                'age_range': (68, 85),
                'event_frequency': 0.4,
                'education_weight': 0.02,
                'career_weight': 0.05,
                'family_weight': 0.1,
                'housing_weight': 0.2,
                'health_weight': 0.4,
                'retirement_weight': 0.23,
                'avg_events_per_year': 1.2
            }
        }
        return configs
    
    def generate_synthetic_client(self, target_age: Optional[int] = None, 
                                target_life_stage: Optional[LifeStage] = None) -> SyntheticClientData:
        """
        Generate a complete synthetic client with lifestyle events
        
        Args:
            target_age: Specific age to target (optional)
            target_life_stage: Specific life stage to target (optional)
            
        Returns:
            SyntheticClientData with complete client information
        """
        # Generate base profile
        if target_age is not None:
            profile = self._generate_profile_for_age(target_age)
        elif target_life_stage is not None:
            profile = self._generate_profile_for_life_stage(target_life_stage)
        else:
            profile = self.synthetic_generator.generate_person_profile()
        
        # Convert to JSON format for vector conversion
        json_data = self._convert_profile_to_json(profile)
        
        # Convert to vector profile
        vector_profile = self.json_converter.convert_json_to_vector_profile(json_data)
        
        # Generate lifestyle events based on age and life stage
        config = self.life_stage_configs[vector_profile.life_stage]
        num_events = int(config['avg_events_per_year'] * 5)  # 5-year horizon
        
        lifestyle_events = self._generate_lifestyle_events_for_profile(
            vector_profile, num_events, config
        )
        
        # Convert to seed events for mesh processing
        seed_events = self.json_converter.convert_events_to_seed_events(lifestyle_events)
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics(vector_profile, lifestyle_events)
        
        return SyntheticClientData(
            client_id=profile.name,
            profile=profile,
            vector_profile=vector_profile,
            lifestyle_events=lifestyle_events,
            seed_events=seed_events,
            financial_metrics=financial_metrics
        )
    
    def _generate_profile_for_age(self, age: int) -> PersonProfile:
        """Generate a profile for a specific age"""
        # Create a profile with the target age
        profile = self.synthetic_generator.generate_person_profile()
        
        # Override age while keeping other characteristics realistic
        return PersonProfile(
            name=profile.name,
            age=age,
            occupation=profile.occupation,
            base_income=profile.base_income,
            family_status=profile.family_status,
            location=profile.location,
            risk_tolerance=profile.risk_tolerance,
            financial_goals=profile.financial_goals,
            current_assets=profile.current_assets,
            debts=profile.debts
        )
    
    def _generate_profile_for_life_stage(self, life_stage: LifeStage) -> PersonProfile:
        """Generate a profile for a specific life stage"""
        config = self.life_stage_configs[life_stage]
        age_range = config['age_range']
        target_age = random.randint(age_range[0], age_range[1])
        
        return self._generate_profile_for_age(target_age)
    
    def _convert_profile_to_json(self, profile: PersonProfile) -> Dict:
        """Convert PersonProfile to JSON format for vector conversion"""
        return {
            'client_id': f"CLIENT_{profile.name}_{profile.age}",
            'age': profile.age,
            'income': profile.base_income,
            'current_assets': profile.current_assets,
            'debts': profile.debts,
            'risk_tolerance': profile.risk_tolerance,
            'family_status': profile.family_status,
            'occupation': profile.occupation
        }
    
    def _generate_lifestyle_events_for_profile(self, vector_profile: ClientVectorProfile, 
                                             num_events: int, config: Dict) -> List[LifestyleEvent]:
        """Generate lifestyle events based on profile and configuration"""
        events = []
        
        # Calculate event category weights for this life stage
        category_weights = {
            EventCategory.EDUCATION: config.get('education_weight', 0.1),
            EventCategory.CAREER: config.get('career_weight', 0.3),
            EventCategory.FAMILY: config.get('family_weight', 0.2),
            EventCategory.HOUSING: config.get('housing_weight', 0.2),
            EventCategory.HEALTH: config.get('health_weight', 0.1),
            EventCategory.RETIREMENT: config.get('retirement_weight', 0.1)
        }
        
        # Normalize weights
        total_weight = sum(category_weights.values())
        if total_weight > 0:
            category_weights = {k: v/total_weight for k, v in category_weights.items()}
        
        # Generate events with weighted category selection
        for i in range(num_events):
            # Select category based on weights
            categories = list(category_weights.keys())
            weights = [category_weights[cat] for cat in categories]
            
            if sum(weights) > 0:
                selected_category = random.choices(categories, weights=weights)[0]
            else:
                selected_category = random.choice(list(EventCategory))
            
            # Create event
            event = self.json_converter._create_lifestyle_event(
                event_id=f"event_{vector_profile.client_id}_{i}",
                category=selected_category,
                age=vector_profile.age,
                income=vector_profile.base_income,
                life_stage=vector_profile.life_stage
            )
            
            events.append(event)
        
        return events
    
    def _calculate_financial_metrics(self, vector_profile: ClientVectorProfile, 
                                   events: List[LifestyleEvent]) -> Dict[str, float]:
        """Calculate comprehensive financial metrics"""
        # Basic metrics
        net_worth = np.sum(vector_profile.current_assets) - np.sum(vector_profile.current_debts)
        total_assets = np.sum(vector_profile.current_assets)
        total_debts = np.sum(vector_profile.current_debts)
        
        # Cash flow metrics
        avg_monthly_cash_flow = np.mean(vector_profile.cash_flow_vector)
        cash_flow_volatility = np.std(vector_profile.cash_flow_vector)
        
        # Discretionary spending metrics
        avg_discretionary = np.mean(vector_profile.discretionary_spending_surface)
        discretionary_volatility = np.std(vector_profile.discretionary_spending_surface)
        
        # Event impact analysis
        total_event_impact = sum(event.base_amount for event in events)
        positive_events = [e for e in events if e.cash_flow_impact == "positive"]
        negative_events = [e for e in events if e.cash_flow_impact == "negative"]
        
        positive_impact = sum(e.base_amount for e in positive_events)
        negative_impact = sum(e.base_amount for e in negative_events)
        
        # Risk metrics
        debt_to_income_ratio = total_debts / vector_profile.base_income if vector_profile.base_income > 0 else 0
        savings_rate = avg_monthly_cash_flow / (vector_profile.base_income / 12) if vector_profile.base_income > 0 else 0
        
        return {
            'net_worth': net_worth,
            'total_assets': total_assets,
            'total_debts': total_debts,
            'avg_monthly_cash_flow': avg_monthly_cash_flow,
            'cash_flow_volatility': cash_flow_volatility,
            'avg_discretionary_spending': avg_discretionary,
            'discretionary_volatility': discretionary_volatility,
            'total_event_impact': total_event_impact,
            'positive_event_impact': positive_impact,
            'negative_event_impact': negative_impact,
            'debt_to_income_ratio': debt_to_income_ratio,
            'savings_rate': savings_rate,
            'risk_tolerance': vector_profile.risk_tolerance
        }
    
    def generate_client_batch(self, num_clients: int, 
                            age_distribution: Optional[Dict[int, float]] = None) -> List[SyntheticClientData]:
        """
        Generate a batch of synthetic clients
        
        Args:
            num_clients: Number of clients to generate
            age_distribution: Optional age distribution (age -> probability)
            
        Returns:
            List of SyntheticClientData objects
        """
        self.logger.info(f"Generating batch of {num_clients} synthetic clients")
        
        clients = []
        
        for i in range(num_clients):
            if age_distribution:
                # Select age based on distribution
                ages = list(age_distribution.keys())
                probabilities = list(age_distribution.values())
                target_age = random.choices(ages, weights=probabilities)[0]
                client = self.generate_synthetic_client(target_age=target_age)
            else:
                client = self.generate_synthetic_client()
            
            clients.append(client)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i + 1}/{num_clients} clients")
        
        self.logger.info(f"Successfully generated {len(clients)} synthetic clients")
        return clients
    
    def process_with_mesh_engine(self, client_data: SyntheticClientData, 
                               num_scenarios: int = 1000, 
                               time_horizon_years: int = 5) -> SyntheticClientData:
        """
        Process client data with mesh engine for financial modeling
        
        Args:
            client_data: Synthetic client data
            num_scenarios: Number of Monte Carlo scenarios
            time_horizon_years: Time horizon for analysis
            
        Returns:
            Updated SyntheticClientData with mesh results
        """
        self.logger.info(f"Processing {client_data.client_id} with mesh engine")
        
        try:
            # Initialize mesh with seed events
            mesh_data, risk_analysis = self.mesh_engine.initialize_mesh_with_time_uncertainty(
                client_data.seed_events,
                num_scenarios=num_scenarios,
                time_horizon_years=time_horizon_years
            )
            
            # Update client data with mesh results
            client_data.mesh_data = mesh_data
            client_data.risk_analysis = risk_analysis
            
            self.logger.info(f"Successfully processed {client_data.client_id} with mesh engine")
            
        except Exception as e:
            self.logger.error(f"Error processing {client_data.client_id} with mesh engine: {e}")
        
        return client_data
    
    def analyze_financial_standing(self, clients: List[SyntheticClientData]) -> Dict:
        """
        Analyze financial standing across all clients
        
        Args:
            clients: List of synthetic client data
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing financial standing for {len(clients)} clients")
        
        # Group by life stage
        life_stage_analysis = {}
        
        for client in clients:
            stage = client.vector_profile.life_stage.value
            if stage not in life_stage_analysis:
                life_stage_analysis[stage] = []
            
            life_stage_analysis[stage].append(client.financial_metrics)
        
        # Calculate statistics for each life stage
        results = {}
        
        for stage, metrics_list in life_stage_analysis.items():
            if metrics_list:
                # Calculate averages
                avg_metrics = {}
                for key in metrics_list[0].keys():
                    values = [m[key] for m in metrics_list]
                    avg_metrics[f'avg_{key}'] = np.mean(values)
                    avg_metrics[f'std_{key}'] = np.std(values)
                    avg_metrics[f'min_{key}'] = np.min(values)
                    avg_metrics[f'max_{key}'] = np.max(values)
                
                results[stage] = {
                    'count': len(metrics_list),
                    'metrics': avg_metrics
                }
        
        return results
    
    def export_synthetic_data(self, clients: List[SyntheticClientData], 
                            filename: str) -> None:
        """
        Export synthetic client data to JSON file
        
        Args:
            clients: List of synthetic client data
            filename: Output filename
        """
        export_data = []
        
        for client in clients:
            client_data = {
                'client_id': client.client_id,
                'profile': {
                    'name': client.profile.name,
                    'age': client.profile.age,
                    'occupation': client.profile.occupation,
                    'base_income': client.profile.base_income,
                    'family_status': client.profile.family_status,
                    'location': client.profile.location,
                    'risk_tolerance': client.profile.risk_tolerance,
                    'financial_goals': client.profile.financial_goals,
                    'current_assets': client.profile.current_assets,
                    'debts': client.profile.debts
                },
                'vector_profile': {
                    'life_stage': client.vector_profile.life_stage.value,
                    'current_assets': client.vector_profile.current_assets.tolist(),
                    'current_debts': client.vector_profile.current_debts.tolist(),
                    'risk_tolerance': client.vector_profile.risk_tolerance,
                    'event_probabilities': client.vector_profile.event_probabilities.tolist(),
                    'cash_flow_vector': client.vector_profile.cash_flow_vector.tolist(),
                    'discretionary_spending_surface': client.vector_profile.discretionary_spending_surface.tolist()
                },
                'lifestyle_events': [
                    {
                        'event_id': event.event_id,
                        'category': event.category.value,
                        'description': event.description,
                        'base_amount': event.base_amount,
                        'base_probability': event.base_probability,
                        'cash_flow_impact': event.cash_flow_impact
                    }
                    for event in client.lifestyle_events
                ],
                'financial_metrics': client.financial_metrics
            }
            
            export_data.append(client_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(clients)} synthetic clients to {filename}")


def create_demo_engine():
    """Create and demonstrate the synthetic lifestyle engine"""
    print("🚀 Synthetic Lifestyle Engine Demo")
    print("=" * 60)
    
    # Create engine
    engine = SyntheticLifestyleEngine(use_gpu=False)  # Use CPU for demo
    
    # Generate clients for different life stages
    life_stages = [LifeStage.EARLY_CAREER, LifeStage.MID_CAREER, LifeStage.ESTABLISHED]
    clients = []
    
    for stage in life_stages:
        print(f"\n👤 Generating client for {stage.value}...")
        client = engine.generate_synthetic_client(target_life_stage=stage)
        clients.append(client)
        
        print(f"   Name: {client.profile.name}")
        print(f"   Age: {client.profile.age}")
        print(f"   Income: ${client.profile.base_income:,.0f}")
        print(f"   Net Worth: ${client.financial_metrics['net_worth']:,.0f}")
        print(f"   Generated {len(client.lifestyle_events)} lifestyle events")
    
    # Process with mesh engine
    print(f"\n🌐 Processing clients with mesh engine...")
    for client in clients:
        client = engine.process_with_mesh_engine(client, num_scenarios=500, time_horizon_years=3)
        
        if client.risk_analysis:
            print(f"   {client.client_id}: Mesh processed successfully")
            if 'min_cash_by_scenario' in client.risk_analysis:
                min_cash = np.min(client.risk_analysis['min_cash_by_scenario'])
                print(f"      Minimum cash: ${min_cash:,.0f}")
    
    # Analyze financial standing
    print(f"\n💰 Financial Standing Analysis")
    analysis = engine.analyze_financial_standing(clients)
    
    for stage, data in analysis.items():
        print(f"\n📊 {stage.upper()}")
        print(f"   Count: {data['count']} clients")
        metrics = data['metrics']
        print(f"   Avg Net Worth: ${metrics['avg_net_worth']:,.0f}")
        print(f"   Avg Risk Tolerance: {metrics['avg_risk_tolerance']:.2f}")
        print(f"   Avg Discretionary Spending: {metrics['avg_avg_discretionary_spending']:.3f}")
    
    # Export data
    engine.export_synthetic_data(clients, 'data/outputs/analysis_data/synthetic_lifestyle_clients.json')
    print(f"\n💾 Exported synthetic client data")
    
    return engine, clients


if __name__ == "__main__":
    engine, clients = create_demo_engine()
    print("\n✅ Synthetic Lifestyle Engine Demo completed successfully!") 