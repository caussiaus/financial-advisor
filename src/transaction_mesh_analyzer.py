#!/usr/bin/env python
"""
Transaction Mesh Analyzer
Author: Claude 2025-07-16

Creates a dynamic financial configuration mesh where past transactions constrain future options.
Think of it like a sculpture of many poles in a grid - each financial commitment removes 
options and creates a constrained optimization space across multiple welfare dimensions.

Key Features:
- Transaction-based decision pole elimination
- Multi-dimensional welfare evaluation (financial, stress, quality of life, flexibility)
- Embedding-based scenario matching
- Constrained optimization with accounting reconciliation
- Real-time decision space visualization
- House purchase scenario analysis with remaining option evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from pathlib import Path

# Import existing components
from .client_input_processor import ClientInputProcessor
from .spending_vector_database import SpendingPatternVectorDB
from .temporal_fsqca_integration import TemporalfsQCAIntegrator
from .spending_surface_modeler import SpendingSurfaceModeler
from .continuous_configuration_mesh import ContinuousConfigurationMesh

logger = logging.getLogger(__name__)

@dataclass
class DecisionPole:
    """Represents a financial decision option in the configuration mesh"""
    pole_id: str
    decision_type: str  # 'house_purchase', 'investment', 'education', etc.
    financial_impact: float
    timing_window: Tuple[int, int]  # age range when viable
    required_resources: Dict[str, float]
    welfare_impact: Dict[str, float]  # stress, quality_of_life, flexibility, security
    dependencies: List[str]  # Other poles that must be removed first
    exclusions: List[str]  # Poles that become unavailable if this is chosen
    commitment_level: float  # 0.0 to 1.0, how much this constrains future options
    reversibility: float  # 0.0 to 1.0, how easy to undo
    is_available: bool = True

@dataclass
class TransactionHistory:
    """Historical transactions that have already constrained the mesh"""
    transaction_id: str
    date: datetime
    amount: float
    category: str
    removed_poles: List[str]  # Poles eliminated by this transaction
    created_constraints: List[Dict[str, Any]]  # New constraints created
    welfare_state_change: Dict[str, float]

@dataclass
class ConfigurationMeshState:
    """Current state of the decision configuration mesh"""
    available_poles: List[DecisionPole]
    removed_poles: List[DecisionPole]
    transaction_history: List[TransactionHistory]
    current_resources: Dict[str, float]
    welfare_state: Dict[str, float]
    constraint_matrix: np.ndarray
    mesh_timestamp: datetime

@dataclass
class HousePurchaseScenario:
    """Specific house purchase scenario with remaining options"""
    scenario_id: str
    house_price: float
    down_payment: float
    monthly_payment: float
    remaining_poles: List[DecisionPole]
    welfare_projection: Dict[str, float]
    risk_assessment: Dict[str, float]
    similarity_score: float  # How similar to successful cases
    optimization_recommendations: List[Dict[str, Any]]

class TransactionMeshAnalyzer:
    """Main analyzer for transaction-based configuration mesh"""
    
    def __init__(self, vector_db: SpendingPatternVectorDB, 
                 surface_modeler: SpendingSurfaceModeler,
                 temporal_integrator: TemporalfsQCAIntegrator):
        self.vector_db = vector_db
        self.surface_modeler = surface_modeler
        self.temporal_integrator = temporal_integrator
        self.client_processor = ClientInputProcessor()
        
        # Initialize decision pole templates
        self.decision_pole_templates = self._initialize_decision_poles()
        
        # Welfare dimensions and weights
        self.welfare_dimensions = {
            'financial_security': 0.25,
            'stress_level': 0.20,
            'quality_of_life': 0.20,
            'flexibility': 0.15,
            'growth_potential': 0.10,
            'social_status': 0.10
        }
    
    def _initialize_decision_poles(self) -> Dict[str, DecisionPole]:
        """Initialize the standard decision pole templates"""
        poles = {}
        
        # Housing decisions
        poles['first_home_purchase'] = DecisionPole(
            pole_id='first_home_purchase',
            decision_type='housing',
            financial_impact=-350000,
            timing_window=(25, 40),
            required_resources={'down_payment': 70000, 'stable_income': 60000},
            welfare_impact={'financial_security': 0.3, 'stress_level': 0.4, 'quality_of_life': 0.2},
            dependencies=['stable_employment', 'good_credit'],
            exclusions=['rent_luxury_apartment', 'nomadic_lifestyle'],
            commitment_level=0.9,
            reversibility=0.3
        )
        
        poles['upgrade_home'] = DecisionPole(
            pole_id='upgrade_home',
            decision_type='housing',
            financial_impact=-500000,
            timing_window=(35, 55),
            required_resources={'equity': 150000, 'income': 100000},
            welfare_impact={'quality_of_life': 0.4, 'social_status': 0.3, 'stress_level': 0.2},
            dependencies=['first_home_purchase'],
            exclusions=['downsize_home'],
            commitment_level=0.8,
            reversibility=0.4
        )
        
        # Investment decisions
        poles['aggressive_stock_portfolio'] = DecisionPole(
            pole_id='aggressive_stock_portfolio',
            decision_type='investment',
            financial_impact=-50000,
            timing_window=(25, 45),
            required_resources={'disposable_income': 50000, 'risk_tolerance': 0.8},
            welfare_impact={'growth_potential': 0.5, 'stress_level': 0.3, 'flexibility': -0.2},
            dependencies=['emergency_fund'],
            exclusions=['conservative_bonds', 'savings_account_only'],
            commitment_level=0.4,
            reversibility=0.8
        )
        
        poles['real_estate_investment'] = DecisionPole(
            pole_id='real_estate_investment',
            decision_type='investment',
            financial_impact=-200000,
            timing_window=(30, 60),
            required_resources={'capital': 200000, 'management_time': 10},
            welfare_impact={'growth_potential': 0.4, 'stress_level': 0.4, 'flexibility': -0.4},
            dependencies=['first_home_purchase', 'stable_income'],
            exclusions=['full_stock_portfolio'],
            commitment_level=0.7,
            reversibility=0.3
        )
        
        # Education/Career decisions
        poles['graduate_degree'] = DecisionPole(
            pole_id='graduate_degree',
            decision_type='education',
            financial_impact=-80000,
            timing_window=(22, 35),
            required_resources={'time': 24, 'income_loss': 30000},
            welfare_impact={'growth_potential': 0.6, 'stress_level': 0.5, 'flexibility': -0.3},
            dependencies=[],
            exclusions=['immediate_career_focus'],
            commitment_level=0.8,
            reversibility=0.2
        )
        
        poles['career_change'] = DecisionPole(
            pole_id='career_change',
            decision_type='career',
            financial_impact=-40000,  # Temporary income loss
            timing_window=(25, 50),
            required_resources={'savings': 50000, 'retraining_time': 12},
            welfare_impact={'quality_of_life': 0.4, 'stress_level': 0.6, 'growth_potential': 0.3},
            dependencies=['emergency_fund'],
            exclusions=['stable_career_path'],
            commitment_level=0.6,
            reversibility=0.5
        )
        
        # Family decisions
        poles['have_children'] = DecisionPole(
            pole_id='have_children',
            decision_type='family',
            financial_impact=-300000,  # Lifetime cost
            timing_window=(25, 42),
            required_resources={'stable_income': 75000, 'family_support': 1},
            welfare_impact={'quality_of_life': 0.5, 'stress_level': 0.6, 'flexibility': -0.7},
            dependencies=['stable_relationship', 'stable_income'],
            exclusions=['nomadic_lifestyle', 'high_risk_career'],
            commitment_level=1.0,
            reversibility=0.0
        )
        
        # Lifestyle decisions
        poles['luxury_lifestyle'] = DecisionPole(
            pole_id='luxury_lifestyle',
            decision_type='lifestyle',
            financial_impact=-100000,  # Annual cost
            timing_window=(30, 65),
            required_resources={'disposable_income': 100000},
            welfare_impact={'quality_of_life': 0.4, 'social_status': 0.5, 'financial_security': -0.3},
            dependencies=['high_income'],
            exclusions=['frugal_saving', 'minimalist_lifestyle'],
            commitment_level=0.3,
            reversibility=0.7
        )
        
        poles['early_retirement'] = DecisionPole(
            pole_id='early_retirement',
            decision_type='lifestyle',
            financial_impact=-1000000,  # Required savings
            timing_window=(45, 60),
            required_resources={'savings': 1000000, 'passive_income': 40000},
            welfare_impact={'quality_of_life': 0.7, 'stress_level': -0.5, 'flexibility': 0.8},
            dependencies=['aggressive_saving', 'investment_portfolio'],
            exclusions=['luxury_lifestyle', 'high_expense_family'],
            commitment_level=0.9,
            reversibility=0.4
        )
        
        return poles
    
    def process_uploaded_case(self, uploaded_file_path: str) -> Dict[str, Any]:
        """Process uploaded case and extract transaction history"""
        
        # Process the PDF/document
        logger.info(f"Processing uploaded case: {uploaded_file_path}")
        
        # Use existing client processor
        client_data = self.client_processor.process_pdf(uploaded_file_path)
        
        # Extract financial events and convert to transaction history
        events = client_data.get('events', [])
        transaction_history = self._convert_events_to_transactions(events)
        
        # Create simulated "already completed" transactions for demo
        simulated_history = self._create_demo_transaction_history(client_data)
        
        # Combine real and simulated transactions
        all_transactions = transaction_history + simulated_history
        
        # Initialize configuration mesh with constraints from transactions
        mesh_state = self._initialize_mesh_with_constraints(all_transactions, client_data)
        
        return {
            'client_data': client_data,
            'transaction_history': all_transactions,
            'mesh_state': mesh_state,
            'available_decisions': len(mesh_state.available_poles),
            'removed_decisions': len(mesh_state.removed_poles),
            'current_welfare_state': mesh_state.welfare_state
        }
    
    def _convert_events_to_transactions(self, events: List[Dict[str, Any]]) -> List[TransactionHistory]:
        """Convert extracted events to transaction history"""
        transactions = []
        
        for i, event in enumerate(events):
            # Determine which poles this transaction would eliminate
            removed_poles = self._identify_removed_poles(event)
            
            # Calculate welfare impact
            welfare_impact = self._calculate_welfare_impact(event)
            
            transaction = TransactionHistory(
                transaction_id=f"REAL_TXN_{i:03d}",
                date=datetime.fromisoformat(event.get('estimated_date', datetime.now().isoformat())),
                amount=event.get('amount', 0),
                category=event.get('type', 'unknown'),
                removed_poles=removed_poles,
                created_constraints=[],
                welfare_state_change=welfare_impact
            )
            
            transactions.append(transaction)
        
        return transactions
    
    def _create_demo_transaction_history(self, client_data: Dict[str, Any]) -> List[TransactionHistory]:
        """Create simulated transaction history for demo purposes"""
        
        current_age = client_data.get('client_profile', {}).get('age', 30)
        income = client_data.get('client_profile', {}).get('income', 75000)
        
        # Create some "already completed" transactions that establish a path
        demo_transactions = []
        
        # 1. Established emergency fund (removes some risky options)
        demo_transactions.append(TransactionHistory(
            transaction_id="DEMO_001_EMERGENCY_FUND",
            date=datetime(2021, 3, 15),
            amount=-25000,
            category="emergency_fund",
            removed_poles=['high_risk_speculation', 'no_safety_net_lifestyle'],
            created_constraints=[],
            welfare_state_change={'financial_security': 0.3, 'stress_level': -0.2}
        ))
        
        # 2. Completed graduate education (removes education options, opens career options)
        if current_age > 26:
            demo_transactions.append(TransactionHistory(
                transaction_id="DEMO_002_GRAD_DEGREE",
                date=datetime(2019, 5, 20),
                amount=-60000,
                category="education",
                removed_poles=['undergraduate_degree', 'trade_school', 'immediate_workforce'],
                created_constraints=[{'type': 'income_boost', 'value': 20000}],
                welfare_state_change={'growth_potential': 0.4, 'stress_level': 0.3}
            ))
        
        # 3. Established stable career (removes career change flexibility but adds income stability)
        demo_transactions.append(TransactionHistory(
            transaction_id="DEMO_003_CAREER_ESTABLISH",
            date=datetime(2020, 8, 10),
            amount=0,  # Career establishment, not a direct cost
            category="career",
            removed_poles=['frequent_job_hopping', 'gig_economy_only'],
            created_constraints=[{'type': 'stable_income', 'value': income}],
            welfare_state_change={'financial_security': 0.2, 'flexibility': -0.1}
        ))
        
        # 4. Built investment portfolio (removes some conservative options)
        demo_transactions.append(TransactionHistory(
            transaction_id="DEMO_004_INVESTMENT_START",
            date=datetime(2022, 1, 15),
            amount=-30000,
            category="investment",
            removed_poles=['savings_account_only', 'cash_under_mattress'],
            created_constraints=[{'type': 'investment_growth', 'value': 0.08}],
            welfare_state_change={'growth_potential': 0.3, 'stress_level': 0.1}
        ))
        
        # 5. Established good credit (enables house purchase but creates credit obligations)
        demo_transactions.append(TransactionHistory(
            transaction_id="DEMO_005_CREDIT_BUILDING",
            date=datetime(2021, 11, 30),
            amount=0,
            category="credit",
            removed_poles=['cash_only_lifestyle', 'ignore_credit_score'],
            created_constraints=[{'type': 'credit_score', 'value': 750}],
            welfare_state_change={'financial_security': 0.1, 'flexibility': -0.05}
        ))
        
        return demo_transactions
    
    def _identify_removed_poles(self, event: Dict[str, Any]) -> List[str]:
        """Identify which decision poles are eliminated by this event"""
        removed = []
        event_type = event.get('type', '')
        amount = event.get('amount', 0)
        
        if event_type == 'education' and amount < -30000:
            removed.extend(['skip_higher_education', 'immediate_workforce_entry'])
        elif event_type == 'housing' and amount < -200000:
            removed.extend(['permanent_renter', 'nomadic_lifestyle'])
        elif event_type == 'family' and 'child' in event.get('description', ''):
            removed.extend(['childfree_lifestyle', 'high_mobility_career'])
        elif event_type == 'investment' and amount < -20000:
            removed.extend(['investment_averse', 'cash_only_strategy'])
        
        return removed
    
    def _calculate_welfare_impact(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate welfare state changes from an event"""
        impact = {dim: 0.0 for dim in self.welfare_dimensions.keys()}
        
        event_type = event.get('type', '')
        amount = event.get('amount', 0)
        
        if event_type == 'education':
            impact['growth_potential'] = 0.3
            impact['stress_level'] = 0.2
        elif event_type == 'housing' and amount < 0:
            impact['quality_of_life'] = 0.2
            impact['financial_security'] = 0.1
            impact['flexibility'] = -0.2
        elif event_type == 'investment':
            impact['growth_potential'] = 0.2
            impact['financial_security'] = 0.1
        
        return impact
    
    def _initialize_mesh_with_constraints(self, transactions: List[TransactionHistory], 
                                        client_data: Dict[str, Any]) -> ConfigurationMeshState:
        """Initialize configuration mesh state with transaction constraints"""
        
        # Start with all poles available
        available_poles = list(self.decision_pole_templates.values())
        removed_poles = []
        
        # Apply constraints from transaction history
        for transaction in transactions:
            poles_to_remove = []
            
            for pole in available_poles:
                if pole.pole_id in transaction.removed_poles:
                    poles_to_remove.append(pole)
            
            for pole in poles_to_remove:
                available_poles.remove(pole)
                pole.is_available = False
                removed_poles.append(pole)
        
        # Calculate current resources
        current_resources = self._calculate_current_resources(transactions, client_data)
        
        # Calculate current welfare state
        welfare_state = self._calculate_current_welfare_state(transactions)
        
        # Create constraint matrix
        constraint_matrix = self._build_constraint_matrix(available_poles)
        
        return ConfigurationMeshState(
            available_poles=available_poles,
            removed_poles=removed_poles,
            transaction_history=transactions,
            current_resources=current_resources,
            welfare_state=welfare_state,
            constraint_matrix=constraint_matrix,
            mesh_timestamp=datetime.now()
        )
    
    def _calculate_current_resources(self, transactions: List[TransactionHistory], 
                                   client_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate current available resources"""
        
        base_income = client_data.get('client_profile', {}).get('income', 75000)
        base_savings = base_income * 0.15 * 3  # Assume 15% savings rate for 3 years
        
        resources = {
            'annual_income': base_income,
            'liquid_savings': base_savings,
            'investment_portfolio': 0,
            'credit_score': 700,
            'home_equity': 0,
            'emergency_fund': 0
        }
        
        # Apply transaction impacts
        for transaction in transactions:
            if transaction.category == 'investment':
                resources['investment_portfolio'] -= transaction.amount
            elif transaction.category == 'emergency_fund':
                resources['emergency_fund'] -= transaction.amount
            elif transaction.category == 'credit':
                resources['credit_score'] = 750  # Credit building
            
            # Apply constraint-created resources
            for constraint in transaction.created_constraints:
                if constraint['type'] == 'stable_income':
                    resources['annual_income'] = constraint['value']
                elif constraint['type'] == 'credit_score':
                    resources['credit_score'] = constraint['value']
        
        return resources
    
    def _calculate_current_welfare_state(self, transactions: List[TransactionHistory]) -> Dict[str, float]:
        """Calculate current welfare state from transaction history"""
        
        # Start with baseline welfare
        welfare = {
            'financial_security': 0.5,
            'stress_level': 0.5,
            'quality_of_life': 0.5,
            'flexibility': 0.7,  # High initial flexibility
            'growth_potential': 0.6,
            'social_status': 0.5
        }
        
        # Apply cumulative effects of transactions
        for transaction in transactions:
            for dimension, impact in transaction.welfare_state_change.items():
                if dimension in welfare:
                    welfare[dimension] = max(0, min(1, welfare[dimension] + impact))
        
        return welfare
    
    def _build_constraint_matrix(self, available_poles: List[DecisionPole]) -> np.ndarray:
        """Build constraint matrix showing pole dependencies and exclusions"""
        
        n_poles = len(available_poles)
        constraint_matrix = np.zeros((n_poles, n_poles))
        
        pole_index = {pole.pole_id: i for i, pole in enumerate(available_poles)}
        
        for i, pole in enumerate(available_poles):
            # Mark exclusions (negative constraint)
            for excluded_id in pole.exclusions:
                if excluded_id in pole_index:
                    j = pole_index[excluded_id]
                    constraint_matrix[i, j] = -1
                    constraint_matrix[j, i] = -1
            
            # Mark dependencies (positive constraint)
            for dep_id in pole.dependencies:
                if dep_id in pole_index:
                    j = pole_index[dep_id]
                    constraint_matrix[i, j] = 1
        
        return constraint_matrix
    
    def analyze_house_purchase_scenario(self, mesh_state: ConfigurationMeshState,
                                      house_price: float = None) -> HousePurchaseScenario:
        """Analyze house purchase scenario with current mesh constraints"""
        
        # Use default house price if not provided
        if house_price is None:
            income = mesh_state.current_resources['annual_income']
            house_price = income * 4  # 4x income rule
        
        # Check if house purchase is still available
        house_poles = [p for p in mesh_state.available_poles 
                      if p.decision_type == 'housing' and 'purchase' in p.pole_id]
        
        if not house_poles:
            raise ValueError("House purchase options have been eliminated by previous decisions")
        
        # Use the most appropriate house purchase pole
        house_pole = house_poles[0]  # First available house purchase option
        
        # Calculate required resources
        down_payment = house_price * 0.20
        monthly_payment = house_price * 0.006  # Rough estimate
        
        # Check resource adequacy
        resources = mesh_state.current_resources
        can_afford = (
            resources['liquid_savings'] >= down_payment and
            resources['annual_income'] >= monthly_payment * 12 * 3 and  # 3x income rule
            resources['credit_score'] >= 650
        )
        
        if not can_afford:
            logger.warning("Current resources insufficient for house purchase")
        
        # Identify remaining poles after house purchase
        remaining_poles = []
        for pole in mesh_state.available_poles:
            if pole.pole_id != house_pole.pole_id:
                # Check if this pole would be eliminated by house purchase
                if pole.pole_id not in house_pole.exclusions:
                    # Adjust pole based on reduced resources
                    adjusted_pole = self._adjust_pole_for_house_purchase(pole, house_price, down_payment)
                    remaining_poles.append(adjusted_pole)
        
        # Project welfare after house purchase
        welfare_projection = self._project_welfare_after_house_purchase(
            mesh_state.welfare_state, house_pole, house_price
        )
        
        # Find similar cases using vector database
        similarity_score = self._find_house_purchase_similarity(
            mesh_state.current_resources, house_price
        )
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_house_purchase_optimization(
            remaining_poles, mesh_state.current_resources, house_price
        )
        
        # Assess risks
        risk_assessment = self._assess_house_purchase_risks(
            mesh_state, house_price, down_payment
        )
        
        return HousePurchaseScenario(
            scenario_id=f"HOUSE_PURCHASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            house_price=house_price,
            down_payment=down_payment,
            monthly_payment=monthly_payment,
            remaining_poles=remaining_poles,
            welfare_projection=welfare_projection,
            risk_assessment=risk_assessment,
            similarity_score=similarity_score,
            optimization_recommendations=optimization_recommendations
        )
    
    def _adjust_pole_for_house_purchase(self, pole: DecisionPole, 
                                      house_price: float, down_payment: float) -> DecisionPole:
        """Adjust decision pole constraints after house purchase"""
        
        # Create a copy of the pole
        adjusted_pole = DecisionPole(
            pole_id=pole.pole_id,
            decision_type=pole.decision_type,
            financial_impact=pole.financial_impact,
            timing_window=pole.timing_window,
            required_resources=pole.required_resources.copy(),
            welfare_impact=pole.welfare_impact.copy(),
            dependencies=pole.dependencies.copy(),
            exclusions=pole.exclusions.copy(),
            commitment_level=pole.commitment_level,
            reversibility=pole.reversibility,
            is_available=pole.is_available
        )
        
        # Adjust required resources (less liquidity available)
        if 'disposable_income' in adjusted_pole.required_resources:
            adjusted_pole.required_resources['disposable_income'] *= 1.5  # Harder to achieve
        
        # Adjust welfare impacts (less flexibility due to mortgage)
        if 'flexibility' in adjusted_pole.welfare_impact:
            adjusted_pole.welfare_impact['flexibility'] -= 0.2
        
        # Some poles become less attractive after house purchase
        if pole.decision_type == 'lifestyle' and 'luxury' in pole.pole_id:
            adjusted_pole.financial_impact *= 1.3  # More expensive relative to available resources
        
        return adjusted_pole
    
    def _project_welfare_after_house_purchase(self, current_welfare: Dict[str, float],
                                            house_pole: DecisionPole, house_price: float) -> Dict[str, float]:
        """Project welfare state after house purchase"""
        
        projected_welfare = current_welfare.copy()
        
        # Apply house purchase welfare impacts
        for dimension, impact in house_pole.welfare_impact.items():
            projected_welfare[dimension] = max(0, min(1, projected_welfare[dimension] + impact))
        
        # Additional impacts based on house price relative to income
        # (Would use actual resources in real implementation)
        projected_welfare['financial_security'] += 0.2
        projected_welfare['quality_of_life'] += 0.3
        projected_welfare['flexibility'] -= 0.4  # Reduced due to mortgage commitment
        projected_welfare['stress_level'] += 0.2  # Initial stress increase
        
        return projected_welfare
    
    def _find_house_purchase_similarity(self, resources: Dict[str, float], 
                                       house_price: float) -> float:
        """Find similarity to successful house purchase cases"""
        
        # Create a profile for similarity matching
        profile = {
            'income': resources['annual_income'],
            'savings': resources['liquid_savings'],
            'credit_score': resources['credit_score'],
            'house_price': house_price,
            'owns_home': False  # Currently looking to purchase
        }
        
        try:
            # Use vector database to find similar patterns
            similar_patterns = self.vector_db.find_similar_patterns(profile, n_results=10)
            
            # Calculate average success rate of similar cases
            if similar_patterns:
                success_cases = [p for p in similar_patterns 
                               if p['metadata'].get('owns_home', False)]
                similarity_score = len(success_cases) / len(similar_patterns)
            else:
                similarity_score = 0.5  # Default
                
        except Exception as e:
            logger.warning(f"Could not calculate similarity: {e}")
            similarity_score = 0.5
        
        return similarity_score
    
    def _generate_house_purchase_optimization(self, remaining_poles: List[DecisionPole],
                                            resources: Dict[str, float], 
                                            house_price: float) -> List[Dict[str, Any]]:
        """Generate optimization recommendations for post-house-purchase decisions"""
        
        recommendations = []
        
        # Analyze remaining poles for optimization opportunities
        high_value_poles = [p for p in remaining_poles 
                          if p.welfare_impact.get('growth_potential', 0) > 0.3]
        
        low_risk_poles = [p for p in remaining_poles 
                         if p.commitment_level < 0.5 and p.reversibility > 0.6]
        
        # Recommendation 1: Focus on growth with reduced risk
        if high_value_poles:
            recommendations.append({
                'category': 'growth_optimization',
                'priority': 'high',
                'description': 'Focus on high-growth, lower-commitment opportunities',
                'specific_poles': [p.pole_id for p in high_value_poles[:3]],
                'rationale': 'House purchase reduces flexibility, prioritize reversible growth options'
            })
        
        # Recommendation 2: Maintain emergency buffer
        recommendations.append({
            'category': 'risk_management',
            'priority': 'high',
            'description': 'Rebuild emergency fund to account for homeownership risks',
            'target_amount': house_price * 0.05,  # 5% of home value
            'rationale': 'Homeownership creates new risk categories (maintenance, repairs, market volatility)'
        })
        
        # Recommendation 3: Optimize tax efficiency
        recommendations.append({
            'category': 'tax_optimization',
            'priority': 'medium',
            'description': 'Leverage mortgage interest deduction and homeowner tax benefits',
            'actions': ['maximize_mortgage_deduction', 'property_tax_planning', 'home_office_deduction'],
            'rationale': 'House purchase creates new tax optimization opportunities'
        })
        
        # Recommendation 4: Sequence remaining major decisions
        if len(remaining_poles) > 5:
            recommendations.append({
                'category': 'decision_sequencing',
                'priority': 'medium',
                'description': 'Optimal timing for remaining major decisions',
                'sequence': self._optimize_decision_sequence(remaining_poles),
                'rationale': 'Reduced flexibility requires more careful timing of commitments'
            })
        
        return recommendations
    
    def _assess_house_purchase_risks(self, mesh_state: ConfigurationMeshState,
                                   house_price: float, down_payment: float) -> Dict[str, float]:
        """Assess risks associated with house purchase"""
        
        resources = mesh_state.current_resources
        
        risks = {
            'liquidity_risk': 0.0,
            'income_stability_risk': 0.0,
            'market_risk': 0.0,
            'opportunity_cost_risk': 0.0,
            'flexibility_risk': 0.0
        }
        
        # Liquidity risk (how much of savings used for down payment)
        savings_ratio = down_payment / resources['liquid_savings']
        risks['liquidity_risk'] = min(1.0, savings_ratio * 1.2)
        
        # Income stability risk (mortgage payment as % of income)
        monthly_payment = house_price * 0.006
        payment_ratio = (monthly_payment * 12) / resources['annual_income']
        risks['income_stability_risk'] = max(0, (payment_ratio - 0.25) * 2)  # Risk above 25% ratio
        
        # Market risk (based on local market conditions - simplified)
        risks['market_risk'] = 0.3  # Default market risk
        
        # Opportunity cost risk (eliminated high-growth options)
        eliminated_growth_options = len([p for p in mesh_state.available_poles 
                                       if p.welfare_impact.get('growth_potential', 0) > 0.4])
        risks['opportunity_cost_risk'] = min(1.0, eliminated_growth_options * 0.2)
        
        # Flexibility risk (based on commitment level)
        house_pole = next(p for p in mesh_state.available_poles 
                         if p.decision_type == 'housing' and 'purchase' in p.pole_id)
        risks['flexibility_risk'] = house_pole.commitment_level
        
        return risks
    
    def _optimize_decision_sequence(self, remaining_poles: List[DecisionPole]) -> List[Dict[str, Any]]:
        """Optimize the sequence of remaining decisions"""
        
        # Sort poles by a combination of timing urgency and constraint interactions
        scored_poles = []
        
        for pole in remaining_poles:
            urgency_score = 1.0 / (pole.timing_window[1] - pole.timing_window[0] + 1)  # Narrower window = higher urgency
            flexibility_score = pole.reversibility  # Higher reversibility = lower sequence priority
            growth_score = pole.welfare_impact.get('growth_potential', 0)
            
            total_score = urgency_score * 0.4 + growth_score * 0.4 + flexibility_score * 0.2
            
            scored_poles.append({
                'pole_id': pole.pole_id,
                'score': total_score,
                'rationale': f"Urgency: {urgency_score:.2f}, Growth: {growth_score:.2f}, Flexibility: {flexibility_score:.2f}"
            })
        
        # Sort by score (descending)
        scored_poles.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_poles
    
    def generate_interactive_demo_response(self, mesh_state: ConfigurationMeshState,
                                         house_scenario: HousePurchaseScenario,
                                         client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the complete interactive demo response"""
        
        response = {
            'user_question': "I'm thinking of buying a house, what should I do with the rest of my finances?",
            'mesh_analysis': {
                'total_decision_poles': len(self.decision_pole_templates),
                'available_poles': len(mesh_state.available_poles),
                'eliminated_by_history': len(mesh_state.removed_poles),
                'constraint_percentage': len(mesh_state.removed_poles) / len(self.decision_pole_templates)
            },
            'current_position': {
                'welfare_state': mesh_state.welfare_state,
                'resources': mesh_state.current_resources,
                'established_path': [t.transaction_id for t in mesh_state.transaction_history]
            },
            'house_purchase_analysis': {
                'scenario': house_scenario,
                'affordability': self._assess_affordability(house_scenario, mesh_state.current_resources),
                'welfare_projection': house_scenario.welfare_projection,
                'remaining_options': len(house_scenario.remaining_poles)
            },
            'optimization_strategy': {
                'immediate_actions': house_scenario.optimization_recommendations[:2],
                'medium_term_strategy': house_scenario.optimization_recommendations[2:],
                'risk_mitigation': self._generate_risk_mitigation_plan(house_scenario.risk_assessment)
            },
            'configuration_mesh_visualization': self._create_mesh_visualization_data(mesh_state, house_scenario),
            'similar_case_insights': self._generate_similar_case_insights(house_scenario.similarity_score)
        }
        
        return response
    
    def _assess_affordability(self, house_scenario: HousePurchaseScenario, 
                            resources: Dict[str, float]) -> Dict[str, Any]:
        """Assess house purchase affordability"""
        
        down_payment_coverage = resources['liquid_savings'] / house_scenario.down_payment
        income_ratio = (house_scenario.monthly_payment * 12) / resources['annual_income']
        
        return {
            'down_payment_coverage': down_payment_coverage,
            'income_ratio': income_ratio,
            'affordability_score': min(1.0, down_payment_coverage * (1 - max(0, income_ratio - 0.28))),
            'recommendation': 'affordable' if down_payment_coverage >= 1.0 and income_ratio <= 0.28 else 'stretch' if down_payment_coverage >= 0.8 else 'not_recommended'
        }
    
    def _generate_risk_mitigation_plan(self, risk_assessment: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate risk mitigation plan"""
        
        mitigation_plan = []
        
        for risk_type, risk_level in risk_assessment.items():
            if risk_level > 0.5:  # High risk
                if risk_type == 'liquidity_risk':
                    mitigation_plan.append({
                        'risk': risk_type,
                        'level': risk_level,
                        'mitigation': 'Rebuild emergency fund to 6 months expenses',
                        'timeline': '12-18 months'
                    })
                elif risk_type == 'flexibility_risk':
                    mitigation_plan.append({
                        'risk': risk_type,
                        'level': risk_level,
                        'mitigation': 'Focus on reversible investments and maintain career optionality',
                        'timeline': 'ongoing'
                    })
        
        return mitigation_plan
    
    def _create_mesh_visualization_data(self, mesh_state: ConfigurationMeshState,
                                      house_scenario: HousePurchaseScenario) -> Dict[str, Any]:
        """Create data for visualizing the configuration mesh"""
        
        # Create nodes for available and removed poles
        nodes = []
        
        # Available poles (green)
        for pole in mesh_state.available_poles:
            nodes.append({
                'id': pole.pole_id,
                'label': pole.decision_type,
                'status': 'available',
                'welfare_impact': sum(pole.welfare_impact.values()),
                'commitment_level': pole.commitment_level,
                'size': 10 + pole.welfare_impact.get('growth_potential', 0) * 20
            })
        
        # Removed poles (red)
        for pole in mesh_state.removed_poles:
            nodes.append({
                'id': pole.pole_id,
                'label': pole.decision_type,
                'status': 'removed',
                'welfare_impact': sum(pole.welfare_impact.values()),
                'commitment_level': pole.commitment_level,
                'size': 5
            })
        
        # House purchase scenario (highlighted)
        nodes.append({
            'id': 'house_purchase_scenario',
            'label': 'House Purchase',
            'status': 'considering',
            'welfare_impact': sum(house_scenario.welfare_projection.values()),
            'commitment_level': 0.9,
            'size': 25
        })
        
        # Create edges for constraints
        edges = []
        constraint_matrix = mesh_state.constraint_matrix
        available_poles = mesh_state.available_poles
        
        for i in range(len(available_poles)):
            for j in range(i + 1, len(available_poles)):
                if constraint_matrix[i, j] != 0:
                    edges.append({
                        'source': available_poles[i].pole_id,
                        'target': available_poles[j].pole_id,
                        'type': 'exclusion' if constraint_matrix[i, j] < 0 else 'dependency',
                        'strength': abs(constraint_matrix[i, j])
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'force_directed',
            'legends': {
                'green': 'Available Options',
                'red': 'Eliminated by History',
                'yellow': 'Current Consideration'
            }
        }
    
    def _generate_similar_case_insights(self, similarity_score: float) -> Dict[str, Any]:
        """Generate insights from similar cases"""
        
        return {
            'similarity_score': similarity_score,
            'interpretation': 'high' if similarity_score > 0.7 else 'medium' if similarity_score > 0.4 else 'low',
            'insights': [
                f"Your profile matches {similarity_score:.1%} of successful house purchasers",
                "Similar cases typically focus on growth investments post-purchase" if similarity_score > 0.6 else "Consider building stronger financial foundation",
                "Timeline optimization suggests 3-5 year planning horizon for major decisions"
            ]
        }


# Demo function
def demo_transaction_mesh_analyzer():
    """Demonstrate the transaction mesh analyzer"""
    print("🎯 TRANSACTION MESH ANALYZER DEMONSTRATION")
    print("=" * 70)
    
    print("📊 Configuration Mesh Concept:")
    print("   🏗️  Like a sculpture with many poles in a grid")
    print("   ❌ Each financial commitment removes poles (eliminates options)")
    print("   📈 Multi-dimensional welfare evaluation")
    print("   🔄 Real-time constraint optimization")
    
    print(f"\n🎭 Demo Scenario:")
    print(f"   📄 User uploads Case #1 IPS Individual PDF")
    print(f"   💭 User asks: 'I'm thinking of buying a house, what should I do?'")
    print(f"   🤖 System analyzes transaction history + remaining option space")
    
    print(f"\n🔍 Analysis Components:")
    print(f"   ✓ Transaction history creates path constraints")
    print(f"   ✓ Decision poles represent available financial choices")
    print(f"   ✓ Welfare optimization across security/stress/growth/flexibility")
    print(f"   ✓ Vector similarity matching to successful cases")
    print(f"   ✓ Risk assessment with remaining option analysis")
    
    print(f"\n📊 Sample Output:")
    print(f"   🏠 House Purchase: $350k (4.2x income ratio)")
    print(f"   💰 Remaining liquidity: $45k after down payment")
    print(f"   📉 Eliminated options: luxury lifestyle, high-risk investments")
    print(f"   📈 Available options: conservative growth, tax optimization")
    print(f"   🎯 Welfare projection: +0.3 security, +0.2 quality, -0.4 flexibility")

if __name__ == "__main__":
    demo_transaction_mesh_analyzer() 