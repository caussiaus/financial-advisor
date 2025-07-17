"""
JSON to Vector Data Converter Engine

This module converts JSON inputs into vector data format for synthetic client data generation.
It incorporates probability modeling for lifestyle events based on age and life stage,
creating a surface of discretionary spending homogeneously sorted.

Key Features:
- Converts JSON client profiles to vectorized event data
- Models event probabilities based on age and life stage
- Generates synthetic lifestyle events with realistic timing
- Creates vectorized cash flow projections
- Integrates with existing mesh engines for financial modeling
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import random
from enum import Enum
import logging

# Import existing components
from src.core.time_uncertainty_mesh import SeedEvent, EventVector, TimeUncertaintyMeshEngine
from src.synthetic_data_generator import PersonProfile


class LifeStage(Enum):
    """Life stages for probability modeling"""
    EARLY_CAREER = "early_career"  # 22-30
    MID_CAREER = "mid_career"      # 31-45
    ESTABLISHED = "established"     # 46-60
    PRE_RETIREMENT = "pre_retirement"  # 61-67
    RETIREMENT = "retirement"       # 68+


class EventCategory(Enum):
    """Categories of lifestyle events"""
    EDUCATION = "education"
    CAREER = "career"
    FAMILY = "family"
    HOUSING = "housing"
    HEALTH = "health"
    RETIREMENT = "retirement"
    INVESTMENT = "investment"
    DEBT = "debt"
    INSURANCE = "insurance"
    CHARITY = "charity"


@dataclass
class LifestyleEvent:
    """Represents a lifestyle event with probability modeling"""
    event_id: str
    category: EventCategory
    description: str
    base_amount: float
    base_probability: float
    age_dependency: Dict[int, float]  # Age -> probability multiplier
    life_stage_dependency: Dict[LifeStage, float]  # Life stage -> probability multiplier
    timing_volatility: float = 0.2
    amount_volatility: float = 0.15
    dependencies: List[str] = field(default_factory=list)
    cash_flow_impact: str = "negative"  # positive, negative, neutral


@dataclass
class ClientVectorProfile:
    """Vectorized client profile for mesh processing"""
    client_id: str
    age: int
    life_stage: LifeStage
    base_income: float
    current_assets: np.ndarray  # Vectorized asset positions
    current_debts: np.ndarray   # Vectorized debt positions
    risk_tolerance: float  # 0-1 scale
    event_probabilities: np.ndarray  # Probability of each event type
    cash_flow_vector: np.ndarray  # Monthly cash flow projections
    discretionary_spending_surface: np.ndarray  # 2D surface of discretionary spending


class JSONToVectorConverter:
    """
    Converts JSON inputs into vector data format for synthetic client generation
    with probability modeling for lifestyle events
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.time_uncertainty_engine = TimeUncertaintyMeshEngine(use_gpu=use_gpu)
        
        # Event probability matrices by age and life stage
        self.event_probability_matrix = self._initialize_event_probability_matrix()
        
        # Life stage transition probabilities
        self.life_stage_transitions = self._initialize_life_stage_transitions()
        
        # Discretionary spending patterns by age and income
        self.discretionary_spending_patterns = self._initialize_discretionary_patterns()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the converter"""
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
    
    def _initialize_event_probability_matrix(self) -> Dict[EventCategory, Dict]:
        """Initialize probability matrix for different event categories by age"""
        matrix = {}
        
        # Education events - higher probability in early career
        matrix[EventCategory.EDUCATION] = {
            'base_probability': 0.3,
            'age_multipliers': {
                22: 2.0, 25: 1.8, 30: 1.2, 35: 0.8, 40: 0.4, 45: 0.2, 50: 0.1
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 2.0,
                LifeStage.MID_CAREER: 1.0,
                LifeStage.ESTABLISHED: 0.3,
                LifeStage.PRE_RETIREMENT: 0.1,
                LifeStage.RETIREMENT: 0.05
            }
        }
        
        # Career events - peak in mid-career
        matrix[EventCategory.CAREER] = {
            'base_probability': 0.4,
            'age_multipliers': {
                22: 0.8, 25: 1.2, 30: 1.5, 35: 1.8, 40: 1.6, 45: 1.2, 50: 0.8, 55: 0.4
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 1.2,
                LifeStage.MID_CAREER: 1.8,
                LifeStage.ESTABLISHED: 1.0,
                LifeStage.PRE_RETIREMENT: 0.6,
                LifeStage.RETIREMENT: 0.2
            }
        }
        
        # Family events - distributed across career stages
        matrix[EventCategory.FAMILY] = {
            'base_probability': 0.25,
            'age_multipliers': {
                22: 0.6, 25: 1.0, 30: 1.4, 35: 1.2, 40: 0.8, 45: 0.4, 50: 0.2
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 1.0,
                LifeStage.MID_CAREER: 1.4,
                LifeStage.ESTABLISHED: 0.8,
                LifeStage.PRE_RETIREMENT: 0.3,
                LifeStage.RETIREMENT: 0.1
            }
        }
        
        # Housing events - higher in established career
        matrix[EventCategory.HOUSING] = {
            'base_probability': 0.2,
            'age_multipliers': {
                22: 0.4, 25: 0.8, 30: 1.2, 35: 1.6, 40: 1.4, 45: 1.0, 50: 0.6
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 0.6,
                LifeStage.MID_CAREER: 1.2,
                LifeStage.ESTABLISHED: 1.6,
                LifeStage.PRE_RETIREMENT: 1.0,
                LifeStage.RETIREMENT: 0.4
            }
        }
        
        # Health events - increase with age
        matrix[EventCategory.HEALTH] = {
            'base_probability': 0.15,
            'age_multipliers': {
                22: 0.3, 25: 0.4, 30: 0.6, 35: 0.8, 40: 1.0, 45: 1.3, 50: 1.6, 55: 2.0, 60: 2.5
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 0.4,
                LifeStage.MID_CAREER: 0.8,
                LifeStage.ESTABLISHED: 1.2,
                LifeStage.PRE_RETIREMENT: 1.8,
                LifeStage.RETIREMENT: 2.2
            }
        }
        
        # Retirement events - peak in pre-retirement
        matrix[EventCategory.RETIREMENT] = {
            'base_probability': 0.1,
            'age_multipliers': {
                22: 0.1, 25: 0.1, 30: 0.2, 35: 0.3, 40: 0.5, 45: 0.8, 50: 1.2, 55: 1.8, 60: 2.5
            },
            'life_stage_multipliers': {
                LifeStage.EARLY_CAREER: 0.1,
                LifeStage.MID_CAREER: 0.3,
                LifeStage.ESTABLISHED: 0.8,
                LifeStage.PRE_RETIREMENT: 2.0,
                LifeStage.RETIREMENT: 1.5
            }
        }
        
        return matrix
    
    def _initialize_life_stage_transitions(self) -> Dict[LifeStage, Dict]:
        """Initialize transition probabilities between life stages"""
        transitions = {
            LifeStage.EARLY_CAREER: {
                LifeStage.MID_CAREER: 0.15,  # 15% chance per year to transition
                LifeStage.ESTABLISHED: 0.05,
                LifeStage.PRE_RETIREMENT: 0.01,
                LifeStage.RETIREMENT: 0.001
            },
            LifeStage.MID_CAREER: {
                LifeStage.ESTABLISHED: 0.12,
                LifeStage.PRE_RETIREMENT: 0.03,
                LifeStage.RETIREMENT: 0.005
            },
            LifeStage.ESTABLISHED: {
                LifeStage.PRE_RETIREMENT: 0.10,
                LifeStage.RETIREMENT: 0.02
            },
            LifeStage.PRE_RETIREMENT: {
                LifeStage.RETIREMENT: 0.20
            },
            LifeStage.RETIREMENT: {
                # No transitions from retirement
            }
        }
        return transitions
    
    def _initialize_discretionary_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize discretionary spending patterns by age and income level"""
        patterns = {}
        
        # Age ranges: 22-30, 31-45, 46-60, 61-67, 68+
        age_ranges = [(22, 30), (31, 45), (46, 60), (61, 67), (68, 85)]
        
        # Income levels: low, medium, high
        income_levels = ['low', 'medium', 'high']
        
        for age_range in age_ranges:
            for income_level in income_levels:
                key = f"{age_range[0]}-{age_range[1]}_{income_level}"
                
                # Create 2D surface: months x spending categories
                # Categories: entertainment, travel, luxury, hobbies, dining, shopping
                categories = 6
                months = 120  # 10 years
                
                # Base discretionary spending as percentage of income
                base_percentages = {
                    'low': 0.08,    # 8% of income
                    'medium': 0.12,  # 12% of income
                    'high': 0.15     # 15% of income
                }
                
                base_percentage = base_percentages[income_level]
                
                # Age-based adjustments
                age_factor = 1.0
                if age_range[0] <= 30:  # Early career - higher discretionary
                    age_factor = 1.2
                elif age_range[0] >= 60:  # Pre-retirement - more conservative
                    age_factor = 0.8
                
                # Create surface with some randomness and trends
                surface = np.random.normal(base_percentage * age_factor, 0.02, (months, categories))
                
                # Add seasonal patterns
                for month in range(months):
                    seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
                    surface[month, :] *= seasonal_factor
                
                # Ensure positive values
                surface = np.maximum(surface, 0.01)
                
                patterns[key] = surface
        
        return patterns
    
    def convert_json_to_vector_profile(self, json_input: Dict) -> ClientVectorProfile:
        """
        Convert JSON input to vectorized client profile
        
        Args:
            json_input: JSON containing client data
            
        Returns:
            ClientVectorProfile with vectorized data
        """
        self.logger.info(f"Converting JSON to vector profile for client: {json_input.get('client_id', 'unknown')}")
        
        # Extract basic profile data
        client_id = json_input.get('client_id', f"client_{random.randint(1000, 9999)}")
        age = json_input.get('age', 35)
        income = json_input.get('income', 75000)
        
        # Determine life stage
        life_stage = self._determine_life_stage(age)
        
        # Convert assets to vector
        assets_data = json_input.get('current_assets', {})
        current_assets = self._convert_assets_to_vector(assets_data)
        
        # Convert debts to vector
        debts_data = json_input.get('debts', {})
        current_debts = self._convert_debts_to_vector(debts_data)
        
        # Calculate risk tolerance
        risk_tolerance = self._calculate_risk_tolerance(json_input)
        
        # Generate event probabilities
        event_probabilities = self._generate_event_probabilities(age, life_stage, income)
        
        # Generate cash flow vector
        cash_flow_vector = self._generate_cash_flow_vector(income, age, life_stage)
        
        # Generate discretionary spending surface
        discretionary_surface = self._generate_discretionary_surface(age, income)
        
        return ClientVectorProfile(
            client_id=client_id,
            age=age,
            life_stage=life_stage,
            base_income=income,
            current_assets=current_assets,
            current_debts=current_debts,
            risk_tolerance=risk_tolerance,
            event_probabilities=event_probabilities,
            cash_flow_vector=cash_flow_vector,
            discretionary_spending_surface=discretionary_surface
        )
    
    def _determine_life_stage(self, age: int) -> LifeStage:
        """Determine life stage based on age"""
        if age <= 30:
            return LifeStage.EARLY_CAREER
        elif age <= 45:
            return LifeStage.MID_CAREER
        elif age <= 60:
            return LifeStage.ESTABLISHED
        elif age <= 67:
            return LifeStage.PRE_RETIREMENT
        else:
            return LifeStage.RETIREMENT
    
    def _convert_assets_to_vector(self, assets_data: Dict) -> np.ndarray:
        """Convert assets dictionary to vector"""
        # Standard asset categories
        asset_categories = ['cash', 'savings', 'investments', 'retirement', 'real_estate', 'other_assets']
        
        vector = np.zeros(len(asset_categories), dtype=np.float32)
        
        for i, category in enumerate(asset_categories):
            vector[i] = assets_data.get(category, 0.0)
        
        return vector
    
    def _convert_debts_to_vector(self, debts_data: Dict) -> np.ndarray:
        """Convert debts dictionary to vector"""
        # Standard debt categories
        debt_categories = ['credit_cards', 'student_loans', 'mortgage', 'auto_loans', 'personal_loans', 'other_debts']
        
        vector = np.zeros(len(debt_categories), dtype=np.float32)
        
        for i, category in enumerate(debt_categories):
            vector[i] = debts_data.get(category, 0.0)
        
        return vector
    
    def _calculate_risk_tolerance(self, json_input: Dict) -> float:
        """Calculate risk tolerance score (0-1) from JSON input"""
        # Extract risk-related factors
        age = json_input.get('age', 35)
        income = json_input.get('income', 75000)
        risk_profile = json_input.get('risk_tolerance', 'moderate')
        
        # Base risk tolerance by profile
        base_risk = {
            'conservative': 0.2,
            'moderate': 0.5,
            'aggressive': 0.8,
            'very_aggressive': 0.9
        }.get(risk_profile.lower(), 0.5)
        
        # Age adjustment (younger = higher risk tolerance)
        age_factor = max(0.3, 1.0 - (age - 25) / 50)
        
        # Income adjustment (higher income = higher risk tolerance)
        income_factor = min(1.2, income / 100000)
        
        # Calculate final risk tolerance
        risk_tolerance = base_risk * age_factor * income_factor
        
        return np.clip(risk_tolerance, 0.0, 1.0)
    
    def _generate_event_probabilities(self, age: int, life_stage: LifeStage, income: float) -> np.ndarray:
        """Generate probability vector for different event categories"""
        event_categories = list(EventCategory)
        probabilities = np.zeros(len(event_categories), dtype=np.float32)
        
        for i, category in enumerate(event_categories):
            if category in self.event_probability_matrix:
                matrix = self.event_probability_matrix[category]
                
                # Base probability
                base_prob = matrix['base_probability']
                
                # Age multiplier
                age_mult = 1.0
                for age_threshold, multiplier in matrix['age_multipliers'].items():
                    if age >= age_threshold:
                        age_mult = multiplier
                        break
                
                # Life stage multiplier
                life_stage_mult = matrix['life_stage_multipliers'].get(life_stage, 1.0)
                
                # Income adjustment (higher income = more events)
                income_factor = min(1.5, income / 75000)
                
                # Calculate final probability
                final_prob = base_prob * age_mult * life_stage_mult * income_factor
                probabilities[i] = np.clip(final_prob, 0.0, 1.0)
        
        return probabilities
    
    def _generate_cash_flow_vector(self, income: float, age: int, life_stage: LifeStage) -> np.ndarray:
        """Generate monthly cash flow vector for 10 years"""
        months = 120  # 10 years
        cash_flow = np.zeros(months, dtype=np.float32)
        
        # Base monthly income
        monthly_income = income / 12
        
        # Estimate monthly expenses based on age and life stage
        expense_ratio = {
            LifeStage.EARLY_CAREER: 0.7,
            LifeStage.MID_CAREER: 0.65,
            LifeStage.ESTABLISHED: 0.6,
            LifeStage.PRE_RETIREMENT: 0.55,
            LifeStage.RETIREMENT: 0.5
        }.get(life_stage, 0.65)
        
        monthly_expenses = monthly_income * expense_ratio
        
        # Generate cash flow with some variability
        for month in range(months):
            # Add some randomness to income and expenses
            income_variation = np.random.normal(1.0, 0.05)
            expense_variation = np.random.normal(1.0, 0.08)
            
            # Seasonal adjustments
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
            
            net_cash_flow = (monthly_income * income_variation * seasonal_factor - 
                           monthly_expenses * expense_variation)
            
            cash_flow[month] = net_cash_flow
        
        return cash_flow
    
    def _generate_discretionary_surface(self, age: int, income: float) -> np.ndarray:
        """Generate discretionary spending surface"""
        # Determine age range and income level
        age_range = None
        for start, end in [(22, 30), (31, 45), (46, 60), (61, 67), (68, 85)]:
            if start <= age <= end:
                age_range = (start, end)
                break
        
        if age_range is None:
            age_range = (31, 45)  # Default to mid-career
        
        # Determine income level
        if income < 50000:
            income_level = 'low'
        elif income < 100000:
            income_level = 'medium'
        else:
            income_level = 'high'
        
        # Get pattern key
        pattern_key = f"{age_range[0]}-{age_range[1]}_{income_level}"
        
        if pattern_key in self.discretionary_spending_patterns:
            return self.discretionary_spending_patterns[pattern_key].copy()
        else:
            # Fallback pattern
            return np.random.normal(0.1, 0.02, (120, 6))
    
    def generate_synthetic_events(self, vector_profile: ClientVectorProfile, 
                                num_events: int = 10) -> List[LifestyleEvent]:
        """
        Generate synthetic lifestyle events based on vector profile
        
        Args:
            vector_profile: Vectorized client profile
            num_events: Number of events to generate
            
        Returns:
            List of LifestyleEvent objects
        """
        self.logger.info(f"Generating {num_events} synthetic events for client {vector_profile.client_id}")
        
        events = []
        event_categories = list(EventCategory)
        
        for i in range(num_events):
            # Select event category based on probabilities
            category_probs = vector_profile.event_probabilities
            selected_category = np.random.choice(event_categories, p=category_probs/sum(category_probs))
            
            # Generate event details
            event = self._create_lifestyle_event(
                event_id=f"event_{vector_profile.client_id}_{i}",
                category=selected_category,
                age=vector_profile.age,
                income=vector_profile.base_income,
                life_stage=vector_profile.life_stage
            )
            
            events.append(event)
        
        return events
    
    def _create_lifestyle_event(self, event_id: str, category: EventCategory, 
                              age: int, income: float, life_stage: LifeStage) -> LifestyleEvent:
        """Create a specific lifestyle event"""
        
        # Event templates by category
        event_templates = {
            EventCategory.EDUCATION: [
                "Graduate school enrollment",
                "Professional certification",
                "Skills training program",
                "Executive education",
                "Online course completion"
            ],
            EventCategory.CAREER: [
                "Job promotion",
                "Career change",
                "Starting a business",
                "Consulting opportunity",
                "Industry conference attendance"
            ],
            EventCategory.FAMILY: [
                "Getting married",
                "Having a child",
                "Caring for aging parents",
                "Family vacation",
                "Child's education planning"
            ],
            EventCategory.HOUSING: [
                "Buying a new home",
                "Home renovation",
                "Moving to a new city",
                "Downsizing",
                "Investment property purchase"
            ],
            EventCategory.HEALTH: [
                "Medical procedure",
                "Health insurance change",
                "Fitness program",
                "Mental health support",
                "Preventive care"
            ],
            EventCategory.RETIREMENT: [
                "Retirement planning consultation",
                "Pension optimization",
                "Social security planning",
                "Retirement community research",
                "Legacy planning"
            ]
        }
        
        # Select description
        descriptions = event_templates.get(category, ["General life event"])
        description = random.choice(descriptions)
        
        # Calculate amount based on category and income
        amount_multipliers = {
            EventCategory.EDUCATION: (0.1, 0.3),
            EventCategory.CAREER: (0.05, 0.15),
            EventCategory.FAMILY: (0.2, 0.5),
            EventCategory.HOUSING: (1.0, 3.0),
            EventCategory.HEALTH: (0.05, 0.2),
            EventCategory.RETIREMENT: (0.1, 0.3)
        }
        
        min_mult, max_mult = amount_multipliers.get(category, (0.1, 0.3))
        base_amount = income * random.uniform(min_mult, max_mult)
        
        # Calculate probability based on age and life stage
        matrix = self.event_probability_matrix.get(category, {})
        base_prob = matrix.get('base_probability', 0.2)
        
        age_mult = 1.0
        for age_threshold, multiplier in matrix.get('age_multipliers', {}).items():
            if age >= age_threshold:
                age_mult = multiplier
                break
        
        life_stage_mult = matrix.get('life_stage_multipliers', {}).get(life_stage, 1.0)
        probability = base_prob * age_mult * life_stage_mult
        
        return LifestyleEvent(
            event_id=event_id,
            category=category,
            description=description,
            base_amount=base_amount,
            base_probability=probability,
            age_dependency=self.event_probability_matrix[category]['age_multipliers'],
            life_stage_dependency=self.event_probability_matrix[category]['life_stage_multipliers'],
            timing_volatility=0.2,
            amount_volatility=0.15,
            cash_flow_impact="negative" if category in [EventCategory.EDUCATION, EventCategory.HOUSING, EventCategory.HEALTH] else "positive"
        )
    
    def convert_events_to_seed_events(self, lifestyle_events: List[LifestyleEvent]) -> List[SeedEvent]:
        """Convert lifestyle events to seed events for mesh processing"""
        seed_events = []
        
        for event in lifestyle_events:
            # Estimate timing based on age and event type
            estimated_date = self._estimate_event_timing(event)
            
            seed_event = SeedEvent(
                event_id=event.event_id,
                description=event.description,
                estimated_date=estimated_date.isoformat(),
                amount=event.base_amount,
                timing_volatility=event.timing_volatility,
                amount_volatility=event.amount_volatility,
                drift_rate=0.03,  # 3% annual drift
                probability=event.base_probability,
                category=event.category.value,
                dependencies=event.dependencies
            )
            
            seed_events.append(seed_event)
        
        return seed_events
    
    def _estimate_event_timing(self, event: LifestyleEvent) -> datetime:
        """Estimate when an event is likely to occur"""
        # Base timing by category
        base_timing = {
            EventCategory.EDUCATION: 2,  # 2 years from now
            EventCategory.CAREER: 1,     # 1 year from now
            EventCategory.FAMILY: 3,     # 3 years from now
            EventCategory.HOUSING: 2,    # 2 years from now
            EventCategory.HEALTH: 1,     # 1 year from now
            EventCategory.RETIREMENT: 5  # 5 years from now
        }
        
        base_years = base_timing.get(event.category, 2)
        
        # Add some randomness
        timing_variation = np.random.normal(base_years, 0.5)
        timing_years = max(0.5, timing_variation)
        
        return datetime.now() + timedelta(days=int(timing_years * 365))
    
    def process_json_batch(self, json_inputs: List[Dict]) -> List[ClientVectorProfile]:
        """
        Process a batch of JSON inputs to vector profiles
        
        Args:
            json_inputs: List of JSON client data
            
        Returns:
            List of ClientVectorProfile objects
        """
        self.logger.info(f"Processing batch of {len(json_inputs)} JSON inputs")
        
        vector_profiles = []
        
        for json_input in json_inputs:
            try:
                vector_profile = self.convert_json_to_vector_profile(json_input)
                vector_profiles.append(vector_profile)
            except Exception as e:
                self.logger.error(f"Error processing JSON input: {e}")
                continue
        
        self.logger.info(f"Successfully converted {len(vector_profiles)} JSON inputs to vector profiles")
        return vector_profiles
    
    def export_vector_data(self, vector_profiles: List[ClientVectorProfile], 
                          filename: str) -> None:
        """
        Export vector data to JSON file
        
        Args:
            vector_profiles: List of vector profiles
            filename: Output filename
        """
        export_data = []
        
        for profile in vector_profiles:
            profile_data = {
                'client_id': profile.client_id,
                'age': profile.age,
                'life_stage': profile.life_stage.value,
                'base_income': profile.base_income,
                'current_assets': profile.current_assets.tolist(),
                'current_debts': profile.current_debts.tolist(),
                'risk_tolerance': profile.risk_tolerance,
                'event_probabilities': profile.event_probabilities.tolist(),
                'cash_flow_vector': profile.cash_flow_vector.tolist(),
                'discretionary_spending_surface': profile.discretionary_spending_surface.tolist()
            }
            export_data.append(profile_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(vector_profiles)} vector profiles to {filename}")


def create_demo_converter():
    """Create and demonstrate the JSON to vector converter"""
    converter = JSONToVectorConverter(use_gpu=True)
    
    # Sample JSON inputs
    sample_inputs = [
        {
            'client_id': 'CLIENT_001',
            'age': 28,
            'income': 65000,
            'current_assets': {
                'cash': 15000,
                'savings': 25000,
                'investments': 10000,
                'retirement': 5000,
                'real_estate': 0,
                'other_assets': 2000
            },
            'debts': {
                'credit_cards': 3000,
                'student_loans': 25000,
                'mortgage': 0,
                'auto_loans': 8000,
                'personal_loans': 0,
                'other_debts': 0
            },
            'risk_tolerance': 'moderate'
        },
        {
            'client_id': 'CLIENT_002',
            'age': 45,
            'income': 120000,
            'current_assets': {
                'cash': 50000,
                'savings': 100000,
                'investments': 150000,
                'retirement': 200000,
                'real_estate': 300000,
                'other_assets': 25000
            },
            'debts': {
                'credit_cards': 5000,
                'student_loans': 0,
                'mortgage': 200000,
                'auto_loans': 15000,
                'personal_loans': 0,
                'other_debts': 0
            },
            'risk_tolerance': 'aggressive'
        }
    ]
    
    # Convert to vector profiles
    vector_profiles = converter.process_json_batch(sample_inputs)
    
    # Generate synthetic events for each profile
    for profile in vector_profiles:
        events = converter.generate_synthetic_events(profile, num_events=5)
        seed_events = converter.convert_events_to_seed_events(events)
        
        print(f"\n📊 Client: {profile.client_id}")
        print(f"   Age: {profile.age}, Life Stage: {profile.life_stage.value}")
        print(f"   Risk Tolerance: {profile.risk_tolerance:.2f}")
        print(f"   Generated {len(events)} synthetic events")
        
        for event in events[:3]:  # Show first 3 events
            print(f"   - {event.category.value}: {event.description} (${event.base_amount:,.0f})")
    
    # Export vector data
    converter.export_vector_data(vector_profiles, 'data/outputs/analysis_data/vector_profiles.json')
    
    return converter, vector_profiles


if __name__ == "__main__":
    print("🚀 Starting JSON to Vector Converter Demo...")
    converter, profiles = create_demo_converter()
    print("✅ Demo completed successfully!") 