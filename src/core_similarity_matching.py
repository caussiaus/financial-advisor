"""
Core Similarity Matching System

This module implements similarity matching with age-based milestone estimation
using the core engines (stochastic, accounting, services) while avoiding
exotic geometry algorithms for now.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import warnings

from .stochastic_mesh_engine import StochasticMeshEngine
from .accounting_reconciliation import AccountingReconciliationEngine
from .unified_cash_flow_model import UnifiedCashFlowModel
from .mesh_vector_database import MeshVectorDatabase
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine

warnings.filterwarnings('ignore')

@dataclass
class AgeBasedMilestone:
    """Age-based milestone with estimated timing"""
    milestone_id: str
    description: str
    estimated_age: int
    estimated_date: datetime
    financial_impact: float
    probability: float
    category: str
    confidence_score: float
    similar_clients_used: List[str] = field(default_factory=list)

@dataclass
class SimilarityMatch:
    """Result of similarity matching"""
    client_id: str
    matched_client_id: str
    similarity_score: float
    matching_factors: List[str]
    age_difference: int
    income_similarity: float
    risk_similarity: float
    milestone_overlap: int
    confidence_score: float

class CoreSimilarityMatcher:
    """
    Core similarity matching system using existing engines
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.vector_db = MeshVectorDatabase()
        self.synthetic_engine = SyntheticLifestyleEngine(use_gpu=use_gpu)
        self.logger = self._setup_logging()
        
        # Age-based milestone templates
        self.age_milestone_templates = self._initialize_age_milestones()
        
        # Similarity thresholds
        self.similarity_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('core_similarity_matcher')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_age_milestones(self) -> Dict[str, Dict]:
        """Initialize age-based milestone templates"""
        return {
            'education': {
                'early_career': {
                    'graduate_school': {'age': 25, 'impact': 50000, 'probability': 0.3},
                    'professional_certification': {'age': 28, 'impact': 10000, 'probability': 0.6},
                    'mba': {'age': 30, 'impact': 80000, 'probability': 0.2}
                },
                'mid_career': {
                    'executive_education': {'age': 35, 'impact': 25000, 'probability': 0.4},
                    'specialized_training': {'age': 38, 'impact': 15000, 'probability': 0.5}
                }
            },
            'career': {
                'early_career': {
                    'job_change': {'age': 26, 'impact': 5000, 'probability': 0.7},
                    'promotion': {'age': 29, 'impact': 15000, 'probability': 0.6}
                },
                'mid_career': {
                    'career_advancement': {'age': 35, 'impact': 25000, 'probability': 0.5},
                    'executive_role': {'age': 40, 'impact': 50000, 'probability': 0.3}
                },
                'established': {
                    'consulting_startup': {'age': 45, 'impact': 100000, 'probability': 0.2},
                    'board_position': {'age': 50, 'impact': 75000, 'probability': 0.3}
                }
            },
            'family': {
                'early_career': {
                    'marriage': {'age': 28, 'impact': 25000, 'probability': 0.6},
                    'first_child': {'age': 30, 'impact': 15000, 'probability': 0.5}
                },
                'mid_career': {
                    'second_child': {'age': 32, 'impact': 15000, 'probability': 0.4},
                    'elder_care': {'age': 45, 'impact': 30000, 'probability': 0.3}
                }
            },
            'housing': {
                'early_career': {
                    'first_home': {'age': 32, 'impact': 50000, 'probability': 0.4},
                    'condo_purchase': {'age': 29, 'impact': 30000, 'probability': 0.5}
                },
                'mid_career': {
                    'upgrade_home': {'age': 38, 'impact': 75000, 'probability': 0.4},
                    'vacation_property': {'age': 45, 'impact': 100000, 'probability': 0.2}
                },
                'established': {
                    'downsizing': {'age': 60, 'impact': -50000, 'probability': 0.6}
                }
            },
            'health': {
                'mid_career': {
                    'major_medical': {'age': 40, 'impact': 20000, 'probability': 0.2},
                    'dental_work': {'age': 35, 'impact': 8000, 'probability': 0.4}
                },
                'established': {
                    'health_insurance_upgrade': {'age': 50, 'impact': 15000, 'probability': 0.3},
                    'long_term_care': {'age': 65, 'impact': 50000, 'probability': 0.4}
                }
            },
            'investment': {
                'early_career': {
                    'first_investment': {'age': 26, 'impact': 10000, 'probability': 0.6},
                    'retirement_start': {'age': 28, 'impact': 5000, 'probability': 0.8}
                },
                'mid_career': {
                    'portfolio_diversification': {'age': 35, 'impact': 25000, 'probability': 0.5},
                    'real_estate_investment': {'age': 40, 'impact': 50000, 'probability': 0.3}
                },
                'established': {
                    'estate_planning': {'age': 55, 'impact': 20000, 'probability': 0.6},
                    'legacy_planning': {'age': 65, 'impact': 100000, 'probability': 0.4}
                }
            }
        }
    
    def find_similar_clients(self, client_profile: Dict[str, Any], top_k: int = 5) -> List[SimilarityMatch]:
        """
        Find similar clients based on profile characteristics
        
        Args:
            client_profile: Client profile data
            top_k: Number of similar clients to return
            
        Returns:
            List of similarity matches
        """
        # Extract key characteristics
        age = client_profile.get('age', 30)
        income = client_profile.get('income', 60000)
        risk_tolerance = client_profile.get('risk_tolerance', 0.5)
        life_stage = client_profile.get('life_stage', 'mid_career')
        
        # Generate synthetic clients for comparison
        synthetic_clients = self.synthetic_engine.generate_client_batch(num_clients=100)
        
        similarities = []
        
        for client in synthetic_clients:
            # Calculate similarity scores
            age_similarity = self._calculate_age_similarity(age, client.profile.age)
            income_similarity = self._calculate_income_similarity(income, client.profile.base_income)
            risk_similarity = self._calculate_risk_similarity(risk_tolerance, client.vector_profile.risk_tolerance)
            stage_similarity = self._calculate_stage_similarity(life_stage, client.vector_profile.life_stage.value)
            
            # Calculate overall similarity
            overall_similarity = (
                0.3 * age_similarity +
                0.3 * income_similarity +
                0.2 * risk_similarity +
                0.2 * stage_similarity
            )
            
            # Identify matching factors
            matching_factors = []
            if age_similarity > 0.8:
                matching_factors.append('age')
            if income_similarity > 0.8:
                matching_factors.append('income')
            if risk_similarity > 0.8:
                matching_factors.append('risk_tolerance')
            if stage_similarity > 0.8:
                matching_factors.append('life_stage')
            
            # Calculate milestone overlap
            milestone_overlap = self._calculate_milestone_overlap(
                client_profile.get('milestones', []),
                [e.description for e in client.lifestyle_events]
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                overall_similarity, len(matching_factors), milestone_overlap
            )
            
            match = SimilarityMatch(
                client_id=client_profile.get('client_id', 'unknown'),
                matched_client_id=client.client_id,
                similarity_score=overall_similarity,
                matching_factors=matching_factors,
                age_difference=abs(age - client.profile.age),
                income_similarity=income_similarity,
                risk_similarity=risk_similarity,
                milestone_overlap=milestone_overlap,
                confidence_score=confidence_score
            )
            
            similarities.append(match)
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def estimate_age_based_milestones(self, client_profile: Dict[str, Any], 
                                    similar_clients: List[SimilarityMatch]) -> List[AgeBasedMilestone]:
        """
        Estimate age-based milestones using similar clients
        
        Args:
            client_profile: Client profile data
            similar_clients: List of similar clients
            
        Returns:
            List of estimated age-based milestones
        """
        age = client_profile.get('age', 30)
        life_stage = client_profile.get('life_stage', 'mid_career')
        income = client_profile.get('income', 60000)
        
        estimated_milestones = []
        
        # Get relevant milestone templates for life stage
        stage_templates = {}
        for category, stages in self.age_milestone_templates.items():
            if life_stage in stages:
                stage_templates[category] = stages[life_stage]
        
        # Estimate milestones based on similar clients
        for category, milestones in stage_templates.items():
            for milestone_name, milestone_data in milestones.items():
                # Adjust age based on similar clients
                adjusted_age = self._adjust_age_based_on_similar_clients(
                    milestone_data['age'], similar_clients, category
                )
                
                # Adjust financial impact based on income
                adjusted_impact = self._adjust_impact_based_on_income(
                    milestone_data['impact'], income, similar_clients
                )
                
                # Calculate probability based on similar clients
                adjusted_probability = self._adjust_probability_based_on_similar_clients(
                    milestone_data['probability'], similar_clients, category
                )
                
                # Calculate confidence score
                confidence_score = self._calculate_milestone_confidence(
                    similar_clients, category, milestone_name
                )
                
                # Estimate date
                years_until_milestone = max(0, adjusted_age - age)
                estimated_date = datetime.now() + timedelta(days=365 * years_until_milestone)
                
                milestone = AgeBasedMilestone(
                    milestone_id=f"{category}_{milestone_name}",
                    description=milestone_name.replace('_', ' ').title(),
                    estimated_age=adjusted_age,
                    estimated_date=estimated_date,
                    financial_impact=adjusted_impact,
                    probability=adjusted_probability,
                    category=category,
                    confidence_score=confidence_score,
                    similar_clients_used=[c.matched_client_id for c in similar_clients[:3]]
                )
                
                estimated_milestones.append(milestone)
        
        return estimated_milestones
    
    def create_floating_mesh_surface(self, client_profile: Dict[str, Any], 
                                   milestones: List[AgeBasedMilestone]) -> Dict[str, Any]:
        """
        Create a floating mesh surface that avoids infeasible solutions
        
        Args:
            client_profile: Client profile data
            milestones: Estimated milestones
            
        Returns:
            Floating mesh surface data
        """
        # Initialize core engines
        initial_state = {
            'cash': client_profile.get('cash', 10000),
            'investments': client_profile.get('investments', 50000),
            'debts': client_profile.get('debts', 20000),
            'total_wealth': client_profile.get('total_wealth', 40000)
        }
        
        mesh_engine = StochasticMeshEngine(initial_state)
        accounting_engine = AccountingReconciliationEngine()
        cash_flow_model = UnifiedCashFlowModel(initial_state)
        
        # Create surface data
        surface_data = {
            'base_surface': self._calculate_base_surface(client_profile),
            'floating_offsets': self._calculate_floating_offsets(milestones),
            'infeasible_zones': self._calculate_infeasible_zones(client_profile, milestones),
            'safe_zones': self._calculate_safe_zones(client_profile, milestones),
            'mesh_elevation': self._calculate_mesh_elevation(milestones)
        }
        
        return surface_data
    
    def _calculate_age_similarity(self, age1: int, age2: int) -> float:
        """Calculate age similarity"""
        age_diff = abs(age1 - age2)
        return max(0, 1 - (age_diff / 20))  # 20 years = 0 similarity
    
    def _calculate_income_similarity(self, income1: float, income2: float) -> float:
        """Calculate income similarity"""
        if income1 == 0 or income2 == 0:
            return 0.0
        ratio = min(income1, income2) / max(income1, income2)
        return ratio
    
    def _calculate_risk_similarity(self, risk1: float, risk2: float) -> float:
        """Calculate risk tolerance similarity"""
        risk_diff = abs(risk1 - risk2)
        return max(0, 1 - risk_diff)
    
    def _calculate_stage_similarity(self, stage1: str, stage2: str) -> float:
        """Calculate life stage similarity"""
        stages = ['early_career', 'mid_career', 'established', 'pre_retirement', 'retirement']
        try:
            idx1 = stages.index(stage1)
            idx2 = stages.index(stage2)
            stage_diff = abs(idx1 - idx2)
            return max(0, 1 - (stage_diff / len(stages)))
        except ValueError:
            return 0.5  # Default similarity if stage not found
    
    def _calculate_milestone_overlap(self, milestones1: List[str], milestones2: List[str]) -> int:
        """Calculate milestone overlap between two clients"""
        set1 = set(milestones1)
        set2 = set(milestones2)
        return len(set1.intersection(set2))
    
    def _calculate_confidence_score(self, similarity: float, matching_factors: int, 
                                  milestone_overlap: int) -> float:
        """Calculate confidence score for similarity match"""
        base_confidence = similarity
        factor_bonus = min(matching_factors * 0.1, 0.3)
        overlap_bonus = min(milestone_overlap * 0.05, 0.2)
        
        confidence = base_confidence + factor_bonus + overlap_bonus
        return max(0.0, min(1.0, confidence))
    
    def _adjust_age_based_on_similar_clients(self, base_age: int, similar_clients: List[SimilarityMatch], 
                                           category: str) -> int:
        """Adjust milestone age based on similar clients"""
        if not similar_clients:
            return base_age
        
        # Calculate weighted average age adjustment
        total_weight = 0
        weighted_age = 0
        
        for client in similar_clients[:3]:  # Use top 3 similar clients
            weight = client.similarity_score
            # Assume similar clients have similar milestone timing
            age_adjustment = np.random.normal(0, 2)  # ±2 years variation
            adjusted_age = base_age + age_adjustment
            
            weighted_age += adjusted_age * weight
            total_weight += weight
        
        if total_weight > 0:
            return int(weighted_age / total_weight)
        else:
            return base_age
    
    def _adjust_impact_based_on_income(self, base_impact: float, income: float, 
                                      similar_clients: List[SimilarityMatch]) -> float:
        """Adjust financial impact based on income level"""
        if not similar_clients:
            return base_impact
        
        # Calculate income adjustment factor
        avg_income_similarity = np.mean([c.income_similarity for c in similar_clients])
        income_factor = 0.8 + (avg_income_similarity * 0.4)  # 0.8 to 1.2 range
        
        return base_impact * income_factor
    
    def _adjust_probability_based_on_similar_clients(self, base_probability: float, 
                                                   similar_clients: List[SimilarityMatch], 
                                                   category: str) -> float:
        """Adjust probability based on similar clients"""
        if not similar_clients:
            return base_probability
        
        # Adjust based on similarity scores
        avg_similarity = np.mean([c.similarity_score for c in similar_clients])
        adjustment_factor = 0.9 + (avg_similarity * 0.2)  # 0.9 to 1.1 range
        
        return min(1.0, base_probability * adjustment_factor)
    
    def _calculate_milestone_confidence(self, similar_clients: List[SimilarityMatch], 
                                      category: str, milestone_name: str) -> float:
        """Calculate confidence score for milestone estimation"""
        if not similar_clients:
            return 0.5
        
        # Base confidence from similarity scores
        avg_similarity = np.mean([c.similarity_score for c in similar_clients])
        
        # Bonus for category-specific factors
        category_bonus = 0.1 if any(c.matching_factors for c in similar_clients) else 0
        
        # Bonus for milestone overlap
        overlap_bonus = min(len(similar_clients) * 0.05, 0.2)
        
        confidence = avg_similarity + category_bonus + overlap_bonus
        return max(0.0, min(1.0, confidence))
    
    def _calculate_base_surface(self, client_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base financial surface"""
        income = client_profile.get('income', 60000)
        age = client_profile.get('age', 30)
        
        # Create time series for surface
        years = np.arange(0, 40)  # 40-year horizon
        ages = age + years
        
        # Base wealth projection
        base_wealth = income * (1 + 0.03 * years)  # 3% annual growth
        
        # Add age-based adjustments
        for i, current_age in enumerate(ages):
            if current_age > 65:
                base_wealth[i] *= 0.8  # Retirement adjustment
            elif current_age > 55:
                base_wealth[i] *= 0.9  # Pre-retirement adjustment
        
        return {
            'years': years.tolist(),
            'ages': ages.tolist(),
            'base_wealth': base_wealth.tolist(),
            'income_projection': (income * (1 + 0.03 * years)).tolist()
        }
    
    def _calculate_floating_offsets(self, milestones: List[AgeBasedMilestone]) -> Dict[str, Any]:
        """Calculate floating offsets to avoid infeasible solutions"""
        offsets = []
        
        for milestone in milestones:
            years_until = (milestone.estimated_date - datetime.now()).days / 365
            
            if years_until > 0:
                # Create offset that increases as milestone approaches
                offset = {
                    'milestone_id': milestone.milestone_id,
                    'years_until': years_until,
                    'financial_impact': milestone.financial_impact,
                    'offset_amount': max(0, milestone.financial_impact * 0.1),  # 10% buffer
                    'confidence': milestone.confidence_score
                }
                offsets.append(offset)
        
        return {'offsets': offsets}
    
    def _calculate_infeasible_zones(self, client_profile: Dict[str, Any], 
                                  milestones: List[AgeBasedMilestone]) -> Dict[str, Any]:
        """Calculate zones where solutions are infeasible"""
        infeasible_zones = []
        
        current_wealth = client_profile.get('total_wealth', 40000)
        monthly_income = client_profile.get('income', 60000) / 12
        
        for milestone in milestones:
            years_until = (milestone.estimated_date - datetime.now()).days / 365
            
            if years_until > 0 and milestone.financial_impact > 0:
                # Calculate minimum required wealth to handle milestone
                required_wealth = milestone.financial_impact + (monthly_income * 6)  # 6 months buffer
                
                if required_wealth > current_wealth:
                    zone = {
                        'milestone_id': milestone.milestone_id,
                        'years_until': years_until,
                        'required_wealth': required_wealth,
                        'current_wealth': current_wealth,
                        'shortfall': required_wealth - current_wealth,
                        'risk_level': 'high' if required_wealth > current_wealth * 2 else 'medium'
                    }
                    infeasible_zones.append(zone)
        
        return {'infeasible_zones': infeasible_zones}
    
    def _calculate_safe_zones(self, client_profile: Dict[str, Any], 
                            milestones: List[AgeBasedMilestone]) -> Dict[str, Any]:
        """Calculate safe zones where solutions are feasible"""
        safe_zones = []
        
        current_wealth = client_profile.get('total_wealth', 40000)
        monthly_income = client_profile.get('income', 60000) / 12
        
        # Calculate periods where wealth is sufficient
        for i in range(40):  # 40-year horizon
            projected_wealth = current_wealth * (1 + 0.05 * i)  # 5% annual growth
            projected_income = monthly_income * 12 * (1 + 0.03 * i)  # 3% income growth
            
            # Check if this period is safe
            is_safe = True
            for milestone in milestones:
                years_until = (milestone.estimated_date - datetime.now()).days / 365
                if 0 <= years_until <= i:
                    if milestone.financial_impact > projected_wealth * 0.3:  # More than 30% of wealth
                        is_safe = False
                        break
            
            if is_safe:
                zone = {
                    'year': i,
                    'projected_wealth': projected_wealth,
                    'projected_income': projected_income,
                    'safety_margin': projected_wealth - (projected_income * 0.3)
                }
                safe_zones.append(zone)
        
        return {'safe_zones': safe_zones}
    
    def _calculate_mesh_elevation(self, milestones: List[AgeBasedMilestone]) -> Dict[str, Any]:
        """Calculate mesh elevation to float above infeasible solutions"""
        elevation_data = []
        
        for milestone in milestones:
            years_until = (milestone.estimated_date - datetime.now()).days / 365
            
            if years_until > 0:
                # Calculate elevation needed to avoid infeasible solutions
                base_elevation = milestone.financial_impact * 0.2  # 20% elevation
                confidence_adjustment = (1 - milestone.confidence_score) * 0.1  # Uncertainty penalty
                
                elevation = {
                    'milestone_id': milestone.milestone_id,
                    'years_until': years_until,
                    'base_elevation': base_elevation,
                    'confidence_adjustment': confidence_adjustment,
                    'total_elevation': base_elevation + confidence_adjustment,
                    'confidence': milestone.confidence_score
                }
                elevation_data.append(elevation)
        
        return {'elevation_data': elevation_data} 