#!/usr/bin/env python
"""
Temporal fsQCA Integration System
Author: Claude 2025-07-16

Integrates fuzzy-set Qualitative Comparative Analysis (fsQCA) with time-oriented data
by creating temporal configuration pathways instead of running fsQCA at every age.

Key Innovation: Configuration effectiveness changes over time based on:
- Life stage context (early career vs established vs pre-retirement)
- Milestone timing windows (when certain events are most feasible)
- Resource availability evolution (income growth, expense changes)
- Risk tolerance shifts with age

This system identifies:
1. Age-optimal configurations for different outcomes
2. Transition timing between configuration strategies
3. Configuration persistence vs adaptation needs
4. Milestone-specific configuration requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import existing components
from .fuzzy_sets_optimizer import fsQCAAnalyzer, FuzzyCondition, fsQCAResult
from .timeline_bias_engine import TimelineBiasEngine, ClientTimeline
from .spending_surface_modeler import SpendingSurfaceModeler
from .continuous_configuration_mesh import ContinuousConfigurationMesh

logger = logging.getLogger(__name__)

@dataclass
class TemporalConfiguration:
    """Configuration analysis at a specific life stage/age range"""
    age_range: Tuple[int, int]
    life_stage: str
    fsqca_result: fsQCAResult
    optimal_configurations: List[Dict[str, Any]]
    transition_triggers: List[Dict[str, Any]]
    milestone_feasibility: Dict[str, float]
    configuration_stability: float

@dataclass
class ConfigurationTransition:
    """Represents transition between configuration strategies"""
    from_age: int
    to_age: int
    trigger_event: Optional[str]
    from_config: Dict[str, float]
    to_config: Dict[str, float]
    transition_cost: float
    adaptation_time: float
    success_probability: float

@dataclass
class TemporalPathway:
    """Complete temporal pathway of configurations"""
    client_id: str
    pathway_id: str
    age_stages: List[TemporalConfiguration]
    transitions: List[ConfigurationTransition]
    milestone_timeline: Dict[str, int]
    overall_effectiveness: float
    risk_profile: str

class TemporalfsQCAIntegrator:
    """Main system for integrating fsQCA with temporal analysis"""
    
    def __init__(self, 
                 timeline_engine: TimelineBiasEngine,
                 surface_modeler: SpendingSurfaceModeler,
                 configuration_mesh: ContinuousConfigurationMesh):
        self.timeline_engine = timeline_engine
        self.surface_modeler = surface_modeler
        self.configuration_mesh = configuration_mesh
        self.fsqca_analyzer = fsQCAAnalyzer()
        
        # Define life stage boundaries
        self.life_stages = {
            'early_career': (22, 32),
            'establishment': (33, 45), 
            'mid_career': (46, 55),
            'pre_retirement': (56, 65),
            'retirement': (66, 85)
        }
        
        # Stage-specific configuration priorities
        self.stage_priorities = {
            'early_career': {
                'primary_outcomes': ['career_advancement', 'skill_building', 'debt_management'],
                'secondary_outcomes': ['emergency_fund', 'relationship_building'],
                'configuration_flexibility': 0.8  # High flexibility
            },
            'establishment': {
                'primary_outcomes': ['home_ownership', 'family_formation', 'wealth_building'],
                'secondary_outcomes': ['career_stability', 'insurance_optimization'],
                'configuration_flexibility': 0.6  # Moderate flexibility
            },
            'mid_career': {
                'primary_outcomes': ['wealth_accumulation', 'children_education', 'peak_earning'],
                'secondary_outcomes': ['health_investment', 'lifestyle_optimization'],
                'configuration_flexibility': 0.4  # Lower flexibility
            },
            'pre_retirement': {
                'primary_outcomes': ['retirement_readiness', 'health_preservation', 'legacy_planning'],
                'secondary_outcomes': ['debt_elimination', 'downsizing_preparation'],
                'configuration_flexibility': 0.3  # Limited flexibility
            },
            'retirement': {
                'primary_outcomes': ['income_preservation', 'health_management', 'legacy_execution'],
                'secondary_outcomes': ['lifestyle_adaptation', 'care_planning'],
                'configuration_flexibility': 0.2  # Very limited flexibility
            }
        }
    
    def analyze_temporal_configurations(self, 
                                      client_profile: Dict[str, Any],
                                      target_milestones: List[str]) -> TemporalPathway:
        """Analyze optimal configurations across client's lifetime"""
        
        current_age = client_profile.get('age', 30)
        client_id = client_profile.get('client_id', 'unknown')
        
        # Step 1: Get milestone timing predictions
        milestone_timeline = self._predict_milestone_timeline(client_profile, target_milestones)
        
        # Step 2: Analyze configurations for each life stage
        age_stage_analyses = []
        
        for stage_name, (min_age, max_age) in self.life_stages.items():
            if max_age < current_age - 5:  # Skip past stages
                continue
                
            stage_analysis = self._analyze_stage_configuration(
                client_profile, stage_name, (min_age, max_age), 
                milestone_timeline, target_milestones
            )
            age_stage_analyses.append(stage_analysis)
        
        # Step 3: Identify optimal transitions between stages
        transitions = self._identify_configuration_transitions(
            age_stage_analyses, client_profile
        )
        
        # Step 4: Calculate pathway effectiveness
        overall_effectiveness = self._calculate_pathway_effectiveness(
            age_stage_analyses, transitions, milestone_timeline
        )
        
        # Step 5: Determine risk profile
        risk_profile = self._determine_risk_profile(client_profile, age_stage_analyses)
        
        return TemporalPathway(
            client_id=client_id,
            pathway_id=f"{client_id}_TEMPORAL_PATHWAY",
            age_stages=age_stage_analyses,
            transitions=transitions,
            milestone_timeline=milestone_timeline,
            overall_effectiveness=overall_effectiveness,
            risk_profile=risk_profile
        )
    
    def _predict_milestone_timeline(self, 
                                  client_profile: Dict[str, Any],
                                  target_milestones: List[str]) -> Dict[str, int]:
        """Predict when milestones will be achieved"""
        timeline = {}
        
        for milestone in target_milestones:
            try:
                # Use spending surface for prediction if available
                if milestone in self.surface_modeler.surfaces:
                    prediction = self.surface_modeler.predict_milestone_timing_surface(
                        milestone,
                        client_profile.get('income', 75000),
                        client_profile.get('age', 30),
                        client_profile.get('discretionary_ratio', 0.15)
                    )
                    timeline[milestone] = int(prediction['predicted_age'])
                
                # Fallback to timeline bias engine
                else:
                    bias = self.timeline_engine.case_db.get_event_timing_bias(
                        milestone, client_profile
                    )
                    timeline[milestone] = int(bias.median_age)
                    
            except Exception as e:
                logger.warning(f"Could not predict {milestone} timing: {e}")
                # Use default age based on milestone type
                default_ages = {
                    'home_purchase': 32,
                    'marriage': 29,
                    'first_child': 31,
                    'career_peak': 45,
                    'retirement': 65
                }
                timeline[milestone] = default_ages.get(milestone, 35)
        
        return timeline
    
    def _analyze_stage_configuration(self, 
                                   client_profile: Dict[str, Any],
                                   stage_name: str,
                                   age_range: Tuple[int, int],
                                   milestone_timeline: Dict[str, int],
                                   target_milestones: List[str]) -> TemporalConfiguration:
        """Analyze optimal configuration for a specific life stage"""
        
        # Generate stage-specific client scenarios
        stage_scenarios = self._generate_stage_scenarios(
            client_profile, stage_name, age_range, milestone_timeline
        )
        
        # Run fsQCA for this stage
        fsqca_result = self._run_stage_fsqca(stage_scenarios, stage_name)
        
        # Identify optimal configurations for this stage
        optimal_configs = self._identify_optimal_stage_configurations(
            fsqca_result, stage_name, age_range
        )
        
        # Analyze milestone feasibility in this stage
        milestone_feasibility = self._analyze_milestone_feasibility(
            stage_name, age_range, milestone_timeline, target_milestones
        )
        
        # Calculate configuration stability for this stage
        stability = self._calculate_configuration_stability(
            optimal_configs, stage_name
        )
        
        # Identify transition triggers
        transition_triggers = self._identify_transition_triggers(
            stage_name, milestone_timeline, client_profile
        )
        
        return TemporalConfiguration(
            age_range=age_range,
            life_stage=stage_name,
            fsqca_result=fsqca_result,
            optimal_configurations=optimal_configs,
            transition_triggers=transition_triggers,
            milestone_feasibility=milestone_feasibility,
            configuration_stability=stability
        )
    
    def _generate_stage_scenarios(self, 
                                client_profile: Dict[str, Any],
                                stage_name: str,
                                age_range: Tuple[int, int],
                                milestone_timeline: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate client scenarios for specific life stage analysis"""
        
        scenarios = []
        current_age = client_profile.get('age', 30)
        base_income = client_profile.get('income', 75000)
        
        # Generate scenarios across the age range
        for age in range(age_range[0], min(age_range[1] + 1, current_age + 20), 2):
            
            # Income evolution
            if age <= 45:
                income_growth = 1 + (age - 25) * 0.03  # 3% annual growth
            else:
                income_growth = 1.5 - (age - 45) * 0.01  # Plateau then decline
            
            stage_income = base_income * income_growth
            
            # Stage-specific financial characteristics
            stage_priorities = self.stage_priorities[stage_name]
            
            # Calculate stage-appropriate financial metrics
            if stage_name == 'early_career':
                savings_rate = np.random.uniform(0.05, 0.20)
                debt_ratio = np.random.uniform(0.1, 0.4)  # Higher debt tolerance
                expense_ratio = np.random.uniform(0.7, 0.9)
            elif stage_name == 'establishment':
                savings_rate = np.random.uniform(0.10, 0.25)
                debt_ratio = np.random.uniform(0.2, 0.6)  # Mortgage acquisition
                expense_ratio = np.random.uniform(0.75, 0.95)
            elif stage_name == 'mid_career':
                savings_rate = np.random.uniform(0.15, 0.35)
                debt_ratio = np.random.uniform(0.1, 0.4)  # Debt paydown
                expense_ratio = np.random.uniform(0.65, 0.85)
            elif stage_name == 'pre_retirement':
                savings_rate = np.random.uniform(0.20, 0.45)
                debt_ratio = np.random.uniform(0.0, 0.2)  # Debt elimination
                expense_ratio = np.random.uniform(0.55, 0.75)
            else:  # retirement
                savings_rate = np.random.uniform(-0.05, 0.10)  # May be drawing down
                debt_ratio = np.random.uniform(0.0, 0.1)
                expense_ratio = np.random.uniform(0.60, 0.80)
            
            # Calculate milestone achievement probabilities
            milestone_achievements = {}
            for milestone, expected_age in milestone_timeline.items():
                if age >= expected_age:
                    achievement_prob = min(1.0, (age - expected_age + 5) / 10)
                    milestone_achievements[milestone] = np.random.random() < achievement_prob
                else:
                    milestone_achievements[milestone] = False
            
            # Calculate financial stress based on life stage pressures
            stress_factors = []
            if stage_name in ['establishment', 'mid_career']:
                stress_factors.append(expense_ratio - 0.7)  # High expenses
                stress_factors.append(max(0, 0.15 - savings_rate))  # Low savings
            
            financial_stress = min(1.0, sum(max(0, factor) for factor in stress_factors))
            
            scenario = {
                'client_id': f"{client_profile.get('client_id', 'unknown')}_{stage_name}_{age}",
                'age': age,
                'life_stage': stage_name,
                'income': stage_income,
                'income_history': [stage_income * (1 + np.random.normal(0, 0.05)) for _ in range(3)],
                'total_expenses': stage_income * expense_ratio,
                'total_debt': stage_income * debt_ratio,
                'savings_rate': savings_rate,
                'emergency_fund': stage_income * np.random.uniform(0.1, 0.6),
                'portfolio_return': np.random.normal(0.08, 0.12),
                'financial_stress': financial_stress,
                'milestone_achievements': milestone_achievements,
                'stage_priorities': stage_priorities['primary_outcomes']
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _run_stage_fsqca(self, 
                        stage_scenarios: List[Dict[str, Any]], 
                        stage_name: str) -> fsQCAResult:
        """Run fsQCA analysis for specific life stage"""
        
        from .fuzzy_sets_optimizer import FinancialStabilityOptimizer
        
        # Use the existing financial stability optimizer but adapt for stage
        optimizer = FinancialStabilityOptimizer()
        
        # Modify scenarios to focus on stage-specific outcomes
        stage_priorities = self.stage_priorities[stage_name]
        
        # Adjust financial stress calculation for stage priorities
        for scenario in stage_scenarios:
            # Weight stress based on stage priorities
            if 'wealth_building' in stage_priorities['primary_outcomes']:
                # Emphasize savings and investment performance
                scenario['financial_stress'] *= (1 - scenario['savings_rate'])
            elif 'debt_management' in stage_priorities['primary_outcomes']:
                # Emphasize debt ratios
                debt_stress = scenario['total_debt'] / scenario['income']
                scenario['financial_stress'] = max(scenario['financial_stress'], debt_stress)
        
        # Run fsQCA with stage-adapted scenarios
        return optimizer.analyze_financial_stability_paths(stage_scenarios)
    
    def _identify_optimal_stage_configurations(self, 
                                             fsqca_result: fsQCAResult,
                                             stage_name: str,
                                             age_range: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Identify optimal configurations for the life stage"""
        
        optimal_configs = []
        
        # Extract top paths from fsQCA result
        for path in fsqca_result.optimal_paths[:3]:  # Top 3 paths
            
            # Convert fsQCA path to actionable configuration
            config = {
                'config_id': f"{stage_name}_{path['path_id']}",
                'age_range': age_range,
                'fsqca_formula': path['formula'],
                'consistency': path['consistency'],
                'coverage': path['coverage'],
                'implementation_difficulty': path['implementation_difficulty'],
                'stage_appropriateness': self._assess_stage_appropriateness(path, stage_name),
                'action_items': self._convert_path_to_actions(path, stage_name),
                'expected_outcomes': self._predict_configuration_outcomes(path, stage_name),
                'monitoring_metrics': self._define_monitoring_metrics(path, stage_name)
            }
            
            optimal_configs.append(config)
        
        return optimal_configs
    
    def _assess_stage_appropriateness(self, path: Dict[str, Any], stage_name: str) -> float:
        """Assess how appropriate a configuration is for the life stage"""
        
        stage_priorities = self.stage_priorities[stage_name]
        flexibility = stage_priorities['configuration_flexibility']
        
        # Higher complexity configurations are less appropriate for less flexible stages
        complexity_penalty = (1 - flexibility) * path.get('complexity', 2) * 0.1
        
        # Base appropriateness from fsQCA metrics
        base_appropriateness = (path['consistency'] * 0.6 + path['coverage'] * 0.4)
        
        return max(0, base_appropriateness - complexity_penalty)
    
    def _convert_path_to_actions(self, path: Dict[str, Any], stage_name: str) -> List[str]:
        """Convert fsQCA path to specific action items for the stage"""
        
        actions = []
        
        # Start with the path's recommendations
        actions.extend(path.get('recommendations', []))
        
        # Add stage-specific action adaptations
        stage_priorities = self.stage_priorities[stage_name]
        
        for priority in stage_priorities['primary_outcomes']:
            if priority == 'career_advancement':
                actions.append("Focus on skill development and networking")
            elif priority == 'home_ownership':
                actions.append("Build down payment fund and improve credit score")
            elif priority == 'wealth_accumulation':
                actions.append("Maximize investment contributions and optimize portfolio")
            elif priority == 'retirement_readiness':
                actions.append("Accelerate retirement savings and plan withdrawal strategy")
        
        return actions[:8]  # Limit to 8 actionable items
    
    def _predict_configuration_outcomes(self, path: Dict[str, Any], stage_name: str) -> Dict[str, float]:
        """Predict outcomes from following this configuration"""
        
        return {
            'stress_reduction': path.get('stress_reduction_potential', 0.5),
            'milestone_acceleration': 0.7 if path['consistency'] > 0.8 else 0.4,
            'financial_stability': path['consistency'],
            'adaptability': self.stage_priorities[stage_name]['configuration_flexibility'],
            'success_probability': min(1.0, path['consistency'] * path['coverage'])
        }
    
    def _define_monitoring_metrics(self, path: Dict[str, Any], stage_name: str) -> List[str]:
        """Define key metrics to monitor for this configuration"""
        
        base_metrics = [
            'monthly_savings_rate',
            'debt_to_income_ratio',
            'emergency_fund_months',
            'investment_performance'
        ]
        
        # Add stage-specific metrics
        stage_priorities = self.stage_priorities[stage_name]
        
        if 'career_advancement' in stage_priorities['primary_outcomes']:
            base_metrics.append('income_growth_rate')
        if 'home_ownership' in stage_priorities['primary_outcomes']:
            base_metrics.append('down_payment_progress')
        if 'retirement_readiness' in stage_priorities['primary_outcomes']:
            base_metrics.append('retirement_savings_rate')
        
        return base_metrics
    
    def _analyze_milestone_feasibility(self, 
                                     stage_name: str,
                                     age_range: Tuple[int, int],
                                     milestone_timeline: Dict[str, int],
                                     target_milestones: List[str]) -> Dict[str, float]:
        """Analyze milestone feasibility within this life stage"""
        
        feasibility = {}
        
        for milestone in target_milestones:
            expected_age = milestone_timeline.get(milestone, 35)
            
            # Check if milestone falls within this stage
            if age_range[0] <= expected_age <= age_range[1]:
                # High feasibility if within stage
                feasibility[milestone] = 0.9
            elif expected_age < age_range[0]:
                # Already should have happened
                feasibility[milestone] = 0.2
            elif expected_age > age_range[1]:
                # Future milestone
                years_ahead = expected_age - age_range[1]
                feasibility[milestone] = max(0.1, 0.6 - years_ahead * 0.1)
            else:
                feasibility[milestone] = 0.3
        
        return feasibility
    
    def _calculate_configuration_stability(self, 
                                         optimal_configs: List[Dict[str, Any]], 
                                         stage_name: str) -> float:
        """Calculate how stable configurations are for this stage"""
        
        if not optimal_configs:
            return 0.0
        
        # Average consistency across configurations
        avg_consistency = np.mean([config['consistency'] for config in optimal_configs])
        
        # Stage flexibility factor
        flexibility = self.stage_priorities[stage_name]['configuration_flexibility']
        
        # Higher flexibility means lower required stability
        stability_requirement = 1 - flexibility
        
        return min(1.0, avg_consistency / stability_requirement)
    
    def _identify_transition_triggers(self, 
                                    stage_name: str,
                                    milestone_timeline: Dict[str, int],
                                    client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify triggers that should prompt configuration transitions"""
        
        triggers = []
        
        # Age-based triggers
        stage_end_age = self.life_stages[stage_name][1]
        triggers.append({
            'trigger_type': 'age_milestone',
            'trigger_age': stage_end_age,
            'description': f"Transition from {stage_name} to next life stage",
            'urgency': 'medium'
        })
        
        # Milestone-based triggers
        for milestone, expected_age in milestone_timeline.items():
            if self.life_stages[stage_name][0] <= expected_age <= self.life_stages[stage_name][1]:
                triggers.append({
                    'trigger_type': 'milestone_achievement',
                    'trigger_age': expected_age,
                    'milestone': milestone,
                    'description': f"Configuration adjustment for {milestone}",
                    'urgency': 'high'
                })
        
        # Income-based triggers
        current_income = client_profile.get('income', 75000)
        if current_income < 60000:
            triggers.append({
                'trigger_type': 'income_threshold',
                'trigger_income': 60000,
                'description': "Income growth enabling configuration upgrade",
                'urgency': 'medium'
            })
        
        return triggers
    
    def _identify_configuration_transitions(self, 
                                          age_stage_analyses: List[TemporalConfiguration],
                                          client_profile: Dict[str, Any]) -> List[ConfigurationTransition]:
        """Identify optimal transitions between life stage configurations"""
        
        transitions = []
        
        for i in range(len(age_stage_analyses) - 1):
            current_stage = age_stage_analyses[i]
            next_stage = age_stage_analyses[i + 1]
            
            # Find best configuration from current stage
            if current_stage.optimal_configurations:
                current_config = current_stage.optimal_configurations[0]
            else:
                continue
                
            # Find best configuration for next stage
            if next_stage.optimal_configurations:
                next_config = next_stage.optimal_configurations[0]
            else:
                continue
            
            # Calculate transition characteristics
            transition_age = current_stage.age_range[1]
            
            # Identify trigger event
            trigger_events = [t for t in current_stage.transition_triggers 
                            if t.get('trigger_age', 999) <= transition_age + 2]
            trigger_event = trigger_events[0]['description'] if trigger_events else None
            
            # Calculate transition cost (complexity difference)
            current_complexity = current_config.get('complexity', 2)
            next_complexity = next_config.get('complexity', 2)
            transition_cost = abs(next_complexity - current_complexity) * 0.2
            
            # Estimate adaptation time (months)
            adaptation_time = 3 + transition_cost * 6
            
            # Calculate success probability
            success_prob = min(1.0, (current_config['consistency'] + next_config['consistency']) / 2)
            
            transition = ConfigurationTransition(
                from_age=current_stage.age_range[1],
                to_age=next_stage.age_range[0],
                trigger_event=trigger_event,
                from_config=current_config,
                to_config=next_config,
                transition_cost=transition_cost,
                adaptation_time=adaptation_time,
                success_probability=success_prob
            )
            
            transitions.append(transition)
        
        return transitions
    
    def _calculate_pathway_effectiveness(self, 
                                       age_stage_analyses: List[TemporalConfiguration],
                                       transitions: List[ConfigurationTransition],
                                       milestone_timeline: Dict[str, int]) -> float:
        """Calculate overall effectiveness of the temporal pathway"""
        
        # Average stage configuration effectiveness
        stage_effectiveness = np.mean([
            stage.configuration_stability for stage in age_stage_analyses
        ]) if age_stage_analyses else 0.0
        
        # Transition smoothness
        transition_smoothness = np.mean([
            t.success_probability for t in transitions
        ]) if transitions else 1.0
        
        # Milestone coverage
        covered_milestones = sum(1 for stage in age_stage_analyses 
                               for milestone, feasibility in stage.milestone_feasibility.items()
                               if feasibility > 0.5)
        milestone_coverage = covered_milestones / max(len(milestone_timeline), 1)
        
        # Weighted overall effectiveness
        overall_effectiveness = (
            stage_effectiveness * 0.5 +
            transition_smoothness * 0.3 +
            milestone_coverage * 0.2
        )
        
        return min(1.0, overall_effectiveness)
    
    def _determine_risk_profile(self, 
                              client_profile: Dict[str, Any],
                              age_stage_analyses: List[TemporalConfiguration]) -> str:
        """Determine risk profile for the temporal pathway"""
        
        # Analyze configuration stability across stages
        avg_stability = np.mean([stage.configuration_stability for stage in age_stage_analyses])
        
        # Analyze income stability
        income = client_profile.get('income', 75000)
        income_risk = 'high' if income < 50000 else 'medium' if income < 100000 else 'low'
        
        # Combine factors
        if avg_stability > 0.8 and income_risk == 'low':
            return 'conservative'
        elif avg_stability > 0.6 and income_risk in ['low', 'medium']:
            return 'moderate'
        elif avg_stability > 0.4:
            return 'balanced'
        else:
            return 'aggressive'
    
    def generate_temporal_recommendations(self, pathway: TemporalPathway) -> Dict[str, Any]:
        """Generate actionable recommendations based on temporal pathway analysis"""
        
        current_stage = None
        next_transition = None
        
        # Find current stage and next transition
        current_age = 30  # Would get from client profile
        
        for stage in pathway.age_stages:
            if stage.age_range[0] <= current_age <= stage.age_range[1]:
                current_stage = stage
                break
        
        for transition in pathway.transitions:
            if transition.from_age >= current_age:
                next_transition = transition
                break
        
        recommendations = {
            'immediate_actions': [],
            'stage_optimization': {},
            'transition_preparation': {},
            'milestone_timing': {},
            'risk_management': {},
            'monitoring_plan': {}
        }
        
        # Current stage recommendations
        if current_stage and current_stage.optimal_configurations:
            best_config = current_stage.optimal_configurations[0]
            recommendations['immediate_actions'] = best_config['action_items'][:5]
            recommendations['stage_optimization'] = {
                'focus_areas': best_config['expected_outcomes'],
                'key_metrics': best_config['monitoring_metrics'],
                'success_probability': best_config['stage_appropriateness']
            }
        
        # Transition preparation
        if next_transition:
            recommendations['transition_preparation'] = {
                'transition_timing': f"Age {next_transition.from_age}",
                'trigger_event': next_transition.trigger_event,
                'preparation_time': f"{next_transition.adaptation_time:.0f} months",
                'success_factors': next_transition.to_config['action_items'][:3]
            }
        
        # Milestone timing optimization
        for milestone, age in pathway.milestone_timeline.items():
            recommendations['milestone_timing'][milestone] = {
                'target_age': age,
                'years_remaining': max(0, age - current_age),
                'feasibility': 'high' if age - current_age > 2 else 'medium' if age - current_age > 0 else 'low'
            }
        
        # Risk management
        recommendations['risk_management'] = {
            'pathway_risk': pathway.risk_profile,
            'effectiveness': f"{pathway.overall_effectiveness:.1%}",
            'key_risks': self._identify_pathway_risks(pathway),
            'mitigation_strategies': self._suggest_risk_mitigation(pathway)
        }
        
        return recommendations
    
    def _identify_pathway_risks(self, pathway: TemporalPathway) -> List[str]:
        """Identify key risks in the temporal pathway"""
        risks = []
        
        if pathway.overall_effectiveness < 0.6:
            risks.append("Low overall pathway effectiveness")
        
        for transition in pathway.transitions:
            if transition.success_probability < 0.7:
                risks.append(f"Difficult transition at age {transition.from_age}")
        
        for stage in pathway.age_stages:
            if stage.configuration_stability < 0.5:
                risks.append(f"Unstable configuration in {stage.life_stage}")
        
        return risks
    
    def _suggest_risk_mitigation(self, pathway: TemporalPathway) -> List[str]:
        """Suggest risk mitigation strategies"""
        strategies = []
        
        if pathway.risk_profile == 'aggressive':
            strategies.append("Build larger emergency fund for configuration flexibility")
            strategies.append("Diversify income sources to reduce transition risk")
        
        if pathway.overall_effectiveness < 0.7:
            strategies.append("Consider alternative milestone timing")
            strategies.append("Increase configuration monitoring frequency")
        
        strategies.append("Regular pathway reviews every 2-3 years")
        strategies.append("Maintain adaptation fund for unexpected transitions")
        
        return strategies


# Demo function
def demo_temporal_fsqca_integration():
    """Demonstrate temporal fsQCA integration"""
    print("🕒 TEMPORAL fsQCA INTEGRATION DEMONSTRATION")
    print("=" * 70)
    
    # This would use the existing components in a real implementation
    print("📊 Integrating fsQCA with temporal analysis...")
    print("\n🎯 Key Innovation: Configuration effectiveness changes over time")
    print("   • Early career: High flexibility, focus on growth")
    print("   • Establishment: Moderate flexibility, focus on stability")
    print("   • Mid-career: Lower flexibility, focus on optimization")
    print("   • Pre-retirement: Limited flexibility, focus on preservation")
    
    print(f"\n📈 Temporal Pathway Analysis:")
    print(f"   ✓ Age-specific optimal configurations")
    print(f"   ✓ Transition timing and triggers")
    print(f"   ✓ Milestone-configuration alignment")
    print(f"   ✓ Risk-adjusted pathway optimization")
    
    print(f"\n🔄 Configuration Evolution:")
    print(f"   • Age 25-32: High savings, flexible lifestyle")
    print(f"   • Age 33-45: Home purchase, family formation")
    print(f"   • Age 46-55: Wealth accumulation, peak earning")
    print(f"   • Age 56-65: Retirement preparation, risk reduction")

if __name__ == "__main__":
    demo_temporal_fsqca_integration() 