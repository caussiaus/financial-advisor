#!/usr/bin/env python3
"""
Behavioral Motivation System
Uses psychological principles to motivate real financial behavior changes
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehavioralIntervention:
    """A behavioral intervention to motivate financial behavior"""
    name: str
    description: str
    psychological_principle: str
    target_behavior: str
    effectiveness_score: float  # 0-1
    implementation_difficulty: str  # "easy", "medium", "hard"
    time_to_impact: str  # "immediate", "short_term", "long_term"
    cost: str  # "free", "low", "medium", "high"

@dataclass
class MotivationTrigger:
    """A trigger that can motivate financial behavior"""
    trigger_type: str  # "loss_aversion", "social_proof", "goal_framing", "nudging"
    description: str
    intensity: float  # 0-1
    target_audience: str
    psychological_basis: str

class BehavioralMotivationEngine:
    """Engine for motivating real financial behavior changes"""
    
    def __init__(self):
        self.interventions = self._load_interventions()
        self.triggers = self._load_triggers()
        self.personality_profiles = self._load_personality_profiles()
        
    def _load_interventions(self) -> List[BehavioralIntervention]:
        """Load behavioral interventions"""
        return [
            BehavioralIntervention(
                name="Automatic Savings Transfer",
                description="Set up automatic transfers to savings account",
                psychological_principle="nudging",
                target_behavior="save",
                effectiveness_score=0.85,
                implementation_difficulty="easy",
                time_to_impact="immediate",
                cost="free"
            ),
            BehavioralIntervention(
                name="Envelope Budgeting",
                description="Use physical envelopes for different spending categories",
                psychological_principle="loss_aversion",
                target_behavior="spend_less",
                effectiveness_score=0.75,
                implementation_difficulty="medium",
                time_to_impact="short_term",
                cost="low"
            ),
            BehavioralIntervention(
                name="24-Hour Purchase Rule",
                description="Wait 24 hours before making non-essential purchases",
                psychological_principle="nudging",
                target_behavior="spend_less",
                effectiveness_score=0.70,
                implementation_difficulty="easy",
                time_to_impact="immediate",
                cost="free"
            ),
            BehavioralIntervention(
                name="Dollar-Cost Averaging",
                description="Invest fixed amounts regularly regardless of market conditions",
                psychological_principle="loss_aversion",
                target_behavior="invest",
                effectiveness_score=0.80,
                implementation_difficulty="easy",
                time_to_impact="long_term",
                cost="low"
            ),
            BehavioralIntervention(
                name="Debt Snowball Method",
                description="Pay off smallest debts first for quick wins",
                psychological_principle="goal_framing",
                target_behavior="debt_reduction",
                effectiveness_score=0.90,
                implementation_difficulty="medium",
                time_to_impact="short_term",
                cost="free"
            ),
            BehavioralIntervention(
                name="Visual Goal Tracker",
                description="Create visual progress charts for financial goals",
                psychological_principle="goal_framing",
                target_behavior="save",
                effectiveness_score=0.75,
                implementation_difficulty="easy",
                time_to_impact="short_term",
                cost="free"
            ),
            BehavioralIntervention(
                name="Social Accountability Partner",
                description="Share financial goals with trusted friend/family",
                psychological_principle="social_proof",
                target_behavior="all",
                effectiveness_score=0.80,
                implementation_difficulty="medium",
                time_to_impact="short_term",
                cost="free"
            ),
            BehavioralIntervention(
                name="Emergency Fund Challenge",
                description="30-day challenge to build emergency fund",
                psychological_principle="goal_framing",
                target_behavior="save",
                effectiveness_score=0.85,
                implementation_difficulty="medium",
                time_to_impact="short_term",
                cost="free"
            ),
            BehavioralIntervention(
                name="Investment Education Program",
                description="Learn about investing through courses/books",
                psychological_principle="social_proof",
                target_behavior="invest",
                effectiveness_score=0.70,
                implementation_difficulty="hard",
                time_to_impact="long_term",
                cost="medium"
            ),
            BehavioralIntervention(
                name="Expense Tracking App",
                description="Use app to track all expenses in real-time",
                psychological_principle="nudging",
                target_behavior="spend_less",
                effectiveness_score=0.80,
                implementation_difficulty="easy",
                time_to_impact="immediate",
                cost="low"
            )
        ]
    
    def _load_triggers(self) -> List[MotivationTrigger]:
        """Load motivation triggers"""
        return [
            MotivationTrigger(
                trigger_type="loss_aversion",
                description="Show potential losses from not saving",
                intensity=0.8,
                target_audience="risk_averse",
                psychological_basis="People feel losses 2-3x more strongly than gains"
            ),
            MotivationTrigger(
                trigger_type="social_proof",
                description="Show how peers are achieving financial goals",
                intensity=0.7,
                target_audience="social_influenced",
                psychological_basis="People follow the behavior of others"
            ),
            MotivationTrigger(
                trigger_type="goal_framing",
                description="Frame goals as positive achievements",
                intensity=0.6,
                target_audience="goal_oriented",
                psychological_basis="Positive framing increases motivation"
            ),
            MotivationTrigger(
                trigger_type="nudging",
                description="Make desired behavior the default option",
                intensity=0.5,
                target_audience="procrastinators",
                psychological_basis="Default options are chosen more often"
            )
        ]
    
    def _load_personality_profiles(self) -> Dict[str, Dict]:
        """Load personality profiles for targeted interventions"""
        return {
            "risk_averse": {
                "fear_of_loss": 0.8,
                "preferred_interventions": ["loss_aversion", "nudging"],
                "motivation_factors": ["security", "stability", "avoiding_regret"]
            },
            "social_influenced": {
                "social_pressure": 0.7,
                "preferred_interventions": ["social_proof", "goal_framing"],
                "motivation_factors": ["peer_comparison", "status", "recognition"]
            },
            "goal_oriented": {
                "patience": 0.8,
                "preferred_interventions": ["goal_framing", "visual_tracking"],
                "motivation_factors": ["achievement", "progress", "milestones"]
            },
            "procrastinator": {
                "immediate_gratification": 0.6,
                "preferred_interventions": ["nudging", "automatic_transfers"],
                "motivation_factors": ["ease", "convenience", "default_options"]
            },
            "optimist": {
                "positive_outlook": 0.7,
                "preferred_interventions": ["goal_framing", "social_proof"],
                "motivation_factors": ["growth", "opportunity", "success"]
            }
        }
    
    def analyze_personality(self, client_data: Dict) -> str:
        """Analyze client personality type"""
        personality = client_data.get("personality", {})
        
        # Calculate personality scores
        fear_of_loss = personality.get("fear_of_loss", 0.5)
        social_pressure = personality.get("social_pressure", 0.5)
        patience = personality.get("patience", 0.5)
        immediate_gratification = 1 - patience  # Inverse of patience
        
        # Determine dominant personality type
        scores = {
            "risk_averse": fear_of_loss,
            "social_influenced": social_pressure,
            "goal_oriented": patience,
            "procrastinator": immediate_gratification,
            "optimist": 1 - fear_of_loss  # Inverse of fear
        }
        
        return max(scores, key=scores.get)
    
    def get_personalized_interventions(self, client_data: Dict, target_behavior: str) -> List[BehavioralIntervention]:
        """Get personalized interventions for client"""
        personality_type = self.analyze_personality(client_data)
        personality_profile = self.personality_profiles.get(personality_type, {})
        
        # Filter interventions by target behavior and personality fit
        suitable_interventions = []
        for intervention in self.interventions:
            if intervention.target_behavior == target_behavior or intervention.target_behavior == "all":
                fit_score = self._calculate_personality_fit(intervention, personality_profile)
                if fit_score > 0.5:  # Only include if good fit
                    suitable_interventions.append(intervention)
        
        # Sort by effectiveness and fit
        suitable_interventions.sort(key=lambda x: x.effectiveness_score, reverse=True)
        return suitable_interventions[:5]  # Return top 5
    
    def _calculate_personality_fit(self, intervention: BehavioralIntervention, personality_profile: Dict) -> float:
        """Calculate how well an intervention fits the personality"""
        preferred_interventions = personality_profile.get("preferred_interventions", [])
        
        if intervention.psychological_principle in preferred_interventions:
            return 0.8
        elif intervention.implementation_difficulty == "easy":
            return 0.6
        else:
            return 0.4
    
    def generate_motivation_message(self, client_data: Dict, target_behavior: str) -> str:
        """Generate personalized motivation message"""
        personality_type = self.analyze_personality(client_data)
        personality_profile = self.personality_profiles.get(personality_type, {})
        
        # Get appropriate trigger
        triggers = [t for t in self.triggers if t.target_audience == personality_type]
        if not triggers:
            triggers = self.triggers
        
        trigger = max(triggers, key=lambda x: x.intensity)
        
        # Generate message based on trigger type
        if trigger.trigger_type == "loss_aversion":
            return f"Don't let financial opportunities slip away. Every day you delay saving, you're missing out on compound growth that could secure your future."
        elif trigger.trigger_type == "social_proof":
            return f"Your peers are building wealth and achieving their financial goals. Join them on the path to financial success."
        elif trigger.trigger_type == "goal_framing":
            return f"Imagine the freedom and security you'll have when you reach your financial goals. Every step forward brings you closer to that vision."
        else:  # nudging
            return f"Make smart financial choices the easy choice. Set up automatic systems that work for you, not against you."
    
    def create_behavioral_plan(self, client_data: Dict) -> Dict:
        """Create comprehensive behavioral plan"""
        personality_type = self.analyze_personality(client_data)
        
        # Get interventions for different behaviors
        save_interventions = self.get_personalized_interventions(client_data, "save")
        invest_interventions = self.get_personalized_interventions(client_data, "invest")
        spend_interventions = self.get_personalized_interventions(client_data, "spend_less")
        
        return {
            "personality_type": personality_type,
            "plan_duration": "90 days",
            "interventions": {
                "saving": [i.name for i in save_interventions],
                "investing": [i.name for i in invest_interventions],
                "spending": [i.name for i in spend_interventions]
            },
            "success_metrics": {
                "savings_rate_increase": "15%",
                "investment_consistency": "Monthly contributions",
                "expense_reduction": "10%"
            },
            "milestones": [
                {"week": 1, "goal": "Set up automatic savings"},
                {"week": 4, "goal": "Complete first month of tracking"},
                {"week": 8, "goal": "Achieve 15% savings rate"},
                {"week": 12, "goal": "Establish consistent investment habit"}
            ]
        }
    
    def track_behavioral_progress(self, client_data: Dict, behavioral_plan: Dict) -> Dict:
        """Track progress on behavioral plan"""
        # Simulate progress tracking
        current_week = 4  # Assume we're in week 4
        
        completed_milestones = []
        upcoming_milestones = []
        
        for milestone in behavioral_plan["milestones"]:
            if milestone["week"] <= current_week:
                completed_milestones.append(milestone)
            else:
                upcoming_milestones.append(milestone)
        
        return {
            "current_week": current_week,
            "completed_milestones": completed_milestones,
            "upcoming_milestones": upcoming_milestones,
            "overall_progress": len(completed_milestones) / len(behavioral_plan["milestones"]),
            "next_milestone": upcoming_milestones[0] if upcoming_milestones else None
        }
    
    def _get_next_milestone(self, savings_progress: float, investment_progress: float, debt_progress: float) -> str:
        """Get next milestone based on progress"""
        if savings_progress < 0.5:
            return "Build emergency fund"
        elif investment_progress < 0.3:
            return "Start regular investing"
        elif debt_progress < 0.7:
            return "Accelerate debt payoff"
        else:
            return "Optimize portfolio allocation"
    
    def apply_psychological_techniques(self, client_data: Dict, target_behavior: str) -> Dict:
        """Apply psychological techniques to motivate behavior"""
        personality_type = self.analyze_personality(client_data)
        
        techniques = {}
        
        if target_behavior == "save":
            techniques.update(self._apply_loss_aversion(client_data, target_behavior))
            techniques.update(self._apply_goal_framing(client_data, target_behavior))
        elif target_behavior == "invest":
            techniques.update(self._apply_social_proof(client_data, target_behavior))
            techniques.update(self._apply_nudging(client_data, target_behavior))
        elif target_behavior == "spend_less":
            techniques.update(self._apply_loss_aversion(client_data, target_behavior))
            techniques.update(self._apply_nudging(client_data, target_behavior))
        
        return techniques
    
    def _apply_loss_aversion(self, client_data: Dict, target_behavior: str) -> Dict:
        """Apply loss aversion technique"""
        income = client_data.get("income", 0)
        age = client_data.get("age", 35)
        
        # Calculate potential losses
        years_to_retirement = 65 - age
        potential_loss = income * 0.15 * years_to_retirement  # 15% savings rate
        
        return {
            "technique": "loss_aversion",
            "message": f"By not saving 15% of your income, you're potentially losing ${potential_loss:,.0f} in retirement savings.",
            "impact_score": 0.8
        }
    
    def _apply_social_proof(self, client_data: Dict, target_behavior: str) -> Dict:
        """Apply social proof technique"""
        age = client_data.get("age", 35)
        
        # Peer comparison data
        peer_savings_rate = 0.12  # Average peer savings rate
        peer_investment_rate = 0.08  # Average peer investment rate
        
        return {
            "technique": "social_proof",
            "message": f"Your peers are saving {peer_savings_rate*100:.0f}% of their income and investing {peer_investment_rate*100:.0f}% in the market.",
            "impact_score": 0.7
        }
    
    def _apply_goal_framing(self, client_data: Dict, target_behavior: str) -> Dict:
        """Apply goal framing technique"""
        age = client_data.get("age", 35)
        income = client_data.get("income", 0)
        
        # Frame as positive achievement
        retirement_goal = income * 10  # 10x income retirement goal
        
        return {
            "technique": "goal_framing",
            "message": f"By saving consistently, you're building toward a ${retirement_goal:,.0f} retirement nest egg.",
            "impact_score": 0.6
        }
    
    def _apply_nudging(self, client_data: Dict, target_behavior: str) -> Dict:
        """Apply nudging technique"""
        return {
            "technique": "nudging",
            "message": "Make saving automatic by setting up recurring transfers. Out of sight, out of mind.",
            "impact_score": 0.5
        }
    
    def calculate_save_motivation(self, client_data: Dict) -> float:
        """Calculate motivation score for saving behavior"""
        personality = client_data.get("personality", {})
        fear_of_loss = personality.get("fear_of_loss", 0.5)
        patience = personality.get("patience", 0.5)
        
        # Base motivation
        base_motivation = 0.5
        
        # Adjust based on personality factors
        if fear_of_loss > 0.7:
            base_motivation += 0.2
        if patience > 0.6:
            base_motivation += 0.15
        
        return min(1.0, base_motivation)
    
    def calculate_invest_motivation(self, client_data: Dict) -> float:
        """Calculate motivation score for investing behavior"""
        personality = client_data.get("personality", {})
        fear_of_loss = personality.get("fear_of_loss", 0.5)
        social_pressure = personality.get("social_pressure", 0.5)
        
        # Base motivation
        base_motivation = 0.4
        
        # Adjust based on personality factors
        if fear_of_loss < 0.4:  # Lower fear = higher investment motivation
            base_motivation += 0.25
        if social_pressure > 0.6:
            base_motivation += 0.2
        
        return min(1.0, base_motivation)
    
    def calculate_spend_motivation(self, client_data: Dict) -> float:
        """Calculate motivation score for spending behavior"""
        personality = client_data.get("personality", {})
        immediate_gratification = 1 - personality.get("patience", 0.5)
        social_pressure = personality.get("social_pressure", 0.5)
        
        # Base motivation (inverse - we want to reduce spending)
        base_motivation = 0.3
        
        # Adjust based on personality factors
        if immediate_gratification > 0.7:
            base_motivation += 0.3
        if social_pressure > 0.6:
            base_motivation += 0.2
        
        return min(1.0, base_motivation)

def main():
    """Main function for testing"""
    engine = BehavioralMotivationEngine()
    
    # Sample client data
    client_data = {
        "name": "Alex Johnson",
        "age": 32,
        "income": 85000,
        "personality": {
            "fear_of_loss": 0.6,
            "social_pressure": 0.3,
            "patience": 0.7,
            "financial_literacy": 0.6
        }
    }
    
    # Test personality analysis
    personality_type = engine.analyze_personality(client_data)
    print(f"Personality type: {personality_type}")
    
    # Test interventions
    interventions = engine.get_personalized_interventions(client_data, "save")
    print(f"Personalized interventions: {[i.name for i in interventions]}")
    
    # Test motivation message
    message = engine.generate_motivation_message(client_data, "save")
    print(f"Motivation message: {message}")
    
    # Test behavioral plan
    plan = engine.create_behavioral_plan(client_data)
    print(f"Behavioral plan: {json.dumps(plan, indent=2, default=str)}")

if __name__ == "__main__":
    main() 