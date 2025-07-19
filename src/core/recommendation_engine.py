#!/usr/bin/env python3
"""
Financial Recommendation Engine
Core recommendation system for financial planning and optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    PURCHASE = "purchase"
    INVESTMENT = "investment"
    DEBT_PAYOFF = "debt_payoff"
    REALLOCATION = "reallocation"
    SAVINGS_INCREASE = "savings_increase"
    EMERGENCY_FUND = "emergency_fund"

@dataclass
class Recommendation:
    """A financial recommendation"""
    recommendation_id: str
    timestamp: datetime
    type: RecommendationType
    title: str
    description: str
    amount: float
    priority: str  # "high", "medium", "low"
    expected_return: float
    risk_level: str
    timeline: str  # "immediate", "short_term", "long_term"
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

class FinancialRecommendationEngine:
    """
    Analyzes financial state and generates intelligent recommendations
    for purchases, investments, and reallocations
    """
    
    def __init__(self, mesh_engine, accounting_engine):
        self.mesh_engine = mesh_engine
        self.accounting_engine = accounting_engine
        self.recommendation_history = []
        self.configuration_matrices = {}
        
        # Investment categories and their characteristics
        self.investment_categories = {
            'emergency_fund': {'risk': 'Very Low', 'liquidity': 'High', 'expected_return': 0.02},
            'high_yield_savings': {'risk': 'Very Low', 'liquidity': 'High', 'expected_return': 0.04},
            'cds': {'risk': 'Low', 'liquidity': 'Medium', 'expected_return': 0.05},
            'government_bonds': {'risk': 'Low', 'liquidity': 'Medium', 'expected_return': 0.04},
            'corporate_bonds': {'risk': 'Medium', 'liquidity': 'Medium', 'expected_return': 0.06},
            'index_funds': {'risk': 'Medium', 'liquidity': 'Medium', 'expected_return': 0.08},
            'growth_stocks': {'risk': 'High', 'liquidity': 'High', 'expected_return': 0.12},
            'real_estate': {'risk': 'Medium', 'liquidity': 'Low', 'expected_return': 0.09},
            'commodities': {'risk': 'High', 'liquidity': 'Medium', 'expected_return': 0.07},
            'crypto': {'risk': 'Very High', 'liquidity': 'High', 'expected_return': 0.15}
        }
        
        # Purchase categories and typical costs
        self.purchase_categories = {
            'home_improvement': {'min_cost': 5000, 'max_cost': 50000, 'urgency': 'Medium'},
            'vehicle': {'min_cost': 15000, 'max_cost': 60000, 'urgency': 'High'},
            'education': {'min_cost': 1000, 'max_cost': 50000, 'urgency': 'High'},
            'healthcare': {'min_cost': 500, 'max_cost': 25000, 'urgency': 'Critical'},
            'vacation': {'min_cost': 2000, 'max_cost': 15000, 'urgency': 'Low'},
            'technology': {'min_cost': 500, 'max_cost': 5000, 'urgency': 'Medium'},
            'insurance': {'min_cost': 1000, 'max_cost': 10000, 'urgency': 'High'}
        }
        
        logger.info("✅ Financial Recommendation Engine initialized")
    
    def analyze_current_position(self, milestones: List[Dict], profile_data: Dict) -> Dict:
        """Analyze current financial position and mesh state"""
        mesh_status = self.mesh_engine.get_mesh_status()
        financial_statement = self.accounting_engine.generate_financial_statement()
        
        # Calculate key financial ratios
        total_assets = financial_statement['summary']['total_assets']
        total_liabilities = financial_statement['summary']['total_liabilities']
        net_worth = financial_statement['summary']['net_worth']
        
        monthly_income = profile_data.get('base_income', 60000) / 12
        current_cash = financial_statement['assets'].get('cash_checking', {}).get('balance', 0)
        
        # Emergency fund ratio
        emergency_fund_ratio = current_cash / (monthly_income * 6)  # 6 months of expenses
        
        # Debt-to-income ratio
        debt_to_income = total_liabilities / (profile_data.get('base_income', 60000))
        
        # Investment allocation
        investment_balance = financial_statement['assets'].get('investments_stocks', {}).get('balance', 0)
        investment_ratio = investment_balance / total_assets if total_assets > 0 else 0
        
        analysis = {
            'financial_health_score': self._calculate_financial_health_score(
                emergency_fund_ratio, debt_to_income, investment_ratio, net_worth
            ),
            'liquidity_position': 'Strong' if emergency_fund_ratio > 1.0 else 'Weak',
            'debt_burden': 'High' if debt_to_income > 0.4 else 'Manageable',
            'investment_diversification': 'Good' if investment_ratio > 0.2 else 'Needs Improvement',
            'mesh_efficiency': mesh_status['visible_future_nodes'] / mesh_status['total_nodes'],
            'recommendation_urgency': self._assess_urgency(milestones),
            'key_metrics': {
                'net_worth': net_worth,
                'emergency_fund_ratio': emergency_fund_ratio,
                'debt_to_income': debt_to_income,
                'investment_ratio': investment_ratio,
                'monthly_surplus': monthly_income * 0.3  # Estimate
            }
        }
        
        return analysis
    
    def _calculate_financial_health_score(self, emergency_fund_ratio: float, 
                                        debt_to_income: float, investment_ratio: float, 
                                        net_worth: float) -> float:
        """Calculate comprehensive financial health score (0-100)"""
        score = 0
        
        # Emergency fund component (25 points)
        if emergency_fund_ratio >= 1.0:
            score += 25
        elif emergency_fund_ratio >= 0.5:
            score += 15
        elif emergency_fund_ratio >= 0.25:
            score += 10
        
        # Debt burden component (25 points)
        if debt_to_income <= 0.2:
            score += 25
        elif debt_to_income <= 0.4:
            score += 15
        elif debt_to_income <= 0.6:
            score += 10
        
        # Investment allocation component (25 points)
        if investment_ratio >= 0.3:
            score += 25
        elif investment_ratio >= 0.2:
            score += 15
        elif investment_ratio >= 0.1:
            score += 10
        
        # Net worth component (25 points)
        if net_worth > 0:
            score += min(25, (net_worth / 100000) * 25)  # Scale based on $100K
        
        return min(100, score)
    
    def _assess_urgency(self, milestones: List[Dict]) -> str:
        """Assess urgency of recommendations based on milestones"""
        if not milestones:
            return "Low"
        
        # Check for critical milestones
        critical_milestones = [m for m in milestones if m.get('urgency') == 'Critical']
        if critical_milestones:
            return "High"
        
        # Check for high urgency milestones
        high_urgency_milestones = [m for m in milestones if m.get('urgency') == 'High']
        if len(high_urgency_milestones) > 1:
            return "Medium"
        
        return "Low"
    
    def generate_monthly_recommendations(self, milestones: List[Dict], 
                                       profile_data: Dict, months_ahead: int = 12) -> List[Recommendation]:
        """Generate monthly recommendations for the next specified months"""
        recommendations = []
        
        # Analyze current position
        current_analysis = self.analyze_current_position(milestones, profile_data)
        
        # Generate recommendations based on analysis
        for month in range(months_ahead):
            month_recommendations = self._generate_month_recommendations(
                current_analysis, profile_data, month + 1
            )
            recommendations.extend(month_recommendations)
        
        # Sort by priority and expected return
        recommendations.sort(key=lambda r: (self._priority_score(r.priority), r.expected_return), reverse=True)
        
        return recommendations[:20]  # Return top 20 recommendations
    
    def _generate_month_recommendations(self, analysis: Dict, profile_data: Dict, month: int) -> List[Recommendation]:
        """Generate recommendations for a specific month"""
        recommendations = []
        
        # Emergency fund recommendations
        if analysis['key_metrics']['emergency_fund_ratio'] < 1.0:
            emergency_rec = self._create_emergency_fund_recommendation(analysis, profile_data, month)
            recommendations.append(emergency_rec)
        
        # Debt payoff recommendations
        if analysis['key_metrics']['debt_to_income'] > 0.4:
            debt_rec = self._create_debt_payoff_recommendation(analysis, profile_data, month)
            recommendations.append(debt_rec)
        
        # Investment recommendations
        if analysis['key_metrics']['investment_ratio'] < 0.2:
            investment_rec = self._create_investment_recommendation(analysis, profile_data, month)
            recommendations.append(investment_rec)
        
        # Savings increase recommendations
        if analysis['financial_health_score'] < 70:
            savings_rec = self._create_savings_increase_recommendation(analysis, profile_data, month)
            recommendations.append(savings_rec)
        
        return recommendations
    
    def _create_emergency_fund_recommendation(self, analysis: Dict, profile_data: Dict, month: int) -> Recommendation:
        """Create emergency fund recommendation"""
        current_ratio = analysis['key_metrics']['emergency_fund_ratio']
        target_ratio = 1.0
        monthly_income = profile_data.get('base_income', 60000) / 12
        
        # Calculate amount needed
        current_emergency_fund = monthly_income * 6 * current_ratio
        target_emergency_fund = monthly_income * 6 * target_ratio
        amount_needed = target_emergency_fund - current_emergency_fund
        
        return Recommendation(
            recommendation_id=f"emergency_fund_{month:02d}",
            timestamp=datetime.now(),
            type=RecommendationType.EMERGENCY_FUND,
            title="Build Emergency Fund",
            description=f"Increase emergency fund to cover 6 months of expenses",
            amount=amount_needed,
            priority="high" if current_ratio < 0.5 else "medium",
            expected_return=0.02,
            risk_level="Very Low",
            timeline="short_term",
            confidence=0.9
        )
    
    def _create_debt_payoff_recommendation(self, analysis: Dict, profile_data: Dict, month: int) -> Recommendation:
        """Create debt payoff recommendation"""
        debt_to_income = analysis['key_metrics']['debt_to_income']
        monthly_income = profile_data.get('base_income', 60000) / 12
        
        # Calculate recommended debt payment
        if debt_to_income > 0.6:
            amount = monthly_income * 0.3  # 30% of income
            priority = "high"
        elif debt_to_income > 0.4:
            amount = monthly_income * 0.2  # 20% of income
            priority = "medium"
        else:
            amount = monthly_income * 0.1  # 10% of income
            priority = "low"
        
        return Recommendation(
            recommendation_id=f"debt_payoff_{month:02d}",
            timestamp=datetime.now(),
            type=RecommendationType.DEBT_PAYOFF,
            title="Accelerate Debt Payoff",
            description=f"Pay extra {amount:,.0f} toward high-interest debt",
            amount=amount,
            priority=priority,
            expected_return=0.08,  # Interest savings
            risk_level="Low",
            timeline="short_term",
            confidence=0.8
        )
    
    def _create_investment_recommendation(self, analysis: Dict, profile_data: Dict, month: int) -> Recommendation:
        """Create investment recommendation"""
        monthly_income = profile_data.get('base_income', 60000) / 12
        risk_tolerance = profile_data.get('risk_tolerance', 'moderate')
        
        # Determine investment amount based on risk tolerance
        if risk_tolerance == 'conservative':
            amount = monthly_income * 0.1
            investment_type = 'index_funds'
        elif risk_tolerance == 'aggressive':
            amount = monthly_income * 0.2
            investment_type = 'growth_stocks'
        else:  # moderate
            amount = monthly_income * 0.15
            investment_type = 'index_funds'
        
        investment_info = self.investment_categories[investment_type]
        
        return Recommendation(
            recommendation_id=f"investment_{month:02d}",
            timestamp=datetime.now(),
            type=RecommendationType.INVESTMENT,
            title=f"Invest in {investment_type.replace('_', ' ').title()}",
            description=f"Start monthly investment of {amount:,.0f}",
            amount=amount,
            priority="medium",
            expected_return=investment_info['expected_return'],
            risk_level=investment_info['risk'],
            timeline="long_term",
            confidence=0.7
        )
    
    def _create_savings_increase_recommendation(self, analysis: Dict, profile_data: Dict, month: int) -> Recommendation:
        """Create savings increase recommendation"""
        monthly_income = profile_data.get('base_income', 60000) / 12
        current_savings = analysis['key_metrics']['monthly_surplus']
        
        # Calculate recommended savings increase
        target_savings = monthly_income * 0.2  # 20% savings rate
        additional_savings = max(0, target_savings - current_savings)
        
        return Recommendation(
            recommendation_id=f"savings_increase_{month:02d}",
            timestamp=datetime.now(),
            type=RecommendationType.SAVINGS_INCREASE,
            title="Increase Monthly Savings",
            description=f"Increase monthly savings by {additional_savings:,.0f}",
            amount=additional_savings,
            priority="medium",
            expected_return=0.02,
            risk_level="Very Low",
            timeline="short_term",
            confidence=0.8
        )
    
    def _priority_score(self, priority: str) -> int:
        """Convert priority string to numeric score"""
        priority_scores = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return priority_scores.get(priority, 1)
    
    def create_configuration_matrix(self, subject_key: str, milestones: List[Dict], 
                                  profile_data: Dict, scenarios: int = 5) -> Dict:
        """Create configuration matrix for analysis"""
        matrix = {
            "subject_key": subject_key,
            "timestamp": datetime.now().isoformat(),
            "scenarios": scenarios,
            "configurations": []
        }
        
        for scenario in range(scenarios):
            # Generate different configurations
            config = {
                "scenario_id": f"scenario_{scenario}",
                "risk_tolerance": random.choice(['conservative', 'moderate', 'aggressive']),
                "time_horizon": random.randint(5, 20),
                "savings_rate": random.uniform(0.1, 0.3),
                "investment_allocation": {
                    "stocks": random.uniform(0.3, 0.8),
                    "bonds": random.uniform(0.1, 0.5),
                    "cash": random.uniform(0.05, 0.2)
                },
                "debt_payoff_priority": random.choice(['high', 'medium', 'low'])
            }
            
            # Generate recommendations for this configuration
            recommendations = self.generate_monthly_recommendations(milestones, profile_data, 12)
            
            config["recommendations"] = [
                {
                    "type": rec.type.value,
                    "title": rec.title,
                    "amount": rec.amount,
                    "priority": rec.priority,
                    "expected_return": rec.expected_return
                } for rec in recommendations[:5]  # Top 5 recommendations
            ]
            
            matrix["configurations"].append(config)
        
        return matrix
    
    def evaluate_mesh_effectiveness(self, milestones: List[Dict], profile_data: Dict, 
                                  recommendations: List[Recommendation]) -> Dict:
        """Evaluate the effectiveness of mesh-based recommendations"""
        if not recommendations:
            return {"error": "No recommendations to evaluate"}
        
        # Calculate effectiveness metrics
        total_recommendations = len(recommendations)
        high_priority_count = len([r for r in recommendations if r.priority == "high"])
        medium_priority_count = len([r for r in recommendations if r.priority == "medium"])
        low_priority_count = len([r for r in recommendations if r.priority == "low"])
        
        # Calculate expected returns
        total_expected_return = sum(r.expected_return * r.amount for r in recommendations)
        average_expected_return = total_expected_return / sum(r.amount for r in recommendations) if sum(r.amount for r in recommendations) > 0 else 0
        
        # Calculate risk distribution
        risk_distribution = {}
        for rec in recommendations:
            risk = rec.risk_level
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        effectiveness = {
            "total_recommendations": total_recommendations,
            "priority_distribution": {
                "high": high_priority_count,
                "medium": medium_priority_count,
                "low": low_priority_count
            },
            "average_expected_return": average_expected_return,
            "total_expected_return": total_expected_return,
            "risk_distribution": risk_distribution,
            "recommendation_types": list(set(r.type.value for r in recommendations)),
            "timeline_distribution": {
                "immediate": len([r for r in recommendations if r.timeline == "immediate"]),
                "short_term": len([r for r in recommendations if r.timeline == "short_term"]),
                "long_term": len([r for r in recommendations if r.timeline == "long_term"])
            }
        }
        
        return effectiveness

def main():
    """Main function for testing"""
    # Mock mesh engine and accounting engine
    class MockMeshEngine:
        def get_mesh_status(self):
            return {
                "total_nodes": 100,
                "visible_future_nodes": 80,
                "total_edges": 200
            }
    
    class MockAccountingEngine:
        def generate_financial_statement(self):
            return {
                "summary": {
                    "total_assets": 200000,
                    "total_liabilities": 80000,
                    "net_worth": 120000
                },
                "assets": {
                    "cash_checking": {"balance": 15000},
                    "investments_stocks": {"balance": 50000}
                }
            }
    
    # Create recommendation engine
    mesh_engine = MockMeshEngine()
    accounting_engine = MockAccountingEngine()
    recommendation_engine = FinancialRecommendationEngine(mesh_engine, accounting_engine)
    
    # Sample data
    milestones = [
        {
            "id": "milestone_1",
            "type": "investment",
            "amount": 10000,
            "urgency": "Medium"
        }
    ]
    
    profile_data = {
        "base_income": 80000,
        "risk_tolerance": "moderate"
    }
    
    # Generate recommendations
    recommendations = recommendation_engine.generate_monthly_recommendations(
        milestones, profile_data, months_ahead=6
    )
    
    print(f"Generated {len(recommendations)} recommendations:")
    for rec in recommendations[:5]:  # Show first 5
        print(f"- {rec.title}: ${rec.amount:,.0f} ({rec.priority} priority)")

if __name__ == "__main__":
    main() 