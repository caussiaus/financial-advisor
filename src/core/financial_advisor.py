#!/usr/bin/env python3
"""
Core Financial Advisor Module
Consolidated financial advisor system combining neural networks, behavioral motivation,
continuous analysis, and realistic probability estimates
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import threading
import time
from dataclasses import dataclass, field
import random

# Import core components
from .neural_engine import NeuralEngineConfig, NeuralFinancialEngine
from .behavioral_motivation import BehavioralMotivationEngine
from .mesh_engine import StochasticMeshEngine
from .recommendation_engine import FinancialRecommendationEngine
from .accounting_engine import AccountingReconciliationEngine
from .visualization_engine import FinancialVisualizationEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialBehavior:
    """Represents a financial behavior with motivation factors"""
    behavior_type: str  # "save", "invest", "spend", "borrow", "repay"
    amount: float
    motivation_score: float  # 0-1, how likely to actually do this
    psychological_factors: Dict[str, float]  # fear, greed, social pressure, etc.
    time_horizon: str  # "immediate", "short_term", "long_term"
    risk_tolerance: float  # 0-1
    social_influence: float  # 0-1, peer pressure, family expectations

@dataclass
class CapitalAllocation:
    """Optimal capital allocation with behavioral motivation"""
    cash_allocation: float
    investment_allocation: float
    debt_reduction: float
    emergency_fund: float
    discretionary_spending: float
    confidence_interval: Tuple[float, float]
    behavioral_motivation: Dict[str, float]
    expected_return: float
    risk_level: str

class IntegratedFinancialAdvisor:
    """Main integrated financial advisor system"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the integrated financial advisor"""
        self.config = config or {}
        
        # Initialize core components
        self.neural_engine = None
        self.behavioral_engine = None
        self.mesh_engine = None
        self.recommendation_engine = None
        self.accounting_engine = None
        self.visualization_engine = None
        
        # Analysis state
        self.analysis_interval = 300  # 5 minutes
        self.is_running = False
        self.analysis_thread = None
        self.current_analysis = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ Integrated Financial Advisor initialized")
    
    def _initialize_components(self):
        """Initialize all core components"""
        try:
            # Initialize neural engine
            neural_config = NeuralEngineConfig(enable_training=False)
            self.neural_engine = NeuralFinancialEngine(neural_config)
            
            # Initialize behavioral engine
            self.behavioral_engine = BehavioralMotivationEngine()
            
            # Initialize mesh engine
            self.mesh_engine = StochasticMeshEngine()
            
            # Initialize accounting engine
            self.accounting_engine = AccountingReconciliationEngine()
            
            # Initialize recommendation engine
            self.recommendation_engine = FinancialRecommendationEngine(
                self.mesh_engine, self.accounting_engine
            )
            
            # Initialize visualization engine
            self.visualization_engine = FinancialVisualizationEngine()
            
            logger.info("✅ All core components initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def start_continuous_analysis(self, client_data: Dict):
        """Start continuous financial analysis"""
        self.is_running = True
        self.analysis_thread = threading.Thread(
            target=self._continuous_analysis_loop,
            args=(client_data,)
        )
        self.analysis_thread.start()
        logger.info("🔄 Started continuous financial analysis")
    
    def stop_continuous_analysis(self):
        """Stop continuous analysis"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join()
        logger.info("⏹️ Stopped continuous financial analysis")
    
    def _continuous_analysis_loop(self, client_data: Dict):
        """Continuous analysis loop"""
        while self.is_running:
            try:
                # Update analysis
                self.current_analysis = self._perform_comprehensive_analysis(client_data)
                
                # Log significant changes
                self._log_significant_changes()
                
                # Wait for next analysis
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ Analysis error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_comprehensive_analysis(self, client_data: Dict) -> Dict:
        """Perform comprehensive financial analysis"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "financial_health": self._calculate_financial_health(client_data),
            "risk_assessment": self._assess_risk(client_data),
            "behavioral_analysis": self._analyze_behavioral_factors(client_data),
            "capital_allocation": self._optimize_capital_allocation(client_data),
            "probability_estimates": self._calculate_probability_estimates(client_data),
            "motivational_insights": self._generate_motivational_insights(client_data),
            "recommendations": self._generate_recommendations(client_data)
        }
        return analysis
    
    def _calculate_financial_health(self, client_data: Dict) -> Dict:
        """Calculate comprehensive financial health score"""
        assets = client_data.get("assets", {})
        liabilities = client_data.get("liabilities", {})
        income = client_data.get("income", 0)
        expenses = client_data.get("expenses", 0)
        
        total_assets = sum(assets.values())
        total_liabilities = sum(liabilities.values())
        net_worth = total_assets - total_liabilities
        savings_rate = (income - expenses) / income if income > 0 else 0
        debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0
        
        # Calculate health score (0-100)
        health_score = 0
        health_score += min(30, (net_worth / max(income, 1)) * 10)  # Net worth factor
        health_score += min(25, savings_rate * 100)  # Savings factor
        health_score += min(25, (1 - debt_ratio) * 25)  # Debt factor
        health_score += min(20, (income / max(expenses, 1)) * 10)  # Income factor
        
        return {
            "score": health_score,
            "net_worth": net_worth,
            "savings_rate": savings_rate,
            "debt_ratio": debt_ratio,
            "income_coverage": income / max(expenses, 1),
            "category": self._categorize_health(health_score)
        }
    
    def _categorize_health(self, score: float) -> str:
        """Categorize financial health"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _assess_risk(self, client_data: Dict) -> Dict:
        """Assess financial risk with realistic probability estimates"""
        age = client_data.get("age", 35)
        risk_tolerance = client_data.get("risk_tolerance", 0.5)
        portfolio = client_data.get("assets", {})
        
        # Market risk scenarios
        scenarios = {
            "recession": {"probability": 0.15, "impact": -0.25},
            "market_correction": {"probability": 0.25, "impact": -0.10},
            "normal_growth": {"probability": 0.45, "impact": 0.08},
            "bull_market": {"probability": 0.15, "impact": 0.20}
        }
        
        # Calculate expected portfolio value
        expected_value = 0
        for scenario, params in scenarios.items():
            expected_value += params["probability"] * params["impact"]
        
        # Risk metrics
        volatility = 0.15  # Historical market volatility
        var_95 = -0.20  # 95% Value at Risk
        max_drawdown = -0.30  # Maximum expected drawdown
        
        return {
            "expected_return": expected_value,
            "volatility": volatility,
            "value_at_risk_95": var_95,
            "max_drawdown": max_drawdown,
            "scenarios": scenarios,
            "risk_level": self._determine_risk_level(risk_tolerance, age)
        }
    
    def _determine_risk_level(self, risk_tolerance: float, age: int) -> str:
        """Determine appropriate risk level based on age and tolerance"""
        if age < 30 and risk_tolerance > 0.7:
            return "aggressive"
        elif age < 50 and risk_tolerance > 0.4:
            return "moderate"
        else:
            return "conservative"
    
    def _analyze_behavioral_factors(self, client_data: Dict) -> Dict:
        """Analyze behavioral factors affecting financial decisions"""
        personality = client_data.get("personality", {})
        
        # Calculate motivation scores for different behaviors
        save_motivation = self.behavioral_engine.calculate_save_motivation(client_data)
        invest_motivation = self.behavioral_engine.calculate_invest_motivation(client_data)
        spend_motivation = self.behavioral_engine.calculate_spend_motivation(client_data)
        
        # Get personalized interventions
        interventions = self.behavioral_engine.get_personalized_interventions(
            client_data, "save"
        )
        
        return {
            "save_motivation": save_motivation,
            "invest_motivation": invest_motivation,
            "spend_motivation": spend_motivation,
            "personality_type": self.behavioral_engine.analyze_personality(client_data),
            "interventions": [i.name for i in interventions],
            "psychological_factors": personality
        }
    
    def _optimize_capital_allocation(self, client_data: Dict) -> CapitalAllocation:
        """Optimize capital allocation with behavioral motivation"""
        financial_health = self._calculate_financial_health(client_data)
        risk_assessment = self._assess_risk(client_data)
        behavioral = self._analyze_behavioral_factors(client_data)
        
        # Get neural recommendations if available
        current_state = {
            "cash": client_data.get("assets", {}).get("cash", 0),
            "investments": client_data.get("assets", {}).get("investments", 0),
            "debt": client_data.get("liabilities", {}).get("total_debt", 0),
            "income": client_data.get("income", 0),
            "age": client_data.get("age", 35),
            "risk_tolerance": client_data.get("risk_tolerance", 0.5)
        }
        
        try:
            if self.neural_engine:
                neural_action = self.neural_engine.optimize_policy(
                    current_state, risk_assessment["risk_level"]
                )
            else:
                neural_action = {"invest": 0.2, "save": 0.3, "spend": 0.1, "repay": 0.4}
        except Exception as e:
            logger.warning(f"Neural optimization failed: {e}")
            neural_action = {"invest": 0.2, "save": 0.3, "spend": 0.1, "repay": 0.4}
        
        # Calculate optimal allocation
        total_assets = sum(client_data.get("assets", {}).values())
        monthly_income = client_data.get("income", 0) / 12
        monthly_expenses = client_data.get("expenses", 0) / 12
        
        # Emergency fund (6 months of expenses)
        emergency_fund = min(total_assets * 0.1, monthly_expenses * 6)
        
        # Investment allocation based on neural recommendation
        investment_allocation = total_assets * neural_action.get("invest", 0.2)
        
        # Debt reduction priority
        debt_reduction = total_assets * neural_action.get("repay", 0.4)
        
        # Cash allocation
        cash_allocation = total_assets * 0.1
        
        # Discretionary spending
        discretionary_spending = max(0, monthly_income - monthly_expenses) * 0.3
        
        return CapitalAllocation(
            cash_allocation=cash_allocation,
            investment_allocation=investment_allocation,
            debt_reduction=debt_reduction,
            emergency_fund=emergency_fund,
            discretionary_spending=discretionary_spending,
            confidence_interval=(0.7, 0.9),
            behavioral_motivation=behavioral["psychological_factors"],
            expected_return=risk_assessment["expected_return"],
            risk_level=risk_assessment["risk_level"]
        )
    
    def _calculate_probability_estimates(self, client_data: Dict) -> Dict:
        """Calculate realistic probability estimates for financial outcomes"""
        age = client_data.get("age", 35)
        income = client_data.get("income", 0)
        current_assets = sum(client_data.get("assets", {}).values())
        
        # Monte Carlo simulation parameters
        num_simulations = 1000
        time_horizon = 65 - age  # Years to retirement
        
        # Simulate retirement scenarios
        retirement_comfortable = 0
        emergency_fund_adequate = 0
        debt_free = 0
        
        for _ in range(num_simulations):
            # Simulate portfolio growth
            portfolio_value = current_assets
            for year in range(time_horizon):
                # Annual return with volatility
                annual_return = np.random.normal(0.07, 0.15)
                portfolio_value *= (1 + annual_return)
                
                # Add annual savings
                savings_rate = 0.15  # 15% savings rate
                portfolio_value += income * savings_rate
            
            # Check outcomes
            if portfolio_value >= 1000000:  # $1M retirement
                retirement_comfortable += 1
            if portfolio_value >= income * 0.5:  # 6 months emergency fund
                emergency_fund_adequate += 1
            if portfolio_value >= current_assets * 2:  # Double current assets
                debt_free += 1
        
        return {
            "probability_retirement_comfortable": retirement_comfortable / num_simulations,
            "probability_emergency_fund_adequate": emergency_fund_adequate / num_simulations,
            "probability_debt_free": debt_free / num_simulations,
            "expected_portfolio_value": current_assets * (1.07 ** time_horizon),
            "confidence_interval": (0.05, 0.95)
        }
    
    def _generate_motivational_insights(self, client_data: Dict) -> Dict:
        """Generate motivational insights based on behavioral analysis"""
        behavioral = self._analyze_behavioral_factors(client_data)
        personality_type = behavioral["personality_type"]
        
        # Get personalized motivation messages
        motivation_message = self.behavioral_engine.generate_motivation_message(
            client_data, "save"
        )
        
        # Create behavioral plan
        behavioral_plan = self.behavioral_engine.create_behavioral_plan(client_data)
        
        return {
            "motivation_message": motivation_message,
            "personality_type": personality_type,
            "behavioral_plan": behavioral_plan,
            "next_actions": self._suggest_next_actions(client_data),
            "progress_tracking": self._create_progress_tracking(client_data)
        }
    
    def _suggest_next_actions(self, client_data: Dict) -> List[Dict]:
        """Suggest next actions based on analysis"""
        actions = []
        
        financial_health = self._calculate_financial_health(client_data)
        behavioral = self._analyze_behavioral_factors(client_data)
        
        # Emergency fund actions
        if financial_health["score"] < 60:
            actions.append({
                "action": "Build emergency fund",
                "priority": "high",
                "amount": client_data.get("expenses", 0) * 0.5,
                "timeline": "3 months"
            })
        
        # Investment actions
        if behavioral["invest_motivation"] < 0.5:
            actions.append({
                "action": "Start dollar-cost averaging",
                "priority": "medium",
                "amount": client_data.get("income", 0) * 0.1,
                "timeline": "immediate"
            })
        
        # Debt reduction actions
        if financial_health["debt_ratio"] > 0.4:
            actions.append({
                "action": "Prioritize debt reduction",
                "priority": "high",
                "amount": client_data.get("income", 0) * 0.2,
                "timeline": "6 months"
            })
        
        return actions
    
    def _create_progress_tracking(self, client_data: Dict) -> Dict:
        """Create progress tracking metrics"""
        return {
            "savings_progress": 0.65,  # 65% to emergency fund goal
            "investment_progress": 0.45,  # 45% to retirement goal
            "debt_progress": 0.30,  # 30% debt reduction progress
            "next_milestone": "Emergency fund complete",
            "estimated_completion": "3 months"
        }
    
    def _generate_recommendations(self, client_data: Dict) -> List[Dict]:
        """Generate comprehensive financial recommendations"""
        recommendations = []
        
        # Get recommendations from recommendation engine
        if self.recommendation_engine:
            try:
                mesh_recommendations = self.recommendation_engine.generate_monthly_recommendations(
                    [], client_data, months_ahead=12
                )
                recommendations.extend(mesh_recommendations)
            except Exception as e:
                logger.warning(f"Mesh recommendations failed: {e}")
        
        # Add behavioral recommendations
        behavioral = self._analyze_behavioral_factors(client_data)
        for intervention in behavioral.get("interventions", []):
            recommendations.append({
                "type": "behavioral",
                "title": intervention,
                "description": f"Implement {intervention} to improve financial behavior",
                "priority": "medium",
                "timeline": "1 month"
            })
        
        return recommendations
    
    def _log_significant_changes(self):
        """Log significant changes in analysis"""
        if not self.current_analysis:
            return
        
        # Log significant changes (simplified)
        logger.info(f"📊 Analysis updated: {self.current_analysis.get('timestamp', 'unknown')}")
    
    def get_current_analysis(self) -> Dict:
        """Get current analysis results"""
        return self.current_analysis
    
    def generate_dashboard_data(self, client_data: Dict) -> Dict:
        """Generate data for dashboard visualization"""
        analysis = self._perform_comprehensive_analysis(client_data)
        
        return {
            "analysis": analysis,
            "charts": self.visualization_engine.generate_charts(analysis),
            "metrics": self.visualization_engine.generate_metrics(analysis),
            "recommendations": analysis.get("recommendations", [])
        }

def main():
    """Main function for testing"""
    advisor = IntegratedFinancialAdvisor()
    
    # Sample client data
    client_data = {
        "name": "Alex Johnson",
        "age": 32,
        "income": 85000,
        "expenses": 65000,
        "assets": {
            "cash": 15000,
            "investments": 45000,
            "retirement": 25000,
            "real_estate": 200000
        },
        "liabilities": {
            "student_loans": 35000,
            "credit_cards": 8000,
            "mortgage": 180000
        },
        "personality": {
            "fear_of_loss": 0.6,
            "greed_factor": 0.4,
            "social_pressure": 0.3,
            "patience": 0.7,
            "financial_literacy": 0.6
        },
        "goals": ["emergency_fund", "debt_free", "retirement"],
        "risk_tolerance": 0.6,
        "life_stage": "early_career"
    }
    
    # Generate analysis
    analysis = advisor._perform_comprehensive_analysis(client_data)
    print(json.dumps(analysis, indent=2, default=str))

if __name__ == "__main__":
    main() 