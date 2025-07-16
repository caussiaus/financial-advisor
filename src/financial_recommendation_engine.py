import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
from enum import Enum
import random

from enhanced_pdf_processor import FinancialMilestone
from stochastic_mesh_engine import StochasticMeshEngine, OmegaNode
from accounting_reconciliation import AccountingReconciliationEngine


class RecommendationType(Enum):
    PURCHASE = "purchase"
    INVESTMENT = "investment"
    DEBT_PAYOFF = "debt_payoff"
    REALLOCATION = "reallocation"
    SAVINGS_INCREASE = "savings_increase"
    EMERGENCY_FUND = "emergency_fund"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MonthlyRecommendation:
    """Represents a monthly financial recommendation"""
    month: int
    year: int
    recommendation_type: RecommendationType
    description: str
    suggested_amount: float
    priority: Priority
    rationale: str
    expected_outcome: str
    risk_level: str
    milestone_impact: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)


@dataclass
class ConfigurationMatrix:
    """Matrix showing different financial configuration paths over time"""
    person_id: str
    time_periods: List[str]
    scenarios: Dict[str, List[Dict]]  # scenario_name -> list of monthly configurations
    probability_weights: Dict[str, float]
    expected_outcomes: Dict[str, Dict]


class FinancialRecommendationEngine:
    """
    Analyzes the Omega mesh system and generates intelligent monthly recommendations
    for purchases, investments, and reallocations based on stochastic modeling
    """
    
    def __init__(self, mesh_engine: StochasticMeshEngine, accounting_engine: AccountingReconciliationEngine):
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

    def analyze_current_position(self, milestones: List[FinancialMilestone], 
                               profile_data: Dict) -> Dict:
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

    def _calculate_financial_health_score(self, emergency_ratio: float, debt_ratio: float, 
                                        investment_ratio: float, net_worth: float) -> int:
        """Calculate overall financial health score (0-100)"""
        score = 50  # Base score
        
        # Emergency fund component (0-25 points)
        if emergency_ratio >= 1.0:
            score += 25
        elif emergency_ratio >= 0.5:
            score += 15
        elif emergency_ratio >= 0.25:
            score += 5
        
        # Debt component (0-25 points)
        if debt_ratio <= 0.2:
            score += 25
        elif debt_ratio <= 0.4:
            score += 15
        elif debt_ratio <= 0.6:
            score += 5
        
        # Investment component (0-25 points)
        if investment_ratio >= 0.3:
            score += 25
        elif investment_ratio >= 0.15:
            score += 15
        elif investment_ratio >= 0.05:
            score += 5
        
        # Net worth component (0-25 points)
        if net_worth >= 500000:
            score += 25
        elif net_worth >= 100000:
            score += 15
        elif net_worth >= 25000:
            score += 10
        elif net_worth >= 0:
            score += 5
        
        return min(100, max(0, score))

    def _assess_urgency(self, milestones: List[FinancialMilestone]) -> Priority:
        """Assess the urgency of financial decisions based on milestones"""
        urgent_milestones = [m for m in milestones if (m.timestamp - datetime.now()).days < 365]
        critical_milestones = [m for m in milestones if (m.timestamp - datetime.now()).days < 180]
        
        if critical_milestones:
            return Priority.CRITICAL
        elif urgent_milestones:
            return Priority.HIGH
        else:
            return Priority.MEDIUM

    def generate_monthly_recommendations(self, milestones: List[FinancialMilestone],
                                       profile_data: Dict, months_ahead: int = 12) -> List[MonthlyRecommendation]:
        """Generate monthly recommendations for the next N months"""
        analysis = self.analyze_current_position(milestones, profile_data)
        recommendations = []
        
        current_date = datetime.now()
        monthly_surplus = analysis['key_metrics']['monthly_surplus']
        risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        
        for month_offset in range(months_ahead):
            target_date = current_date + timedelta(days=30 * month_offset)
            month_recommendations = self._generate_month_recommendations(
                target_date, analysis, milestones, profile_data, monthly_surplus
            )
            recommendations.extend(month_recommendations)
        
        return recommendations

    def _generate_month_recommendations(self, target_date: datetime, analysis: Dict,
                                      milestones: List[FinancialMilestone], profile_data: Dict,
                                      monthly_surplus: float) -> List[MonthlyRecommendation]:
        """Generate recommendations for a specific month"""
        month_recs = []
        risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        
        # Check for upcoming milestones
        upcoming_milestones = [
            m for m in milestones 
            if abs((m.timestamp - target_date).days) < 60
        ]
        
        # Emergency fund recommendation
        if analysis['key_metrics']['emergency_fund_ratio'] < 1.0:
            emergency_amount = min(monthly_surplus * 0.4, 2000)
            month_recs.append(MonthlyRecommendation(
                month=target_date.month,
                year=target_date.year,
                recommendation_type=RecommendationType.EMERGENCY_FUND,
                description=f"Increase emergency fund by ${emergency_amount:,.0f}",
                suggested_amount=emergency_amount,
                priority=Priority.HIGH,
                rationale="Emergency fund below 6 months of expenses",
                expected_outcome="Improved financial security and liquidity",
                risk_level="Very Low",
                alternative_options=["High-yield savings account", "Money market account"]
            ))
        
        # Debt payoff recommendations
        if analysis['debt_burden'] == 'High':
            debt_payment = min(monthly_surplus * 0.5, 1500)
            month_recs.append(MonthlyRecommendation(
                month=target_date.month,
                year=target_date.year,
                recommendation_type=RecommendationType.DEBT_PAYOFF,
                description=f"Extra debt payment of ${debt_payment:,.0f}",
                suggested_amount=debt_payment,
                priority=Priority.HIGH,
                rationale="High debt-to-income ratio affecting financial flexibility",
                expected_outcome="Reduced interest payments and improved credit",
                risk_level="Very Low",
                alternative_options=["Debt consolidation", "Balance transfer"]
            ))
        
        # Investment recommendations based on risk tolerance
        investment_amount = self._calculate_investment_allocation(
            monthly_surplus, risk_tolerance, analysis
        )
        
        if investment_amount > 100:
            investment_type = self._select_investment_type(risk_tolerance, analysis)
            month_recs.append(MonthlyRecommendation(
                month=target_date.month,
                year=target_date.year,
                recommendation_type=RecommendationType.INVESTMENT,
                description=f"Invest ${investment_amount:,.0f} in {investment_type}",
                suggested_amount=investment_amount,
                priority=Priority.MEDIUM,
                rationale=f"Based on {risk_tolerance.lower()} risk tolerance and current allocation",
                expected_outcome=f"Expected annual return: {self.investment_categories[investment_type]['expected_return']:.1%}",
                risk_level=self.investment_categories[investment_type]['risk'],
                alternative_options=self._get_alternative_investments(investment_type)
            ))
        
        # Milestone-specific recommendations
        for milestone in upcoming_milestones:
            milestone_amount = milestone.financial_impact or 0
            months_until = max(1, (milestone.timestamp - target_date).days / 30)
            monthly_savings_needed = milestone_amount / months_until
            
            if monthly_savings_needed > 0:
                month_recs.append(MonthlyRecommendation(
                    month=target_date.month,
                    year=target_date.year,
                    recommendation_type=RecommendationType.SAVINGS_INCREASE,
                    description=f"Save ${monthly_savings_needed:,.0f} for {milestone.event_type}",
                    suggested_amount=monthly_savings_needed,
                    priority=Priority.HIGH if months_until < 6 else Priority.MEDIUM,
                    rationale=f"Milestone approaching in {months_until:.1f} months",
                    expected_outcome=f"On track to meet {milestone.event_type} goal",
                    risk_level="Low",
                    milestone_impact=[milestone.event_type]
                ))
        
        # Reallocation recommendations
        reallocation_rec = self._generate_reallocation_recommendation(
            target_date, analysis, profile_data
        )
        if reallocation_rec:
            month_recs.append(reallocation_rec)
        
        return month_recs

    def _calculate_investment_allocation(self, monthly_surplus: float, risk_tolerance: str,
                                       analysis: Dict) -> float:
        """Calculate how much to allocate to investments"""
        allocation_percentages = {
            'Conservative': 0.3,
            'Moderate': 0.5,
            'Aggressive': 0.7,
            'Very Aggressive': 0.8
        }
        
        base_allocation = monthly_surplus * allocation_percentages.get(risk_tolerance, 0.5)
        
        # Adjust based on financial health
        health_score = analysis['financial_health_score']
        if health_score > 80:
            base_allocation *= 1.2
        elif health_score < 50:
            base_allocation *= 0.7
        
        return max(0, base_allocation)

    def _select_investment_type(self, risk_tolerance: str, analysis: Dict) -> str:
        """Select appropriate investment type based on risk tolerance"""
        investment_options = {
            'Conservative': ['emergency_fund', 'high_yield_savings', 'cds', 'government_bonds'],
            'Moderate': ['corporate_bonds', 'index_funds', 'real_estate'],
            'Aggressive': ['growth_stocks', 'index_funds', 'real_estate'],
            'Very Aggressive': ['growth_stocks', 'commodities', 'crypto', 'real_estate']
        }
        
        options = investment_options.get(risk_tolerance, investment_options['Moderate'])
        
        # Weight selection based on current portfolio
        current_investment_ratio = analysis['key_metrics']['investment_ratio']
        if current_investment_ratio < 0.1:
            # Start with safer options
            return random.choice(options[:2])
        else:
            return random.choice(options)

    def _get_alternative_investments(self, primary_type: str) -> List[str]:
        """Get alternative investment options"""
        all_types = list(self.investment_categories.keys())
        alternatives = [t for t in all_types if t != primary_type]
        return random.sample(alternatives, min(3, len(alternatives)))

    def _generate_reallocation_recommendation(self, target_date: datetime, analysis: Dict,
                                            profile_data: Dict) -> Optional[MonthlyRecommendation]:
        """Generate portfolio reallocation recommendations"""
        investment_ratio = analysis['key_metrics']['investment_ratio']
        risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        
        target_ratios = {
            'Conservative': {'stocks': 0.3, 'bonds': 0.6, 'cash': 0.1},
            'Moderate': {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1},
            'Aggressive': {'stocks': 0.8, 'bonds': 0.15, 'cash': 0.05},
            'Very Aggressive': {'stocks': 0.9, 'bonds': 0.05, 'cash': 0.05}
        }
        
        target = target_ratios.get(risk_tolerance, target_ratios['Moderate'])
        
        # Simplified reallocation logic
        if investment_ratio > 0.2:  # Only if significant portfolio exists
            return MonthlyRecommendation(
                month=target_date.month,
                year=target_date.year,
                recommendation_type=RecommendationType.REALLOCATION,
                description=f"Rebalance portfolio to {risk_tolerance.lower()} allocation",
                suggested_amount=0,  # No additional money needed
                priority=Priority.LOW,
                rationale="Periodic rebalancing to maintain target allocation",
                expected_outcome="Optimized risk-return profile",
                risk_level="Low",
                alternative_options=["Quarterly rebalancing", "Threshold-based rebalancing"]
            )
        
        return None

    def create_configuration_matrix(self, person_id: str, milestones: List[FinancialMilestone],
                                  profile_data: Dict, scenarios: int = 5) -> ConfigurationMatrix:
        """Create a matrix showing different financial configuration paths over time"""
        
        # Generate time periods (monthly for next 2 years)
        current_date = datetime.now()
        time_periods = []
        for i in range(24):
            period_date = current_date + timedelta(days=30 * i)
            time_periods.append(f"{period_date.year}-{period_date.month:02d}")
        
        # Generate different scenarios
        scenario_configurations = {}
        scenario_probabilities = {}
        expected_outcomes = {}
        
        for scenario_idx in range(scenarios):
            scenario_name = f"scenario_{scenario_idx + 1}"
            
            # Create variations in key parameters
            income_multiplier = random.uniform(0.8, 1.2)
            expense_multiplier = random.uniform(0.9, 1.1)
            market_performance = random.uniform(-0.2, 0.3)  # -20% to +30% annual
            
            scenario_config = []
            current_wealth = sum(profile_data.get('current_assets', {}).values())
            
            for period in time_periods:
                year, month = period.split('-')
                
                # Simulate monthly changes
                monthly_income = profile_data.get('base_income', 60000) / 12 * income_multiplier
                monthly_expenses = monthly_income * 0.7 * expense_multiplier
                monthly_surplus = monthly_income - monthly_expenses
                
                # Apply market performance to investments
                investment_growth = current_wealth * 0.3 * (1 + market_performance/12) - current_wealth * 0.3
                current_wealth += monthly_surplus + investment_growth
                
                # Generate configuration for this period
                config = self._generate_period_configuration(
                    int(year), int(month), current_wealth, monthly_surplus, 
                    milestones, profile_data, scenario_idx
                )
                
                scenario_config.append(config)
            
            scenario_configurations[scenario_name] = scenario_config
            scenario_probabilities[scenario_name] = 1.0 / scenarios  # Equal probability for now
            
            # Calculate expected outcomes for this scenario
            final_wealth = scenario_config[-1]['total_wealth']
            milestones_achieved = sum(1 for config in scenario_config if config['milestone_progress'] > 0)
            
            expected_outcomes[scenario_name] = {
                'final_wealth': final_wealth,
                'milestones_achieved': milestones_achieved,
                'average_monthly_surplus': sum(c['monthly_surplus'] for c in scenario_config) / len(scenario_config),
                'risk_adjusted_return': (final_wealth - current_wealth) / current_wealth
            }
        
        return ConfigurationMatrix(
            person_id=person_id,
            time_periods=time_periods,
            scenarios=scenario_configurations,
            probability_weights=scenario_probabilities,
            expected_outcomes=expected_outcomes
        )

    def _generate_period_configuration(self, year: int, month: int, wealth: float,
                                     surplus: float, milestones: List[FinancialMilestone],
                                     profile_data: Dict, scenario_idx: int) -> Dict:
        """Generate configuration for a specific time period"""
        
        # Asset allocation based on scenario
        allocation_variations = [
            {'cash': 0.3, 'stocks': 0.4, 'bonds': 0.2, 'real_estate': 0.1},
            {'cash': 0.2, 'stocks': 0.6, 'bonds': 0.1, 'real_estate': 0.1},
            {'cash': 0.4, 'stocks': 0.2, 'bonds': 0.3, 'real_estate': 0.1},
            {'cash': 0.1, 'stocks': 0.7, 'bonds': 0.1, 'real_estate': 0.1},
            {'cash': 0.25, 'stocks': 0.45, 'bonds': 0.2, 'real_estate': 0.1}
        ]
        
        allocation = allocation_variations[scenario_idx % len(allocation_variations)]
        
        # Check milestone progress
        period_date = datetime(year, month, 1)
        milestone_progress = 0
        active_milestones = []
        
        for milestone in milestones:
            if abs((milestone.timestamp - period_date).days) < 365:
                milestone_progress += 1
                active_milestones.append(milestone.event_type)
        
        # Generate recommended actions
        actions = self._generate_period_actions(surplus, allocation, active_milestones, profile_data)
        
        return {
            'year': year,
            'month': month,
            'total_wealth': wealth,
            'monthly_surplus': surplus,
            'asset_allocation': allocation,
            'milestone_progress': milestone_progress,
            'active_milestones': active_milestones,
            'recommended_actions': actions,
            'risk_level': self._calculate_period_risk(allocation),
            'liquidity_ratio': allocation['cash'] + allocation['bonds'] * 0.8,
            'growth_potential': allocation['stocks'] * 1.0 + allocation['real_estate'] * 0.8
        }

    def _generate_period_actions(self, surplus: float, allocation: Dict, 
                               milestones: List[str], profile_data: Dict) -> List[str]:
        """Generate recommended actions for a specific period"""
        actions = []
        risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        
        if surplus > 1000:
            if allocation['cash'] < 0.15:
                actions.append(f"Increase emergency fund by ${min(surplus * 0.3, 1000):.0f}")
            
            if allocation['stocks'] < 0.5 and risk_tolerance in ['Aggressive', 'Very Aggressive']:
                actions.append(f"Invest ${surplus * 0.4:.0f} in growth stocks")
            elif allocation['bonds'] < 0.3 and risk_tolerance == 'Conservative':
                actions.append(f"Invest ${surplus * 0.5:.0f} in bonds")
            else:
                actions.append(f"Invest ${surplus * 0.6:.0f} in index funds")
        
        if milestones:
            actions.append(f"Save ${surplus * 0.3:.0f} for upcoming {milestones[0]}")
        
        if len(actions) == 0:
            actions.append("Maintain current allocation")
        
        return actions

    def _calculate_period_risk(self, allocation: Dict) -> str:
        """Calculate risk level for a given allocation"""
        risk_score = (
            allocation['stocks'] * 0.8 +
            allocation['real_estate'] * 0.6 +
            allocation['bonds'] * 0.3 +
            allocation['cash'] * 0.1
        )
        
        if risk_score > 0.7:
            return "High"
        elif risk_score > 0.4:
            return "Medium"
        else:
            return "Low"

    def evaluate_mesh_effectiveness(self, milestones: List[FinancialMilestone],
                                  profile_data: Dict, recommendations: List[MonthlyRecommendation]) -> Dict:
        """Evaluate how well the mesh system is working"""
        
        mesh_status = self.mesh_engine.get_mesh_status()
        
        # Calculate metrics
        coverage_score = len([r for r in recommendations if r.priority in [Priority.HIGH, Priority.CRITICAL]]) / max(1, len(milestones))
        diversification_score = len(set(r.recommendation_type for r in recommendations)) / len(RecommendationType)
        
        # Mesh efficiency metrics
        path_efficiency = mesh_status['solidified_nodes'] / max(1, mesh_status['total_nodes'])
        visibility_utilization = mesh_status['visible_future_nodes'] / max(1, mesh_status['total_nodes'])
        
        # Prediction accuracy (simulated for demonstration)
        prediction_accuracy = random.uniform(0.75, 0.95)  # Would be calculated from historical data
        
        evaluation = {
            'overall_effectiveness': (coverage_score + diversification_score + path_efficiency) / 3,
            'recommendation_coverage': coverage_score,
            'strategy_diversification': diversification_score,
            'mesh_path_efficiency': path_efficiency,
            'mesh_visibility_utilization': visibility_utilization,
            'prediction_accuracy': prediction_accuracy,
            'recommendations_per_month': len(recommendations) / 12,
            'high_priority_percentage': len([r for r in recommendations if r.priority == Priority.HIGH]) / max(1, len(recommendations)),
            'risk_distribution': self._analyze_risk_distribution(recommendations),
            'action_feasibility': self._assess_action_feasibility(recommendations, profile_data)
        }
        
        return evaluation

    def _analyze_risk_distribution(self, recommendations: List[MonthlyRecommendation]) -> Dict[str, float]:
        """Analyze the risk distribution of recommendations"""
        risk_counts = {'Very Low': 0, 'Low': 0, 'Medium': 0, 'High': 0, 'Very High': 0}
        
        for rec in recommendations:
            risk_counts[rec.risk_level] = risk_counts.get(rec.risk_level, 0) + 1
        
        total = max(1, len(recommendations))
        return {risk: count / total for risk, count in risk_counts.items()}

    def _assess_action_feasibility(self, recommendations: List[MonthlyRecommendation], 
                                 profile_data: Dict) -> float:
        """Assess how feasible the recommended actions are"""
        monthly_income = profile_data.get('base_income', 60000) / 12
        
        feasible_count = 0
        for rec in recommendations:
            if rec.suggested_amount <= monthly_income * 0.5:  # Less than 50% of income
                feasible_count += 1
        
        return feasible_count / max(1, len(recommendations))

    def export_analysis_results(self, person_id: str, analysis: Dict, 
                              recommendations: List[MonthlyRecommendation],
                              config_matrix: ConfigurationMatrix,
                              evaluation: Dict, output_dir: str):
        """Export all analysis results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export recommendations
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                'month': rec.month,
                'year': rec.year,
                'type': rec.recommendation_type.value,
                'description': rec.description,
                'amount': rec.suggested_amount,
                'priority': rec.priority.name,
                'rationale': rec.rationale,
                'expected_outcome': rec.expected_outcome,
                'risk_level': rec.risk_level,
                'milestone_impact': rec.milestone_impact,
                'alternatives': rec.alternative_options
            })
        
        with open(f"{output_dir}/{person_id}_recommendations.json", 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        
        # Export configuration matrix
        matrix_data = {
            'person_id': config_matrix.person_id,
            'time_periods': config_matrix.time_periods,
            'scenarios': config_matrix.scenarios,
            'probabilities': config_matrix.probability_weights,
            'expected_outcomes': config_matrix.expected_outcomes
        }
        
        with open(f"{output_dir}/{person_id}_configuration_matrix.json", 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        # Export evaluation
        evaluation_data = {
            'person_id': person_id,
            'analysis_date': datetime.now().isoformat(),
            'financial_analysis': analysis,
            'mesh_evaluation': evaluation
        }
        
        with open(f"{output_dir}/{person_id}_evaluation.json", 'w') as f:
            json.dump(evaluation_data, f, indent=2)


if __name__ == "__main__":
    # This would normally be called from the main pipeline
    print("Financial Recommendation Engine initialized")
    print("Use this engine within the Omega mesh integration system")