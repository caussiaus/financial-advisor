#!/usr/bin/env python3
"""
Mesh-Market Integration System

Integrates the Omega mesh personal finance system with real market tracking
and backtesting to create a comprehensive financial planning and analysis tool.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from ..market_tracking_backtest import (
    MarketDataTracker, PersonalFinanceInvestmentMapper, BacktestEngine, 
    BacktestAnalyzer, PersonalFinanceAction, InvestmentDecision, BacktestResult
)
from ..core.stochastic_mesh_engine import StochasticMeshEngine
from ..enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone
from ..accounting_reconciliation import AccountingReconciliationEngine
from ..financial_recommendation_engine import FinancialRecommendationEngine

class MeshMarketIntegration:
    """
    Integrates Omega mesh personal finance system with market tracking and backtesting
    """
    
    def __init__(self):
        # Initialize market tracking components
        self.market_tracker = MarketDataTracker()
        self.mapper = PersonalFinanceInvestmentMapper()
        self.backtest_engine = BacktestEngine(self.market_tracker, self.mapper)
        self.analyzer = BacktestAnalyzer()
        
        # Initialize mesh components
        self.pdf_processor = EnhancedPDFProcessor()
        self.accounting_engine = AccountingReconciliationEngine()
        
        # Integration state
        self.mesh_actions = []
        self.market_decisions = []
        self.backtest_results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for integration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def process_mesh_to_market_actions(self, mesh_engine: StochasticMeshEngine) -> List[PersonalFinanceAction]:
        """
        Extract personal finance actions from the Omega mesh and convert them
        to market-trackable actions
        """
        self.logger.info("🔄 Processing mesh actions to market actions...")
        
        mesh_actions = []
        
        # Extract payment events from mesh
        for node_id, node in mesh_engine.memory_manager.batch_retrieve(list(mesh_engine.omega_mesh.nodes())):
            if node and hasattr(node, 'payment_opportunities'):
                for payment_id, payment_data in node.payment_opportunities.items():
                    action = self._convert_payment_to_action(payment_data, node)
                    if action:
                        mesh_actions.append(action)
        
        # Extract milestone events
        for milestone in mesh_engine.milestones:
            action = self._convert_milestone_to_action(milestone)
            if action:
                mesh_actions.append(action)
        
        # Extract state transitions
        for state_change in mesh_engine.state_history:
            action = self._convert_state_change_to_action(state_change)
            if action:
                mesh_actions.append(action)
        
        self.mesh_actions = mesh_actions
        self.logger.info(f"✅ Converted {len(mesh_actions)} mesh actions to market actions")
        
        return mesh_actions
    
    def _convert_payment_to_action(self, payment_data: Dict, node) -> Optional[PersonalFinanceAction]:
        """Convert mesh payment opportunity to market action"""
        try:
            amount = payment_data.get('amount', 0)
            if amount <= 0:
                return None
                
            action_type = self._determine_action_type(payment_data)
            category = payment_data.get('category', 'general')
            
            return PersonalFinanceAction(
                timestamp=node.timestamp,
                action_type=action_type,
                amount=amount,
                description=f"Payment: {payment_data.get('description', 'Unknown')}",
                category=category,
                confidence=payment_data.get('probability', 0.5),
                impact_duration=payment_data.get('duration_days', 30)
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Error converting payment to action: {e}")
            return None
    
    def _convert_milestone_to_action(self, milestone) -> Optional[PersonalFinanceAction]:
        """Convert mesh milestone to market action"""
        try:
            action_type = self._determine_action_type_from_milestone(milestone)
            amount = milestone.financial_impact if hasattr(milestone, 'financial_impact') else 0
            
            return PersonalFinanceAction(
                timestamp=milestone.timestamp,
                action_type=action_type,
                amount=amount,
                description=milestone.description,
                category=milestone.event_type,
                confidence=milestone.probability if hasattr(milestone, 'probability') else 0.7,
                impact_duration=365  # Milestones typically have long-term impact
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Error converting milestone to action: {e}")
            return None
    
    def _convert_state_change_to_action(self, state_change: Dict) -> Optional[PersonalFinanceAction]:
        """Convert mesh state change to market action"""
        try:
            # Extract key changes from state
            old_state = state_change.get('previous_state', {})
            new_state = state_change.get('current_state', {})
            
            # Calculate net change
            old_wealth = sum(old_state.values()) if isinstance(old_state, dict) else 0
            new_wealth = sum(new_state.values()) if isinstance(new_state, dict) else 0
            wealth_change = new_wealth - old_wealth
            
            if abs(wealth_change) < 1000:  # Ignore small changes
                return None
                
            action_type = 'income_increase' if wealth_change > 0 else 'income_decrease'
            
            return PersonalFinanceAction(
                timestamp=state_change.get('timestamp', datetime.now()),
                action_type=action_type,
                amount=abs(wealth_change),
                description=f"State change: {state_change.get('description', 'Portfolio adjustment')}",
                category='portfolio_change',
                confidence=0.8,
                impact_duration=90
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Error converting state change to action: {e}")
            return None
    
    def _determine_action_type(self, payment_data: Dict) -> str:
        """Determine action type from payment data"""
        category = payment_data.get('category', '').lower()
        amount = payment_data.get('amount', 0)
        
        if 'income' in category or 'salary' in category:
            return 'income_increase'
        elif 'expense' in category or 'cost' in category:
            return 'major_expense'
        elif 'debt' in category or 'loan' in category:
            return 'debt_payment'
        elif 'milestone' in category or 'goal' in category:
            return 'milestone_achievement'
        else:
            # Default based on amount
            if amount > 10000:
                return 'major_expense'
            elif amount < -5000:
                return 'income_decrease'
            else:
                return 'milestone_achievement'
    
    def _determine_action_type_from_milestone(self, milestone) -> str:
        """Determine action type from milestone"""
        event_type = milestone.event_type.lower()
        
        if 'education' in event_type:
            return 'major_expense'
        elif 'career' in event_type or 'promotion' in event_type:
            return 'income_increase'
        elif 'family' in event_type or 'wedding' in event_type:
            return 'major_expense'
        elif 'housing' in event_type or 'mortgage' in event_type:
            return 'major_expense'
        elif 'investment' in event_type:
            return 'milestone_achievement'
        else:
            return 'milestone_achievement'
    
    def run_mesh_backtest(self, 
                         mesh_engine: StochasticMeshEngine,
                         start_date: datetime,
                         end_date: datetime,
                         initial_portfolio: Dict[str, float],
                         risk_tolerance: str = 'moderate') -> BacktestResult:
        """
        Run backtest using actions extracted from the Omega mesh
        """
        self.logger.info("🚀 Starting mesh-based backtest...")
        
        # Extract actions from mesh
        mesh_actions = self.process_mesh_to_market_actions(mesh_engine)
        
        if not mesh_actions:
            self.logger.warning("⚠️ No actions found in mesh, using sample actions")
            mesh_actions = self._create_sample_actions()
        
        # Run backtest
        result = self.backtest_engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_portfolio=initial_portfolio,
            personal_finance_actions=mesh_actions,
            risk_tolerance=risk_tolerance,
            rebalance_frequency='monthly'
        )
        
        self.backtest_results.append(result)
        self.logger.info(f"✅ Mesh backtest completed. Final value: ${result.final_portfolio_value:,.2f}")
        
        return result
    
    def _create_sample_actions(self) -> List[PersonalFinanceAction]:
        """Create sample actions if mesh is empty"""
        return [
            PersonalFinanceAction(
                timestamp=datetime(2020, 3, 15),
                action_type='income_increase',
                amount=15000,
                description='Sample promotion and salary increase',
                category='career',
                confidence=0.9,
                impact_duration=365
            ),
            PersonalFinanceAction(
                timestamp=datetime(2020, 6, 10),
                action_type='major_expense',
                amount=25000,
                description='Sample home renovation project',
                category='housing',
                confidence=0.8,
                impact_duration=180
            )
        ]
    
    def analyze_mesh_performance(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Analyze how well the mesh-based decisions performed
        """
        self.logger.info("📊 Analyzing mesh performance...")
        
        # Basic analysis
        analysis = self.analyzer.analyze_backtest_result(result)
        
        # Mesh-specific analysis
        mesh_analysis = {
            'mesh_actions_processed': len(self.mesh_actions),
            'action_types_distribution': self._analyze_action_distribution(),
            'mesh_decision_effectiveness': self._analyze_mesh_decisions(result),
            'market_outperformance': analysis['market_analysis']['outperformance'],
            'risk_adjusted_performance': analysis['risk_analysis']['risk_adjusted_metrics']
        }
        
        # Combine analyses
        full_analysis = {**analysis, 'mesh_specific': mesh_analysis}
        
        return full_analysis
    
    def _analyze_action_distribution(self) -> Dict[str, int]:
        """Analyze distribution of action types from mesh"""
        distribution = {}
        for action in self.mesh_actions:
            action_type = action.action_type
            distribution[action_type] = distribution.get(action_type, 0) + 1
        return distribution
    
    def _analyze_mesh_decisions(self, result: BacktestResult) -> Dict[str, float]:
        """Analyze effectiveness of mesh-based decisions"""
        if not result.action_decisions:
            return {}
        
        # Group decisions by trigger action type
        decision_effectiveness = {}
        for decision in result.action_decisions:
            trigger_type = decision.trigger_action.action_type
            if trigger_type not in decision_effectiveness:
                decision_effectiveness[trigger_type] = []
            decision_effectiveness[trigger_type].append(decision.expected_return)
        
        # Calculate average returns by trigger type
        effectiveness = {}
        for trigger_type, returns in decision_effectiveness.items():
            effectiveness[trigger_type] = sum(returns) / len(returns)
        
        return effectiveness
    
    def generate_mesh_market_report(self, result: BacktestResult, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report combining mesh and market analysis
        """
        self.logger.info("📋 Generating mesh-market integration report...")
        
        # Create visualizations
        visualizations = self.analyzer.generate_visualizations(result)
        
        # Generate report
        report = {
            'backtest_summary': {
                'period': f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
                'initial_value': f"${result.initial_portfolio_value:,.2f}",
                'final_value': f"${result.final_portfolio_value:,.2f}",
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}"
            },
            'mesh_analysis': {
                'actions_processed': len(self.mesh_actions),
                'decisions_made': len(result.action_decisions),
                'action_distribution': self._analyze_action_distribution(),
                'decision_effectiveness': self._analyze_mesh_decisions(result)
            },
            'market_analysis': analysis['market_analysis'],
            'performance_analysis': analysis['performance_summary'],
            'risk_analysis': analysis['risk_analysis'],
            'recommendations': self._generate_mesh_recommendations(result, analysis)
        }
        
        return report
    
    def _generate_mesh_recommendations(self, result: BacktestResult, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on mesh-market integration"""
        recommendations = []
        
        # Performance-based recommendations
        if result.sharpe_ratio < 0.5:
            recommendations.append("Consider reducing portfolio volatility through better diversification")
        
        if result.max_drawdown > 0.2:
            recommendations.append("Implement stricter risk management to reduce maximum drawdown")
        
        # Action-based recommendations
        action_distribution = self._analyze_action_distribution()
        if action_distribution.get('major_expense', 0) > 3:
            recommendations.append("High frequency of major expenses - consider building larger emergency fund")
        
        if action_distribution.get('income_increase', 0) > 2:
            recommendations.append("Multiple income increases detected - consider increasing equity allocation")
        
        # Market-based recommendations
        market_outperformance = analysis['market_analysis'].get('outperformance', 0)
        if market_outperformance < 0:
            recommendations.append("Portfolio underperforming market - review asset allocation strategy")
        elif market_outperformance > 0.05:
            recommendations.append("Strong outperformance - consider locking in gains through rebalancing")
        
        return recommendations
    
    def create_mesh_market_dashboard(self, result: BacktestResult, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create interactive dashboard data for mesh-market integration
        """
        self.logger.info("📊 Creating mesh-market dashboard...")
        
        # Generate visualizations
        visualizations = self.analyzer.generate_visualizations(result)
        
        # Create dashboard data
        dashboard_data = {
            'portfolio_timeline': self._create_portfolio_timeline_data(result),
            'action_analysis': self._create_action_analysis_data(result),
            'performance_metrics': self._create_performance_metrics_data(result),
            'mesh_insights': self._create_mesh_insights_data(result),
            'visualizations': {
                'portfolio_timeline': visualizations.get('portfolio_timeline', {}),
                'returns_distribution': visualizations.get('returns_distribution', {}),
                'drawdown_chart': visualizations.get('drawdown_chart', {}),
                'decision_analysis': visualizations.get('decision_analysis', {})
            }
        }
        
        return dashboard_data
    
    def _create_portfolio_timeline_data(self, result: BacktestResult) -> List[Dict]:
        """Create portfolio timeline data for dashboard"""
        # Simplified timeline - in practice, this would use actual portfolio history
        timeline_data = []
        
        # Create monthly data points
        current_date = result.start_date
        current_value = result.initial_portfolio_value
        
        while current_date <= result.end_date:
            # Simulate monthly growth
            monthly_return = result.annualized_return / 12
            current_value *= (1 + monthly_return)
            
            timeline_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'portfolio_value': current_value,
                'cumulative_return': (current_value - result.initial_portfolio_value) / result.initial_portfolio_value
            })
            
            current_date += timedelta(days=30)
        
        return timeline_data
    
    def _create_action_analysis_data(self, result: BacktestResult) -> Dict[str, Any]:
        """Create action analysis data for dashboard"""
        action_distribution = self._analyze_action_distribution()
        decision_effectiveness = self._analyze_mesh_decisions(result)
        
        return {
            'action_distribution': action_distribution,
            'decision_effectiveness': decision_effectiveness,
            'total_actions': len(self.mesh_actions),
            'total_decisions': len(result.action_decisions),
            'profitable_decisions': len([d for d in result.action_decisions if d.expected_return > 0])
        }
    
    def _create_performance_metrics_data(self, result: BacktestResult) -> Dict[str, Any]:
        """Create performance metrics data for dashboard"""
        return {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profitable_trades': result.profitable_trades
        }
    
    def _create_mesh_insights_data(self, result: BacktestResult) -> Dict[str, Any]:
        """Create mesh-specific insights data for dashboard"""
        return {
            'mesh_actions_processed': len(self.mesh_actions),
            'action_types': list(set(action.action_type for action in self.mesh_actions)),
            'average_action_confidence': np.mean([action.confidence for action in self.mesh_actions]) if self.mesh_actions else 0,
            'mesh_decision_success_rate': result.profitable_trades / result.total_trades if result.total_trades > 0 else 0
        }

def run_comprehensive_mesh_market_analysis():
    """
    Run comprehensive analysis integrating Omega mesh with market tracking
    """
    print("🚀 Starting Comprehensive Mesh-Market Integration Analysis...")
    
    # Initialize integration system
    integration = MeshMarketIntegration()
    
    # Create sample mesh engine (in practice, this would be your actual mesh)
    initial_state = {
        'cash': 50000,
        'investments': 100000,
        'debts': 20000
    }
    
    mesh_engine = StochasticMeshEngine(initial_state)
    
    # Create sample milestones
    milestones = [
        FinancialMilestone(
            timestamp=datetime(2020, 6, 15),
            event_type='education',
            description='Graduate school tuition payment',
            financial_impact=25000,
            probability=0.9,
            entity='client'
        ),
        FinancialMilestone(
            timestamp=datetime(2021, 3, 10),
            event_type='career',
            description='Promotion and salary increase',
            financial_impact=15000,
            probability=0.8,
            entity='client'
        ),
        FinancialMilestone(
            timestamp=datetime(2022, 8, 20),
            event_type='housing',
            description='Home renovation project',
            financial_impact=30000,
            probability=0.7,
            entity='client'
        )
    ]
    
    # Initialize mesh with milestones
    mesh_engine.initialize_mesh(milestones)
    
    # Run mesh-based backtest
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_portfolio = {
        'equity': 60000,
        'bonds': 30000,
        'cash': 10000
    }
    
    result = integration.run_mesh_backtest(
        mesh_engine=mesh_engine,
        start_date=start_date,
        end_date=end_date,
        initial_portfolio=initial_portfolio,
        risk_tolerance='moderate'
    )
    
    # Analyze results
    analysis = integration.analyze_mesh_performance(result)
    
    # Generate report
    report = integration.generate_mesh_market_report(result, analysis)
    
    # Create dashboard
    dashboard = integration.create_mesh_market_dashboard(result, analysis)
    
    # Print results
    print("\n📊 Mesh-Market Integration Results:")
    print(f"Initial Portfolio Value: ${result.initial_portfolio_value:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    
    print("\n🎯 Mesh Actions Processed:")
    print(f"Total Actions: {len(integration.mesh_actions)}")
    action_distribution = integration._analyze_action_distribution()
    for action_type, count in action_distribution.items():
        print(f"  {action_type}: {count}")
    
    print("\n💡 Recommendations:")
    recommendations = integration._generate_mesh_recommendations(result, analysis)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Save results
    results_data = {
        'backtest_result': {
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_portfolio_value': result.initial_portfolio_value,
            'final_portfolio_value': result.final_portfolio_value,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate
        },
        'analysis': analysis,
        'report': report,
        'dashboard': dashboard
    }
    
    with open('mesh_market_integration_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("\n✅ Analysis completed! Results saved to mesh_market_integration_results.json")
    
    return result, analysis, report, dashboard

if __name__ == "__main__":
    run_comprehensive_mesh_market_analysis() 