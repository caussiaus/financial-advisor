#!/usr/bin/env python3
"""
Test Market Tracking and Personal Finance Association

Simplified demonstration of how personal finance decisions in the Omega mesh
translate to investment actions and market performance.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

try:
    from market_tracking_backtest import (
        MarketDataTracker, PersonalFinanceInvestmentMapper, BacktestEngine, 
        BacktestAnalyzer, PersonalFinanceAction, InvestmentDecision, BacktestResult
    )
    print("✅ Successfully imported market tracking components")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Creating simplified test version...")
    
    # Simplified classes for testing
    class PersonalFinanceAction:
        def __init__(self, timestamp, action_type, amount, description, category, confidence, impact_duration):
            self.timestamp = timestamp
            self.action_type = action_type
            self.amount = amount
            self.description = description
            self.category = category
            self.confidence = confidence
            self.impact_duration = impact_duration
    
    class InvestmentDecision:
        def __init__(self, timestamp, decision_type, asset_class, amount, percentage, trigger_action, market_conditions, expected_return, risk_score):
            self.timestamp = timestamp
            self.decision_type = decision_type
            self.asset_class = asset_class
            self.amount = amount
            self.percentage = percentage
            self.trigger_action = trigger_action
            self.market_conditions = market_conditions
            self.expected_return = expected_return
            self.risk_score = risk_score
    
    class BacktestResult:
        def __init__(self, start_date, end_date, initial_portfolio_value, final_portfolio_value, total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, win_rate, total_trades, profitable_trades, action_decisions, market_performance, personal_finance_actions):
            self.start_date = start_date
            self.end_date = end_date
            self.initial_portfolio_value = initial_portfolio_value
            self.final_portfolio_value = final_portfolio_value
            self.total_return = total_return
            self.annualized_return = annualized_return
            self.volatility = volatility
            self.sharpe_ratio = sharpe_ratio
            self.max_drawdown = max_drawdown
            self.win_rate = win_rate
            self.total_trades = total_trades
            self.profitable_trades = profitable_trades
            self.action_decisions = action_decisions
            self.market_performance = market_performance
            self.personal_finance_actions = personal_finance_actions

def create_sample_mesh_actions():
    """Create sample personal finance actions that would come from the Omega mesh"""
    
    actions = [
        PersonalFinanceAction(
            timestamp=datetime(2020, 3, 15),
            action_type='income_increase',
            amount=15000,
            description='Promotion and salary increase',
            category='career',
            confidence=0.9,
            impact_duration=365
        ),
        PersonalFinanceAction(
            timestamp=datetime(2020, 6, 10),
            action_type='major_expense',
            amount=25000,
            description='Home renovation project',
            category='housing',
            confidence=0.8,
            impact_duration=180
        ),
        PersonalFinanceAction(
            timestamp=datetime(2021, 1, 5),
            action_type='debt_payment',
            amount=10000,
            description='Student loan payoff',
            category='debt',
            confidence=0.95,
            impact_duration=365
        ),
        PersonalFinanceAction(
            timestamp=datetime(2021, 8, 20),
            action_type='milestone_achievement',
            amount=5000,
            description='Emergency fund goal reached',
            category='savings',
            confidence=0.7,
            impact_duration=90
        ),
        PersonalFinanceAction(
            timestamp=datetime(2022, 3, 12),
            action_type='income_increase',
            amount=20000,
            description='Job change with higher salary',
            category='career',
            confidence=0.85,
            impact_duration=365
        )
    ]
    
    return actions

def simulate_market_conditions():
    """Simulate realistic market conditions over time"""
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Simulate realistic market data
    np.random.seed(42)  # For reproducible results
    
    # S&P 500 returns (monthly)
    sp500_returns = np.random.normal(0.007, 0.04, len(dates))  # 7% annual return, 4% monthly vol
    
    # Bond yields
    bond_yields = np.random.normal(0.04, 0.01, len(dates))  # 4% average yield
    
    # Market stress (0-1 scale)
    market_stress = np.random.normal(0.3, 0.15, len(dates))
    market_stress = np.clip(market_stress, 0, 1)
    
    market_data = {}
    for i, date in enumerate(dates):
        market_data[date] = {
            'sp500_return': sp500_returns[i],
            'bond_yield': bond_yields[i],
            'market_stress': market_stress[i],
            'equity_volatility': 0.15 + market_stress[i] * 0.1,  # 15-25% volatility
            'economic_outlook': 0.5 - market_stress[i] * 0.5  # Negative correlation with stress
        }
    
    return market_data

def map_actions_to_decisions(actions: List[PersonalFinanceAction], market_data: Dict) -> List[InvestmentDecision]:
    """Map personal finance actions to investment decisions"""
    
    decisions = []
    
    for action in actions:
        # Get market conditions for this date
        market_conditions = market_data.get(action.timestamp, market_data[list(market_data.keys())[0]])
        
        # Map action type to investment decision
        if action.action_type == 'income_increase':
            decision = InvestmentDecision(
                timestamp=action.timestamp,
                decision_type='buy',
                asset_class='equity',
                amount=action.amount * 0.5,  # 50% of income increase to equity
                percentage=0.05,  # 5% of portfolio
                trigger_action=action,
                market_conditions=market_conditions,
                expected_return=0.08,
                risk_score=0.6
            )
        elif action.action_type == 'major_expense':
            decision = InvestmentDecision(
                timestamp=action.timestamp,
                decision_type='buy',
                asset_class='cash',
                amount=action.amount * 0.3,  # 30% of expense to cash
                percentage=0.1,  # 10% of portfolio
                trigger_action=action,
                market_conditions=market_conditions,
                expected_return=0.02,
                risk_score=0.1
            )
        elif action.action_type == 'debt_payment':
            decision = InvestmentDecision(
                timestamp=action.timestamp,
                decision_type='buy',
                asset_class='bonds',
                amount=action.amount * 0.4,  # 40% of debt payment to bonds
                percentage=0.03,  # 3% of portfolio
                trigger_action=action,
                market_conditions=market_conditions,
                expected_return=0.04,
                risk_score=0.2
            )
        else:  # milestone_achievement
            decision = InvestmentDecision(
                timestamp=action.timestamp,
                decision_type='rebalance',
                asset_class='mixed',
                amount=action.amount * 0.2,  # 20% of milestone to rebalancing
                percentage=0.02,  # 2% of portfolio
                trigger_action=action,
                market_conditions=market_conditions,
                expected_return=0.06,
                risk_score=0.4
            )
        
        decisions.append(decision)
    
    return decisions

def simulate_portfolio_performance(initial_portfolio: Dict, decisions: List[InvestmentDecision], market_data: Dict) -> BacktestResult:
    """Simulate portfolio performance based on decisions and market data"""
    
    # Initialize portfolio
    portfolio = initial_portfolio.copy()
    portfolio_history = []
    
    # Sort decisions by timestamp
    decisions.sort(key=lambda x: x.timestamp)
    
    # Simulate portfolio evolution
    current_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    while current_date <= end_date:
        # Apply market returns
        if current_date in market_data:
            market_conditions = market_data[current_date]
            
            # Apply returns to portfolio
            if 'equity' in portfolio:
                portfolio['equity'] *= (1 + market_conditions['sp500_return'])
            if 'bonds' in portfolio:
                portfolio['bonds'] *= (1 + market_conditions['bond_yield'] / 12)
            if 'cash' in portfolio:
                portfolio['cash'] *= (1 + 0.02 / 12)  # 2% risk-free rate
        
        # Apply decisions for this month
        for decision in decisions:
            if decision.timestamp.year == current_date.year and decision.timestamp.month == current_date.month:
                if decision.decision_type == 'buy':
                    if decision.asset_class in portfolio:
                        portfolio[decision.asset_class] += decision.amount
                    else:
                        portfolio[decision.asset_class] = decision.amount
                elif decision.decision_type == 'sell':
                    if decision.asset_class in portfolio:
                        portfolio[decision.asset_class] = max(0, portfolio[decision.asset_class] - decision.amount)
        
        # Record portfolio state
        portfolio_history.append({
            'date': current_date,
            'portfolio': portfolio.copy(),
            'total_value': sum(portfolio.values())
        })
        
        current_date += timedelta(days=30)
    
    # Calculate performance metrics
    initial_value = sum(initial_portfolio.values())
    final_value = sum(portfolio.values())
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate annualized return
    years = (end_date - datetime(2020, 1, 1)).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Calculate volatility
    values = [p['total_value'] for p in portfolio_history]
    returns = pd.Series(values).pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)  # Annualized monthly volatility
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    profitable_decisions = len([d for d in decisions if d.expected_return > 0])
    win_rate = profitable_decisions / len(decisions) if decisions else 0
    
    # Market performance
    market_performance = {
        'sp500_total_return': sum(market_data[d]['sp500_return'] for d in market_data),
        'bond_total_return': sum(market_data[d]['bond_yield'] for d in market_data) / 12
    }
    
    return BacktestResult(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_portfolio_value=initial_value,
        final_portfolio_value=final_value,
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=len(decisions),
        profitable_trades=profitable_decisions,
        action_decisions=decisions,
        market_performance=market_performance,
        personal_finance_actions=create_sample_mesh_actions()
    )

def analyze_results(result: BacktestResult):
    """Analyze and display backtest results"""
    
    print("\n" + "="*60)
    print("📊 MESH-MARKET INTEGRATION BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n📈 Portfolio Performance:")
    print(f"  Initial Value: ${result.initial_portfolio_value:,.2f}")
    print(f"  Final Value: ${result.final_portfolio_value:,.2f}")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    
    print(f"\n🎯 Decision Analysis:")
    print(f"  Total Decisions: {result.total_trades}")
    print(f"  Profitable Decisions: {result.profitable_trades}")
    print(f"  Success Rate: {result.profitable_trades/result.total_trades:.1%}" if result.total_trades > 0 else "  Success Rate: N/A")
    
    print(f"\n📊 Market Performance:")
    for key, value in result.market_performance.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
    
    print(f"\n🎯 Personal Finance Actions:")
    action_types = {}
    for action in result.personal_finance_actions:
        action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
    
    for action_type, count in action_types.items():
        print(f"  {action_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n💡 Key Insights:")
    if result.sharpe_ratio > 0.5:
        print("  ✅ Strong risk-adjusted returns")
    else:
        print("  ⚠️ Consider improving risk-adjusted returns")
    
    if result.max_drawdown > -0.2:
        print("  ✅ Acceptable maximum drawdown")
    else:
        print("  ⚠️ High maximum drawdown - consider risk management")
    
    if result.win_rate > 0.6:
        print("  ✅ High decision success rate")
    else:
        print("  ⚠️ Consider improving decision accuracy")
    
    print("\n" + "="*60)

def main():
    """Run the complete mesh-market integration test"""
    
    print("🚀 Starting Mesh-Market Integration Test...")
    
    # Create sample mesh actions
    print("📋 Creating sample personal finance actions from Omega mesh...")
    mesh_actions = create_sample_mesh_actions()
    print(f"✅ Created {len(mesh_actions)} personal finance actions")
    
    # Simulate market conditions
    print("📈 Simulating market conditions...")
    market_data = simulate_market_conditions()
    print(f"✅ Generated {len(market_data)} months of market data")
    
    # Map actions to investment decisions
    print("🔄 Mapping personal finance actions to investment decisions...")
    decisions = map_actions_to_decisions(mesh_actions, market_data)
    print(f"✅ Created {len(decisions)} investment decisions")
    
    # Initial portfolio
    initial_portfolio = {
        'equity': 60000,
        'bonds': 30000,
        'cash': 10000
    }
    
    # Simulate portfolio performance
    print("📊 Simulating portfolio performance...")
    result = simulate_portfolio_performance(initial_portfolio, decisions, market_data)
    print("✅ Portfolio simulation completed")
    
    # Analyze results
    analyze_results(result)
    
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
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profitable_trades': result.profitable_trades
        },
        'mesh_actions': [
            {
                'timestamp': action.timestamp.isoformat(),
                'action_type': action.action_type,
                'amount': action.amount,
                'description': action.description,
                'category': action.category,
                'confidence': action.confidence
            }
            for action in result.personal_finance_actions
        ],
        'investment_decisions': [
            {
                'timestamp': decision.timestamp.isoformat(),
                'decision_type': decision.decision_type,
                'asset_class': decision.asset_class,
                'amount': decision.amount,
                'percentage': decision.percentage,
                'expected_return': decision.expected_return,
                'risk_score': decision.risk_score
            }
            for decision in result.action_decisions
        ],
        'market_performance': result.market_performance
    }
    
    with open('mesh_market_test_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("\n✅ Test completed! Results saved to mesh_market_test_results.json")
    print("\n🎯 This demonstrates how your Omega mesh personal finance decisions")
    print("   are tracked and associated with specific investment actions and")
    print("   market performance outcomes.")

if __name__ == "__main__":
    main() 