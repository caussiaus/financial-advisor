#!/usr/bin/env python
# Dynamic Portfolio Engine Demonstration
# Shows real-time portfolio rebalancing with life events and market changes
# Author: ChatGPT 2025-01-16

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

from dynamic_portfolio_engine import DynamicPortfolioEngine

def create_sample_client_config():
    """Create a sample client configuration"""
    return {
        'income': 250000,
        'disposable_cash': 8000,
        'allowable_var': 0.15,
        'age': 42,
        'risk_profile': 3,  # Moderate risk tolerance
        'portfolio_value': 1500000,
        'target_allocation': {
            'equity': 0.58,
            'bonds': 0.32,
            'cash': 0.10
        }
    }

def simulate_market_scenarios():
    """Simulate different market scenarios over time"""
    scenarios = [
        # Normal market conditions
        {'equity_volatility': 0.16, 'bond_yields': 0.04, 'economic_outlook': 0.0, 'market_stress': 0.3},
        # Market stress period
        {'equity_volatility': 0.25, 'bond_yields': 0.03, 'economic_outlook': -0.3, 'market_stress': 0.8},
        # Recovery period
        {'equity_volatility': 0.18, 'bond_yields': 0.05, 'economic_outlook': 0.2, 'market_stress': 0.4},
        # Bull market
        {'equity_volatility': 0.14, 'bond_yields': 0.06, 'economic_outlook': 0.6, 'market_stress': 0.2},
        # Volatile period
        {'equity_volatility': 0.22, 'bond_yields': 0.04, 'economic_outlook': 0.1, 'market_stress': 0.6}
    ]
    return scenarios

def simulate_life_events():
    """Simulate life events that affect portfolio decisions"""
    events = [
        {
            'date': '2024-03-15',
            'type': 'career_change',
            'description': 'Promotion to senior role - increased income stability',
            'impact_score': 0.3  # Positive impact - can be more aggressive
        },
        {
            'date': '2024-06-20',
            'type': 'family_expansion',
            'description': 'Second child born - increased expenses and need for stability',
            'impact_score': -0.4  # Negative impact - more conservative
        },
        {
            'date': '2024-09-10',
            'type': 'market_crash',
            'description': 'Major market correction - client concerned about losses',
            'impact_score': -0.6  # Very negative impact - defensive positioning
        },
        {
            'date': '2024-12-05',
            'type': 'inheritance',
            'description': 'Received inheritance - increased portfolio value and flexibility',
            'impact_score': 0.2  # Positive impact - can take more risk
        },
        {
            'date': '2025-02-15',
            'type': 'health_concern',
            'description': 'Family health issue - need for more conservative approach',
            'impact_score': -0.3  # Negative impact - more conservative
        }
    ]
    return events

def run_dynamic_portfolio_simulation():
    """Run a comprehensive simulation of the dynamic portfolio engine"""
    print("🚀 Starting Dynamic Portfolio Engine Simulation")
    print("=" * 60)
    
    # Initialize the engine
    client_config = create_sample_client_config()
    engine = DynamicPortfolioEngine(client_config)
    
    print(f"📊 Initial Client Configuration:")
    print(f"   Age: {client_config['age']}")
    print(f"   Portfolio Value: ${client_config['portfolio_value']:,}")
    print(f"   Risk Profile: {client_config['risk_profile']}/5")
    print(f"   Initial Allocation: {client_config['target_allocation']}")
    print()
    
    # Get initial snapshot
    initial_snapshot = engine.get_portfolio_snapshot()
    print(f"📈 Initial Portfolio State:")
    print(f"   Equity: {initial_snapshot['portfolio']['equity']:.1%}")
    print(f"   Bonds: {initial_snapshot['portfolio']['bonds']:.1%}")
    print(f"   Cash: {initial_snapshot['portfolio']['cash']:.1%}")
    print(f"   Comfort Score: {initial_snapshot['comfort_metrics']['comfort_score']:.2f}")
    print()
    
    # Simulate market scenarios over time
    market_scenarios = simulate_market_scenarios()
    dates = ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01', '2025-01-01']
    
    print("🌍 Simulating Market Scenarios...")
    for i, (date, scenario) in enumerate(zip(dates, market_scenarios)):
        print(f"\n📅 {date} - Market Update:")
        print(f"   Equity Volatility: {scenario['equity_volatility']:.1%}")
        print(f"   Market Stress: {scenario['market_stress']:.1f}")
        print(f"   Economic Outlook: {scenario['economic_outlook']:+.1f}")
        
        # Update market data
        recommendations = engine.update_market_data(scenario)
        
        if recommendations['rebalancing_needed']:
            print("   ⚠️  Rebalancing Recommended!")
            for asset, change in recommendations['changes'].items():
                if abs(change) > 0.01:
                    print(f"      {asset.title()}: {change:+.1%}")
            print(f"   Comfort Score: {recommendations['comfort_metrics']['comfort_score']:.2f}")
        else:
            print("   ✅ No rebalancing needed")
        
        # Get snapshot
        engine.get_portfolio_snapshot()
    
    # Simulate life events
    life_events = simulate_life_events()
    print(f"\n👨‍👩‍👧‍👦 Simulating Life Events...")
    
    for event in life_events:
        print(f"\n📅 {event['date']} - {event['type'].replace('_', ' ').title()}:")
        print(f"   {event['description']}")
        print(f"   Impact Score: {event['impact_score']:+.1f}")
        
        # Add life event
        engine.add_life_event(
            event['type'],
            event['description'],
            event['impact_score'],
            event['date']
        )
        
        # Get recommendations after life event
        recommendations = engine.recommend_rebalancing()
        
        if recommendations['rebalancing_needed']:
            print("   ⚠️  Portfolio Adjustment Needed!")
            for asset, change in recommendations['changes'].items():
                if abs(change) > 0.01:
                    print(f"      {asset.title()}: {change:+.1%}")
        else:
            print("   ✅ No adjustment needed")
        
        # Get snapshot
        engine.get_portfolio_snapshot()
    
    # Create interactive dashboard
    print(f"\n📊 Creating Interactive Dashboard...")
    dashboard = engine.create_interactive_dashboard()
    
    # Save dashboard
    dashboard.write_html("dynamic_portfolio_dashboard.html")
    print("   ✅ Dashboard saved as 'dynamic_portfolio_dashboard.html'")
    
    # Export data for analysis
    data = engine.export_data()
    
    # Create summary report
    print(f"\n📋 Portfolio Evolution Summary:")
    print(f"   Total Snapshots: {len(data['portfolio_snapshots'])}")
    print(f"   Rebalancing Events: {len(data['rebalancing_history'])}")
    print(f"   Life Events: {len(data['life_events_log'])}")
    
    # Final portfolio state
    final_snapshot = data['portfolio_snapshots'][-1]
    print(f"\n🎯 Final Portfolio State:")
    print(f"   Equity: {final_snapshot['portfolio']['equity']:.1%}")
    print(f"   Bonds: {final_snapshot['portfolio']['bonds']:.1%}")
    print(f"   Cash: {final_snapshot['portfolio']['cash']:.1%}")
    print(f"   Comfort Score: {final_snapshot['comfort_metrics']['comfort_score']:.2f}")
    
    # Save detailed data
    with open('dynamic_portfolio_data.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print("   ✅ Detailed data saved as 'dynamic_portfolio_data.json'")
    
    return engine, dashboard

def create_contribution_analysis(engine):
    """Create contribution analysis showing how each factor affects the portfolio"""
    data = engine.export_data()
    
    # Create contribution timeline
    contributions = []
    
    for i, snapshot in enumerate(data['portfolio_snapshots']):
        if i == 0:
            continue
        
        prev_snapshot = data['portfolio_snapshots'][i-1]
        
        # Calculate changes
        equity_change = snapshot['portfolio']['equity'] - prev_snapshot['portfolio']['equity']
        bonds_change = snapshot['portfolio']['bonds'] - prev_snapshot['portfolio']['bonds']
        cash_change = snapshot['portfolio']['cash'] - prev_snapshot['portfolio']['cash']
        
        # Find life events in this period
        period_events = [e for e in data['life_events_log'] 
                        if prev_snapshot['date'] <= e['date'] <= snapshot['date']]
        
        contributions.append({
            'date': snapshot['date'],
            'equity_change': equity_change,
            'bonds_change': bonds_change,
            'cash_change': cash_change,
            'market_stress': snapshot['market_data']['market_stress'],
            'comfort_score': snapshot['comfort_metrics']['comfort_score'],
            'life_events': period_events
        })
    
    # Create contribution visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolio Allocation Changes', 'Contributing Factors'),
        vertical_spacing=0.15
    )
    
    dates = [c['date'] for c in contributions]
    equity_changes = [c['equity_change'] for c in contributions]
    bonds_changes = [c['bonds_change'] for c in contributions]
    cash_changes = [c['cash_change'] for c in contributions]
    market_stress = [c['market_stress'] for c in contributions]
    comfort_scores = [c['comfort_score'] for c in contributions]
    
    # Allocation changes
    fig.add_trace(
        go.Bar(x=dates, y=equity_changes, name='Equity Change', 
               marker_color='#1f77b4'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=dates, y=bonds_changes, name='Bonds Change', 
               marker_color='#ff7f0e'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=dates, y=cash_changes, name='Cash Change', 
               marker_color='#2ca02c'),
        row=1, col=1
    )
    
    # Contributing factors
    fig.add_trace(
        go.Scatter(x=dates, y=market_stress, name='Market Stress', 
                  line=dict(color='#d62728')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=comfort_scores, name='Comfort Score', 
                  line=dict(color='#9467bd')),
        row=2, col=1
    )
    
    # Add life event annotations
    for contribution in contributions:
        if contribution['life_events']:
            event = contribution['life_events'][0]  # Show first event if multiple
            fig.add_annotation(
                x=contribution['date'], y=0.8,
                text=f"📅 {event['type']}<br>{event['description'][:30]}...",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0, ay=-40,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
    
    fig.update_layout(
        title="Portfolio Contribution Analysis",
        height=600,
        showlegend=True,
        barmode='stack'
    )
    
    fig.update_yaxes(title_text="Allocation Change", row=1, col=1)
    fig.update_yaxes(title_text="Factor Values", row=2, col=1)
    
    return fig

if __name__ == "__main__":
    print("🎯 Dynamic Portfolio Engine Demonstration")
    print("=" * 50)
    
    # Run the simulation
    engine, dashboard = run_dynamic_portfolio_simulation()
    
    # Create contribution analysis
    print(f"\n📊 Creating Contribution Analysis...")
    contribution_fig = create_contribution_analysis(engine)
    contribution_fig.write_html("portfolio_contribution_analysis.html")
    print("   ✅ Contribution analysis saved as 'portfolio_contribution_analysis.html'")
    
    print(f"\n🎉 Demonstration Complete!")
    print(f"   Open 'dynamic_portfolio_dashboard.html' for the main dashboard")
    print(f"   Open 'portfolio_contribution_analysis.html' for contribution analysis")
    print(f"   Check 'dynamic_portfolio_data.json' for detailed data") 