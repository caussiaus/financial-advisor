#!/usr/bin/env python
"""
Stress Monitoring System Demo for Portfolio Managers
Author: ChatGPT 2025-07-16

This script demonstrates how portfolio managers can use the stress monitoring
system to provide real-time guidance and automated interventions for clients.

Usage:
    python stress_monitoring_demo.py

Features demonstrated:
- Real-time stress monitoring
- Automated alert generation
- Intervention recommendations
- Portfolio rebalancing
- Client communication templates
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import our stress monitoring system
from ips_model import (
    StressMonitor, simulate_intervention_impact,
    generate_stress_monitoring_report, STRESS_MONITORING,
    INTERVENTION_IMPACTS, PORTFOLIO_RULES
)

def load_client_data():
    """Load sample client configurations and historical data"""
    clients = [
        {
            'client_id': 'CLIENT_001',
            'name': 'John & Sarah Mitchell',
            'baseline_config': {
                'ED_PATH': 'JohnsHopkins',
                'HEL_WORK': 'Full-time',
                'BONUS_PCT': 0.30,
                'DON_STYLE': 0,
                'RISK_BAND': 3,
                'FX_SCENARIO': 'Base'
            },
            'baseline_stress': 0.435,
            'portfolio_value': 250000,
            'monthly_expenses': 8500,
            'last_review': '2024-10-01'
        },
        {
            'client_id': 'CLIENT_002',
            'name': 'David & Lisa Chen',
            'baseline_config': {
                'ED_PATH': 'McGill',
                'HEL_WORK': 'Part-time',
                'BONUS_PCT': 0.15,
                'DON_STYLE': 1,
                'RISK_BAND': 2,
                'FX_SCENARIO': 'Base'
            },
            'baseline_stress': 0.147,
            'portfolio_value': 180000,
            'monthly_expenses': 6200,
            'last_review': '2024-11-15'
        }
    ]
    return clients

def simulate_market_scenario(scenario_type='mild_stress'):
    """Simulate different market/economic scenarios"""
    scenarios = {
        'mild_stress': {
            'market_decline': -0.10,
            'income_impact': -0.05,
            'expense_increase': 0.03,
            'description': 'Mild market correction with minor income impact'
        },
        'moderate_stress': {
            'market_decline': -0.20,
            'income_impact': -0.10,
            'expense_increase': 0.05,
            'description': 'Moderate recession with income reduction'
        },
        'severe_stress': {
            'market_decline': -0.35,
            'income_impact': -0.20,
            'expense_increase': 0.08,
            'description': 'Severe economic downturn'
        },
        'inflation_spike': {
            'market_decline': -0.05,
            'income_impact': 0.02,
            'expense_increase': 0.15,
            'description': 'Inflation spike scenario'
        }
    }
    return scenarios.get(scenario_type, scenarios['mild_stress'])

def calculate_stress_increase(client, market_scenario):
    """Calculate how market scenario impacts client stress"""
    # Base stress increase from market scenario
    base_increase = abs(market_scenario['market_decline']) * 0.3
    
    # Additional stress from income/expense impacts
    income_stress = abs(market_scenario['income_impact']) * 0.4
    expense_stress = market_scenario['expense_increase'] * 0.3
    
    # Risk tolerance adjustment
    risk_multiplier = 1.5 if client['baseline_config']['RISK_BAND'] == 1 else 1.0
    
    total_increase = (base_increase + income_stress + expense_stress) * risk_multiplier
    return min(total_increase, 0.5)  # Cap at 50% increase

def generate_client_communication(client, monitor_response, scenario):
    """Generate client communication templates"""
    
    stress_level = monitor_response['stress_level']
    recommendations = monitor_response['recommendations'][:3]
    
    # Determine communication urgency
    if stress_level > STRESS_MONITORING['stress_critical_threshold']:
        urgency = "URGENT"
        subject = f"🚨 Immediate Action Required - Financial Stress Alert"
    elif stress_level > STRESS_MONITORING['stress_alert_threshold']:
        urgency = "HIGH"
        subject = f"⚠️ Portfolio Review Recommended - Stress Level Elevated"
    else:
        urgency = "NORMAL"
        subject = f"📊 Quarterly Portfolio Update"
    
    message = f"""
Dear {client['name']},

{scenario['description']} has impacted your financial plan. Here's your current status:

📊 FINANCIAL STRESS UPDATE:
   Current Stress Level: {stress_level:.1%}
   Change from Baseline: +{monitor_response['stress_change']:.1%}
   Status: {urgency}

💡 RECOMMENDED ACTIONS:
"""
    
    for i, rec in enumerate(recommendations):
        message += f"""
   {i+1}. {rec['description']}
      • Estimated stress reduction: {rec['estimated_stress_reduction']:.1%}
      • Implementation time: {rec['time_to_implement_months']} months
      • Feasibility: {rec['feasibility']:.0%}
"""
    
    if monitor_response['auto_interventions']:
        message += f"""
🤖 AUTOMATIC ADJUSTMENTS MADE:
"""
        for auto_int in monitor_response['auto_interventions']:
            message += f"   ✅ {auto_int['description']}\n"
    
    message += f"""
📱 NEXT STEPS:
   1. Review attached detailed analysis
   2. Schedule consultation if stress level remains elevated
   3. Monitor implementation of recommended actions
   4. Next review scheduled: {(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')}

Best regards,
Your Portfolio Management Team
"""
    
    return {
        'subject': subject,
        'urgency': urgency,
        'message': message,
        'attachments': ['detailed_analysis.pdf', 'intervention_roadmap.pdf']
    }

def run_stress_monitoring_demo():
    """Run comprehensive stress monitoring demonstration"""
    
    print("🎯 PORTFOLIO MANAGER STRESS MONITORING DEMONSTRATION")
    print("=" * 60)
    
    # Load client data
    clients = load_client_data()
    
    # Test different scenarios
    scenarios = ['mild_stress', 'moderate_stress', 'severe_stress', 'inflation_spike']
    
    demo_results = []
    
    for scenario_name in scenarios:
        print(f"\n📈 SCENARIO: {scenario_name.upper().replace('_', ' ')}")
        print("-" * 40)
        
        scenario = simulate_market_scenario(scenario_name)
        print(f"Description: {scenario['description']}")
        
        scenario_results = []
        
        for client in clients:
            print(f"\n👤 CLIENT: {client['name']} ({client['client_id']})")
            
            # Create stress monitor
            baseline_metrics = {'Financial_Stress_Rank': client['baseline_stress']}
            monitor = StressMonitor(client['baseline_config'], baseline_metrics)
            
            # Calculate stress increase from scenario
            stress_increase = calculate_stress_increase(client, scenario)
            new_stress = client['baseline_stress'] + stress_increase
            
            # Update stress monitor
            current_metrics = {'Financial_Stress_Rank': new_stress}
            monitor_response = monitor.update_stress(current_metrics)
            
            print(f"   Baseline Stress: {client['baseline_stress']:.1%}")
            print(f"   Current Stress:  {monitor_response['stress_level']:.1%} (+{monitor_response['stress_change']:.1%})")
            
            # Generate alerts and recommendations
            if monitor_response['alerts']:
                print(f"   🚨 Alerts: {len(monitor_response['alerts'])}")
                for alert in monitor_response['alerts']:
                    print(f"      {alert['level']}: {alert['message']}")
            
            if monitor_response['recommendations']:
                print(f"   💡 Top Recommendation: {monitor_response['recommendations'][0]['description']}")
                print(f"      Impact: {monitor_response['recommendations'][0]['estimated_stress_reduction']:.1%} stress reduction")
            
            # Generate client communication
            communication = generate_client_communication(client, monitor_response, scenario)
            print(f"   📧 Communication: {communication['urgency']} priority")
            
            # Store results
            client_result = {
                'scenario': scenario_name,
                'client_id': client['client_id'],
                'client_name': client['name'],
                'baseline_stress': client['baseline_stress'],
                'new_stress': monitor_response['stress_level'],
                'stress_increase': monitor_response['stress_change'],
                'alert_count': len(monitor_response['alerts']),
                'top_recommendation': monitor_response['recommendations'][0]['intervention'] if monitor_response['recommendations'] else None,
                'communication_urgency': communication['urgency'],
                'auto_interventions': len(monitor_response['auto_interventions']) if monitor_response['auto_interventions'] else 0
            }
            scenario_results.append(client_result)
        
        demo_results.extend(scenario_results)
    
    # Save demonstration results
    demo_df = pd.DataFrame(demo_results)
    demo_df.to_csv('stress_monitoring_demo_results.csv', index=False)
    
    # Generate summary report
    print(f"\n📊 DEMONSTRATION SUMMARY")
    print("=" * 40)
    
    for scenario_name in scenarios:
        scenario_data = demo_df[demo_df['scenario'] == scenario_name]
        avg_stress_increase = scenario_data['stress_increase'].mean()
        high_priority_clients = len(scenario_data[scenario_data['communication_urgency'].isin(['URGENT', 'HIGH'])])
        
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print(f"   Average stress increase: {avg_stress_increase:.1%}")
        print(f"   High-priority clients: {high_priority_clients}/{len(scenario_data)}")
        print(f"   Auto-interventions triggered: {scenario_data['auto_interventions'].sum()}")
    
    print(f"\n💼 PORTFOLIO MANAGER WORKFLOW:")
    print(f"   1. ✅ Monitor market conditions and client stress levels")
    print(f"   2. ✅ Generate automated alerts for elevated stress")
    print(f"   3. ✅ Rank intervention recommendations by impact/feasibility")
    print(f"   4. ✅ Implement high-feasibility interventions automatically")
    print(f"   5. ✅ Generate client communications with specific action plans")
    print(f"   6. ✅ Track intervention effectiveness over time")
    
    print(f"\n🔄 AUTOMATION CAPABILITIES:")
    print(f"   • Real-time stress level monitoring")
    print(f"   • Automatic portfolio rebalancing based on stress thresholds")
    print(f"   • Prioritized client communication based on urgency")
    print(f"   • Intervention impact simulation and tracking")
    print(f"   • Compliance reporting and audit trails")
    
    return demo_results

def create_intervention_dashboard_data():
    """Create sample data for intervention dashboard"""
    
    dashboard_data = {
        'clients_monitored': 50,
        'avg_stress_level': 0.23,
        'alerts_today': 7,
        'interventions_pending': 12,
        'auto_adjustments_made': 3,
        'high_stress_clients': 8,
        'review_overdue': 4,
        'stress_trend': 'STABLE',
        'top_interventions': [
            {'intervention': 'increase_portfolio_conservatism', 'clients': 15, 'avg_impact': 0.06},
            {'intervention': 'build_emergency_fund', 'clients': 12, 'avg_impact': 0.20},
            {'intervention': 'reduce_bonus_dependency', 'clients': 8, 'avg_impact': 0.08}
        ]
    }
    
    # Save dashboard data
    with open('stress_dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"\n📊 DASHBOARD DATA CREATED:")
    print(f"   Total clients monitored: {dashboard_data['clients_monitored']}")
    print(f"   Average stress level: {dashboard_data['avg_stress_level']:.1%}")
    print(f"   Active alerts: {dashboard_data['alerts_today']}")
    print(f"   Pending interventions: {dashboard_data['interventions_pending']}")
    
    return dashboard_data

if __name__ == "__main__":
    # Run the demonstration
    demo_results = run_stress_monitoring_demo()
    
    # Create dashboard data
    dashboard_data = create_intervention_dashboard_data()
    
    print(f"\n✅ Demonstration complete!")
    print(f"   📁 Results saved to: stress_monitoring_demo_results.csv")
    print(f"   📊 Dashboard data: stress_dashboard_data.json")
    print(f"\n🚀 Ready for integration with portfolio management systems!") 