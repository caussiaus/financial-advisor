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
import sys

# Import our stress monitoring system
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from ips_model import (
    StressMonitor, simulate_intervention_impact, STRESS_MONITORING,
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
    
    # Create final summary report
    summary_df = pd.DataFrame(demo_results)
    
    output_dir = Path("ips_output")
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / "stress_monitoring_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("📊 STRESS MONITORING DEMO COMPLETE")
    print(f"   Summary report saved to: {summary_path}")
    print(f"   Scenarios tested: {len(scenarios)}")
    print(f"   Total client simulations: {len(summary_df)}")
    print("="*60)

def main():
    """Main function to run the demo"""
    run_stress_monitoring_demo()

if __name__ == "__main__":
    main() 