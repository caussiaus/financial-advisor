#!/usr/bin/env python
"""
End-to-End Life Planning Pipeline Demo
Author: ChatGPT 2025-01-27

Demonstrates the complete pipeline from client input processing
to optimal decision recommendations.

Pipeline:
1. Client sends email/PDF with life updates
2. AI extracts and timestamps life events
3. System models stress scenarios and rules out impossible paths
4. Balance tracking across all scenarios
5. Optimal decision recommendations
6. Real-time monitoring and alerts
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Import our modules
from integrated_life_planner import IntegratedLifePlanner
from client_input_processor import ClientInputProcessor
from ips_model import load_config, generate_scenarios, analyze_configuration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_client_data():
    """Create sample client data for demonstration"""
    return {
        'client_id': 'DEMO_CLIENT_001',
        'baseline_config': {
            'ED_PATH': 'McGill',
            'HEL_WORK': 'Full-time',
            'BONUS_PCT': 0.10,
            'DON_STYLE': 0,
            'MOM_GIFT_USE': 0,
            'FX_SCENARIO': 'Base',
            'RISK_BAND': 2
        },
        'initial_assets': {
            'checking': 50000,
            'savings': 100000,
            'investment': 250000,
            'retirement': 150000
        }
    }

def create_sample_client_updates():
    """Create sample client updates over time"""
    return [
        {
            'date': datetime.now(),
            'type': 'email',
            'subject': 'Education Decision Update',
            'content': """
            Hi,
            
            We've made some important decisions about our child's education. 
            We've decided to send our child to Johns Hopkins University starting next year.
            The tuition will be $110,000 per year for 4 years. This is a significant 
            investment but we believe it's the right choice for their future.
            
            My wife is planning to work part-time starting in 3 months to help with 
            childcare costs and provide more flexibility for our family.
            
            We're also considering buying a new house in 2 years for around $2.5 million.
            This would be a major purchase but we think the timing works well with 
            our current financial situation.
            
            Thanks,
            Client
            """
        },
        {
            'date': datetime.now() + timedelta(days=30),
            'type': 'email',
            'subject': 'Career and Renovation Update',
            'content': """
            Great news! I received a promotion with a 20% bonus increase. 
            This should help significantly with the education costs and overall 
            financial planning.
            
            However, we're also planning a major home renovation in 6 months 
            that will cost about $150,000. This is necessary to accommodate 
            our growing family needs.
            
            We're also considering increasing our charitable giving this year 
            to take advantage of the higher income.
            
            Best regards,
            Client
            """
        },
        {
            'date': datetime.now() + timedelta(days=90),
            'type': 'email',
            'subject': 'Investment and Retirement Update',
            'content': """
            We've been reviewing our investment strategy and retirement planning.
            
            Given the upcoming education costs and home renovation, we're thinking 
            about adjusting our portfolio to be more conservative. We want to ensure 
            we have sufficient liquidity for these major expenses.
            
            We're also considering deferring some of our charitable giving until 
            after the education costs are behind us.
            
            On a positive note, my wife's part-time work arrangement is working 
            well and providing good work-life balance.
            
            Regards,
            Client
            """
        }
    ]

def demo_event_extraction():
    """Demo the AI-powered event extraction"""
    print("\n" + "="*60)
    print("DEMO: AI-Powered Event Extraction")
    print("="*60)
    
    # Initialize processor
    processor = ClientInputProcessor("DEMO_CLIENT_001")
    
    # Sample email content
    sample_email = """
    Hi,
    
    We've decided to send our child to Johns Hopkins University starting next year.
    The tuition will be $110,000 per year. My wife is planning to work part-time
    starting in 3 months to help with childcare costs.
    
    We're also considering buying a new house in 2 years for around $2.5 million.
    
    Thanks,
    Client
    """
    
    # Extract events
    events = processor.process_email(sample_email)
    
    print(f"\nExtracted {len(events)} events from email:")
    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(f"  Type: {event.event_type}")
        print(f"  Description: {event.description[:100]}...")
        print(f"  Date: {event.planned_date.strftime('%Y-%m-%d')}")
        print(f"  Impact: {event.cash_flow_impact} ${event.impact_amount:,.2f}")
        print(f"  Confidence: {event.confidence:.2f}")

def demo_scenario_modeling():
    """Demo scenario modeling and stress analysis"""
    print("\n" + "="*60)
    print("DEMO: Scenario Modeling and Stress Analysis")
    print("="*60)
    
    # Load configuration
    YEARS, PARAM, FACTOR_SPACE, RISK_SPLITS, FX_SCENARIOS = load_config()
    
    # Generate scenarios
    scenarios = generate_scenarios(iterations=5)
    
    print(f"\nGenerated {len(scenarios)} scenarios")
    
    # Analyze each scenario
    scenario_analyses = []
    for i, scenario in enumerate(scenarios[:3]):  # Show first 3
        analysis = analyze_configuration(scenario, scenarios)
        scenario_analyses.append(analysis)
        
        print(f"\nScenario {i+1} Analysis:")
        print(f"  QOL Score: {analysis.get('qol_score', 0):.3f}")
        print(f"  Stress Level: {analysis.get('stress_metrics', {}).get('overall_stress', 0):.3f}")
        print(f"  Feasible: {analysis.get('is_feasible', True)}")
        
        if analysis.get('stress_metrics', {}).get('overall_stress', 0) > 0.4:
            print(f"  ⚠️  HIGH STRESS SCENARIO")

def demo_balance_tracking():
    """Demo balance tracking across scenarios"""
    print("\n" + "="*60)
    print("DEMO: Balance Tracking Across Scenarios")
    print("="*60)
    
    # Initialize processor
    processor = ClientInputProcessor("DEMO_CLIENT_001")
    
    # Create sample events
    from client_input_processor import LifeEvent
    
    events = [
        LifeEvent(
            event_type='education',
            description='Johns Hopkins tuition payment',
            planned_date=datetime.now() + timedelta(days=365),
            confidence=0.9,
            cash_flow_impact='negative',
            impact_amount=110000,
            source_text='Tuition payment',
            extracted_date=datetime.now()
        ),
        LifeEvent(
            event_type='work',
            description='Promotion with bonus increase',
            planned_date=datetime.now() + timedelta(days=30),
            confidence=0.8,
            cash_flow_impact='positive',
            impact_amount=30000,
            source_text='Promotion bonus',
            extracted_date=datetime.now()
        )
    ]
    
    # Process events and track balances
    for event in events:
        processor._queue_balance_recalculation([event])
    
    # Get balance summary
    balances = processor.balance_tracker.get_all_balances()
    
    print(f"\nTracking balances across {len(balances)} scenarios")
    
    # Show sample balance updates
    for scenario_id, balance_entries in list(balances.items())[:2]:
        print(f"\nScenario {scenario_id}:")
        for entry in balance_entries:
            print(f"  {entry['date']}: ${entry['balance']:,.2f} ({entry['description']})")

def demo_optimal_decisions():
    """Demo optimal decision generation"""
    print("\n" + "="*60)
    print("DEMO: Optimal Decision Generation")
    print("="*60)
    
    # Initialize planner
    planner = IntegratedLifePlanner("DEMO_CLIENT_001")
    
    # Process sample update
    sample_update = """
    Hi,
    
    We've decided to send our child to Johns Hopkins University starting next year.
    The tuition will be $110,000 per year. My wife is planning to work part-time
    starting in 3 months to help with childcare costs.
    
    We're also considering buying a new house in 2 years for around $2.5 million.
    
    Thanks,
    Client
    """
    
    result = planner.process_client_update(sample_update, 'email')
    
    print(f"\nGenerated optimal decisions for {len(result['optimal_decisions'])} scenarios")
    
    # Show sample decisions
    for scenario_id, decisions in list(result['optimal_decisions'].items())[:2]:
        print(f"\nScenario {scenario_id}:")
        print(f"  Current Config: {decisions['current_config']}")
        print(f"  Optimal Adjustments: {decisions['optimal_adjustments']}")
        print(f"  Confidence Score: {decisions['confidence_score']:.3f}")
        print(f"  Stress Level: {decisions['stress_level']:.3f}")
        
        if decisions['recommended_actions']:
            print(f"  Recommended Actions:")
            for action in decisions['recommended_actions'][:3]:
                print(f"    - {action}")

def demo_real_time_monitoring():
    """Demo real-time monitoring and alerts"""
    print("\n" + "="*60)
    print("DEMO: Real-Time Monitoring and Alerts")
    print("="*60)
    
    # Initialize planner
    planner = IntegratedLifePlanner("DEMO_CLIENT_001")
    
    # Add some events that would trigger alerts
    from client_input_processor import LifeEvent
    
    upcoming_events = [
        LifeEvent(
            event_type='education',
            description='Johns Hopkins tuition due',
            planned_date=datetime.now() + timedelta(days=15),  # Soon!
            confidence=0.9,
            cash_flow_impact='negative',
            impact_amount=110000,
            source_text='Tuition payment due soon',
            extracted_date=datetime.now()
        ),
        LifeEvent(
            event_type='housing',
            description='House purchase closing',
            planned_date=datetime.now() + timedelta(days=45),
            confidence=0.8,
            cash_flow_impact='negative',
            impact_amount=2500000,
            source_text='House purchase',
            extracted_date=datetime.now()
        )
    ]
    
    # Process events
    for event in upcoming_events:
        planner.events_tracker.log_actual_event(
            event.event_type,
            event.planned_date,
            event.description,
            event.cash_flow_impact,
            event.impact_amount
        )
    
    # Simulate monitoring check
    print("\nMonitoring upcoming events...")
    
    for event in upcoming_events:
        days_until = (event.planned_date - datetime.now()).days
        if days_until <= 30:
            print(f"⚠️  ALERT: {event.event_type} event in {days_until} days")
            print(f"   Amount: ${event.impact_amount:,.2f}")
            print(f"   Impact: {event.cash_flow_impact}")
            
            # Get recommendations
            recommendations = planner._get_event_recommendations({
                'event_type': event.event_type,
                'cash_flow_impact': event.cash_flow_impact,
                'impact_amount': event.impact_amount
            })
            
            print(f"   Recommendations:")
            for rec in recommendations:
                print(f"     - {rec}")

def demo_comprehensive_pipeline():
    """Demo the complete end-to-end pipeline"""
    print("\n" + "="*60)
    print("DEMO: Complete End-to-End Pipeline")
    print("="*60)
    
    # Initialize planner
    planner = IntegratedLifePlanner("DEMO_CLIENT_001")
    
    # Get sample updates
    updates = create_sample_client_updates()
    
    print(f"\nProcessing {len(updates)} client updates over time...")
    
    # Process each update
    for i, update in enumerate(updates, 1):
        print(f"\n--- Update {i} ({update['date'].strftime('%Y-%m-%d')}) ---")
        
        result = planner.process_client_update(
            update['content'],
            update['type'],
            update['date']
        )
        
        print(f"✅ Extracted {len(result['extracted_events'])} events")
        print(f"✅ Generated {len(result['optimal_decisions'])} optimal decisions")
        
        # Show stress analysis
        stress = result['stress_analysis']
        print(f"📊 Stress Analysis:")
        print(f"   - Total scenarios: {stress['total_scenarios']}")
        print(f"   - Stressful scenarios: {stress['stressful_scenarios']}")
        print(f"   - Ruled out scenarios: {stress['ruled_out_scenarios']}")
        print(f"   - Average stress level: {stress['average_stress_level']:.3f}")
        
        # Show key recommendations
        recs = result['recommendations']
        if recs['immediate_actions']:
            print(f"🚨 Immediate Actions:")
            for action in recs['immediate_actions'][:2]:
                print(f"   - {action}")
    
    # Export comprehensive report
    report_path = planner.export_comprehensive_report()
    print(f"\n📄 Comprehensive report exported to: {report_path}")
    
    # Show final summary
    history = planner.get_decision_history()
    print(f"📈 Decision history: {len(history)} entries")
    
    scenarios = planner.input_processor.scenario_manager.get_all_scenarios()
    feasible = planner.input_processor.scenario_manager.get_feasible_scenarios()
    print(f"🎯 Scenario space: {len(feasible)}/{len(scenarios)} scenarios feasible")

def main():
    """Run the complete demo"""
    print("🚀 INTEGRATED LIFE PLANNING SYSTEM DEMO")
    print("="*60)
    print("This demo shows the complete end-to-end pipeline from client input")
    print("to optimal decision recommendations.")
    print("="*60)
    
    try:
        # Run individual demos
        demo_event_extraction()
        demo_scenario_modeling()
        demo_balance_tracking()
        demo_optimal_decisions()
        demo_real_time_monitoring()
        
        # Run comprehensive pipeline demo
        demo_comprehensive_pipeline()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• AI-powered event extraction from natural language")
        print("• Real-time scenario modeling and stress analysis")
        print("• Balance tracking across multiple life paths")
        print("• Optimal decision recommendations")
        print("• Real-time monitoring and alerts")
        print("• Comprehensive reporting and analysis")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check that all dependencies are installed and modules are available.")

if __name__ == "__main__":
    main() 