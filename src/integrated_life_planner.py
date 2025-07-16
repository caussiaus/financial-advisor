#!/usr/bin/env python
"""
Integrated Life Planner - End-to-End Client Input to Optimal Decisions
Author: ChatGPT 2025-01-27

Connects client input processing with IPS modeling to create optimal
life decisions as events unfold.

Features:
- Email/PDF ingestion and event extraction
- Real-time scenario modeling and stress analysis
- Optimal decision recommendations
- Balance tracking across life paths
- Configuration space management
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import logging

# Import our modules
from client_input_processor import ClientInputProcessor, LifeEvent
from ips_model import load_config, calculate_cashflow, generate_scenarios, analyze_configuration
from life_events_tracker import LifeEventsTracker
from stress_analyzer import StressAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedLifePlanner:
    """Main class for integrated life planning and decision optimization"""
    
    def __init__(self, client_id: str, config_path: str = "config/ips_config.json"):
        self.client_id = client_id
        self.config_path = config_path
        
        # Load configuration
        self.YEARS, self.PARAM, self.FACTOR_SPACE, self.RISK_SPLITS, self.FX_SCENARIOS = load_config(config_path)
        
        # Initialize components
        self.input_processor = ClientInputProcessor(client_id, config_path)
        self.events_tracker = LifeEventsTracker(client_id, self._get_baseline_config())
        self.stress_analyzer = StressAnalyzer()
        
        # Decision tracking
        self.decision_history = []
        self.optimal_decisions_cache = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"Initialized IntegratedLifePlanner for client {client_id}")
    
    def _get_baseline_config(self) -> Dict:
        """Get baseline configuration for the client"""
        # This would be loaded from client profile
        return {
            'ED_PATH': 'McGill',
            'HEL_WORK': 'Full-time',
            'BONUS_PCT': 0.10,
            'DON_STYLE': 0,
            'MOM_GIFT_USE': 0,
            'FX_SCENARIO': 'Base',
            'RISK_BAND': 2
        }
    
    def process_client_update(self, input_text: str, input_type: str = 'email', 
                           input_date: datetime = None) -> Dict:
        """Process a client update and return optimal decisions"""
        logger.info(f"Processing {input_type} update for client {self.client_id}")
        
        # Extract events from input
        if input_type == 'email':
            events = self.input_processor.process_email(input_text, input_date)
        elif input_type == 'pdf':
            events = self.input_processor.process_pdf(input_text)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        # Log events in tracker
        for event in events:
            self.events_tracker.log_actual_event(
                event.event_type,
                event.planned_date,
                event.description,
                event.cash_flow_impact,
                event.impact_amount
            )
        
        # Update scenario space
        self._update_scenario_space(events)
        
        # Calculate optimal decisions
        optimal_decisions = self._calculate_optimal_decisions()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(events, optimal_decisions)
        
        # Start monitoring if not already active
        if not self.monitoring_active:
            self._start_monitoring()
        
        return {
            'extracted_events': [event.to_dict() for event in events],
            'optimal_decisions': optimal_decisions,
            'recommendations': recommendations,
            'stress_analysis': self._get_stress_analysis(),
            'scenario_summary': self._get_scenario_summary()
        }
    
    def _update_scenario_space(self, new_events: List[LifeEvent]):
        """Update the scenario space based on new events"""
        logger.info(f"Updating scenario space with {len(new_events)} new events")
        
        # Get current scenarios
        scenarios = self.input_processor.scenario_manager.get_all_scenarios()
        
        # Update each scenario with new events
        for scenario_id, scenario in scenarios.items():
            # Add new events to scenario
            scenario['events'] = scenario.get('events', []) + new_events
            
            # Recalculate cash flows
            updated_cashflows = self._calculate_scenario_cashflows(scenario)
            
            # Check for stress scenarios
            stress_level = self._calculate_scenario_stress(updated_cashflows)
            
            if stress_level > 0.4:  # High stress threshold
                self.input_processor.scenario_manager.flag_stress_scenario(scenario_id, stress_level)
                
                if stress_level > 0.6:  # Very high stress
                    self.input_processor.scenario_manager.rule_out_scenario(
                        scenario_id, f"Excessive stress level: {stress_level:.2f}"
                    )
    
    def _calculate_scenario_cashflows(self, scenario: Dict) -> List[Dict]:
        """Calculate cash flows for a specific scenario"""
        cashflows = []
        
        for year in range(self.YEARS):
            # Use the existing cash flow calculation from ips_model
            cashflow = calculate_cashflow(year, scenario)
            cashflows.append(cashflow)
        
        return cashflows
    
    def _calculate_scenario_stress(self, cashflows: List[Dict]) -> float:
        """Calculate stress level for a scenario"""
        # Calculate total cash flows
        total_income = sum(cf.get('H_SAL', 0) + cf.get('H_BONUS', 0) + cf.get('HEL_SAL', 0) 
                          for cf in cashflows)
        total_expenses = sum(abs(cf.get('Daycare', 0)) + abs(cf.get('Tuition', 0)) + 
                          abs(cf.get('Mortgage', 0)) + abs(cf.get('Ppty', 0)) + 
                          abs(cf.get('Healthcare', 0)) + abs(cf.get('Charity', 0))
                          for cf in cashflows)
        
        if total_income == 0:
            return 1.0
        
        # Stress ratio
        stress_ratio = total_expenses / total_income
        
        # Additional stress factors
        negative_years = sum(1 for cf in cashflows 
                           if cf.get('H_SAL', 0) + cf.get('H_BONUS', 0) + cf.get('HEL_SAL', 0) < 
                              abs(cf.get('Daycare', 0)) + abs(cf.get('Tuition', 0)) + 
                              abs(cf.get('Mortgage', 0)) + abs(cf.get('Ppty', 0)) + 
                              abs(cf.get('Healthcare', 0)) + abs(cf.get('Charity', 0)))
        
        negative_penalty = negative_years / len(cashflows) * 0.3
        
        return min(1.0, stress_ratio + negative_penalty)
    
    def _calculate_optimal_decisions(self) -> Dict:
        """Calculate optimal decisions for all feasible scenarios"""
        feasible_scenarios = self.input_processor.scenario_manager.get_feasible_scenarios()
        
        optimal_decisions = {}
        
        for scenario in feasible_scenarios:
            # Analyze configuration using existing IPS model
            analysis = analyze_configuration(scenario, generate_scenarios(10))
            
            # Calculate optimal decisions
            decisions = self._optimize_decisions_for_scenario(scenario, analysis)
            optimal_decisions[scenario['scenario_id']] = decisions
        
        return optimal_decisions
    
    def _optimize_decisions_for_scenario(self, scenario: Dict, analysis: Dict) -> Dict:
        """Optimize decisions for a specific scenario"""
        # Get current configuration
        current_config = {
            'ED_PATH': scenario.get('ED_PATH', 'McGill'),
            'HEL_WORK': scenario.get('HEL_WORK', 'Full-time'),
            'BONUS_PCT': scenario.get('BONUS_PCT', 0.10),
            'DON_STYLE': scenario.get('DON_STYLE', 0),
            'RISK_BAND': scenario.get('RISK_BAND', 2)
        }
        
        # Calculate optimal adjustments
        optimal_adjustments = self._calculate_optimal_adjustments(scenario, analysis)
        
        return {
            'current_config': current_config,
            'optimal_adjustments': optimal_adjustments,
            'confidence_score': analysis.get('qol_score', 0.5),
            'stress_level': analysis.get('stress_metrics', {}).get('overall_stress', 0.0),
            'recommended_actions': self._generate_recommended_actions(scenario, analysis)
        }
    
    def _calculate_optimal_adjustments(self, scenario: Dict, analysis: Dict) -> Dict:
        """Calculate optimal adjustments for a scenario"""
        adjustments = {}
        
        # Education path optimization
        if analysis.get('stress_metrics', {}).get('overall_stress', 0) > 0.4:
            if scenario.get('ED_PATH') == 'JohnsHopkins':
                adjustments['ED_PATH'] = 'McGill'  # Switch to cheaper option
        
        # Work arrangement optimization
        if analysis.get('stress_metrics', {}).get('income_volatility', 0) > 0.3:
            if scenario.get('HEL_WORK') == 'Full-time':
                adjustments['HEL_WORK'] = 'Part-time'  # More stable income
        
        # Risk band optimization
        stress_level = analysis.get('stress_metrics', {}).get('overall_stress', 0)
        current_risk = scenario.get('RISK_BAND', 2)
        
        if stress_level > 0.5:
            adjustments['RISK_BAND'] = max(1, current_risk - 1)  # More conservative
        elif stress_level < 0.2:
            adjustments['RISK_BAND'] = min(3, current_risk + 1)  # More aggressive
        
        # Charitable giving optimization
        if analysis.get('stress_metrics', {}).get('cash_flow_stress', 0) > 0.4:
            if scenario.get('DON_STYLE') != 0:
                adjustments['DON_STYLE'] = 0  # Regular giving instead of lump sum
        
        return adjustments
    
    def _generate_recommended_actions(self, scenario: Dict, analysis: Dict) -> List[str]:
        """Generate recommended actions for a scenario"""
        actions = []
        stress_metrics = analysis.get('stress_metrics', {})
        
        if stress_metrics.get('overall_stress', 0) > 0.4:
            actions.append("Consider reducing discretionary expenses")
            actions.append("Build emergency fund")
        
        if stress_metrics.get('income_volatility', 0) > 0.3:
            actions.append("Diversify income sources")
            actions.append("Consider part-time work arrangement")
        
        if stress_metrics.get('cash_flow_stress', 0) > 0.4:
            actions.append("Review education funding strategy")
            actions.append("Consider deferring major purchases")
        
        if analysis.get('qol_score', 0) < 0.6:
            actions.append("Review life goals and priorities")
            actions.append("Consider lifestyle adjustments")
        
        return actions
    
    def _generate_recommendations(self, events: List[LifeEvent], 
                                optimal_decisions: Dict) -> Dict:
        """Generate comprehensive recommendations"""
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_considerations': [],
            'risk_mitigation': [],
            'opportunity_identification': []
        }
        
        # Analyze events for immediate actions
        for event in events:
            if event.cash_flow_impact == 'negative' and event.impact_amount > 50000:
                recommendations['immediate_actions'].append(
                    f"Plan for {event.event_type} impact: ${event.impact_amount:,.2f}"
                )
            
            if event.event_type == 'education':
                recommendations['short_term_goals'].append(
                    "Review education funding strategy"
                )
            
            if event.event_type == 'housing':
                recommendations['long_term_considerations'].append(
                    "Consider mortgage refinancing options"
                )
        
        # Add recommendations from optimal decisions
        for scenario_id, decisions in optimal_decisions.items():
            if decisions.get('stress_level', 0) > 0.4:
                recommendations['risk_mitigation'].extend(
                    decisions.get('recommended_actions', [])
                )
        
        return recommendations
    
    def _get_stress_analysis(self) -> Dict:
        """Get comprehensive stress analysis"""
        scenarios = self.input_processor.scenario_manager.get_all_scenarios()
        
        stress_summary = {
            'total_scenarios': len(scenarios),
            'stressful_scenarios': len(self.input_processor.scenario_manager.stress_scenarios),
            'ruled_out_scenarios': len(self.input_processor.scenario_manager.ruled_out_scenarios),
            'average_stress_level': 0.0,
            'high_stress_events': []
        }
        
        if scenarios:
            total_stress = sum(
                scenario.get('stress_level', 0) 
                for scenario in scenarios.values()
            )
            stress_summary['average_stress_level'] = total_stress / len(scenarios)
        
        return stress_summary
    
    def _get_scenario_summary(self) -> Dict:
        """Get summary of all scenarios"""
        scenarios = self.input_processor.scenario_manager.get_all_scenarios()
        feasible_scenarios = self.input_processor.scenario_manager.get_feasible_scenarios()
        
        return {
            'total_scenarios': len(scenarios),
            'feasible_scenarios': len(feasible_scenarios),
            'scenario_distribution': {
                'education_paths': {},
                'work_arrangements': {},
                'risk_bands': {}
            }
        }
    
    def _start_monitoring(self):
        """Start real-time monitoring of client situation"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started real-time monitoring")
    
    def _monitoring_worker(self):
        """Worker thread for real-time monitoring"""
        while self.monitoring_active:
            try:
                # Check for upcoming events
                upcoming_events = self.events_tracker._get_upcoming_events(months_ahead=3)
                
                for event in upcoming_events:
                    # Check if event needs attention
                    if self._event_needs_attention(event):
                        self._alert_about_event(event)
                
                # Sleep for monitoring interval
                threading.Event().wait(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
    
    def _event_needs_attention(self, event: Dict) -> bool:
        """Check if an event needs immediate attention"""
        # Events within 30 days need attention
        days_until_event = (event['planned_date'] - datetime.now()).days
        
        return days_until_event <= 30
    
    def _alert_about_event(self, event: Dict):
        """Generate alert for an upcoming event"""
        logger.warning(f"ALERT: Upcoming event {event['event_type']} in {event['planned_date']}")
        
        # This would integrate with notification system
        alert = {
            'event_type': event['event_type'],
            'planned_date': event['planned_date'],
            'description': event['description'],
            'cash_flow_impact': event['cash_flow_impact'],
            'impact_amount': event['impact_amount'],
            'recommended_actions': self._get_event_recommendations(event)
        }
        
        # Store alert for client review
        self.decision_history.append({
            'timestamp': datetime.now(),
            'type': 'event_alert',
            'data': alert
        })
    
    def _get_event_recommendations(self, event: Dict) -> List[str]:
        """Get recommendations for handling an upcoming event"""
        recommendations = []
        
        if event['cash_flow_impact'] == 'negative':
            recommendations.append("Review cash flow projections")
            recommendations.append("Consider funding alternatives")
        
        if event['event_type'] == 'education':
            recommendations.append("Review education funding strategy")
            recommendations.append("Consider scholarship opportunities")
        
        if event['event_type'] == 'housing':
            recommendations.append("Review mortgage options")
            recommendations.append("Consider refinancing opportunities")
        
        return recommendations
    
    def get_decision_history(self) -> List[Dict]:
        """Get history of all decisions and alerts"""
        return self.decision_history
    
    def export_comprehensive_report(self, output_path: str = None) -> str:
        """Export comprehensive life planning report"""
        if output_path is None:
            output_path = f"ips_output/{self.client_id}_comprehensive_report.json"
        
        report = {
            'client_id': self.client_id,
            'report_date': datetime.now().isoformat(),
            'extracted_events': [event.to_dict() for event in self.input_processor.extracted_events],
            'scenarios': self.input_processor.scenario_manager.get_all_scenarios(),
            'optimal_decisions': self._calculate_optimal_decisions(),
            'stress_analysis': self._get_stress_analysis(),
            'decision_history': self.get_decision_history(),
            'balance_summary': self.input_processor.balance_tracker.get_all_balances(),
            'recommendations': self._generate_recommendations(
                self.input_processor.extracted_events,
                self._calculate_optimal_decisions()
            )
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report exported to: {output_path}")
        return output_path

def demo_integrated_life_planning():
    """Demo the integrated life planning system"""
    print("=== Integrated Life Planning Demo ===")
    
    # Initialize planner
    planner = IntegratedLifePlanner("DEMO_CLIENT_001")
    
    # Sample client updates
    updates = [
        {
            'text': """
            Hi,
            
            We've decided to send our child to Johns Hopkins University starting next year.
            The tuition will be $110,000 per year. My wife is planning to work part-time
            starting in 3 months to help with childcare costs.
            
            We're also considering buying a new house in 2 years for around $2.5 million.
            
            Thanks,
            Client
            """,
            'type': 'email',
            'date': datetime.now()
        },
        {
            'text': """
            Update: I received a promotion with a 20% bonus increase. This should help
            with the education costs. However, we're also planning a major renovation
            in 6 months that will cost about $150,000.
            """,
            'type': 'email',
            'date': datetime.now() + timedelta(days=30)
        }
    ]
    
    # Process each update
    for i, update in enumerate(updates):
        print(f"\n--- Processing Update {i+1} ---")
        
        result = planner.process_client_update(
            update['text'],
            update['type'],
            update['date']
        )
        
        print(f"Extracted {len(result['extracted_events'])} events")
        print(f"Generated {len(result['optimal_decisions'])} optimal decisions")
        print(f"Created {len(result['recommendations'])} recommendation categories")
        
        # Show some key recommendations
        if result['recommendations']['immediate_actions']:
            print("\nImmediate Actions:")
            for action in result['recommendations']['immediate_actions'][:3]:
                print(f"- {action}")
    
    # Export comprehensive report
    report_path = planner.export_comprehensive_report()
    print(f"\nComprehensive report exported to: {report_path}")
    
    # Show decision history
    history = planner.get_decision_history()
    print(f"\nDecision history contains {len(history)} entries")

if __name__ == "__main__":
    demo_integrated_life_planning() 