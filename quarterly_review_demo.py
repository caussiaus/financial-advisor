#!/usr/bin/env python
"""
Quarterly Review Demonstration - Complete Advisor Workflow
Author: ChatGPT 2025-07-16

Demonstrates the complete 15-minute advisor-client quarterly review workflow
using the IPS toolkit with visual timelines and stress monitoring.

Usage:
    python quarterly_review_demo.py

Simulates:
- Client life update session
- Configuration matching
- Visual timeline presentation
- Stress monitoring analysis
- Intervention recommendations
- Next steps planning
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import our toolkit components
from ips_model import StressMonitor, simulate_intervention_impact
from visual_timeline_builder import TimelineVisualizer, load_enhanced_configurations

class QuarterlyReviewSession:
    """Simulate a complete quarterly review session"""
    
    def __init__(self):
        self.session_date = datetime.now()
        self.advisor_name = "Sarah Johnson, CFP"
        self.firm_name = "Pinnacle Wealth Management"
        
        # Load configuration data
        self.configurations = load_enhanced_configurations()
        self.visualizer = TimelineVisualizer()
        
        # Sample client profiles
        self.sample_clients = [
            {
                'client_id': 'CLIENT_001',
                'name': 'John & Sarah Mitchell',
                'age': 32,
                'current_situation': {
                    'ED_PATH': 'JohnsHopkins',
                    'HEL_WORK': 'Full-time',
                    'BONUS_PCT': 0.30,
                    'DON_STYLE': 0,
                    'RISK_BAND': 3,
                    'FX_SCENARIO': 'Base'
                },
                'life_updates': {
                    'considering_education_change': True,
                    'work_stress_level': 'High',
                    'bonus_uncertainty': True,
                    'market_concerns': 'Moderate'
                },
                'last_review': '2024-10-01'
            },
            {
                'client_id': 'CLIENT_002', 
                'name': 'David & Lisa Chen',
                'age': 29,
                'current_situation': {
                    'ED_PATH': 'McGill',
                    'HEL_WORK': 'Part-time',
                    'BONUS_PCT': 0.15,
                    'DON_STYLE': 1,
                    'RISK_BAND': 2,
                    'FX_SCENARIO': 'Base'
                },
                'life_updates': {
                    'considering_full_time': True,
                    'charitable_timing_questions': True,
                    'risk_tolerance_increased': True,
                    'market_concerns': 'Low'
                },
                'last_review': '2024-11-15'
            }
        ]
    
    def conduct_review_session(self, client):
        """Conduct complete quarterly review for a client"""
        
        print(f"\n{'='*80}")
        print(f"📅 QUARTERLY REVIEW SESSION")
        print(f"{'='*80}")
        print(f"📊 Firm: {self.firm_name}")
        print(f"👤 Advisor: {self.advisor_name}")
        print(f"📅 Date: {self.session_date.strftime('%B %d, %Y')}")
        print(f"🏠 Client: {client['name']} ({client['client_id']})")
        print(f"⏱️  Duration: 15 minutes (projected)")
        
        # Step 1: Client Life Update (5 minutes)
        print(f"\n🔍 STEP 1: CLIENT LIFE UPDATE (5 minutes)")
        print("-" * 50)
        self._gather_life_updates(client)
        
        # Step 2: Configuration Matching (3 minutes)
        print(f"\n🎯 STEP 2: CONFIGURATION MATCHING (3 minutes)")
        print("-" * 50)
        optimal_configs = self._match_configurations(client)
        
        # Step 3: Visual Timeline Presentation (7 minutes)
        print(f"\n📈 STEP 3: VISUAL TIMELINE PRESENTATION (7 minutes)")
        print("-" * 50)
        self._present_visual_analysis(client, optimal_configs)
        
        # Step 4: Intervention Recommendations & Next Steps
        print(f"\n💡 STEP 4: RECOMMENDATIONS & NEXT STEPS")
        print("-" * 50)
        self._generate_action_plan(client, optimal_configs)
        
        # Generate session summary
        return self._create_session_summary(client, optimal_configs)
    
    def _gather_life_updates(self, client):
        """Simulate gathering client life updates"""
        
        print(f"📋 Current Configuration Assessment:")
        current = client['current_situation']
        print(f"   • Education Path: {current['ED_PATH']}")
        print(f"   • Work Arrangement: {current['HEL_WORK']}")
        print(f"   • Bonus Expectation: {current['BONUS_PCT']:.0%}")
        print(f"   • Charitable Strategy: {['Regular', 'Deferred', 'Immediate'][current['DON_STYLE']]}")
        print(f"   • Risk Tolerance: {current['RISK_BAND']}/3")
        
        print(f"\n📢 Life Updates Since Last Review:")
        updates = client['life_updates']
        
        for update, status in updates.items():
            icon = "🔄" if status == True else "⚠️" if status == "High" else "✅"
            update_text = update.replace('_', ' ').title()
            print(f"   {icon} {update_text}: {status}")
        
        # Advisor questions simulation
        print(f"\n🗣️  Advisor Questions:")
        if updates.get('considering_education_change'):
            print(f"   Q: 'Are you reconsidering the Johns Hopkins path due to cost concerns?'")
            print(f"   A: 'Yes, we're looking at alternatives that might reduce financial stress.'")
        
        if updates.get('work_stress_level') == 'High':
            print(f"   Q: 'How is the full-time work arrangement affecting your family life?'")
            print(f"   A: 'It's challenging. We're open to exploring part-time options.'")
        
        if updates.get('bonus_uncertainty'):
            print(f"   Q: 'Has your bonus outlook changed this quarter?'")
            print(f"   A: 'Less certain now. Maybe we should plan more conservatively.'")
    
    def _match_configurations(self, client):
        """Match client situation to optimal configurations"""
        
        print(f"🔍 Analyzing 480+ predetermined configurations...")
        
        # Find current configuration
        current_config = None
        for cfg in self.configurations:
            if (cfg.get('ED_PATH') == client['current_situation']['ED_PATH'] and
                cfg.get('HEL_WORK') == client['current_situation']['HEL_WORK'] and
                abs(cfg.get('BONUS_PCT', 0) - client['current_situation']['BONUS_PCT']) < 0.01):
                current_config = cfg
                break
        
        # If exact match not found, find closest match
        if not current_config and self.configurations:
            print(f"⚠️  Exact configuration not found, finding closest match...")
            current_config = min(self.configurations, 
                                key=lambda cfg: (
                                    abs(cfg.get('BONUS_PCT', 0) - client['current_situation']['BONUS_PCT']) +
                                    (0 if cfg.get('ED_PATH') == client['current_situation']['ED_PATH'] else 0.5) +
                                    (0 if cfg.get('HEL_WORK') == client['current_situation']['HEL_WORK'] else 0.3)
                                ))
        
        if current_config:
            print(f"✅ Current configuration found: {current_config['cfg_id']}")
            print(f"   Financial Stress Level: {current_config.get('Financial_Stress_Rank', 0):.1%}")
            print(f"   Quality of Life Score: {current_config.get('QoL_Score', 0):.2f}")
        else:
            # Create a default config for demo purposes
            current_config = {
                'cfg_id': 'DEMO_DEFAULT',
                'ED_PATH': client['current_situation']['ED_PATH'],
                'HEL_WORK': client['current_situation']['HEL_WORK'],
                'BONUS_PCT': client['current_situation']['BONUS_PCT'],
                'Financial_Stress_Rank': 0.35,  # Moderate stress
                'QoL_Score': 0.60  # Moderate QoL
            }
            print(f"⚠️  Using demo default configuration")
        
        # Find alternative configurations based on life updates
        print(f"\n🎯 Alternative Configurations Based on Life Updates:")
        
        alternatives = []
        
        # If considering education change
        if client['life_updates'].get('considering_education_change'):
            mcgill_configs = [cfg for cfg in self.configurations 
                            if cfg.get('ED_PATH') == 'McGill' and
                            cfg.get('HEL_WORK') == client['current_situation']['HEL_WORK']]
            if mcgill_configs:
                best_mcgill = min(mcgill_configs, key=lambda x: x.get('Financial_Stress_Rank', 1))
                alternatives.append(('Education Alternative', best_mcgill))
        
        # If considering work change
        if client['life_updates'].get('work_stress_level') == 'High':
            pt_configs = [cfg for cfg in self.configurations 
                         if cfg.get('HEL_WORK') == 'Part-time' and
                         cfg.get('ED_PATH') == client['current_situation']['ED_PATH']]
            if pt_configs:
                best_pt = min(pt_configs, key=lambda x: x.get('Financial_Stress_Rank', 1))
                alternatives.append(('Work-Life Balance Alternative', best_pt))
        
        # Show alternatives
        for alt_name, alt_config in alternatives:
            stress_improvement = current_config.get('Financial_Stress_Rank', 0) - alt_config.get('Financial_Stress_Rank', 0)
            qol_improvement = alt_config.get('QoL_Score', 0) - current_config.get('QoL_Score', 0)
            
            print(f"\n   📊 {alt_name}: {alt_config['cfg_id']}")
            print(f"      Stress Change: {stress_improvement:+.1%} ({'Better' if stress_improvement > 0 else 'Worse'})")
            print(f"      QoL Change: {qol_improvement:+.2f} ({'Better' if qol_improvement > 0 else 'Worse'})")
            print(f"      Key Changes: {alt_config.get('ED_PATH', 'N/A')}, {alt_config.get('HEL_WORK', 'N/A')}")
        
        return {
            'current': current_config,
            'alternatives': alternatives
        }
    
    def _present_visual_analysis(self, client, configs):
        """Present visual timeline analysis to client"""
        
        print(f"📊 VISUAL TIMELINE PRESENTATION")
        print(f"   (Showing client: visual_timelines/timeline_{configs['current']['cfg_id']}.png)")
        
        current = configs['current']
        
        print(f"\n🎨 Current Path Visualization Highlights:")
        print(f"   📈 40-Year Financial Journey Chart Generated")
        print(f"   📅 Major Life Events Timeline:")
        
        # Simulate describing key timeline events
        if current.get('ED_PATH') == 'JohnsHopkins':
            print(f"      🎓 Years 5-10: Johns Hopkins education costs (~$400K total)")
            print(f"      💰 Higher education debt impacts stress levels significantly")
        else:
            print(f"      🎓 Years 5-10: McGill education costs (~$200K total)")
            print(f"      💰 Lower education costs provide financial breathing room")
        
        if current.get('HEL_WORK') == 'Full-time':
            print(f"      👨‍💼 Years 0-5: Full-time work with daycare costs")
            print(f"      💼 Higher income but reduced flexibility")
        else:
            print(f"      👨‍💼 Part-time work arrangement")
            print(f"      💼 Lower income but better work-life balance")
        
        # Stress progression
        stress_level = current.get('Financial_Stress_Rank', 0)
        if stress_level > 0.3:
            print(f"      🚨 Financial stress: HIGH ({stress_level:.1%})")
            print(f"         Red zones indicate periods of financial pressure")
        elif stress_level > 0.2:
            print(f"      ⚠️  Financial stress: MODERATE ({stress_level:.1%})")
            print(f"         Yellow zones indicate caution periods")
        else:
            print(f"      ✅ Financial stress: LOW ({stress_level:.1%})")
            print(f"         Green zones indicate comfortable periods")
        
        # Show alternatives if available
        if configs['alternatives']:
            print(f"\n📊 Alternative Path Comparison:")
            print(f"   (Showing configuration_comparison.png dashboard)")
            
            for alt_name, alt_config in configs['alternatives']:
                print(f"   🔄 {alt_name}:")
                print(f"      • Stress Level: {alt_config.get('Financial_Stress_Rank', 0):.1%}")
                print(f"      • QoL Score: {alt_config.get('QoL_Score', 0):.2f}")
                print(f"      • Visual timeline: timeline_{alt_config['cfg_id']}.png")
    
    def _generate_action_plan(self, client, configs):
        """Generate intervention recommendations and action plan"""
        
        current = configs['current']
        stress_level = current.get('Financial_Stress_Rank', 0)
        
        print(f"💡 PERSONALIZED ACTION PLAN")
        
        # Create stress monitor for real-time recommendations
        baseline_metrics = {'Financial_Stress_Rank': stress_level}
        monitor = StressMonitor(client['current_situation'], baseline_metrics)
        
        # Simulate current market stress (mild uptick)
        simulated_stress = stress_level + 0.05  # 5% stress increase
        current_metrics = {'Financial_Stress_Rank': simulated_stress}
        
        monitor_response = monitor.update_stress(current_metrics)
        
        if monitor_response['alerts']:
            print(f"\n🚨 IMMEDIATE ALERTS:")
            for alert in monitor_response['alerts']:
                print(f"   {alert['level']}: {alert['message']}")
        
        if monitor_response['recommendations']:
            print(f"\n📋 RANKED INTERVENTION RECOMMENDATIONS:")
            for i, rec in enumerate(monitor_response['recommendations'][:3]):
                urgency = "🔥 HIGH" if rec['time_to_implement_months'] <= 1 else "⏳ MEDIUM"
                print(f"\n   {i+1}. {rec['description']}")
                print(f"      Impact: {rec['estimated_stress_reduction']:.1%} stress reduction")
                print(f"      Feasibility: {rec['feasibility']:.0%}")
                print(f"      Timeline: {rec['time_to_implement_months']} months")
                print(f"      Urgency: {urgency}")
                print(f"      Next Steps:")
                for step in rec['specific_actions'][:2]:  # Show first 2 steps
                    print(f"         • {step}")
        
        # Portfolio adjustments
        if monitor_response['auto_interventions']:
            print(f"\n🤖 AUTOMATIC PORTFOLIO ADJUSTMENTS:")
            for auto_int in monitor_response['auto_interventions']:
                print(f"   ✅ {auto_int['description']}")
            
            print(f"\n📊 New Portfolio Allocation:")
            allocation = monitor_response['new_portfolio_allocation']
            print(f"   • Equity: {allocation['equity']:.1%}")
            print(f"   • Bonds: {allocation['bonds']:.1%}")
            print(f"   • Cash: {allocation['cash']:.1%}")
        
        # Next review scheduling
        next_review = self.session_date + timedelta(days=90)
        print(f"\n📅 NEXT STEPS & SCHEDULING:")
        print(f"   • Implementation timeline: 30-90 days")
        print(f"   • Next quarterly review: {next_review.strftime('%B %d, %Y')}")
        print(f"   • Stress monitoring: Continuous (alerts if >25% stress)")
        print(f"   • Emergency consultation: Available if stress >40%")
    
    def _create_session_summary(self, client, configs):
        """Create session summary report"""
        
        summary = {
            'session_date': self.session_date.isoformat(),
            'client_id': client['client_id'],
            'client_name': client['name'],
            'advisor': self.advisor_name,
            'current_configuration': configs['current']['cfg_id'],
            'current_stress_level': configs['current'].get('Financial_Stress_Rank', 0),
            'current_qol_score': configs['current'].get('QoL_Score', 0),
            'alternatives_considered': len(configs['alternatives']),
            'recommendations_count': 3,  # Top 3 recommendations
            'next_review_date': (self.session_date + timedelta(days=90)).isoformat(),
            'session_duration_minutes': 15,
            'visual_files_generated': [
                f"timeline_{configs['current']['cfg_id']}.png",
                "configuration_comparison.png"
            ],
            'action_items': [
                "Review education cost alternatives",
                "Consider work arrangement optimization", 
                "Implement portfolio rebalancing",
                "Monitor stress levels quarterly"
            ]
        }
        
        return summary

def run_quarterly_review_demo():
    """Run complete quarterly review demonstration"""
    
    print("🏢 QUARTERLY REVIEW DEMONSTRATION")
    print("Pinnacle Wealth Management - Financial Planning Division")
    print("=" * 80)
    
    review_session = QuarterlyReviewSession()
    
    # Conduct reviews for sample clients
    session_summaries = []
    
    for client in review_session.sample_clients:
        summary = review_session.conduct_review_session(client)
        session_summaries.append(summary)
        
        print(f"\n✅ SESSION COMPLETE")
        print(f"   📁 Visual files ready for client")
        print(f"   📊 Action plan generated")
        print(f"   📅 Next review scheduled")
        
        print(f"\n" + "="*80)
    
    # Save session summaries
    summaries_df = pd.DataFrame(session_summaries)
    summaries_df.to_csv('quarterly_review_sessions.csv', index=False)
    
    # Generate final report
    print(f"\n📊 QUARTERLY REVIEW SESSION SUMMARY")
    print(f"=" * 50)
    print(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"👥 Clients Reviewed: {len(session_summaries)}")
    print(f"⏱️  Average Session Duration: 15 minutes")
    print(f"📊 Visual Timelines Generated: {len(session_summaries) * 2}")
    print(f"💡 Action Plans Created: {len(session_summaries)}")
    print(f"📈 Stress Improvements Identified: {sum(1 for s in session_summaries if s['alternatives_considered'] > 0)}")
    
    print(f"\n🎯 KEY OUTCOMES:")
    for summary in session_summaries:
        client_name = summary['client_name']
        stress_level = summary['current_stress_level']
        alternatives = summary['alternatives_considered']
        
        if stress_level > 0.3:
            status = "🚨 HIGH STRESS - Immediate intervention needed"
        elif stress_level > 0.2:
            status = "⚠️ MODERATE STRESS - Monitoring recommended"
        else:
            status = "✅ LOW STRESS - Maintain current path"
        
        print(f"   • {client_name}: {status}")
        if alternatives > 0:
            print(f"     📊 {alternatives} alternative configuration(s) identified")
    
    print(f"\n📁 SESSION OUTPUTS:")
    print(f"   📊 quarterly_review_sessions.csv - Session summaries")
    print(f"   🎨 visual_timelines/ - Client presentation materials")
    print(f"   📈 Individual timeline charts for each configuration")
    print(f"   📊 Configuration comparison dashboards")
    
    print(f"\n💼 ADVISOR WORKFLOW EFFICIENCY:")
    print(f"   ⏱️  15 minutes per client (vs 60+ minutes traditional)")
    print(f"   🎯 100% scenario coverage (480+ configurations pre-modeled)")
    print(f"   📊 Professional visual presentations ready instantly")
    print(f"   💡 AI-powered intervention recommendations")
    print(f"   🤖 Automated portfolio adjustments when needed")
    
    return session_summaries

if __name__ == "__main__":
    session_summaries = run_quarterly_review_demo() 