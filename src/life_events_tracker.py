#!/usr/bin/env python
"""
Life Events Tracker - Real vs. Planned Event Logging
Author: ChatGPT 2025-07-16

Tracks actual life events vs. modeled timeline to reduce tracking error.
The IPS model assumes time-bound configurations, but real clients make decisions
at different times, affecting cash flow projections.

Usage:
    python life_events_tracker.py

Features:
- Planned vs. actual event logging
- Cash flow impact analysis
- Configuration drift detection
- Quarterly review updates
- Historical event timeline
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class LifeEventsTracker:
    """Track actual life events vs. planned configuration timeline"""
    
    def __init__(self, client_id, baseline_config):
        self.client_id = client_id
        self.baseline_config = baseline_config
        self.start_date = datetime.now()
        
        # Initialize event tracking database
        self.events_log = []
        self.planned_events = self._generate_planned_timeline()
        self.quarterly_reviews = []
        
        # Configuration drift tracking
        self.config_drift_log = []
        self.current_config = baseline_config.copy()
        
    def _generate_planned_timeline(self):
        """Generate planned event timeline from baseline configuration"""
        planned_events = []
        
        # Standard life events based on configuration
        base_events = [
            {
                'event_type': 'daycare_start',
                'planned_year': 0,
                'description': 'Daycare begins',
                'cash_flow_impact': 'negative',
                'impact_amount': -12000,  # Annual daycare cost
                'applies_if': self.baseline_config.get('HEL_WORK') == 'Full-time'
            },
            {
                'event_type': 'daycare_end', 
                'planned_year': 5,
                'description': 'Daycare ends',
                'cash_flow_impact': 'positive',
                'impact_amount': 12000,
                'applies_if': self.baseline_config.get('HEL_WORK') == 'Full-time'
            },
            {
                'event_type': 'education_start',
                'planned_year': 5,
                'description': f"{self.baseline_config.get('ED_PATH', 'Unknown')} education begins",
                'cash_flow_impact': 'negative',
                'impact_amount': -40000 if self.baseline_config.get('ED_PATH') == 'JohnsHopkins' else -20000,
                'applies_if': True
            },
            {
                'event_type': 'education_end',
                'planned_year': 10,
                'description': f"{self.baseline_config.get('ED_PATH', 'Unknown')} education completed",
                'cash_flow_impact': 'positive',
                'impact_amount': 40000 if self.baseline_config.get('ED_PATH') == 'JohnsHopkins' else 20000,
                'applies_if': True
            },
            {
                'event_type': 'charitable_giving',
                'planned_year': 0 if self.baseline_config.get('DON_STYLE') == 2 else 10,
                'description': 'Charitable donation',
                'cash_flow_impact': 'negative',
                'impact_amount': -25000,
                'applies_if': True
            },
            {
                'event_type': 'mortgage_payoff',
                'planned_year': 30,
                'description': 'Mortgage paid off',
                'cash_flow_impact': 'positive',
                'impact_amount': 36000,  # Annual mortgage payment
                'applies_if': True
            }
        ]
        
        # Filter events that apply to this configuration
        for event in base_events:
            if event['applies_if']:
                planned_events.append({
                    'event_id': f"PLANNED_{event['event_type'].upper()}",
                    'event_type': event['event_type'],
                    'planned_date': self.start_date + timedelta(days=event['planned_year']*365),
                    'planned_year': event['planned_year'],
                    'description': event['description'],
                    'cash_flow_impact': event['cash_flow_impact'],
                    'impact_amount': event['impact_amount'],
                    'status': 'planned'
                })
        
        return planned_events
    
    def log_actual_event(self, event_type, actual_date, description, 
                        cash_flow_impact, impact_amount, notes=""):
        """Log an actual life event that occurred"""
        
        event_entry = {
            'event_id': f"ACTUAL_{event_type.upper()}_{len(self.events_log)}",
            'event_type': event_type,
            'actual_date': actual_date,
            'description': description,
            'cash_flow_impact': cash_flow_impact,
            'impact_amount': impact_amount,
            'notes': notes,
            'logged_date': datetime.now(),
            'status': 'actual'
        }
        
        self.events_log.append(event_entry)
        
        # Check for configuration drift
        self._detect_config_drift(event_type, actual_date)
        
        return event_entry
    
    def _detect_config_drift(self, event_type, actual_date):
        """Detect if actual events are drifting from planned configuration"""
        
        # Find corresponding planned event
        planned_event = None
        for event in self.planned_events:
            if event['event_type'] == event_type:
                planned_event = event
                break
        
        if planned_event:
            planned_date = planned_event['planned_date']
            drift_days = (actual_date - planned_date).days
            drift_years = drift_days / 365.25
            
            if abs(drift_years) > 0.5:  # More than 6 months drift
                drift_entry = {
                    'drift_id': f"DRIFT_{len(self.config_drift_log)}",
                    'event_type': event_type,
                    'planned_date': planned_date,
                    'actual_date': actual_date,
                    'drift_days': drift_days,
                    'drift_years': drift_years,
                    'drift_severity': 'high' if abs(drift_years) > 2 else 'medium',
                    'detected_date': datetime.now()
                }
                
                self.config_drift_log.append(drift_entry)
                
                # Update current configuration if needed
                self._update_current_config(event_type, actual_date)
    
    def _update_current_config(self, event_type, actual_date):
        """Update current configuration based on actual events"""
        
        if event_type == 'education_path_change':
            # Client changed education path
            old_path = self.current_config.get('ED_PATH')
            # This would be determined from the event details
            # For now, assume switching to the other option
            new_path = 'McGill' if old_path == 'JohnsHopkins' else 'JohnsHopkins'
            self.current_config['ED_PATH'] = new_path
            
        elif event_type == 'work_arrangement_change':
            # Client changed work arrangement
            old_work = self.current_config.get('HEL_WORK')
            new_work = 'Part-time' if old_work == 'Full-time' else 'Full-time'
            self.current_config['HEL_WORK'] = new_work
        
        elif event_type == 'bonus_change':
            # Bonus expectations changed
            # This would be updated based on actual event details
            pass
    
    def conduct_quarterly_review(self, review_date, advisor_notes=""):
        """Log quarterly review and update event tracking"""
        
        review_entry = {
            'review_id': f"REVIEW_{len(self.quarterly_reviews)}",
            'review_date': review_date,
            'events_since_last_review': [],
            'config_changes': [],
            'drift_alerts': [],
            'advisor_notes': advisor_notes,
            'next_review_date': review_date + timedelta(days=90)
        }
        
        # Identify events since last review
        last_review_date = (self.quarterly_reviews[-1]['review_date'] 
                           if self.quarterly_reviews else self.start_date)
        
        recent_events = [
            event for event in self.events_log
            if event['logged_date'] > last_review_date
        ]
        
        review_entry['events_since_last_review'] = recent_events
        
        # Check for new configuration drifts
        recent_drifts = [
            drift for drift in self.config_drift_log
            if drift['detected_date'] > last_review_date
        ]
        
        review_entry['drift_alerts'] = recent_drifts
        
        self.quarterly_reviews.append(review_entry)
        
        return review_entry
    
    def generate_tracking_report(self):
        """Generate comprehensive tracking report"""
        
        report = {
            'client_id': self.client_id,
            'baseline_config': self.baseline_config,
            'current_config': self.current_config,
            'tracking_period_days': (datetime.now() - self.start_date).days,
            'total_events_logged': len(self.events_log),
            'total_drifts_detected': len(self.config_drift_log),
            'quarterly_reviews_conducted': len(self.quarterly_reviews),
            'configuration_accuracy': self._calculate_config_accuracy(),
            'cash_flow_variance': self._calculate_cash_flow_variance(),
            'next_scheduled_events': self._get_upcoming_events()
        }
        
        return report
    
    def _calculate_config_accuracy(self):
        """Calculate how accurate the original configuration has been"""
        
        if not self.events_log:
            return 1.0  # Perfect accuracy if no events yet
        
        total_events = len(self.planned_events)
        drifted_events = len(self.config_drift_log)
        
        return max(0, (total_events - drifted_events) / total_events) if total_events > 0 else 1.0
    
    def _calculate_cash_flow_variance(self):
        """Calculate variance between planned and actual cash flows"""
        
        planned_total = sum(event['impact_amount'] for event in self.planned_events)
        actual_total = sum(event['impact_amount'] for event in self.events_log)
        
        if planned_total == 0:
            return 0
        
        return (actual_total - planned_total) / abs(planned_total)
    
    def _get_upcoming_events(self, months_ahead=12):
        """Get events planned for the next N months"""
        
        cutoff_date = datetime.now() + timedelta(days=months_ahead*30)
        
        upcoming = [
            event for event in self.planned_events
            if datetime.now() < event['planned_date'] <= cutoff_date
        ]
        
        return sorted(upcoming, key=lambda x: x['planned_date'])
    
    def create_timeline_visualization(self):
        """Create visual timeline of planned vs. actual events"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Event timeline
        planned_dates = [event['planned_date'] for event in self.planned_events]
        planned_amounts = [event['impact_amount'] for event in self.planned_events]
        
        actual_dates = [event['actual_date'] for event in self.events_log if 'actual_date' in event]
        actual_amounts = [event['impact_amount'] for event in self.events_log if 'actual_date' in event]
        
        ax1.scatter(planned_dates, planned_amounts, c='blue', alpha=0.7, s=100, label='Planned Events')
        ax1.scatter(actual_dates, actual_amounts, c='red', alpha=0.7, s=100, label='Actual Events')
        
        # Connect planned to actual events
        for planned_event in self.planned_events:
            for actual_event in self.events_log:
                if (planned_event['event_type'] == actual_event['event_type'] and 
                    'actual_date' in actual_event):
                    ax1.plot([planned_event['planned_date'], actual_event['actual_date']],
                            [planned_event['impact_amount'], actual_event['impact_amount']],
                            'gray', alpha=0.5, linestyle='--')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Life Events Timeline: Planned vs. Actual', fontweight='bold')
        ax1.set_ylabel('Cash Flow Impact ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Configuration drift over time
        if self.config_drift_log:
            drift_dates = [drift['detected_date'] for drift in self.config_drift_log]
            drift_years = [abs(drift['drift_years']) for drift in self.config_drift_log]
            
            ax2.bar(drift_dates, drift_years, alpha=0.7, color='orange')
            ax2.set_title('Configuration Drift Detection', fontweight='bold')
            ax2.set_ylabel('Drift (Years)')
            ax2.set_xlabel('Detection Date')
        else:
            ax2.text(0.5, 0.5, 'No Configuration Drift Detected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Configuration Drift Detection', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = f'life_events_timeline_{self.client_id}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def export_tracking_data(self):
        """Export all tracking data to CSV files"""
        
        # Events log
        if self.events_log:
            events_df = pd.DataFrame(self.events_log)
            events_df.to_csv(f'life_events_log_{self.client_id}.csv', index=False)
        
        # Planned events
        planned_df = pd.DataFrame(self.planned_events)
        planned_df.to_csv(f'planned_events_{self.client_id}.csv', index=False)
        
        # Configuration drift
        if self.config_drift_log:
            drift_df = pd.DataFrame(self.config_drift_log)
            drift_df.to_csv(f'config_drift_log_{self.client_id}.csv', index=False)
        
        # Quarterly reviews
        if self.quarterly_reviews:
            reviews_df = pd.DataFrame(self.quarterly_reviews)
            reviews_df.to_csv(f'quarterly_reviews_{self.client_id}.csv', index=False)
        
        return {
            'events_log': f'life_events_log_{self.client_id}.csv',
            'planned_events': f'planned_events_{self.client_id}.csv',
            'config_drift': f'config_drift_log_{self.client_id}.csv',
            'quarterly_reviews': f'quarterly_reviews_{self.client_id}.csv'
        }

def demo_life_events_tracking():
    """Demonstrate life events tracking system"""
    
    print("📋 LIFE EVENTS TRACKING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create tracker for sample client
    baseline_config = {
        'ED_PATH': 'JohnsHopkins',
        'HEL_WORK': 'Full-time',
        'BONUS_PCT': 0.30,
        'DON_STYLE': 0,
        'RISK_BAND': 3,
        'FX_SCENARIO': 'Base'
    }
    
    tracker = LifeEventsTracker('CLIENT_001', baseline_config)
    
    print(f"🎯 Client: CLIENT_001")
    print(f"📊 Baseline Configuration:")
    for key, value in baseline_config.items():
        print(f"   • {key}: {value}")
    
    print(f"\n📅 Planned Events Timeline:")
    for event in tracker.planned_events:
        print(f"   Year {event['planned_year']}: {event['description']} "
              f"(${event['impact_amount']:,} impact)")
    
    # Simulate some actual events with timing differences
    print(f"\n🔄 SIMULATING ACTUAL LIFE EVENTS:")
    
    # Education decision made 6 months early
    early_education_decision = datetime.now() + timedelta(days=4*365 + 180)  # 4.5 years instead of 5
    tracker.log_actual_event(
        event_type='education_path_change',
        actual_date=early_education_decision,
        description='Decided to switch to McGill University',
        cash_flow_impact='positive',
        impact_amount=20000,  # Savings from switching
        notes='Concerned about Johns Hopkins costs, switched to McGill'
    )
    
    # Work arrangement change
    work_change = datetime.now() + timedelta(days=2*365)  # 2 years from now
    tracker.log_actual_event(
        event_type='work_arrangement_change',
        actual_date=work_change,
        description='Switched to part-time work',
        cash_flow_impact='mixed',
        impact_amount=-15000,  # Income reduction but daycare savings
        notes='Work-life balance concerns, reduced hours'
    )
    
    # Unexpected bonus change
    bonus_change = datetime.now() + timedelta(days=365)  # 1 year from now
    tracker.log_actual_event(
        event_type='bonus_change',
        actual_date=bonus_change,
        description='Bonus reduced to 15%',
        cash_flow_impact='negative',
        impact_amount=-7500,  # 15% reduction in expected bonus
        notes='Company performance affected bonus structure'
    )
    
    print(f"   ✅ Education path change logged (6 months early)")
    print(f"   ✅ Work arrangement change logged")
    print(f"   ✅ Bonus expectation change logged")
    
    # Conduct quarterly reviews
    print(f"\n📊 QUARTERLY REVIEWS:")
    
    # First review (3 months)
    review1_date = datetime.now() + timedelta(days=90)
    review1 = tracker.conduct_quarterly_review(
        review1_date, 
        "Client expressing concerns about education costs"
    )
    print(f"   📅 Q1 Review: {review1['review_date'].strftime('%B %Y')}")
    print(f"       Events since last: {len(review1['events_since_last_review'])}")
    print(f"       Drift alerts: {len(review1['drift_alerts'])}")
    
    # Second review (6 months)
    review2_date = datetime.now() + timedelta(days=180)
    review2 = tracker.conduct_quarterly_review(
        review2_date,
        "Work-life balance discussions ongoing"
    )
    print(f"   📅 Q2 Review: {review2['review_date'].strftime('%B %Y')}")
    print(f"       Events since last: {len(review2['events_since_last_review'])}")
    print(f"       Drift alerts: {len(review2['drift_alerts'])}")
    
    # Generate tracking report
    print(f"\n📊 TRACKING REPORT:")
    report = tracker.generate_tracking_report()
    
    print(f"   📈 Configuration Accuracy: {report['configuration_accuracy']:.1%}")
    print(f"   💰 Cash Flow Variance: {report['cash_flow_variance']:+.1%}")
    print(f"   🚨 Total Drifts Detected: {report['total_drifts_detected']}")
    print(f"   📋 Events Logged: {report['total_events_logged']}")
    
    print(f"\n🎯 CONFIGURATION DRIFT ANALYSIS:")
    if tracker.config_drift_log:
        for drift in tracker.config_drift_log:
            print(f"   ⚠️  {drift['event_type']}: {drift['drift_years']:.1f} years drift")
    else:
        print(f"   ✅ No significant configuration drift detected")
    
    print(f"\n📅 UPCOMING EVENTS (Next 12 months):")
    upcoming = report['next_scheduled_events']
    if upcoming:
        for event in upcoming:
            print(f"   📌 {event['planned_date'].strftime('%B %Y')}: {event['description']}")
    else:
        print(f"   📌 No major events scheduled in next 12 months")
    
    # Create visualizations and export data
    print(f"\n📊 GENERATING OUTPUTS:")
    timeline_path = tracker.create_timeline_visualization()
    export_files = tracker.export_tracking_data()
    
    print(f"   🎨 Timeline visualization: {timeline_path}")
    print(f"   📁 Exported tracking data:")
    for file_type, filename in export_files.items():
        if Path(filename).exists():
            print(f"      • {file_type}: {filename}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🎯 Real life events deviate from planned timelines")
    print(f"   📊 Tracking actual events prevents configuration drift")
    print(f"   🔄 Quarterly reviews essential for course correction")
    print(f"   💰 Cash flow impacts can be significantly different")
    print(f"   📈 Visual timelines help communicate variance to clients")
    
    return tracker

if __name__ == "__main__":
    tracker = demo_life_events_tracking() 