#!/usr/bin/env python
"""
Timeline Bias Engine - Case Data-Driven Event Timing Estimation
Author: Claude 2025-07-16

Uses database of historical client cases to bias timeline estimates for realistic 
life event forecasting when users don't provide exact dates. Controls for income,
demographics, and life circumstances to provide accurate probabilistic timing.

Key Features:
- Income-controlled timeline bias from case database
- Demographic similarity matching for event timing
- Probabilistic event windows with confidence intervals
- Lifestyle choice impact on timing patterns
- Continuous forecasting with uncertainty quantification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class TimelineBias:
    """Represents bias information for event timing"""
    event_type: str
    median_age: float
    std_dev: float
    income_correlation: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    demographic_factors: Dict[str, float]

@dataclass
class ClientTimeline:
    """Client timeline with bias-adjusted estimates"""
    client_id: str
    events: List[Dict[str, Any]]
    bias_adjustments: List[TimelineBias]
    confidence_score: float
    similar_cases: List[str]

class CaseDatabase:
    """Database of historical client cases for timeline bias"""
    
    def __init__(self, db_path: str = "data/case_database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the case database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                client_age INTEGER,
                income_level TEXT,
                income_amount REAL,
                education_level TEXT,
                family_status TEXT,
                location TEXT,
                career_stage TEXT,
                creation_date TEXT
            )
        ''')
        
        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS life_events (
                event_id TEXT PRIMARY KEY,
                case_id TEXT,
                event_type TEXT,
                event_date TEXT,
                client_age_at_event REAL,
                financial_impact REAL,
                stress_level REAL,
                confidence REAL,
                description TEXT,
                FOREIGN KEY (case_id) REFERENCES cases (case_id)
            )
        ''')
        
        # Insert sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM cases")
        if cursor.fetchone()[0] == 0:
            self._populate_sample_data(conn)
        
        conn.commit()
        conn.close()
    
    def _populate_sample_data(self, conn):
        """Populate database with realistic sample cases"""
        import random
        random.seed(42)
        np.random.seed(42)
        
        # Generate 1000 sample cases
        cases = []
        events = []
        
        for i in range(1000):
            case_id = f"CASE_{i:04d}"
            
            # Generate client profile
            age = np.random.normal(35, 8)
            age = max(22, min(65, age))
            
            income_level = np.random.choice(['low', 'middle', 'high', 'ultra_high'], 
                                          p=[0.2, 0.5, 0.25, 0.05])
            
            income_ranges = {
                'low': (30000, 60000),
                'middle': (60000, 120000), 
                'high': (120000, 300000),
                'ultra_high': (300000, 1000000)
            }
            income = np.random.uniform(*income_ranges[income_level])
            
            education = np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], 
                                       p=[0.1, 0.4, 0.4, 0.1])
            
            family_status = np.random.choice(['single', 'married', 'divorced', 'widowed'],
                                           p=[0.3, 0.6, 0.08, 0.02])
            
            location = np.random.choice(['urban', 'suburban', 'rural'], p=[0.4, 0.5, 0.1])
            
            if age < 30:
                career_stage = 'early_career'
            elif age < 45:
                career_stage = 'mid_career'
            elif age < 55:
                career_stage = 'established'
            else:
                career_stage = 'pre_retirement'
            
            cases.append((case_id, int(age), income_level, income, education, 
                         family_status, location, career_stage, 
                         datetime.now().isoformat()))
            
            # Generate realistic life events for this case
            case_events = self._generate_case_events(case_id, age, income, income_level, 
                                                   education, family_status, career_stage)
            events.extend(case_events)
        
        # Insert into database
        conn.executemany('''
            INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', cases)
        
        conn.executemany('''
            INSERT INTO life_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', events)
    
    def _generate_case_events(self, case_id: str, current_age: float, income: float, 
                            income_level: str, education: str, family_status: str, 
                            career_stage: str) -> List[Tuple]:
        """Generate realistic life events for a case"""
        events = []
        event_counter = 0
        
        # Event probability and timing patterns based on demographics
        event_patterns = {
            'house_purchase': {
                'probability': 0.7,
                'age_range': (25, 40),
                'income_factor': 1.2 if income > 80000 else 0.8,
                'timing_std': 3.0
            },
            'car_purchase': {
                'probability': 0.9,
                'age_range': (22, 65),
                'income_factor': 1.0,
                'timing_std': 2.0
            },
            'education_completion': {
                'probability': 0.3 if education in ['masters', 'phd'] else 0.1,
                'age_range': (22, 35),
                'income_factor': 0.9,
                'timing_std': 2.5
            },
            'career_advancement': {
                'probability': 0.8,
                'age_range': (25, 50),
                'income_factor': 1.1 if income > 100000 else 0.9,
                'timing_std': 4.0
            },
            'marriage': {
                'probability': 0.6 if family_status == 'single' else 0.0,
                'age_range': (25, 40),
                'income_factor': 1.0,
                'timing_std': 5.0
            },
            'child_birth': {
                'probability': 0.5 if family_status == 'married' else 0.2,
                'age_range': (25, 42),
                'income_factor': 0.9,
                'timing_std': 4.0
            },
            'investment_milestone': {
                'probability': 0.6,
                'age_range': (30, 55),
                'income_factor': 1.3 if income > 120000 else 0.7,
                'timing_std': 5.0
            },
            'health_event': {
                'probability': 0.3,
                'age_range': (35, 65),
                'income_factor': 1.0,
                'timing_std': 8.0
            }
        }
        
        for event_type, pattern in event_patterns.items():
            if np.random.random() < pattern['probability']:
                # Calculate event age based on pattern and demographics
                min_age, max_age = pattern['age_range']
                
                # Adjust for income
                income_adj = pattern['income_factor']
                if income_adj > 1.0:  # High income events happen earlier
                    target_age = min_age + (max_age - min_age) * 0.3
                else:  # Low income events happen later
                    target_age = min_age + (max_age - min_age) * 0.7
                
                # Add randomness
                event_age = np.random.normal(target_age, pattern['timing_std'])
                event_age = max(min_age, min(max_age, event_age))
                
                # Only include events that could have happened by now
                if event_age <= current_age:
                    event_date = datetime.now() - timedelta(days=(current_age - event_age) * 365)
                    
                    # Calculate financial impact based on event type and income
                    impact_ranges = {
                        'house_purchase': (-400000, -200000),
                        'car_purchase': (-40000, -15000),
                        'education_completion': (-60000, -20000),
                        'career_advancement': (20000, 80000),
                        'marriage': (-15000, 5000),
                        'child_birth': (-25000, -10000),
                        'investment_milestone': (50000, 200000),
                        'health_event': (-15000, -5000)
                    }
                    
                    min_impact, max_impact = impact_ranges[event_type]
                    financial_impact = np.random.uniform(min_impact, max_impact)
                    
                    # Adjust impact for income level
                    income_multiplier = {
                        'low': 0.6, 'middle': 1.0, 'high': 1.5, 'ultra_high': 2.5
                    }
                    financial_impact *= income_multiplier[income_level]
                    
                    # Calculate stress level (higher for negative events)
                    if financial_impact < 0:
                        stress_level = min(1.0, abs(financial_impact) / income * 2)
                    else:
                        stress_level = max(0.1, 0.3 - financial_impact / income)
                    
                    confidence = np.random.uniform(0.7, 0.95)
                    
                    event_id = f"{case_id}_EVENT_{event_counter:03d}"
                    event_counter += 1
                    
                    events.append((
                        event_id, case_id, event_type, event_date.isoformat(),
                        event_age, financial_impact, stress_level, confidence,
                        f"{event_type.replace('_', ' ').title()} at age {event_age:.1f}"
                    ))
        
        return events
    
    def get_similar_cases(self, target_profile: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Find cases similar to target profile"""
        conn = sqlite3.connect(self.db_path)
        
        # Build similarity query
        age_range = 5  # ±5 years
        income_tolerance = 0.3  # ±30%
        
        query = '''
            SELECT * FROM cases 
            WHERE client_age BETWEEN ? AND ?
            AND income_amount BETWEEN ? AND ?
            AND income_level = ?
            LIMIT ?
        '''
        
        target_age = target_profile.get('age', 35)
        target_income = target_profile.get('income', 75000)
        target_income_level = target_profile.get('income_level', 'middle')
        
        params = (
            target_age - age_range,
            target_age + age_range,
            target_income * (1 - income_tolerance),
            target_income * (1 + income_tolerance),
            target_income_level,
            limit
        )
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df.to_dict('records')
    
    def get_event_timing_bias(self, event_type: str, client_profile: Dict[str, Any]) -> TimelineBias:
        """Get timing bias for specific event type based on similar cases"""
        similar_cases = self.get_similar_cases(client_profile)
        
        if not similar_cases:
            # Fallback to general patterns
            return self._get_default_bias(event_type, client_profile)
        
        case_ids = [case['case_id'] for case in similar_cases]
        
        conn = sqlite3.connect(self.db_path)
        
        # Get events of this type from similar cases
        placeholders = ','.join('?' * len(case_ids))
        query = f'''
            SELECT client_age_at_event, financial_impact, stress_level, confidence
            FROM life_events 
            WHERE case_id IN ({placeholders}) AND event_type = ?
        '''
        
        params = case_ids + [event_type]
        event_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if event_df.empty:
            return self._get_default_bias(event_type, client_profile)
        
        # Calculate statistics
        ages = event_df['client_age_at_event'].values
        median_age = np.median(ages)
        std_dev = np.std(ages)
        confidence_interval = np.percentile(ages, [25, 75])
        
        # Calculate income correlation
        income_correlation = 0.0  # Would need more sophisticated analysis
        
        # Calculate demographic factors
        demographic_factors = {
            'income_sensitivity': np.corrcoef(
                [case['income_amount'] for case in similar_cases], 
                ages
            )[0, 1] if len(ages) > 1 else 0.0
        }
        
        return TimelineBias(
            event_type=event_type,
            median_age=median_age,
            std_dev=std_dev,
            income_correlation=income_correlation,
            confidence_interval=confidence_interval,
            sample_size=len(ages),
            demographic_factors=demographic_factors
        )
    
    def _get_default_bias(self, event_type: str, client_profile: Dict[str, Any]) -> TimelineBias:
        """Fallback bias when no similar cases found"""
        default_patterns = {
            'house_purchase': {'median_age': 32, 'std_dev': 6},
            'car_purchase': {'median_age': 28, 'std_dev': 8},
            'education_completion': {'median_age': 26, 'std_dev': 4},
            'career_advancement': {'median_age': 35, 'std_dev': 7},
            'marriage': {'median_age': 29, 'std_dev': 5},
            'child_birth': {'median_age': 31, 'std_dev': 5},
            'investment_milestone': {'median_age': 40, 'std_dev': 8},
            'health_event': {'median_age': 45, 'std_dev': 12}
        }
        
        pattern = default_patterns.get(event_type, {'median_age': 35, 'std_dev': 10})
        
        return TimelineBias(
            event_type=event_type,
            median_age=pattern['median_age'],
            std_dev=pattern['std_dev'],
            income_correlation=0.0,
            confidence_interval=(pattern['median_age'] - pattern['std_dev'], 
                               pattern['median_age'] + pattern['std_dev']),
            sample_size=0,
            demographic_factors={}
        )

class TimelineBiasEngine:
    """Main engine for timeline bias estimation"""
    
    def __init__(self, case_db_path: str = "data/case_database.db"):
        self.case_db = CaseDatabase(case_db_path)
        self.scaler = StandardScaler()
        
    def estimate_event_timeline_with_bias(self, events: List[Dict[str, Any]], 
                                        client_profile: Dict[str, Any]) -> ClientTimeline:
        """Estimate event timeline using case data bias"""
        
        # Find similar cases for this client profile
        similar_cases = self.case_db.get_similar_cases(client_profile)
        
        # Get bias adjustments for each event type
        bias_adjustments = []
        adjusted_events = []
        
        current_date = datetime.now()
        current_age = client_profile.get('age', 35)
        
        for event in events:
            event_type = event.get('type', 'unknown')
            
            # Get bias for this event type
            bias = self.case_db.get_event_timing_bias(event_type, client_profile)
            bias_adjustments.append(bias)
            
            # If event doesn't have a date, estimate using bias
            if 'estimated_date' not in event or not event['estimated_date']:
                
                # Calculate probable age for this event
                probable_age = bias.median_age
                
                # Adjust for client's current situation
                if current_age >= probable_age:
                    # Event likely happened in the past
                    years_ago = np.random.normal(
                        current_age - probable_age, 
                        bias.std_dev / 2
                    )
                    years_ago = max(0, years_ago)
                    event_age = current_age - years_ago
                else:
                    # Event likely to happen in the future
                    years_from_now = probable_age - current_age
                    event_age = probable_age
                    years_ago = -years_from_now  # Negative for future events
                
                # Calculate event date
                if years_ago >= 0:
                    event_date = current_date - timedelta(days=int(years_ago * 365))
                else:
                    event_date = current_date + timedelta(days=int(abs(years_ago) * 365))
                
                # Add bias-adjusted information to event
                adjusted_event = event.copy()
                adjusted_event.update({
                    'estimated_date': event_date.isoformat(),
                    'estimated_age': event_age,
                    'years_ago': years_ago,
                    'bias_median_age': bias.median_age,
                    'bias_confidence_interval': bias.confidence_interval,
                    'bias_sample_size': bias.sample_size,
                    'timeline_confidence': min(1.0, bias.sample_size / 10),  # Higher confidence with more data
                    'similar_cases_count': len(similar_cases)
                })
            else:
                adjusted_event = event.copy()
            
            adjusted_events.append(adjusted_event)
        
        # Calculate overall confidence score
        total_sample_size = sum(bias.sample_size for bias in bias_adjustments)
        confidence_score = min(1.0, total_sample_size / (len(events) * 20))
        
        return ClientTimeline(
            client_id=client_profile.get('client_id', 'unknown'),
            events=adjusted_events,
            bias_adjustments=bias_adjustments,
            confidence_score=confidence_score,
            similar_cases=[case['case_id'] for case in similar_cases[:10]]
        )
    
    def forecast_future_events(self, client_profile: Dict[str, Any], 
                             forecast_years: int = 10) -> List[Dict[str, Any]]:
        """Forecast probable future events based on case data"""
        
        current_age = client_profile.get('age', 35)
        forecast_age = current_age + forecast_years
        
        # Get typical events for similar clients in this age range
        similar_cases = self.case_db.get_similar_cases(client_profile)
        
        if not similar_cases:
            return []
        
        # Query events from similar cases that occurred in the forecast age range
        case_ids = [case['case_id'] for case in similar_cases]
        conn = sqlite3.connect(self.case_db.db_path)
        
        placeholders = ','.join('?' * len(case_ids))
        query = f'''
            SELECT event_type, AVG(client_age_at_event) as avg_age, 
                   COUNT(*) as frequency, AVG(financial_impact) as avg_impact
            FROM life_events 
            WHERE case_id IN ({placeholders}) 
            AND client_age_at_event BETWEEN ? AND ?
            GROUP BY event_type
            HAVING frequency >= 3
            ORDER BY frequency DESC
        '''
        
        params = case_ids + [current_age, forecast_age]
        forecast_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert to forecast events
        forecast_events = []
        current_date = datetime.now()
        
        for _, row in forecast_df.iterrows():
            probability = min(1.0, row['frequency'] / len(similar_cases))
            
            if probability > 0.2:  # Only include events with >20% probability
                years_from_now = row['avg_age'] - current_age
                event_date = current_date + timedelta(days=int(years_from_now * 365))
                
                forecast_events.append({
                    'type': row['event_type'],
                    'description': f"Forecasted {row['event_type'].replace('_', ' ')}",
                    'estimated_date': event_date.isoformat(),
                    'estimated_age': row['avg_age'],
                    'years_from_now': years_from_now,
                    'probability': probability,
                    'avg_financial_impact': row['avg_impact'],
                    'source': 'case_data_forecast',
                    'similar_cases_frequency': int(row['frequency'])
                })
        
        return forecast_events
    
    def get_timeline_uncertainty(self, events: List[Dict[str, Any]], 
                               client_profile: Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate uncertainty metrics for timeline estimates"""
        
        timeline = self.estimate_event_timeline_with_bias(events, client_profile)
        
        uncertainty_metrics = {
            'overall_confidence': timeline.confidence_score,
            'similar_cases_available': len(timeline.similar_cases),
            'event_uncertainties': []
        }
        
        for event, bias in zip(timeline.events, timeline.bias_adjustments):
            event_uncertainty = {
                'event_type': event['type'],
                'sample_size': bias.sample_size,
                'timing_std_dev': bias.std_dev,
                'confidence_interval_width': bias.confidence_interval[1] - bias.confidence_interval[0],
                'uncertainty_level': 'low' if bias.sample_size > 20 else 'medium' if bias.sample_size > 5 else 'high'
            }
            uncertainty_metrics['event_uncertainties'].append(event_uncertainty)
        
        return uncertainty_metrics

# Usage example and testing
def demo_timeline_bias_engine():
    """Demonstrate timeline bias engine functionality"""
    print("🎯 TIMELINE BIAS ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize engine
    engine = TimelineBiasEngine()
    
    # Sample client profile
    client_profile = {
        'client_id': 'TEST_CLIENT_001',
        'age': 32,
        'income': 95000,
        'income_level': 'middle',
        'education': 'bachelors',
        'family_status': 'married',
        'location': 'suburban'
    }
    
    # Sample events without exact dates
    events = [
        {'type': 'house_purchase', 'description': 'Planning to buy a house', 'amount': -350000},
        {'type': 'child_birth', 'description': 'Planning to have children', 'amount': -20000},
        {'type': 'car_purchase', 'description': 'Need a new car', 'amount': -30000},
        {'type': 'career_advancement', 'description': 'Expecting promotion', 'amount': 25000}
    ]
    
    print(f"📊 Client Profile:")
    for key, value in client_profile.items():
        print(f"   • {key}: {value}")
    
    print(f"\n📅 Events to Timeline (without exact dates):")
    for event in events:
        print(f"   • {event['description']} (${event['amount']:,})")
    
    # Get timeline with bias
    timeline = engine.estimate_event_timeline_with_bias(events, client_profile)
    
    print(f"\n🎯 BIAS-ADJUSTED TIMELINE:")
    print(f"Overall Confidence: {timeline.confidence_score:.2%}")
    print(f"Similar Cases Used: {len(timeline.similar_cases)}")
    
    for event in timeline.events:
        print(f"\n📍 {event['description']}:")
        print(f"   Estimated Date: {event['estimated_date'][:10]}")
        print(f"   Estimated Age: {event['estimated_age']:.1f}")
        print(f"   Bias Median Age: {event.get('bias_median_age', 'N/A')}")
        print(f"   Sample Size: {event.get('bias_sample_size', 0)}")
        print(f"   Timeline Confidence: {event.get('timeline_confidence', 0):.1%}")
    
    # Get future forecasts
    print(f"\n🔮 FUTURE EVENT FORECASTS:")
    future_events = engine.forecast_future_events(client_profile, forecast_years=10)
    
    for event in future_events[:5]:  # Show top 5
        print(f"   • {event['description']} ({event['probability']:.1%} probability)")
        print(f"     Expected at age {event['estimated_age']:.1f}")
        print(f"     Avg impact: ${event['avg_financial_impact']:,.0f}")
    
    # Get uncertainty metrics
    print(f"\n📊 UNCERTAINTY ANALYSIS:")
    uncertainty = engine.get_timeline_uncertainty(events, client_profile)
    
    print(f"Overall Confidence: {uncertainty['overall_confidence']:.1%}")
    print(f"Similar Cases: {uncertainty['similar_cases_available']}")
    
    for eu in uncertainty['event_uncertainties']:
        print(f"   • {eu['event_type']}: {eu['uncertainty_level']} uncertainty")
        print(f"     (sample size: {eu['sample_size']}, std: {eu['timing_std_dev']:.1f} years)")

if __name__ == "__main__":
    demo_timeline_bias_engine() 