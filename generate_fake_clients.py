#!/usr/bin/env python
"""
Generate Fake Client Data for Portfolio Simulation
Author: ChatGPT 2025-07-16

Generates realistic fake client data for testing portfolio backtesting and simulation.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

class FakeClientGenerator:
    """Generates realistic fake client data"""
    
    def __init__(self):
        self.client_types = {
            'young_professional': {
                'age_range': (25, 35),
                'income_range': (60000, 120000),
                'life_stage': 'early_career',
                'typical_events': ['education', 'work', 'housing', 'family']
            },
            'mid_career_executive': {
                'age_range': (35, 50),
                'income_range': (100000, 250000),
                'life_stage': 'mid_career',
                'typical_events': ['work', 'housing', 'family', 'financial', 'retirement']
            },
            'established_investor': {
                'age_range': (45, 60),
                'income_range': (150000, 500000),
                'life_stage': 'established',
                'typical_events': ['financial', 'retirement', 'charity', 'health']
            },
            'pre_retirement': {
                'age_range': (55, 65),
                'income_range': (80000, 200000),
                'life_stage': 'pre_retirement',
                'typical_events': ['retirement', 'health', 'charity', 'financial']
            },
            'retiree': {
                'age_range': (65, 80),
                'income_range': (40000, 150000),
                'life_stage': 'retirement',
                'typical_events': ['retirement', 'health', 'charity']
            }
        }
        
        self.event_templates = {
            'education': [
                {'description': 'MBA Degree', 'amount_range': (40000, 80000)},
                {'description': 'Professional Certification', 'amount_range': (2000, 10000)},
                {'description': 'Graduate School', 'amount_range': (30000, 60000)},
                {'description': 'Executive Education', 'amount_range': (5000, 25000)}
            ],
            'work': [
                {'description': 'Promotion', 'amount_range': (5000, 25000)},
                {'description': 'Job Change', 'amount_range': (10000, 50000)},
                {'description': 'Bonus', 'amount_range': (5000, 50000)},
                {'description': 'Stock Options', 'amount_range': (10000, 100000)}
            ],
            'housing': [
                {'description': 'House Purchase', 'amount_range': (200000, 800000)},
                {'description': 'Down Payment', 'amount_range': (40000, 160000)},
                {'description': 'Home Renovation', 'amount_range': (10000, 100000)},
                {'description': 'Property Tax', 'amount_range': (2000, 15000)}
            ],
            'family': [
                {'description': 'Wedding', 'amount_range': (15000, 50000)},
                {'description': 'Child Birth', 'amount_range': (2000, 10000)},
                {'description': 'Child Education', 'amount_range': (10000, 50000)},
                {'description': 'Family Vacation', 'amount_range': (3000, 20000)}
            ],
            'financial': [
                {'description': 'Investment Portfolio', 'amount_range': (50000, 500000)},
                {'description': 'Emergency Fund', 'amount_range': (10000, 50000)},
                {'description': 'Debt Payoff', 'amount_range': (5000, 50000)},
                {'description': 'Tax Payment', 'amount_range': (5000, 50000)}
            ],
            'retirement': [
                {'description': '401k Contribution', 'amount_range': (5000, 25000)},
                {'description': 'IRA Investment', 'amount_range': (3000, 15000)},
                {'description': 'Pension Plan', 'amount_range': (10000, 100000)},
                {'description': 'Retirement Planning', 'amount_range': (5000, 25000)}
            ],
            'health': [
                {'description': 'Medical Procedure', 'amount_range': (5000, 50000)},
                {'description': 'Health Insurance', 'amount_range': (2000, 15000)},
                {'description': 'Dental Work', 'amount_range': (1000, 10000)},
                {'description': 'Fitness Membership', 'amount_range': (500, 3000)}
            ],
            'charity': [
                {'description': 'Charitable Donation', 'amount_range': (1000, 25000)},
                {'description': 'Foundation Gift', 'amount_range': (5000, 100000)},
                {'description': 'Community Support', 'amount_range': (500, 10000)},
                {'description': 'Scholarship Fund', 'amount_range': (2000, 50000)}
            ]
        }
    
    def generate_client_profile(self, client_type: str = None) -> Dict[str, Any]:
        """Generate a realistic client profile"""
        if client_type is None:
            client_type = random.choice(list(self.client_types.keys()))
        
        profile_template = self.client_types[client_type]
        
        age = random.randint(*profile_template['age_range'])
        income = random.randint(*profile_template['income_range'])
        
        # Adjust income based on age and life stage
        if age < 30:
            income *= 0.8
        elif age > 60:
            income *= 0.7
        
        return {
            'client_type': client_type,
            'age': age,
            'income': int(income),
            'life_stage': profile_template['life_stage'],
            'income_level': self._categorize_income(income),
            'confidence': random.uniform(0.7, 0.95)
        }
    
    def _categorize_income(self, income: float) -> str:
        """Categorize income level"""
        if income < 50000:
            return 'low_income'
        elif income < 100000:
            return 'middle_income'
        elif income < 200000:
            return 'high_income'
        else:
            return 'ultra_high_income'
    
    def generate_life_events(self, profile: Dict[str, Any], num_events: int = 8) -> List[Dict[str, Any]]:
        """Generate realistic life events for the client"""
        events = []
        current_date = datetime.now()
        
        # Get typical events for this client type
        typical_events = self.client_types[profile['client_type']]['typical_events']
        
        # Generate events over the client's life
        for i in range(num_events):
            # Choose event type
            event_type = random.choice(typical_events)
            
            # Get event template
            template = random.choice(self.event_templates[event_type])
            
            # Generate amount based on income
            amount_range = template['amount_range']
            income_factor = profile['income'] / 100000  # Normalize to 100k
            adjusted_range = (
                int(amount_range[0] * income_factor * 0.5),
                int(amount_range[1] * income_factor * 1.5)
            )
            amount = random.randint(*adjusted_range)
            
            # Estimate when this event occurred
            years_ago = random.uniform(0, profile['age'] - 18)  # Assume adult life starts at 18
            event_date = current_date - timedelta(days=int(years_ago * 365))
            
            # Calculate estimated age when event occurred
            estimated_age = profile['age'] - years_ago
            
            events.append({
                'type': event_type,
                'description': template['description'],
                'amount': amount,
                'estimated_date': event_date.isoformat(),
                'years_ago': round(years_ago, 1),
                'estimated_age': int(estimated_age),
                'confidence': random.uniform(0.7, 0.95),
                'life_stage_when_occurred': self._get_life_stage_for_age(estimated_age),
                'timeline_confidence': random.uniform(0.6, 0.9)
            })
        
        # Sort by estimated date (most recent first)
        events.sort(key=lambda x: x['estimated_date'], reverse=True)
        
        return events
    
    def _get_life_stage_for_age(self, age: float) -> str:
        """Get life stage for a given age"""
        if age < 30:
            return 'early_career'
        elif age < 45:
            return 'mid_career'
        elif age < 55:
            return 'established'
        elif age < 65:
            return 'pre_retirement'
        else:
            return 'retirement'
    
    def generate_portfolio_data(self, profile: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate portfolio data for backtesting"""
        # Calculate portfolio value based on income and events
        base_portfolio = profile['income'] * 0.3  # Assume 30% of income in investments
        
        # Adjust based on financial events
        financial_events = [e for e in events if e['type'] == 'financial']
        for event in financial_events:
            if 'Investment' in event['description']:
                base_portfolio += event['amount']
        
        # Generate historical portfolio values
        portfolio_history = []
        current_value = base_portfolio
        
        for i in range(60):  # 5 years of monthly data
            # Add some market volatility
            monthly_return = random.uniform(-0.05, 0.08)  # -5% to +8% monthly
            current_value *= (1 + monthly_return)
            
            # Add contributions based on income
            monthly_contribution = profile['income'] * 0.1 / 12  # 10% of income monthly
            current_value += monthly_contribution
            
            portfolio_history.append({
                'date': (datetime.now() - timedelta(days=30*i)).isoformat(),
                'value': max(0, current_value),
                'contribution': monthly_contribution,
                'return': monthly_return
            })
        
        return {
            'current_value': current_value,
            'historical_data': portfolio_history,
            'allocation': {
                'stocks': random.uniform(0.4, 0.8),
                'bonds': random.uniform(0.1, 0.4),
                'cash': random.uniform(0.05, 0.2),
                'alternatives': random.uniform(0.0, 0.1)
            }
        }
    
    def generate_client_data(self, num_clients: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple client datasets"""
        clients = []
        
        for i in range(num_clients):
            # Generate profile
            profile = self.generate_client_profile()
            
            # Generate events
            events = self.generate_life_events(profile)
            
            # Generate portfolio data
            portfolio = self.generate_portfolio_data(profile, events)
            
            client_data = {
                'client_id': f"CLIENT_{i+1:03d}",
                'profile': profile,
                'events': events,
                'portfolio': portfolio,
                'generated_at': datetime.now().isoformat()
            }
            
            clients.append(client_data)
        
        return clients

def main():
    """Generate fake client data"""
    generator = FakeClientGenerator()
    
    print("🎲 Generating Fake Client Data...")
    
    # Generate 5 clients
    clients = generator.generate_client_data(5)
    
    # Save to file
    output_file = "data/outputs/analysis_data/fake_clients.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(clients, f, indent=2)
    
    print(f"✅ Generated {len(clients)} fake clients")
    print(f"📁 Saved to: {output_file}")
    
    # Print summary
    for client in clients:
        profile = client['profile']
        events = client['events']
        portfolio = client['portfolio']
        
        print(f"\n👤 {client['client_id']}:")
        print(f"   Age: {profile['age']}, Income: ${profile['income']:,}")
        print(f"   Life Stage: {profile['life_stage']}")
        print(f"   Events: {len(events)}")
        print(f"   Portfolio Value: ${portfolio['current_value']:,.0f}")

if __name__ == "__main__":
    import os
    main() 