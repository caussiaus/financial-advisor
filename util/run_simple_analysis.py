#!/usr/bin/env python3
"""
Simple analysis script that leverages existing working modules from the codebase.
"""

import os
import json
import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append('src')
sys.path.append('.')

def load_person_data(person_dir: str) -> Dict:
    """Load all data for a single person"""
    person_data = {}
    
    # Load each JSON file
    for filename in ['profile.json', 'financial_state.json', 'goals.json', 'life_events.json', 'preferences.json']:
        filepath = os.path.join(person_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                person_data[filename.replace('.json', '')] = json.load(f)
    
    return person_data

def analyze_financial_profiles(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Analyze financial profiles using basic statistics"""
    print(f"💰 Analyzing financial profiles for {len(people_data)} people...")
    
    analysis = {
        'total_people': len(people_data),
        'age_distribution': [],
        'income_distribution': [],
        'wealth_distribution': [],
        'risk_tolerance_distribution': {},
        'family_status_distribution': {},
        'asset_allocation': {},
        'liability_distribution': [],
        'goal_analysis': {},
        'life_events_analysis': {}
    }
    
    for person_id, person_data in people_data:
        profile = person_data['profile']
        financial_state = person_data['financial_state']
        goals = person_data['goals']
        life_events = person_data['life_events']
        preferences = person_data['preferences']
        
        # Age analysis
        analysis['age_distribution'].append(profile['age'])
        
        # Income analysis
        annual_salary = financial_state['income']['annual_salary']
        analysis['income_distribution'].append(annual_salary)
        
        # Wealth analysis
        assets = sum(financial_state['assets'].values())
        liabilities = sum(financial_state['liabilities'].values())
        net_worth = assets - liabilities
        analysis['wealth_distribution'].append(net_worth)
        
        # Risk tolerance
        risk = profile['risk_tolerance']
        analysis['risk_tolerance_distribution'][risk] = analysis['risk_tolerance_distribution'].get(risk, 0) + 1
        
        # Family status
        family_status = profile['family_status']
        analysis['family_status_distribution'][family_status] = analysis['family_status_distribution'].get(family_status, 0) + 1
        
        # Asset allocation
        for asset_type, amount in financial_state['assets'].items():
            if amount > 0:
                analysis['asset_allocation'][asset_type] = analysis['asset_allocation'].get(asset_type, 0) + 1
        
        # Liability analysis
        analysis['liability_distribution'].append(liabilities)
        
        # Goal analysis
        total_goals = len(goals.get('short_term_goals', [])) + len(goals.get('medium_term_goals', [])) + len(goals.get('long_term_goals', []))
        analysis['goal_analysis']['total_goals'] = analysis['goal_analysis'].get('total_goals', 0) + total_goals
        
        # Life events analysis
        total_events = len(life_events.get('past_events', [])) + len(life_events.get('planned_events', []))
        analysis['life_events_analysis']['total_events'] = analysis['life_events_analysis'].get('total_events', 0) + total_events
    
    # Calculate statistics
    analysis['age_stats'] = {
        'mean': float(np.mean(analysis['age_distribution'])),
        'std': float(np.std(analysis['age_distribution'])),
        'min': float(np.min(analysis['age_distribution'])),
        'max': float(np.max(analysis['age_distribution']))
    }
    
    analysis['income_stats'] = {
        'mean': float(np.mean(analysis['income_distribution'])),
        'std': float(np.std(analysis['income_distribution'])),
        'min': float(np.min(analysis['income_distribution'])),
        'max': float(np.max(analysis['income_distribution']))
    }
    
    analysis['wealth_stats'] = {
        'mean': float(np.mean(analysis['wealth_distribution'])),
        'std': float(np.std(analysis['wealth_distribution'])),
        'min': float(np.min(analysis['wealth_distribution'])),
        'max': float(np.max(analysis['wealth_distribution']))
    }
    
    return analysis

def analyze_investment_preferences(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Analyze investment preferences across people"""
    print(f"📈 Analyzing investment preferences...")
    
    analysis = {
        'preferred_asset_classes': {},
        'avoided_asset_classes': {},
        'target_allocations': [],
        'liquidity_needs': [],
        'tax_considerations': {},
        'constraints': {}
    }
    
    for person_id, person_data in people_data:
        preferences = person_data['preferences']
        
        # Preferred asset classes
        for asset_class in preferences['investment_preferences']['preferred_asset_classes']:
            analysis['preferred_asset_classes'][asset_class] = analysis['preferred_asset_classes'].get(asset_class, 0) + 1
        
        # Avoided asset classes
        for asset_class in preferences['investment_preferences']['avoided_asset_classes']:
            analysis['avoided_asset_classes'][asset_class] = analysis['avoided_asset_classes'].get(asset_class, 0) + 1
        
        # Target allocations
        analysis['target_allocations'].append(preferences['investment_preferences']['target_asset_allocation'])
        
        # Liquidity needs
        analysis['liquidity_needs'].append(preferences['liquidity_needs']['emergency_fund_target'])
        
        # Tax considerations
        tax_bracket = preferences['tax_considerations']['tax_bracket']
        analysis['tax_considerations'][tax_bracket] = analysis['tax_considerations'].get(tax_bracket, 0) + 1
        
        # Constraints
        ethical_investing = preferences['constraints']['ethical_investing']
        analysis['constraints']['ethical_investing'] = analysis['constraints'].get('ethical_investing', 0) + (1 if ethical_investing else 0)
    
    return analysis

def analyze_life_events_patterns(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Analyze patterns in life events"""
    print(f"📅 Analyzing life events patterns...")
    
    analysis = {
        'event_categories': {},
        'financial_impacts': [],
        'timing_patterns': {},
        'probability_distribution': []
    }
    
    for person_id, person_data in people_data:
        life_events = person_data['life_events']
        
        # Past events
        for event in life_events.get('past_events', []):
            category = event['category']
            analysis['event_categories'][category] = analysis['event_categories'].get(category, 0) + 1
            analysis['financial_impacts'].append(event['financial_impact'])
        
        # Planned events
        for event in life_events.get('planned_events', []):
            category = event['category']
            analysis['event_categories'][category] = analysis['event_categories'].get(category, 0) + 1
            analysis['financial_impacts'].append(event['expected_impact'])
            analysis['probability_distribution'].append(event['probability'])
            
            # Timing analysis
            event_date = datetime.fromisoformat(event['date'])
            days_from_now = (event_date - datetime.now()).days
            if days_from_now > 0:
                if days_from_now <= 365:
                    timing_key = 'within_1_year'
                elif days_from_now <= 1095:
                    timing_key = '1_3_years'
                elif days_from_now <= 1825:
                    timing_key = '3_5_years'
                else:
                    timing_key = '5+_years'
                analysis['timing_patterns'][timing_key] = analysis['timing_patterns'].get(timing_key, 0) + 1
    
    # Calculate financial impact statistics
    if analysis['financial_impacts']:
        analysis['financial_impact_stats'] = {
            'mean': float(np.mean(analysis['financial_impacts'])),
            'std': float(np.std(analysis['financial_impacts'])),
            'min': float(np.min(analysis['financial_impacts'])),
            'max': float(np.max(analysis['financial_impacts']))
        }
    
    # Calculate probability statistics
    if analysis['probability_distribution']:
        analysis['probability_stats'] = {
            'mean': float(np.mean(analysis['probability_distribution'])),
            'std': float(np.std(analysis['probability_distribution'])),
            'min': float(np.min(analysis['probability_distribution'])),
            'max': float(np.max(analysis['probability_distribution']))
        }
    
    return analysis

def create_similarity_matrix(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Create similarity matrix based on key characteristics"""
    print(f"🔗 Creating similarity matrix...")
    
    # Extract key characteristics for similarity
    characteristics = []
    for person_id, person_data in people_data:
        profile = person_data['profile']
        financial_state = person_data['financial_state']
        
        char_vector = [
            profile['age'],
            profile['income_level'] == 'high',  # Binary encoding
            profile['risk_tolerance'] == 'high',
            sum(financial_state['assets'].values()),
            sum(financial_state['liabilities'].values()),
            len(person_data['life_events']['planned_events'])
        ]
        characteristics.append(char_vector)
    
    # Calculate similarity matrix
    n = len(characteristics)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # Simple Euclidean distance-based similarity
                distance = np.linalg.norm(np.array(characteristics[i]) - np.array(characteristics[j]))
                similarity_matrix[i][j] = 1.0 / (1.0 + distance)
    
    return {
        'similarity_matrix': similarity_matrix.tolist(),
        'characteristics': characteristics,
        'person_ids': [person_id for person_id, _ in people_data]
    }

def main(people_dir: str):
    """Main analysis pipeline using existing codebase patterns"""
    print(f"🚀 Starting simple analysis on people data in {people_dir}")
    
    # Load all people data
    people_data = []
    for person_dir in sorted(os.listdir(people_dir)):
        person_path = os.path.join(people_dir, person_dir)
        if os.path.isdir(person_path):
            person_data = load_person_data(person_path)
            if person_data:
                people_data.append((person_dir, person_data))
    
    print(f"📋 Loaded {len(people_data)} people for analysis")
    
    # Run different types of analysis
    results = {
        'financial_profiles': analyze_financial_profiles(people_data),
        'investment_preferences': analyze_investment_preferences(people_data),
        'life_events_patterns': analyze_life_events_patterns(people_data),
        'similarity_matrix': create_similarity_matrix(people_data)
    }
    
    # Save results
    output_file = f"data/outputs/analysis_data/simple_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"   - Total people analyzed: {results['financial_profiles']['total_people']}")
    print(f"   - Average age: {results['financial_profiles']['age_stats']['mean']:.1f}")
    print(f"   - Average income: ${results['financial_profiles']['income_stats']['mean']:,.0f}")
    print(f"   - Average net worth: ${results['financial_profiles']['wealth_stats']['mean']:,.0f}")
    print(f"   - Most common risk tolerance: {max(results['financial_profiles']['risk_tolerance_distribution'], key=results['financial_profiles']['risk_tolerance_distribution'].get)}")
    print(f"   - Total life events: {results['life_events_patterns']['event_categories']}")
    print(f"   - Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people-dir", type=str, default="data/inputs/people/current",
                       help="Directory containing people data")
    args = parser.parse_args()
    
    main(args.people_dir) 