#!/usr/bin/env python3
"""
Single-use utility to run mesh analysis on generated people data.

Usage:
    python util/run_mesh_analysis.py --people-dir data/inputs/people/current
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

try:
    from src.core.stochastic_mesh_engine import StochasticMeshEngine
    from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine, SeedEvent
    from src.analysis.mesh_congruence_engine import MeshCongruenceEngine
    from src.analysis.mesh_vector_database import MeshVectorDatabase
    from src.utilities.adaptive_mesh_generator import AdaptiveMeshGenerator
    from src.accounting_reconciliation import AccountingReconciliationEngine
    from src.financial_recommendation_engine import FinancialRecommendationEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the src/ directory")
    sys.exit(1)

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

def convert_life_events_to_milestones(life_events: Dict) -> List[Dict]:
    """Convert life events to milestone format for mesh engine"""
    milestones = []
    
    # Add past events
    for event in life_events.get('past_events', []):
        milestones.append({
            'timestamp': datetime.fromisoformat(event['date']),
            'event_type': event['category'],
            'description': event['description'],
            'financial_impact': event['financial_impact'],
            'probability': 1.0,  # Past events are certain
            'entity': 'client'
        })
    
    # Add planned events
    for event in life_events.get('planned_events', []):
        milestones.append({
            'timestamp': datetime.fromisoformat(event['date']),
            'event_type': event['category'],
            'description': event['description'],
            'financial_impact': event['expected_impact'],
            'probability': event['probability'],
            'entity': 'client'
        })
    
    return milestones

def create_initial_financial_state(financial_state: Dict) -> Dict[str, float]:
    """Create initial financial state for mesh engine"""
    assets = financial_state.get('assets', {})
    liabilities = financial_state.get('liabilities', {})
    
    return {
        'cash': float(assets.get('cash', 0)),
        'investments': float(assets.get('investments', 0)),
        'real_estate': float(assets.get('real_estate', 0)),
        'retirement_accounts': float(assets.get('retirement_accounts', 0)),
        'other_assets': float(assets.get('other_assets', 0)),
        'mortgage': float(liabilities.get('mortgage', 0)),
        'student_loans': float(liabilities.get('student_loans', 0)),
        'credit_cards': float(liabilities.get('credit_cards', 0)),
        'other_debt': float(liabilities.get('other_debt', 0)),
        'total_wealth': sum(assets.values()) - sum(liabilities.values())
    }

def run_stochastic_mesh_analysis(person_data: Dict, person_id: str) -> Dict:
    """Run stochastic mesh analysis for a single person"""
    print(f"🔍 Running stochastic mesh analysis for {person_id}...")
    
    # Create initial financial state
    initial_state = create_initial_financial_state(person_data['financial_state'])
    
    # Convert life events to milestones
    milestones = convert_life_events_to_milestones(person_data['life_events'])
    
    # Initialize stochastic mesh engine
    mesh_engine = StochasticMeshEngine(initial_state)
    
    # Initialize mesh
    mesh_status = mesh_engine.initialize_mesh(milestones, time_horizon_years=10)
    
    # Get payment options
    payment_options = mesh_engine.get_payment_options()
    
    # Generate recommendations
    accounting_engine = AccountingReconciliationEngine()
    recommendation_engine = FinancialRecommendationEngine(mesh_engine, accounting_engine)
    
    profile_data = {
        'base_income': person_data['financial_state']['income']['annual_salary'],
        'risk_tolerance': person_data['profile']['risk_tolerance'],
        'age': person_data['profile']['age'],
        'family_status': person_data['profile']['family_status'],
        'current_assets': sum(person_data['financial_state']['assets'].values()),
        'debts': sum(person_data['financial_state']['liabilities'].values())
    }
    
    # Generate monthly recommendations
    recommendations = recommendation_engine.generate_monthly_recommendations(
        milestones, profile_data, months_ahead=24
    )
    
    return {
        'person_id': person_id,
        'mesh_status': mesh_status,
        'payment_options': len(payment_options),
        'recommendations': len(recommendations),
        'initial_state': initial_state,
        'milestones': len(milestones)
    }

def run_time_uncertainty_analysis(person_data: Dict, person_id: str) -> Dict:
    """Run time uncertainty mesh analysis for a single person"""
    print(f"⏰ Running time uncertainty analysis for {person_id}...")
    
    # Create initial financial state
    initial_state = create_initial_financial_state(person_data['financial_state'])
    
    # Convert life events to seed events
    seed_events = []
    for event in person_data['life_events'].get('planned_events', []):
        seed_events.append(SeedEvent(
            event_id=event['id'],
            event_type=event['category'],
            expected_date=datetime.fromisoformat(event['date']),
            expected_amount=event['expected_impact'],
            uncertainty_amount=abs(event['expected_impact']) * 0.2,  # 20% uncertainty
            uncertainty_timing=365,  # 1 year uncertainty
            probability=event['probability']
        ))
    
    # Initialize time uncertainty engine
    time_engine = TimeUncertaintyMeshEngine(use_gpu=True)
    
    # Initialize mesh with time uncertainty
    mesh_data, risk_analysis = time_engine.initialize_mesh_with_time_uncertainty(
        seed_events, num_scenarios=1000, time_horizon_years=10
    )
    
    return {
        'person_id': person_id,
        'seed_events': len(seed_events),
        'scenarios': len(mesh_data.get('states', [])),
        'risk_metrics': len(risk_analysis),
        'time_steps': len(mesh_data.get('time_steps', []))
    }

def run_mesh_congruence_analysis(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Run mesh congruence analysis across all people"""
    print(f"🔗 Running mesh congruence analysis for {len(people_data)} people...")
    
    congruence_engine = MeshCongruenceEngine()
    
    # Create client profiles for congruence analysis
    client_profiles = []
    for person_id, person_data in people_data:
        profile = {
            'id': person_id,
            'age': person_data['profile']['age'],
            'income_level': person_data['profile']['income_level'],
            'risk_tolerance': person_data['profile']['risk_tolerance'],
            'family_status': person_data['profile']['family_status'],
            'total_wealth': sum(person_data['financial_state']['assets'].values()) - 
                           sum(person_data['financial_state']['liabilities'].values()),
            'financial_state': person_data['financial_state']
        }
        client_profiles.append(profile)
    
    # Run congruence analysis
    congruence_results = []
    for i, profile1 in enumerate(client_profiles):
        for j, profile2 in enumerate(client_profiles[i+1:], i+1):
            result = congruence_engine.analyze_mesh_congruence(
                profile1, profile2, num_scenarios=100
            )
            congruence_results.append(result)
    
    return {
        'total_comparisons': len(congruence_results),
        'average_congruence': np.mean([r.congruence_score for r in congruence_results]),
        'congruence_std': np.std([r.congruence_score for r in congruence_results])
    }

def run_vector_database_analysis(people_data: List[Tuple[str, Dict]]) -> Dict:
    """Run vector database analysis for similarity matching"""
    print(f"📊 Running vector database analysis for {len(people_data)} people...")
    
    vector_db = MeshVectorDatabase()
    
    # Convert people to vector embeddings
    embeddings = []
    for person_id, person_data in people_data:
        embedding = vector_db.create_client_embedding({
            'id': person_id,
            'profile': person_data['profile'],
            'financial_state': person_data['financial_state'],
            'goals': person_data['goals'],
            'life_events': person_data['life_events'],
            'preferences': person_data['preferences']
        })
        embeddings.append(embedding)
    
    # Store embeddings in database
    for embedding in embeddings:
        vector_db.store_embedding(embedding)
    
    # Find similar clients for each person
    similarity_results = []
    for person_id, person_data in people_data:
        similar_clients = vector_db.find_similar_clients(
            person_data['profile'], top_k=3
        )
        similarity_results.append({
            'person_id': person_id,
            'similar_clients': len(similar_clients),
            'avg_similarity': np.mean([s.similarity_score for s in similar_clients]) if similar_clients else 0
        })
    
    return {
        'total_embeddings': len(embeddings),
        'avg_similarity': np.mean([r['avg_similarity'] for r in similarity_results]),
        'similarity_std': np.std([r['avg_similarity'] for r in similarity_results])
    }

def main(people_dir: str):
    """Main analysis pipeline"""
    print(f"🚀 Starting mesh analysis on people data in {people_dir}")
    
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
        'stochastic_mesh': [],
        'time_uncertainty': [],
        'congruence': {},
        'vector_database': {}
    }
    
    # Run stochastic mesh analysis for each person
    for person_id, person_data in people_data:
        try:
            result = run_stochastic_mesh_analysis(person_data, person_id)
            results['stochastic_mesh'].append(result)
        except Exception as e:
            print(f"❌ Error in stochastic mesh analysis for {person_id}: {e}")
    
    # Run time uncertainty analysis for each person
    for person_id, person_data in people_data:
        try:
            result = run_time_uncertainty_analysis(person_data, person_id)
            results['time_uncertainty'].append(result)
        except Exception as e:
            print(f"❌ Error in time uncertainty analysis for {person_id}: {e}")
    
    # Run congruence analysis across all people
    try:
        results['congruence'] = run_mesh_congruence_analysis(people_data)
    except Exception as e:
        print(f"❌ Error in congruence analysis: {e}")
    
    # Run vector database analysis
    try:
        results['vector_database'] = run_vector_database_analysis(people_data)
    except Exception as e:
        print(f"❌ Error in vector database analysis: {e}")
    
    # Save results
    output_file = f"data/outputs/analysis_data/mesh_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Analysis Summary:")
    print(f"   - Stochastic mesh analysis: {len(results['stochastic_mesh'])} people")
    print(f"   - Time uncertainty analysis: {len(results['time_uncertainty'])} people")
    print(f"   - Congruence analysis: {results['congruence'].get('total_comparisons', 0)} comparisons")
    print(f"   - Vector database: {results['vector_database'].get('total_embeddings', 0)} embeddings")
    print(f"   - Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--people-dir", type=str, default="data/inputs/people/current",
                       help="Directory containing people data")
    args = parser.parse_args()
    
    main(args.people_dir) 