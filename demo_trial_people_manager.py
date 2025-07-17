#!/usr/bin/env python3
"""
Trial People Manager Demo

This script demonstrates the trial people manager with 5 trial people:
1. Sets up file upload structure
2. Creates sample trial people data
3. Processes with mesh engine
4. Interpolates surfaces across the group
5. Identifies less dense sections for stress testing
6. Visualizes high-dimensional topology
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trial_people_manager import TrialPeopleManager, TrialPerson


def create_sample_trial_people():
    """Create sample trial people data"""
    print("📝 Creating sample trial people data...")
    
    # Define 5 trial people with diverse characteristics
    trial_people = [
        {
            'name': 'alex_chen',
            'personal_info': {
                'name': 'Alex Chen',
                'age': 28,
                'income': 65000,
                'net_worth': 25000,
                'risk_tolerance': 0.8,
                'life_stage': 'early_career'
            },
            'lifestyle_events': {
                'events': [
                    {
                        'event_type': 'career_change',
                        'expected_age': 30,
                        'estimated_cost': 3000,
                        'probability': 0.8
                    },
                    {
                        'event_type': 'education',
                        'expected_age': 29,
                        'estimated_cost': 15000,
                        'probability': 0.6
                    },
                    {
                        'event_type': 'housing',
                        'expected_age': 32,
                        'estimated_cost': 50000,
                        'probability': 0.7
                    }
                ]
            },
            'financial_profile': {
                'monthly_income': 5417,
                'monthly_expenses': 3500,
                'savings_rate': 0.15,
                'debt_to_income_ratio': 0.4,
                'investment_portfolio': {
                    'stocks': 0.8,
                    'bonds': 0.1,
                    'cash': 0.1
                }
            },
            'goals': {
                'short_term_goals': ['emergency_fund', 'debt_payoff'],
                'medium_term_goals': ['house_down_payment', 'career_advancement'],
                'long_term_goals': ['retirement_savings', 'investment_portfolio']
            }
        },
        {
            'name': 'sarah_johnson',
            'personal_info': {
                'name': 'Sarah Johnson',
                'age': 35,
                'income': 85000,
                'net_worth': 120000,
                'risk_tolerance': 0.6,
                'life_stage': 'mid_career'
            },
            'lifestyle_events': {
                'events': [
                    {
                        'event_type': 'family',
                        'expected_age': 36,
                        'estimated_cost': 8000,
                        'probability': 0.9
                    },
                    {
                        'event_type': 'housing',
                        'expected_age': 37,
                        'estimated_cost': 80000,
                        'probability': 0.8
                    },
                    {
                        'event_type': 'career_change',
                        'expected_age': 40,
                        'estimated_cost': 5000,
                        'probability': 0.5
                    }
                ]
            },
            'financial_profile': {
                'monthly_income': 7083,
                'monthly_expenses': 4500,
                'savings_rate': 0.25,
                'debt_to_income_ratio': 0.25,
                'investment_portfolio': {
                    'stocks': 0.6,
                    'bonds': 0.3,
                    'cash': 0.1
                }
            },
            'goals': {
                'short_term_goals': ['family_expenses', 'house_purchase'],
                'medium_term_goals': ['education_fund', 'career_growth'],
                'long_term_goals': ['retirement_savings', 'estate_planning']
            }
        },
        {
            'name': 'michael_rodriguez',
            'personal_info': {
                'name': 'Michael Rodriguez',
                'age': 42,
                'income': 110000,
                'net_worth': 350000,
                'risk_tolerance': 0.4,
                'life_stage': 'established'
            },
            'lifestyle_events': {
                'events': [
                    {
                        'event_type': 'education',
                        'expected_age': 44,
                        'estimated_cost': 25000,
                        'probability': 0.4
                    },
                    {
                        'event_type': 'retirement',
                        'expected_age': 65,
                        'estimated_cost': 0,
                        'probability': 0.9
                    }
                ]
            },
            'financial_profile': {
                'monthly_income': 9167,
                'monthly_expenses': 5500,
                'savings_rate': 0.3,
                'debt_to_income_ratio': 0.15,
                'investment_portfolio': {
                    'stocks': 0.5,
                    'bonds': 0.4,
                    'cash': 0.1
                }
            },
            'goals': {
                'short_term_goals': ['tax_optimization', 'investment_rebalancing'],
                'medium_term_goals': ['education_funding', 'business_development'],
                'long_term_goals': ['retirement_planning', 'wealth_preservation']
            }
        },
        {
            'name': 'emily_thompson',
            'personal_info': {
                'name': 'Emily Thompson',
                'age': 58,
                'income': 95000,
                'net_worth': 800000,
                'risk_tolerance': 0.3,
                'life_stage': 'pre_retirement'
            },
            'lifestyle_events': {
                'events': [
                    {
                        'event_type': 'retirement',
                        'expected_age': 65,
                        'estimated_cost': 0,
                        'probability': 0.95
                    },
                    {
                        'event_type': 'health',
                        'expected_age': 60,
                        'estimated_cost': 15000,
                        'probability': 0.6
                    }
                ]
            },
            'financial_profile': {
                'monthly_income': 7917,
                'monthly_expenses': 4000,
                'savings_rate': 0.35,
                'debt_to_income_ratio': 0.1,
                'investment_portfolio': {
                    'stocks': 0.3,
                    'bonds': 0.6,
                    'cash': 0.1
                }
            },
            'goals': {
                'short_term_goals': ['health_planning', 'tax_efficiency'],
                'medium_term_goals': ['retirement_transition', 'legacy_planning'],
                'long_term_goals': ['retirement_income', 'estate_planning']
            }
        },
        {
            'name': 'david_wilson',
            'personal_info': {
                'name': 'David Wilson',
                'age': 68,
                'income': 45000,
                'net_worth': 1200000,
                'risk_tolerance': 0.2,
                'life_stage': 'retirement'
            },
            'lifestyle_events': {
                'events': [
                    {
                        'event_type': 'health',
                        'expected_age': 70,
                        'estimated_cost': 25000,
                        'probability': 0.7
                    },
                    {
                        'event_type': 'family',
                        'expected_age': 72,
                        'estimated_cost': 10000,
                        'probability': 0.4
                    }
                ]
            },
            'financial_profile': {
                'monthly_income': 3750,
                'monthly_expenses': 3000,
                'savings_rate': 0.1,
                'debt_to_income_ratio': 0.05,
                'investment_portfolio': {
                    'stocks': 0.2,
                    'bonds': 0.7,
                    'cash': 0.1
                }
            },
            'goals': {
                'short_term_goals': ['income_optimization', 'health_care'],
                'medium_term_goals': ['legacy_planning', 'family_support'],
                'long_term_goals': ['estate_planning', 'wealth_transfer']
            }
        }
    ]
    
    return trial_people


def setup_trial_people_files(manager: TrialPeopleManager, trial_people: list):
    """Set up trial people files in the upload directory"""
    print("📁 Setting up trial people files...")
    
    for person_data in trial_people:
        person_folder = manager.upload_dir / person_data['name']
        person_folder.mkdir(exist_ok=True)
        
        # Create PERSONAL_INFO.json
        with open(person_folder / 'PERSONAL_INFO.json', 'w') as f:
            json.dump(person_data['personal_info'], f, indent=2)
        
        # Create LIFESTYLE_EVENTS.json
        with open(person_folder / 'LIFESTYLE_EVENTS.json', 'w') as f:
            json.dump(person_data['lifestyle_events'], f, indent=2)
        
        # Create FINANCIAL_PROFILE.json
        with open(person_folder / 'FINANCIAL_PROFILE.json', 'w') as f:
            json.dump(person_data['financial_profile'], f, indent=2)
        
        # Create GOALS.json
        with open(person_folder / 'GOALS.json', 'w') as f:
            json.dump(person_data['goals'], f, indent=2)
        
        print(f"   ✅ Created files for {person_data['personal_info']['name']}")


def run_comprehensive_trial_analysis(manager: TrialPeopleManager):
    """Run comprehensive analysis on trial people"""
    print("\n🚀 Running Comprehensive Trial Analysis")
    print("=" * 60)
    
    # Scan for uploaded people
    print("\n🔍 Scanning for uploaded trial people...")
    people_folders = manager.scan_upload_directory()
    
    if not people_folders:
        print("❌ No trial people found. Please set up files first.")
        return
    
    print(f"✅ Found {len(people_folders)} trial people: {people_folders}")
    
    # Ingest trial people
    print("\n📥 Ingesting trial people...")
    for folder in people_folders:
        person = manager.ingest_trial_person(folder)
        print(f"   ✅ Ingested {person.name} ({person.person_id})")
    
    # Process with mesh engine
    print("\n🌐 Processing with mesh engine...")
    for person in manager.trial_people.values():
        print(f"   🔄 Processing {person.name}...")
        person = manager.process_trial_person_with_mesh(person)
        print(f"   ✅ Completed mesh processing for {person.name}")
    
    # Interpolate surfaces
    print("\n📊 Interpolating surfaces across group...")
    surfaces = manager.interpolate_surfaces_across_group()
    print(f"   ✅ Generated {len(surfaces)} interpolated surfaces:")
    for surface_type, surface in surfaces.items():
        print(f"      - {surface_type}: {len(surface.contributing_people)} contributors")
    
    # Schedule tasks
    print("\n📅 Scheduling analysis tasks...")
    tasks = manager.schedule_tasks()
    print(f"   ✅ Scheduled {len(tasks)} tasks:")
    for task in tasks:
        print(f"      - {task.task_type} (Priority: {task.priority}, Duration: {task.estimated_duration})")
    
    # Identify less dense sections
    print("\n🎯 Identifying less dense mesh sections...")
    density_analysis = manager.identify_less_dense_sections()
    if density_analysis:
        print(f"   ✅ Found {density_analysis['outliers_found']} outliers")
        print(f"   ✅ Found {len(density_analysis['low_density_people'])} low-density people")
        print(f"   ✅ Identified {density_analysis['clusters_found']} clusters")
        
        if density_analysis['outlier_people']:
            print(f"   📍 Outlier people: {', '.join(density_analysis['outlier_people'])}")
        if density_analysis['low_density_people']:
            print(f"   📍 Low-density people: {', '.join(density_analysis['low_density_people'])}")
    
    # Create visualizations
    print("\n📈 Creating high-dimensional topology visualizations...")
    viz_files = manager.visualize_high_dimensional_topology()
    print(f"   ✅ Created {len(viz_files)} visualizations:")
    for viz_type, file_path in viz_files.items():
        print(f"      - {viz_type}: {file_path}")
    
    # Save results
    print("\n💾 Saving analysis results...")
    results_file = manager.save_analysis_results()
    print(f"   ✅ Saved results to {results_file}")
    
    return surfaces, density_analysis, viz_files


def analyze_vector_embedding_training(manager: TrialPeopleManager):
    """Analyze vector embedding training opportunities"""
    print("\n🧠 Vector Embedding Training Analysis")
    print("=" * 50)
    
    # Analyze embedding quality
    embeddings = []
    people_info = []
    
    for person in manager.trial_people.values():
        if person.vector_embedding is not None:
            embeddings.append(person.vector_embedding)
            people_info.append({
                'id': person.person_id,
                'name': person.name,
                'age': person.age,
                'life_stage': person.life_stage.value
            })
    
    if len(embeddings) < 2:
        print("❌ Need at least 2 people with embeddings for analysis")
        return
    
    embeddings = np.array(embeddings)
    
    print(f"📊 Embedding Analysis:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Mean embedding norm: {np.mean([np.linalg.norm(emb) for emb in embeddings]):.3f}")
    print(f"   Embedding std: {np.std(embeddings):.3f}")
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                similarity = np.dot(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
    
    print(f"\n📈 Similarity Analysis:")
    print(f"   Mean similarity: {np.mean(similarity_matrix[similarity_matrix > 0]):.3f}")
    print(f"   Std similarity: {np.std(similarity_matrix[similarity_matrix > 0]):.3f}")
    print(f"   Min similarity: {np.min(similarity_matrix[similarity_matrix > 0]):.3f}")
    print(f"   Max similarity: {np.max(similarity_matrix):.3f}")
    
    # Identify training opportunities
    print(f"\n🎯 Training Opportunities:")
    
    # Find most similar pairs for positive training examples
    max_similarity_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_similarity_idx]
    print(f"   Most similar pair: {people_info[max_similarity_idx[0]]['name']} & {people_info[max_similarity_idx[1]]['name']} (similarity: {max_similarity:.3f})")
    
    # Find least similar pairs for negative training examples
    min_similarity_idx = np.unravel_index(np.argmin(similarity_matrix[similarity_matrix > 0]), similarity_matrix.shape)
    min_similarity = similarity_matrix[min_similarity_idx]
    print(f"   Least similar pair: {people_info[min_similarity_idx[0]]['name']} & {people_info[min_similarity_idx[1]]['name']} (similarity: {min_similarity:.3f})")
    
    # Analyze by life stage
    life_stages = {}
    for i, info in enumerate(people_info):
        stage = info['life_stage']
        if stage not in life_stages:
            life_stages[stage] = []
        life_stages[stage].append(i)
    
    print(f"\n📊 Life Stage Analysis:")
    for stage, indices in life_stages.items():
        if len(indices) > 1:
            stage_similarities = []
            for i in indices:
                for j in indices:
                    if i != j:
                        stage_similarities.append(similarity_matrix[i, j])
            
            print(f"   {stage}: {np.mean(stage_similarities):.3f} ± {np.std(stage_similarities):.3f} (n={len(indices)})")


def main():
    """Main demo function"""
    print("🚀 Trial People Manager Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the trial people manager with 5 diverse trial people,")
    print("including file upload setup, mesh processing, surface interpolation,")
    print("and high-dimensional topology visualization.")
    print("=" * 80)
    
    # Create trial people manager
    manager = TrialPeopleManager()
    
    # Show upload instructions
    print("\n📁 Upload Instructions:")
    print(manager.get_upload_instructions())
    
    # Create sample trial people
    trial_people = create_sample_trial_people()
    print(f"\n📝 Created {len(trial_people)} sample trial people")
    
    # Set up files
    setup_trial_people_files(manager, trial_people)
    
    # Run comprehensive analysis
    surfaces, density_analysis, viz_files = run_comprehensive_trial_analysis(manager)
    
    # Analyze vector embedding training opportunities
    analyze_vector_embedding_training(manager)
    
    # Summary
    print(f"\n📊 Final Summary")
    print("=" * 40)
    print(f"Trial people processed: {len(manager.trial_people)}")
    print(f"Interpolated surfaces: {len(surfaces)}")
    print(f"Visualizations created: {len(viz_files)}")
    print(f"Vector database embeddings: {len(manager.vector_db.embeddings)}")
    
    if density_analysis:
        print(f"Outliers identified: {density_analysis['outliers_found']}")
        print(f"Low-density people: {len(density_analysis['low_density_people'])}")
    
    print("\n✅ Trial People Manager Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   - File upload and ingestion for multiple people")
    print("   - Surface interpolation across the group")
    print("   - Resource management and task scheduling")
    print("   - Identification of less dense mesh sections")
    print("   - High-dimensional topological space visualization")
    print("   - Vector embedding training analysis")


if __name__ == "__main__":
    main() 