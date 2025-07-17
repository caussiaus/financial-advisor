#!/usr/bin/env python3
"""
Real People Integration Demo

This script integrates the provided real people data into the trial people manager
for vector database processing and similarity matching.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.real_people_integrator import RealPeopleIntegrator
from src.trial_people_manager import TrialPeopleManager


def load_real_people_data():
    """Load the real people data provided by the user"""
    real_people_data = [
        {
            "events": [
                {
                    "event_id": "evt_101",
                    "description": "Annual salary for software engineer",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 120000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.20, "state": 0.06},
                    "probability": 1.0,
                    "source_text": "Currently earning an annual salary of $120,000 as a software engineer."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_101",
                    "description": "Plans to pursue MBA",
                    "modulator_type": "education",
                    "date": "2026-09-01",
                    "profile_change": "Temporary income loss during MBA studies",
                    "accounts_impacted": ["salary_income", "education_expenses"],
                    "source_text": "Considering leaving current job to pursue an MBA in September 2026."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_101",
                    "name": "Alex",
                    "entity_type": "person",
                    "initial_balances": {"salary": 120000, "savings": 30000},
                    "relationships": [],
                    "source_text": "Alex, a software engineer, has an annual income of $120,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_102",
                    "description": "Annual freelance graphic design income",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 85000,
                    "amount_type": "annual",
                    "account_affected": "freelance_income",
                    "tax_implications": {"federal": 0.18, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Freelance graphic designer earning $85,000 annually."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_102",
                    "description": "Planning to buy a new studio",
                    "modulator_type": "asset_purchase",
                    "date": "2025-05-01",
                    "profile_change": "One-time expense of $300,000",
                    "accounts_impacted": ["real_estate_asset", "savings"],
                    "source_text": "Plans to purchase a studio space for $300,000 in May 2025."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_102",
                    "name": "Jordan",
                    "entity_type": "person",
                    "initial_balances": {"freelance_income": 85000, "savings": 45000},
                    "relationships": [],
                    "source_text": "Jordan is a freelance graphic designer with annual earnings of $85,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_103",
                    "description": "Annual teacher salary",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 65000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.15, "state": 0.04},
                    "probability": 1.0,
                    "source_text": "Annual salary as a high school teacher is $65,000."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_103",
                    "description": "Retirement planned at age 60",
                    "modulator_type": "retirement",
                    "date": "2030-01-01",
                    "profile_change": "Expected pension of 60% of final salary",
                    "accounts_impacted": ["salary_income", "retirement_pension"],
                    "source_text": "Plans to retire at 60 with pension equal to 60% of final salary."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_103",
                    "name": "Taylor",
                    "entity_type": "person",
                    "initial_balances": {"salary": 65000, "savings": 20000},
                    "relationships": [],
                    "source_text": "Taylor works as a high school teacher, earning $65,000 annually."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_104",
                    "description": "Annual salary for nurse",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 78000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.17, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Annual income as a registered nurse is $78,000."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_104",
                    "description": "Marriage planned in 2025",
                    "modulator_type": "marriage",
                    "date": "2025-06-15",
                    "profile_change": "Joint household income increases by spouse's salary",
                    "accounts_impacted": ["salary_income", "household_expenses"],
                    "source_text": "Plans to marry in June 2025, increasing household income."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_104",
                    "name": "Casey",
                    "entity_type": "person",
                    "initial_balances": {"salary": 78000, "savings": 25000},
                    "relationships": [],
                    "source_text": "Casey is a registered nurse with an annual income of $78,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_105",
                    "description": "Annual income from restaurant",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 95000,
                    "amount_type": "annual",
                    "account_affected": "business_income",
                    "tax_implications": {"federal": 0.19, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Owns a restaurant generating $95,000 annually."
                }
            ],
            "modulators": [],
            "entities": [
                {
                    "entity_id": "ent_105",
                    "name": "Sam",
                    "entity_type": "person",
                    "initial_balances": {"business_income": 95000, "savings": 40000},
                    "relationships": [],
                    "source_text": "Sam owns a successful restaurant generating $95,000 annually."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_106",
                    "description": "Annual salary for data scientist",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 130000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.21, "state": 0.07},
                    "probability": 1.0,
                    "source_text": "Data scientist earning $130,000 per year."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_106",
                    "description": "Sabbatical for research",
                    "modulator_type": "career_break",
                    "date": "2025-06-01",
                    "profile_change": "Unpaid leave for 6 months",
                    "accounts_impacted": ["salary_income", "savings"],
                    "source_text": "Planning a six-month paid sabbatical starting June 2025."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_106",
                    "name": "Morgan",
                    "entity_type": "person",
                    "initial_balances": {"salary": 130000, "savings": 50000},
                    "relationships": [],
                    "source_text": "Morgan works as a data scientist with an annual income of $130,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_107",
                    "description": "Annual revenue from retail business",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 200000,
                    "amount_type": "annual",
                    "account_affected": "business_income",
                    "tax_implications": {"federal": 0.22, "state": 0.06},
                    "probability": 1.0,
                    "source_text": "Small retail business generating $200,000 in annual revenue."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_107",
                    "description": "Business expansion loan",
                    "modulator_type": "liability",
                    "date": "2024-07-01",
                    "profile_change": "Took a loan of $100,000 for expansion",
                    "accounts_impacted": ["business_income", "loan_liability"],
                    "source_text": "Secured a $100,000 business loan for expansion in July 2024."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_107",
                    "name": "Drew",
                    "entity_type": "person",
                    "initial_balances": {"business_income": 200000, "savings": 60000},
                    "relationships": [],
                    "source_text": "Drew owns a small retail business with $200,000 annual revenue."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_201",
                    "description": "Combined salary of household",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 150000,
                    "amount_type": "annual",
                    "account_affected": "household_income",
                    "tax_implications": {"federal": 0.19, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Luis and Maria earn a combined $150,000 annually."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_201",
                    "description": "Purchase of family car",
                    "modulator_type": "asset_purchase",
                    "date": "2024-03-15",
                    "profile_change": "One-time expense of $35,000",
                    "accounts_impacted": ["transportation_expenses", "savings"],
                    "source_text": "Bought a family SUV for $35,000 in March 2024."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_201",
                    "name": "Garcia Family",
                    "entity_type": "household",
                    "initial_balances": {"household_income": 150000, "savings": 40000},
                    "relationships": ["Luis spouse of Maria", "Maria spouse of Luis"],
                    "source_text": "The Garcia Family has two working adults, Luis and Maria, with combined earnings of $150,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_202",
                    "description": "Pension income of retired couple",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 80000,
                    "amount_type": "annual",
                    "account_affected": "pension_income",
                    "tax_implications": {"federal": 0.15},
                    "probability": 1.0,
                    "source_text": "John and Ellen receive a combined pension of $80,000 annually."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_202",
                    "description": "Downsizing home",
                    "modulator_type": "asset_sale",
                    "date": "2025-06-01",
                    "profile_change": "Proceeds of $250,000 from home sale",
                    "accounts_impacted": ["real_estate_asset", "savings"],
                    "source_text": "Plan to downsize and sell home for $250,000 in June 2025."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_202",
                    "name": "Johnson Family",
                    "entity_type": "household",
                    "initial_balances": {"pension_income": 80000, "savings": 200000},
                    "relationships": ["spouses John and Ellen Johnson"],
                    "source_text": "John and Ellen Johnson are retired and receive pension income."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_203",
                    "description": "Single parent income",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 60000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.18, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Priya earns $60,000 annually as a school counselor."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_203",
                    "description": "Childcare costs",
                    "modulator_type": "expense",
                    "date": "2024-01-01",
                    "profile_change": "Annual childcare expense of $15,000",
                    "accounts_impacted": ["childcare_expenses"],
                    "source_text": "Pays $15,000 per year in childcare."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_203",
                    "name": "Patel Household",
                    "entity_type": "household",
                    "initial_balances": {"salary_income": 60000, "savings": 10000},
                    "relationships": ["single parent Priya and child Arjun"],
                    "source_text": "Priya Patel is a single parent earning $60,000 with one child, Arjun."
                }
            ]
        }
    ]
    
    return real_people_data


def run_real_people_integration():
    """Run the real people integration demo"""
    print("🚀 Real People Integration Demo")
    print("=" * 60)
    print("Integrating real people data into the trial people manager")
    print("for vector database processing and similarity matching.")
    print("=" * 60)
    
    # Load real people data
    real_people_data = load_real_people_data()
    print(f"\n📋 Loaded {len(real_people_data)} real people:")
    
    for i, person_data in enumerate(real_people_data):
        entity = person_data['entities'][0]
        name = entity['name']
        income = max(entity['initial_balances'].values())
        print(f"   {i+1}. {name} - ${income:,.0f} annual income")
    
    # Create trial people manager
    print("\n🏗️ Creating trial people manager...")
    trial_manager = TrialPeopleManager()
    
    # Create integrator
    print("🔗 Creating real people integrator...")
    integrator = RealPeopleIntegrator(trial_manager)
    
    # Process real people with analysis
    print("\n🔄 Processing real people with complete analysis...")
    results = integrator.process_real_people_with_analysis(real_people_data)
    
    # Display results
    print(f"\n📊 Integration Results:")
    print(f"   ✅ Created {len(results['created_folders'])} trial person folders")
    print(f"   ✅ Generated {len(results['surfaces'])} interpolated surfaces")
    print(f"   ✅ Created {len(results['viz_files'])} visualizations")
    
    if results['density_analysis']:
        print(f"   ✅ Found {results['density_analysis']['outliers_found']} outliers")
        print(f"   ✅ Found {len(results['density_analysis']['low_density_people'])} low-density people")
        print(f"   ✅ Identified {results['density_analysis']['clusters_found']} clusters")
        
        if results['density_analysis']['outlier_people']:
            print(f"   📍 Outlier people: {', '.join(results['density_analysis']['outlier_people'])}")
        if results['density_analysis']['low_density_people']:
            print(f"   📍 Low-density people: {', '.join(results['density_analysis']['low_density_people'])}")
    
    print(f"   💾 Results saved to {results['results_file']}")
    
    # Show created folders
    print(f"\n📁 Created Trial Person Folders:")
    for folder in results['created_folders']:
        print(f"   - {folder}")
    
    # Show visualizations
    if results['viz_files']:
        print(f"\n📈 Generated Visualizations:")
        for viz_type, file_path in results['viz_files'].items():
            print(f"   - {viz_type}: {file_path}")
    
    return integrator, results


def analyze_vector_embedding_training(integrator, results):
    """Analyze vector embedding training opportunities"""
    print(f"\n🧠 Vector Embedding Training Analysis")
    print("=" * 50)
    
    # Get trial people from manager
    trial_people = integrator.trial_manager.trial_people
    
    print(f"📊 Trial People Analysis:")
    print(f"   Total people: {len(trial_people)}")
    
    # Analyze by income levels
    income_levels = {}
    for person in trial_people.values():
        if person.income < 50000:
            level = "Low Income"
        elif person.income < 80000:
            level = "Medium Income"
        elif person.income < 120000:
            level = "High Income"
        else:
            level = "Very High Income"
        
        if level not in income_levels:
            income_levels[level] = []
        income_levels[level].append(person.name)
    
    print(f"\n💰 Income Distribution:")
    for level, people in income_levels.items():
        print(f"   {level}: {len(people)} people")
        for person in people:
            print(f"      - {person}")
    
    # Analyze by life stages
    life_stages = {}
    for person in trial_people.values():
        stage = person.life_stage.value
        if stage not in life_stages:
            life_stages[stage] = []
        life_stages[stage].append(person.name)
    
    print(f"\n📅 Life Stage Distribution:")
    for stage, people in life_stages.items():
        print(f"   {stage}: {len(people)} people")
        for person in people:
            print(f"      - {person}")
    
    # Analyze vector database
    vector_db = integrator.trial_manager.vector_db
    print(f"\n🔍 Vector Database Analysis:")
    print(f"   Total embeddings: {len(vector_db.embeddings)}")
    print(f"   Embedding dimension: {vector_db.embedding_dim}")
    
    # Test similarity matching
    if len(trial_people) > 1:
        test_person = list(trial_people.values())[0]
        similar_clients = vector_db.find_similar_clients(test_person.person_id, top_k=3)
        
        print(f"\n🎯 Similarity Matching Test:")
        print(f"   Test person: {test_person.name}")
        print(f"   Found {len(similar_clients)} similar clients:")
        
        for i, match in enumerate(similar_clients):
            matched_person = trial_people.get(match.matched_client_id)
            if matched_person:
                print(f"      {i+1}. {matched_person.name} (similarity: {match.similarity_score:.3f})")
                print(f"         Matching factors: {', '.join(match.matching_factors)}")
    
    # Analyze mesh data
    mesh_data_count = sum(1 for person in trial_people.values() if person.mesh_data is not None)
    print(f"\n🌐 Mesh Data Analysis:")
    print(f"   People with mesh data: {mesh_data_count}/{len(trial_people)}")
    
    # Analyze surface points
    surface_points_count = sum(1 for person in trial_people.values() if person.surface_points is not None)
    print(f"   People with surface points: {surface_points_count}/{len(trial_people)}")
    
    # Analyze vector embeddings
    embedding_count = sum(1 for person in trial_people.values() if person.vector_embedding is not None)
    print(f"   People with vector embeddings: {embedding_count}/{len(trial_people)}")


def main():
    """Main demo function"""
    print("🚀 Real People Integration Comprehensive Demo")
    print("=" * 80)
    print("This demo integrates real people data into the trial people manager")
    print("for vector database processing, similarity matching, and uncertainty estimation.")
    print("=" * 80)
    
    # Run integration
    integrator, results = run_real_people_integration()
    
    # Analyze vector embedding training
    analyze_vector_embedding_training(integrator, results)
    
    # Summary
    print(f"\n📊 Final Summary")
    print("=" * 40)
    print(f"Real people integrated: {len(results['created_folders'])}")
    print(f"Interpolated surfaces: {len(results['surfaces'])}")
    print(f"Visualizations created: {len(results['viz_files'])}")
    print(f"Vector database embeddings: {len(integrator.trial_manager.vector_db.embeddings)}")
    
    if results['density_analysis']:
        print(f"Outliers identified: {results['density_analysis']['outliers_found']}")
        print(f"Low-density people: {len(results['density_analysis']['low_density_people'])}")
    
    print("\n✅ Real People Integration Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   - Real people data integration")
    print("   - Vector embedding generation")
    print("   - Similarity matching across diverse profiles")
    print("   - Uncertainty estimation for less dense sections")
    print("   - High-dimensional topology visualization")
    print("   - Vector embedding training analysis")


if __name__ == "__main__":
    main() 