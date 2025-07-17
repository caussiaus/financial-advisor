#!/usr/bin/env python3
"""
Demo script for JSON to Vector Converter Engine

This script demonstrates:
1. Converting JSON client data to vectorized profiles
2. Generating synthetic lifestyle events with probability modeling
3. Integrating with existing mesh engines
4. Creating surfaces of discretionary spending
5. Analyzing financial standing across different life stages
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.json_to_vector_converter import (
    JSONToVectorConverter, 
    ClientVectorProfile, 
    LifeStage, 
    EventCategory
)
from src.time_uncertainty_mesh import TimeUncertaintyMeshEngine
from src.synthetic_data_generator import SyntheticFinancialDataGenerator


def create_sample_json_data():
    """Create sample JSON data for testing"""
    sample_data = [
        {
            'client_id': 'CLIENT_001',
            'age': 28,
            'income': 65000,
            'current_assets': {
                'cash': 15000,
                'savings': 25000,
                'investments': 10000,
                'retirement': 5000,
                'real_estate': 0,
                'other_assets': 2000
            },
            'debts': {
                'credit_cards': 3000,
                'student_loans': 25000,
                'mortgage': 0,
                'auto_loans': 8000,
                'personal_loans': 0,
                'other_debts': 0
            },
            'risk_tolerance': 'moderate',
            'family_status': 'Single',
            'occupation': 'Software Engineer'
        },
        {
            'client_id': 'CLIENT_002',
            'age': 45,
            'income': 120000,
            'current_assets': {
                'cash': 50000,
                'savings': 100000,
                'investments': 150000,
                'retirement': 200000,
                'real_estate': 300000,
                'other_assets': 25000
            },
            'debts': {
                'credit_cards': 5000,
                'student_loans': 0,
                'mortgage': 200000,
                'auto_loans': 15000,
                'personal_loans': 0,
                'other_debts': 0
            },
            'risk_tolerance': 'aggressive',
            'family_status': 'Married with 2 children',
            'occupation': 'Marketing Manager'
        },
        {
            'client_id': 'CLIENT_003',
            'age': 62,
            'income': 85000,
            'current_assets': {
                'cash': 75000,
                'savings': 150000,
                'investments': 200000,
                'retirement': 400000,
                'real_estate': 250000,
                'other_assets': 50000
            },
            'debts': {
                'credit_cards': 2000,
                'student_loans': 0,
                'mortgage': 100000,
                'auto_loans': 0,
                'personal_loans': 0,
                'other_debts': 0
            },
            'risk_tolerance': 'conservative',
            'family_status': 'Married',
            'occupation': 'Accountant'
        }
    ]
    
    return sample_data


def analyze_vector_profiles(converter, vector_profiles):
    """Analyze the vector profiles and generate insights"""
    print("\n📊 Vector Profile Analysis")
    print("=" * 50)
    
    # Create analysis data
    analysis_data = []
    
    for profile in vector_profiles:
        # Calculate net worth
        net_worth = np.sum(profile.current_assets) - np.sum(profile.current_debts)
        
        # Calculate discretionary spending capacity
        avg_discretionary = np.mean(profile.discretionary_spending_surface)
        
        # Find most likely events
        event_categories = list(EventCategory)
        most_likely_event_idx = np.argmax(profile.event_probabilities)
        most_likely_event = event_categories[most_likely_event_idx]
        
        analysis_data.append({
            'client_id': profile.client_id,
            'age': profile.age,
            'life_stage': profile.life_stage.value,
            'income': profile.base_income,
            'net_worth': net_worth,
            'risk_tolerance': profile.risk_tolerance,
            'avg_discretionary': avg_discretionary,
            'most_likely_event': most_likely_event.value,
            'event_probability': profile.event_probabilities[most_likely_event_idx]
        })
    
    # Display analysis
    for data in analysis_data:
        print(f"\n👤 {data['client_id']}")
        print(f"   Age: {data['age']} ({data['life_stage']})")
        print(f"   Income: ${data['income']:,.0f}")
        print(f"   Net Worth: ${data['net_worth']:,.0f}")
        print(f"   Risk Tolerance: {data['risk_tolerance']:.2f}")
        print(f"   Avg Discretionary Spending: {data['avg_discretionary']:.3f}")
        print(f"   Most Likely Event: {data['most_likely_event']} ({data['event_probability']:.2f})")
    
    return analysis_data


def generate_synthetic_events_demo(converter, vector_profiles):
    """Demonstrate synthetic event generation"""
    print("\n🎯 Synthetic Event Generation Demo")
    print("=" * 50)
    
    for profile in vector_profiles:
        print(f"\n📋 Generating events for {profile.client_id} (Age: {profile.age})")
        
        # Generate synthetic events
        events = converter.generate_synthetic_events(profile, num_events=5)
        
        # Convert to seed events for mesh processing
        seed_events = converter.convert_events_to_seed_events(events)
        
        print(f"   Generated {len(events)} synthetic events:")
        
        for i, event in enumerate(events):
            print(f"   {i+1}. {event.category.value}: {event.description}")
            print(f"      Amount: ${event.base_amount:,.0f}")
            print(f"      Probability: {event.base_probability:.2f}")
            print(f"      Cash Flow Impact: {event.cash_flow_impact}")
        
        # Show event distribution by category
        category_counts = {}
        for event in events:
            category = event.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\n   Event Distribution:")
        for category, count in category_counts.items():
            print(f"      {category}: {count} events")


def create_discretionary_spending_visualization(vector_profiles):
    """Create visualization of discretionary spending surfaces"""
    print("\n📈 Creating Discretionary Spending Visualizations")
    print("=" * 50)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(vector_profiles), figsize=(15, 5))
    if len(vector_profiles) == 1:
        axes = [axes]
    
    spending_categories = ['Entertainment', 'Travel', 'Luxury', 'Hobbies', 'Dining', 'Shopping']
    
    for i, profile in enumerate(vector_profiles):
        ax = axes[i]
        
        # Get discretionary spending surface
        surface = profile.discretionary_spending_surface
        
        # Create heatmap
        im = ax.imshow(surface.T, aspect='auto', cmap='YlOrRd')
        ax.set_title(f'{profile.client_id}\nAge: {profile.age}, Life Stage: {profile.life_stage.value}')
        ax.set_xlabel('Months')
        ax.set_ylabel('Spending Categories')
        ax.set_yticks(range(len(spending_categories)))
        ax.set_yticklabels(spending_categories)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Discretionary Spending Ratio')
    
    plt.tight_layout()
    plt.savefig('data/outputs/visuals/discretionary_spending_surfaces.png', dpi=300, bbox_inches='tight')
    print("💾 Saved discretionary spending visualization to data/outputs/visuals/discretionary_spending_surfaces.png")


def integrate_with_mesh_engine(converter, vector_profiles):
    """Integrate with existing mesh engine for financial modeling"""
    print("\n🌐 Mesh Engine Integration Demo")
    print("=" * 50)
    
    for profile in vector_profiles:
        print(f"\n🔗 Processing {profile.client_id} with mesh engine...")
        
        # Generate synthetic events
        events = converter.generate_synthetic_events(profile, num_events=3)
        seed_events = converter.convert_events_to_seed_events(events)
        
        # Initialize mesh engine with current financial state
        current_state = {
            'cash': float(profile.current_assets[0]),
            'investments': float(profile.current_assets[2]),
            'debts': float(profile.current_debts[2]),  # mortgage
            'income': profile.base_income,
            'expenses': profile.base_income * 0.6  # Estimate expenses
        }
        
        # Create time uncertainty mesh
        mesh_engine = TimeUncertaintyMeshEngine(use_gpu=False)  # Use CPU for demo
        
        try:
            # Initialize mesh with events
            mesh_data, risk_analysis = mesh_engine.initialize_mesh_with_time_uncertainty(
                seed_events, 
                num_scenarios=1000,  # Reduced for demo
                time_horizon_years=5
            )
            
            print(f"   ✅ Mesh initialized successfully")
            print(f"   📊 {len(seed_events)} events processed")
            print(f"   🎲 {len(mesh_data['scenario_weights'])} scenarios generated")
            print(f"   📅 {len(mesh_data['time_steps'])} time steps")
            
            # Extract key risk metrics
            if 'min_cash_by_scenario' in risk_analysis:
                min_cash = np.min(risk_analysis['min_cash_by_scenario'])
                max_drawdown = np.max(risk_analysis['max_drawdown_by_scenario'])
                
                print(f"   💰 Minimum cash across scenarios: ${min_cash:,.0f}")
                print(f"   📉 Maximum drawdown: ${max_drawdown:,.0f}")
            
        except Exception as e:
            print(f"   ⚠️ Error initializing mesh: {e}")


def create_financial_standing_analysis(vector_profiles):
    """Analyze financial standing across different life stages"""
    print("\n💰 Financial Standing Analysis")
    print("=" * 50)
    
    # Group by life stage
    life_stage_data = {}
    
    for profile in vector_profiles:
        stage = profile.life_stage.value
        if stage not in life_stage_data:
            life_stage_data[stage] = []
        
        net_worth = np.sum(profile.current_assets) - np.sum(profile.current_debts)
        discretionary_capacity = np.mean(profile.discretionary_spending_surface)
        
        life_stage_data[stage].append({
            'age': profile.age,
            'income': profile.base_income,
            'net_worth': net_worth,
            'discretionary_capacity': discretionary_capacity,
            'risk_tolerance': profile.risk_tolerance
        })
    
    # Analyze each life stage
    for stage, profiles in life_stage_data.items():
        print(f"\n📊 {stage.upper()} Life Stage Analysis")
        print(f"   Number of profiles: {len(profiles)}")
        
        if profiles:
            avg_age = np.mean([p['age'] for p in profiles])
            avg_income = np.mean([p['income'] for p in profiles])
            avg_net_worth = np.mean([p['net_worth'] for p in profiles])
            avg_discretionary = np.mean([p['discretionary_capacity'] for p in profiles])
            avg_risk = np.mean([p['risk_tolerance'] for p in profiles])
            
            print(f"   Average Age: {avg_age:.1f}")
            print(f"   Average Income: ${avg_income:,.0f}")
            print(f"   Average Net Worth: ${avg_net_worth:,.0f}")
            print(f"   Average Discretionary Capacity: {avg_discretionary:.3f}")
            print(f"   Average Risk Tolerance: {avg_risk:.2f}")


def main():
    """Main demo function"""
    print("🚀 JSON to Vector Converter Demo")
    print("=" * 60)
    
    # Create converter
    converter = JSONToVectorConverter(use_gpu=False)  # Use CPU for demo
    
    # Create sample data
    sample_data = create_sample_json_data()
    print(f"📋 Created {len(sample_data)} sample client profiles")
    
    # Convert to vector profiles
    print("\n🔄 Converting JSON to vector profiles...")
    vector_profiles = converter.process_json_batch(sample_data)
    print(f"✅ Successfully converted {len(vector_profiles)} profiles")
    
    # Analyze vector profiles
    analysis_data = analyze_vector_profiles(converter, vector_profiles)
    
    # Generate synthetic events
    generate_synthetic_events_demo(converter, vector_profiles)
    
    # Create visualizations
    try:
        create_discretionary_spending_visualization(vector_profiles)
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Integrate with mesh engine
    integrate_with_mesh_engine(converter, vector_profiles)
    
    # Analyze financial standing
    create_financial_standing_analysis(vector_profiles)
    
    # Export vector data
    try:
        converter.export_vector_data(vector_profiles, 'data/outputs/analysis_data/vector_profiles_demo.json')
        print(f"\n💾 Exported vector data to data/outputs/analysis_data/vector_profiles_demo.json")
    except Exception as e:
        print(f"⚠️ Could not export data: {e}")
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   - JSON to vector conversion with probability modeling")
    print("   - Age and life stage based event generation")
    print("   - Discretionary spending surface creation")
    print("   - Integration with existing mesh engines")
    print("   - Financial standing analysis across life stages")


if __name__ == "__main__":
    main() 