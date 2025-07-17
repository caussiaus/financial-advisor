#!/usr/bin/env python3
"""
Comprehensive Synthetic Lifestyle Engine Demo

This script demonstrates the complete pipeline for generating synthetic client data
with realistic lifestyle events that naturally occur based on age and life stage.
It shows how the system creates a surface of discretionary spending homogeneously sorted
and determines if people reach congruent financial standing despite daily life fluctuations.

Key Features Demonstrated:
1. JSON-to-vector conversion with probability modeling
2. Age and life stage based event generation
3. Discretionary spending surface creation
4. Integration with existing mesh engines
5. Financial standing analysis across life stages
6. Research design framework simulation (referencing the paper)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.json_to_vector_converter import LifeStage, EventCategory


def create_age_distribution():
    """Create realistic age distribution for synthetic client generation"""
    # Based on typical financial planning client demographics
    age_distribution = {
        25: 0.05,   # 5% early career
        28: 0.08,   # 8% early career
        32: 0.12,   # 12% mid career
        35: 0.15,   # 15% mid career
        38: 0.12,   # 12% mid career
        42: 0.10,   # 10% established
        45: 0.08,   # 8% established
        48: 0.06,   # 6% established
        52: 0.05,   # 5% established
        55: 0.04,   # 4% pre-retirement
        58: 0.03,   # 3% pre-retirement
        62: 0.03,   # 3% pre-retirement
        65: 0.02,   # 2% retirement
        68: 0.02,   # 2% retirement
        72: 0.01    # 1% retirement
    }
    return age_distribution


def analyze_congruent_financial_standing(clients: List[SyntheticClientData]) -> Dict:
    """
    Analyze if clients reach congruent financial standing despite life fluctuations
    
    This analysis examines whether clients maintain consistent financial standing
    across different life stages and events, similar to the research design framework
    in the referenced paper.
    """
    print("\n🎯 Congruent Financial Standing Analysis")
    print("=" * 60)
    
    # Group clients by life stage
    life_stage_groups = {}
    for client in clients:
        stage = client.vector_profile.life_stage.value
        if stage not in life_stage_groups:
            life_stage_groups[stage] = []
        life_stage_groups[stage].append(client)
    
    analysis_results = {}
    
    for stage, stage_clients in life_stage_groups.items():
        print(f"\n📊 {stage.upper()} Life Stage Analysis")
        print(f"   Number of clients: {len(stage_clients)}")
        
        # Calculate financial standing metrics
        net_worths = [client.financial_metrics['net_worth'] for client in stage_clients]
        risk_tolerances = [client.financial_metrics['risk_tolerance'] for client in stage_clients]
        discretionary_spending = [client.financial_metrics['avg_discretionary_spending'] for client in stage_clients]
        
        # Calculate congruence metrics
        net_worth_cv = np.std(net_worths) / np.mean(net_worths) if np.mean(net_worths) > 0 else 0
        risk_tolerance_cv = np.std(risk_tolerances) / np.mean(risk_tolerances) if np.mean(risk_tolerances) > 0 else 0
        discretionary_cv = np.std(discretionary_spending) / np.mean(discretionary_spending) if np.mean(discretionary_spending) > 0 else 0
        
        # Overall congruence score (lower = more congruent)
        congruence_score = (net_worth_cv + risk_tolerance_cv + discretionary_cv) / 3
        
        print(f"   Average Net Worth: ${np.mean(net_worths):,.0f}")
        print(f"   Net Worth CV: {net_worth_cv:.3f}")
        print(f"   Average Risk Tolerance: {np.mean(risk_tolerances):.3f}")
        print(f"   Risk Tolerance CV: {risk_tolerance_cv:.3f}")
        print(f"   Average Discretionary Spending: {np.mean(discretionary_spending):.3f}")
        print(f"   Discretionary Spending CV: {discretionary_cv:.3f}")
        print(f"   Congruence Score: {congruence_score:.3f}")
        
        # Determine congruence level
        if congruence_score < 0.2:
            congruence_level = "HIGH"
        elif congruence_score < 0.4:
            congruence_level = "MEDIUM"
        else:
            congruence_level = "LOW"
        
        print(f"   Congruence Level: {congruence_level}")
        
        analysis_results[stage] = {
            'count': len(stage_clients),
            'avg_net_worth': np.mean(net_worths),
            'avg_risk_tolerance': np.mean(risk_tolerances),
            'avg_discretionary': np.mean(discretionary_spending),
            'net_worth_cv': net_worth_cv,
            'risk_tolerance_cv': risk_tolerance_cv,
            'discretionary_cv': discretionary_cv,
            'congruence_score': congruence_score,
            'congruence_level': congruence_level
        }
    
    return analysis_results


def create_discretionary_spending_analysis(clients: List[SyntheticClientData]):
    """Create comprehensive analysis of discretionary spending surfaces"""
    print("\n📈 Discretionary Spending Surface Analysis")
    print("=" * 60)
    
    # Create visualization of discretionary spending patterns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    spending_categories = ['Entertainment', 'Travel', 'Luxury', 'Hobbies', 'Dining', 'Shopping']
    
    # Group by life stage for analysis
    life_stages = [LifeStage.EARLY_CAREER, LifeStage.MID_CAREER, LifeStage.ESTABLISHED, 
                   LifeStage.PRE_RETIREMENT, LifeStage.RETIREMENT]
    
    for i, stage in enumerate(life_stages):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get clients for this life stage
        stage_clients = [c for c in clients if c.vector_profile.life_stage == stage]
        
        if stage_clients:
            # Calculate average discretionary spending surface for this life stage
            surfaces = [client.vector_profile.discretionary_spending_surface for client in stage_clients]
            avg_surface = np.mean(surfaces, axis=0)
            
            # Create heatmap
            im = ax.imshow(avg_surface.T, aspect='auto', cmap='YlOrRd')
            ax.set_title(f'{stage.value.replace("_", " ").title()}\n({len(stage_clients)} clients)')
            ax.set_xlabel('Months')
            ax.set_ylabel('Spending Categories')
            ax.set_yticks(range(len(spending_categories)))
            ax.set_yticklabels(spending_categories)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Discretionary Spending Ratio')
            
            # Calculate and display statistics
            avg_discretionary = np.mean(avg_surface)
            ax.text(0.02, 0.98, f'Avg: {avg_discretionary:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/outputs/visuals/discretionary_spending_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Saved discretionary spending analysis to data/outputs/visuals/discretionary_spending_analysis.png")


def analyze_event_probability_patterns(clients: List[SyntheticClientData]):
    """Analyze event probability patterns across different life stages"""
    print("\n🎯 Event Probability Pattern Analysis")
    print("=" * 60)
    
    # Group by life stage
    life_stage_events = {}
    
    for client in clients:
        stage = client.vector_profile.life_stage.value
        if stage not in life_stage_events:
            life_stage_events[stage] = []
        
        # Count events by category
        event_counts = {}
        for event in client.lifestyle_events:
            category = event.category.value
            event_counts[category] = event_counts.get(category, 0) + 1
        
        life_stage_events[stage].append(event_counts)
    
    # Analyze patterns
    for stage, event_lists in life_stage_events.items():
        print(f"\n📊 {stage.upper()} Event Patterns")
        print(f"   Number of clients: {len(event_lists)}")
        
        if event_lists:
            # Calculate average event counts by category
            all_categories = set()
            for event_list in event_lists:
                all_categories.update(event_list.keys())
            
            avg_counts = {}
            for category in all_categories:
                counts = [event_list.get(category, 0) for event_list in event_lists]
                avg_counts[category] = np.mean(counts)
            
            # Sort by average count
            sorted_categories = sorted(avg_counts.items(), key=lambda x: x[1], reverse=True)
            
            print("   Average events per client by category:")
            for category, avg_count in sorted_categories:
                print(f"      {category}: {avg_count:.2f}")


def create_research_design_simulation(clients: List[SyntheticClientData]):
    """
    Simulate research design framework similar to the referenced paper
    
    This creates a systematic analysis of how different "design choices" 
    (lifestyle events) impact financial outcomes across various scenarios.
    """
    print("\n🔬 Research Design Framework Simulation")
    print("=" * 60)
    
    # Create design choice combinations (similar to the paper's approach)
    design_choices = {
        'event_frequency': ['low', 'medium', 'high'],
        'income_level': ['low', 'medium', 'high'],
        'life_stage': ['early_career', 'mid_career', 'established', 'pre_retirement'],
        'risk_tolerance': ['conservative', 'moderate', 'aggressive'],
        'event_types': ['education_focused', 'career_focused', 'family_focused', 'balanced']
    }
    
    # Simulate different design combinations
    simulation_results = []
    
    for freq in design_choices['event_frequency']:
        for income in design_choices['income_level']:
            for stage in design_choices['life_stage']:
                for risk in design_choices['risk_tolerance']:
                    for event_type in design_choices['event_types']:
                        
                        # Filter clients matching this design combination
                        matching_clients = []
                        for client in clients:
                            # Simple matching logic (in practice, this would be more sophisticated)
                            if (client.vector_profile.life_stage.value == stage and
                                client.financial_metrics['risk_tolerance'] < 0.4 if risk == 'conservative' else
                                client.financial_metrics['risk_tolerance'] > 0.6 if risk == 'aggressive' else True):
                                matching_clients.append(client)
                        
                        if matching_clients:
                            # Calculate outcome metrics for this design combination
                            net_worths = [c.financial_metrics['net_worth'] for c in matching_clients]
                            congruence_scores = [c.financial_metrics.get('congruence_score', 0) for c in matching_clients]
                            
                            result = {
                                'design_combination': f"{freq}_{income}_{stage}_{risk}_{event_type}",
                                'event_frequency': freq,
                                'income_level': income,
                                'life_stage': stage,
                                'risk_tolerance': risk,
                                'event_types': event_type,
                                'num_clients': len(matching_clients),
                                'avg_net_worth': np.mean(net_worths),
                                'net_worth_std': np.std(net_worths),
                                'avg_congruence': np.mean(congruence_scores),
                                'outcome_score': np.mean(net_worths) * (1 - np.mean(congruence_scores))  # Higher net worth, lower congruence = better
                            }
                            
                            simulation_results.append(result)
    
    # Sort by outcome score
    simulation_results.sort(key=lambda x: x['outcome_score'], reverse=True)
    
    print("Top 10 Design Combinations by Outcome Score:")
    for i, result in enumerate(simulation_results[:10]):
        print(f"   {i+1}. {result['design_combination']}")
        print(f"      Net Worth: ${result['avg_net_worth']:,.0f}")
        print(f"      Congruence: {result['avg_congruence']:.3f}")
        print(f"      Outcome Score: {result['outcome_score']:.2f}")
    
    return simulation_results


def main():
    """Main comprehensive demo function"""
    print("🚀 Comprehensive Synthetic Lifestyle Engine Demo")
    print("=" * 80)
    print("This demo showcases the complete pipeline for generating synthetic client data")
    print("with realistic lifestyle events and analyzing financial standing congruence.")
    print("=" * 80)
    
    # Create engine
    engine = SyntheticLifestyleEngine(use_gpu=False)  # Use CPU for demo
    
    # Generate synthetic clients with realistic age distribution
    print("\n📋 Generating synthetic client population...")
    age_distribution = create_age_distribution()
    
    # Generate a larger sample for comprehensive analysis
    clients = engine.generate_client_batch(
        num_clients=50, 
        age_distribution=age_distribution
    )
    
    print(f"✅ Generated {len(clients)} synthetic clients")
    
    # Process subset with mesh engine (for performance)
    print(f"\n🌐 Processing subset with mesh engine...")
    mesh_clients = clients[:10]  # Process first 10 for mesh analysis
    
    for i, client in enumerate(mesh_clients):
        print(f"   Processing {client.client_id} ({i+1}/{len(mesh_clients)})...")
        client = engine.process_with_mesh_engine(
            client, 
            num_scenarios=500,  # Reduced for demo
            time_horizon_years=3
        )
    
    # Analyze congruent financial standing
    congruence_analysis = analyze_congruent_financial_standing(clients)
    
    # Create discretionary spending analysis
    try:
        create_discretionary_spending_analysis(clients)
    except Exception as e:
        print(f"⚠️ Could not create discretionary spending analysis: {e}")
    
    # Analyze event probability patterns
    analyze_event_probability_patterns(clients)
    
    # Create research design simulation
    simulation_results = create_research_design_simulation(clients)
    
    # Export comprehensive results
    try:
        # Export client data
        engine.export_synthetic_data(clients, 'data/outputs/analysis_data/comprehensive_synthetic_clients.json')
        
        # Export analysis results
        analysis_export = {
            'congruence_analysis': congruence_analysis,
            'simulation_results': simulation_results,
            'timestamp': datetime.now().isoformat(),
            'total_clients': len(clients),
            'mesh_processed_clients': len(mesh_clients)
        }
        
        with open('data/outputs/analysis_data/comprehensive_analysis_results.json', 'w') as f:
            json.dump(analysis_export, f, indent=2)
        
        print(f"\n💾 Exported comprehensive results to data/outputs/analysis_data/")
        
    except Exception as e:
        print(f"⚠️ Could not export results: {e}")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics")
    print("=" * 40)
    print(f"Total clients generated: {len(clients)}")
    print(f"Clients processed with mesh: {len(mesh_clients)}")
    print(f"Life stages represented: {len(set(c.vector_profile.life_stage.value for c in clients))}")
    print(f"Total lifestyle events generated: {sum(len(c.lifestyle_events) for c in clients)}")
    
    # Congruence summary
    high_congruence = sum(1 for stage_data in congruence_analysis.values() 
                         if stage_data['congruence_level'] == 'HIGH')
    medium_congruence = sum(1 for stage_data in congruence_analysis.values() 
                           if stage_data['congruence_level'] == 'MEDIUM')
    low_congruence = sum(1 for stage_data in congruence_analysis.values() 
                        if stage_data['congruence_level'] == 'LOW')
    
    print(f"Life stages with HIGH congruence: {high_congruence}")
    print(f"Life stages with MEDIUM congruence: {medium_congruence}")
    print(f"Life stages with LOW congruence: {low_congruence}")
    
    print("\n✅ Comprehensive demo completed successfully!")
    print("\n🎯 Key Insights:")
    print("   - Synthetic client generation with realistic demographics")
    print("   - Age and life stage based event probability modeling")
    print("   - Discretionary spending surface creation and analysis")
    print("   - Financial standing congruence analysis across life stages")
    print("   - Research design framework simulation for systematic analysis")


if __name__ == "__main__":
    main() 