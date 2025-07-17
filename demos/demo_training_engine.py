#!/usr/bin/env python3
"""
Demo: Mesh Training Engine

This demo shows how the training engine:
1. Generates synthetic people with realistic financial profiles
2. Applies financial shocks to test resilience
3. Runs cash flows through mesh systems
4. Tracks commutator routes and edge paths
5. Learns optimal recovery strategies

Usage:
    python demos/demo_training_engine.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# Use the training controller for clean imports
from src.training.training_controller import (
    get_training_controller,
    run_training_session,
    get_training_status
)


def demo_basic_training():
    """Demo basic training functionality"""
    print("🚀 Mesh Training Engine Demo")
    print("=" * 60)
    
    # Get training controller
    controller = get_training_controller()
    
    print("📊 Generating synthetic training scenarios...")
    
    # Generate a small batch of scenarios for demo
    scenarios = controller.generate_scenarios_only(num_scenarios=10)
    
    print(f"✅ Generated {len(scenarios)} training scenarios")
    
    # Show sample scenario
    sample_scenario = scenarios[0]
    print(f"\n📋 Sample Scenario: {sample_scenario.scenario_id}")
    print(f"  Person: {sample_scenario.person.profile.name} ({sample_scenario.person.profile.age} years old)")
    print(f"  Occupation: {sample_scenario.person.profile.occupation}")
    print(f"  Income: ${sample_scenario.person.profile.base_income:,.0f}")
    print(f"  Net Worth: ${sample_scenario.person.financial_metrics['net_worth']:,.0f}")
    print(f"  Risk Tolerance: {sample_scenario.person.profile.risk_tolerance}")
    
    print(f"\n⚡ Financial Shocks Applied:")
    for i, shock in enumerate(sample_scenario.shocks):
        print(f"  {i+1}. {shock['type'].replace('_', ' ').title()}")
        print(f"     Magnitude: {shock['magnitude']:.1%}")
        print(f"     Timing: {shock['timing'].strftime('%Y-%m-%d')}")
        print(f"     Category: {shock['category']}")
    
    return controller, scenarios


def demo_shock_analysis():
    """Demo shock type analysis"""
    print("\n🔍 Shock Type Analysis")
    print("-" * 40)
    
    controller = get_training_controller()
    
    # Generate scenarios with different age distributions
    age_distributions = {
        'young_professionals': {25: 0.4, 30: 0.4, 35: 0.2},
        'mid_career': {35: 0.3, 40: 0.4, 45: 0.3},
        'established': {45: 0.3, 50: 0.4, 55: 0.3},
        'pre_retirement': {55: 0.3, 60: 0.4, 65: 0.3}
    }
    
    for age_group, distribution in age_distributions.items():
        print(f"\n📊 {age_group.replace('_', ' ').title()}:")
        scenarios = controller.generate_scenarios_only(
            num_scenarios=20, 
            age_distribution=distribution
        )
        
        # Analyze shock patterns
        shock_counts = {}
        for scenario in scenarios:
            for shock in scenario.shocks:
                shock_type = shock['type']
                shock_counts[shock_type] = shock_counts.get(shock_type, 0) + 1
        
        print("  Most common shocks:")
        for shock_type, count in sorted(shock_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {shock_type.replace('_', ' ').title()}: {count} occurrences")


def demo_commutator_learning():
    """Demo commutator route learning"""
    print("\n🧠 Commutator Route Learning")
    print("-" * 40)
    
    # Run a training session using the controller
    print("🔄 Running training session...")
    result = run_training_session(num_scenarios=50)
    
    print(f"\n📈 Training Results:")
    print(f"  Total Scenarios: {result.num_scenarios}")
    print(f"  Successful Recoveries: {result.successful_recoveries}")
    print(f"  Failed Recoveries: {result.failed_recoveries}")
    print(f"  Success Rate: {result.successful_recoveries / result.num_scenarios:.1%}")
    print(f"  Average Recovery Time: {result.average_recovery_time:.1f} days")
    
    print(f"\n🎯 Best Performing Shock Types:")
    for shock_type, success_rate in sorted(result.shock_type_performance.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {shock_type.replace('_', ' ').title()}: {success_rate:.1%}")
    
    print(f"\n🔧 Most Successful Commutator Operations:")
    for op_type, count in result.mesh_optimization_insights.get('most_frequent_operations', [])[:5]:
        print(f"  {op_type.replace('_', ' ').title()}: {count} times")
    
    # Show best routes
    print(f"\n🏆 Top 3 Successful Routes:")
    for i, route in enumerate(result.best_commutator_routes[:3]):
        print(f"  {i+1}. Route {route.route_id}")
        print(f"     Shock: {route.shock_type.replace('_', ' ').title()} ({route.shock_magnitude:.1%})")
        print(f"     Success Score: {route.success_score:.3f}")
        print(f"     Recovery Time: {route.recovery_time} days")
        print(f"     Commutator Steps: {len(route.commutator_sequence)}")
        print(f"     Edge Path: {' -> '.join(route.edge_path[:3])}...")
    
    return result


def demo_route_analysis():
    """Demo analysis of successful commutator routes"""
    print("\n🔬 Commutator Route Analysis")
    print("-" * 40)
    
    controller = get_training_controller()
    
    # Analyze existing training results
    analysis = controller.analyze_training_results()
    
    if analysis['total_routes'] > 0:
        print(f"📊 Analyzing {analysis['total_routes']} routes...")
        
        print(f"\n📊 Route Statistics:")
        print(f"  Total Routes: {analysis['total_routes']}")
        print(f"  Successful Routes: {analysis['successful_routes']}")
        print(f"  Failed Routes: {analysis['failed_routes']}")
        print(f"  Success Rate: {analysis['success_rate']:.1%}")
        
        if 'average_success_score' in analysis:
            print(f"  Average Success Score: {analysis['average_success_score']:.3f}")
            print(f"  Average Recovery Time: {analysis['average_recovery_time']:.1f} days")
            print(f"  Best Success Score: {analysis['best_success_score']:.3f}")
            print(f"  Worst Success Score: {analysis['worst_success_score']:.3f}")
        
        print(f"  Training Sessions: {analysis['training_history']}")
    
    else:
        print("❌ No training results found. Run training session first.")


def demo_system_status():
    """Demo training system status"""
    print("\n🔧 Training System Status")
    print("-" * 40)
    
    status = get_training_status()
    
    print(f"📊 Import Status:")
    print(f"  Import Success: {status['import_success']}")
    if not status['import_success']:
        print(f"  Import Error: {status['import_error']}")
    
    print(f"\n🔧 Component Status:")
    for component, initialized in status['components_initialized'].items():
        status_icon = "✅" if initialized else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")
    
    print(f"\n📈 Training Metrics:")
    metrics = status['training_metrics']
    print(f"  Scenarios Generated: {metrics['scenarios_generated']}")
    print(f"  Successful Routes: {metrics['successful_routes']}")
    print(f"  Failed Routes: {metrics['failed_routes']}")
    print(f"  Training Sessions: {metrics['training_sessions']}")


def demo_visualization():
    """Demo visualization of training results"""
    print("\n📊 Training Results Visualization")
    print("-" * 40)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load training results
        training_results_dir = Path("data/outputs/training")
        routes_file = training_results_dir / "successful_routes.json"
        
        if routes_file.exists():
            with open(routes_file, 'r') as f:
                routes_data = json.load(f)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Success rates by shock type
            shock_success = {}
            for route in routes_data:
                shock_type = route['shock_type']
                if shock_type not in shock_success:
                    shock_success[shock_type] = []
                shock_success[shock_type].append(route['success_score'])
            
            shock_types = list(shock_success.keys())
            avg_scores = [np.mean(shock_success[st]) for st in shock_types]
            
            axes[0, 0].bar(range(len(shock_types)), avg_scores)
            axes[0, 0].set_title('Average Success Score by Shock Type')
            axes[0, 0].set_xticks(range(len(shock_types)))
            axes[0, 0].set_xticklabels([st.replace('_', ' ').title() for st in shock_types], rotation=45)
            axes[0, 0].set_ylabel('Success Score')
            
            # 2. Recovery time distribution
            recovery_times = [route['recovery_time'] for route in routes_data]
            axes[0, 1].hist(recovery_times, bins=20, alpha=0.7)
            axes[0, 1].set_title('Recovery Time Distribution')
            axes[0, 1].set_xlabel('Recovery Time (days)')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. Route length vs success score
            route_lengths = [len(route['commutator_sequence']) for route in routes_data]
            success_scores = [route['success_score'] for route in routes_data]
            
            axes[1, 0].scatter(route_lengths, success_scores, alpha=0.6)
            axes[1, 0].set_title('Route Length vs Success Score')
            axes[1, 0].set_xlabel('Route Length (steps)')
            axes[1, 0].set_ylabel('Success Score')
            
            # 4. Shock magnitude vs recovery time
            shock_magnitudes = [abs(route['shock_magnitude']) for route in routes_data]
            recovery_times = [route['recovery_time'] for route in routes_data]
            
            axes[1, 1].scatter(shock_magnitudes, recovery_times, alpha=0.6)
            axes[1, 1].set_title('Shock Magnitude vs Recovery Time')
            axes[1, 1].set_xlabel('Shock Magnitude')
            axes[1, 1].set_ylabel('Recovery Time (days)')
            
            plt.tight_layout()
            
            # Save plot
            output_dir = Path("data/outputs/training")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "training_analysis.png", dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {output_dir / 'training_analysis.png'}")
            
        else:
            print("❌ No training results found. Run training session first.")
    
    except ImportError:
        print("❌ Matplotlib/Seaborn not available for visualization")


def main():
    """Run all training demos"""
    print("🎓 Mesh Training Engine - Comprehensive Demo")
    print("=" * 80)
    
    # Demo 1: System status
    demo_system_status()
    
    # Demo 2: Basic training functionality
    controller, scenarios = demo_basic_training()
    
    # Demo 3: Shock analysis
    demo_shock_analysis()
    
    # Demo 4: Commutator learning
    result = demo_commutator_learning()
    
    # Demo 5: Route analysis
    demo_route_analysis()
    
    # Demo 6: Visualization
    demo_visualization()
    
    print("\n✅ Training demo completed successfully!")
    print("\n📚 Key Insights:")
    print("  • Synthetic people with realistic financial profiles are generated")
    print("  • Financial shocks test resilience and recovery strategies")
    print("  • Commutator routes track optimal recovery paths")
    print("  • Edge paths show the sequence of financial moves")
    print("  • Training data can be applied to real financial situations")
    
    print(f"\n💾 Results saved to: data/outputs/training/")
    print(f"📊 Visualization: data/outputs/training/training_analysis.png")


if __name__ == "__main__":
    main() 