#!/usr/bin/env python3
"""
Comprehensive Demo: Vector-Friendly Time Uncertainty Mesh System

This demo showcases the complete time uncertainty mesh system with:
1. Vectorized GBM-based time uncertainty modeling
2. GPU acceleration (Metal/CUDA/CPU)
3. Integration with existing stochastic mesh engine
4. Full data rejuvenation and export capabilities
5. Risk analysis and scenario management
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from src.time_uncertainty_mesh import (
    TimeUncertaintyMeshEngine, 
    SeedEvent, 
    create_sample_events,
    demo_time_uncertainty_mesh
)
from src.time_uncertainty_integration import (
    TimeUncertaintyIntegration,
    demo_integrated_mesh
)

def create_comprehensive_events() -> list:
    """Create comprehensive set of events for full system demo"""
    return [
        # Education Events
        SeedEvent(
            event_id="college_start",
            description="College starts",
            estimated_date="2027-09-01",
            amount=80000,
            timing_volatility=0.2,
            amount_volatility=0.1,
            drift_rate=0.05,
            probability=0.9,
            category="education"
        ),
        SeedEvent(
            event_id="graduate_school",
            description="Graduate school",
            estimated_date="2031-09-01",
            amount=120000,
            timing_volatility=0.3,
            amount_volatility=0.15,
            drift_rate=0.04,
            probability=0.6,
            category="education"
        ),
        
        # Life Events
        SeedEvent(
            event_id="wedding",
            description="Wedding expenses",
            estimated_date="2028-06-15",
            amount=25000,
            timing_volatility=0.3,
            amount_volatility=0.15,
            drift_rate=0.03,
            probability=0.7,
            category="life_event"
        ),
        SeedEvent(
            event_id="honeymoon",
            description="Honeymoon",
            estimated_date="2028-07-01",
            amount=15000,
            timing_volatility=0.2,
            amount_volatility=0.1,
            drift_rate=0.02,
            probability=0.8,
            category="life_event"
        ),
        
        # Housing Events
        SeedEvent(
            event_id="house_down_payment",
            description="House down payment",
            estimated_date="2030-03-01",
            amount=150000,
            timing_volatility=0.5,
            amount_volatility=0.2,
            drift_rate=0.04,
            probability=0.8,
            category="housing"
        ),
        SeedEvent(
            event_id="home_renovation",
            description="Home renovation",
            estimated_date="2031-06-01",
            amount=50000,
            timing_volatility=0.4,
            amount_volatility=0.25,
            drift_rate=0.03,
            probability=0.6,
            category="housing"
        ),
        
        # Family Events
        SeedEvent(
            event_id="first_child",
            description="First child expenses",
            estimated_date="2029-03-01",
            amount=30000,
            timing_volatility=0.4,
            amount_volatility=0.2,
            drift_rate=0.03,
            probability=0.7,
            category="family"
        ),
        SeedEvent(
            event_id="second_child",
            description="Second child expenses",
            estimated_date="2032-06-01",
            amount=25000,
            timing_volatility=0.5,
            amount_volatility=0.15,
            drift_rate=0.02,
            probability=0.5,
            category="family"
        ),
        
        # Career Events
        SeedEvent(
            event_id="career_change",
            description="Career change/transition",
            estimated_date="2033-01-01",
            amount=-50000,  # Income reduction
            timing_volatility=0.6,
            amount_volatility=0.3,
            drift_rate=-0.02,
            probability=0.4,
            category="career"
        ),
        SeedEvent(
            event_id="business_startup",
            description="Business startup investment",
            estimated_date="2034-06-01",
            amount=100000,
            timing_volatility=0.8,
            amount_volatility=0.4,
            drift_rate=0.08,
            probability=0.3,
            category="career"
        ),
        
        # Retirement Events
        SeedEvent(
            event_id="early_retirement",
            description="Early retirement",
            estimated_date="2040-01-01",
            amount=-100000,  # Income reduction
            timing_volatility=1.0,
            amount_volatility=0.2,
            drift_rate=-0.03,
            probability=0.2,
            category="retirement"
        ),
        SeedEvent(
            event_id="retirement",
            description="Standard retirement",
            estimated_date="2045-01-01",
            amount=-150000,  # Income reduction
            timing_volatility=0.8,
            amount_volatility=0.1,
            drift_rate=-0.02,
            probability=0.95,
            category="retirement"
        ),
        
        # Healthcare Events
        SeedEvent(
            event_id="major_medical",
            description="Major medical expense",
            estimated_date="2035-01-01",
            amount=75000,
            timing_volatility=0.7,
            amount_volatility=0.5,
            drift_rate=0.06,
            probability=0.3,
            category="healthcare"
        ),
        
        # Legacy Events
        SeedEvent(
            event_id="inheritance",
            description="Inheritance received",
            estimated_date="2036-01-01",
            amount=200000,
            timing_volatility=0.9,
            amount_volatility=0.3,
            drift_rate=0.01,
            probability=0.4,
            category="legacy"
        )
    ]

def demo_standalone_time_uncertainty():
    """Demo the standalone time uncertainty mesh system"""
    print("\n" + "="*60)
    print("🎯 STANDALONE TIME UNCERTAINTY MESH DEMO")
    print("="*60)
    
    # Create comprehensive events
    events = create_comprehensive_events()
    print(f"📋 Created {len(events)} comprehensive events")
    
    # Initialize engine
    engine = TimeUncertaintyMeshEngine(use_gpu=True)
    
    # Initialize mesh with time uncertainty
    mesh_data, risk_analysis = engine.initialize_mesh_with_time_uncertainty(
        events, 
        num_scenarios=10000,
        time_horizon_years=20
    )
    
    # Get scenario summary
    summary = engine.get_scenario_summary()
    print("\n📊 Scenario Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Export mesh data
    engine.export_mesh_data("comprehensive_mesh_data.json")
    
    return engine, mesh_data, risk_analysis

def demo_integrated_system():
    """Demo the integrated mesh system"""
    print("\n" + "="*60)
    print("🔗 INTEGRATED MESH SYSTEM DEMO")
    print("="*60)
    
    # Create sample financial state
    initial_state = {
        'cash': 1000000.0,
        'investments': 500000.0,
        'debts': 0.0,
        'income': 150000.0,
        'expenses': 0.0
    }
    
    # Create milestones for integration
    milestones = [
        {
            'id': 'college_start',
            'description': 'College starts',
            'estimated_date': '2027-09-01',
            'amount': 80000,
            'timing_uncertainty': 0.2,
            'amount_uncertainty': 0.1,
            'drift_rate': 0.05,
            'probability': 0.9,
            'category': 'education'
        },
        {
            'id': 'house_down_payment',
            'description': 'House down payment',
            'estimated_date': '2030-03-01',
            'amount': 150000,
            'timing_uncertainty': 0.5,
            'amount_uncertainty': 0.2,
            'drift_rate': 0.04,
            'probability': 0.8,
            'category': 'housing'
        },
        {
            'id': 'retirement',
            'description': 'Retirement',
            'estimated_date': '2045-01-01',
            'amount': 0,  # Income reduction handled separately
            'timing_uncertainty': 0.8,
            'amount_uncertainty': 0.1,
            'drift_rate': -0.02,
            'probability': 0.95,
            'category': 'retirement'
        }
    ]
    
    # Initialize integrated mesh
    integration = TimeUncertaintyIntegration(initial_state)
    
    # Initialize integrated mesh
    integrated_data, risk_analysis = integration.initialize_integrated_mesh(
        milestones, 
        num_scenarios=10000,
        time_horizon_years=20
    )
    
    # Get payment options
    payment_options = integration.get_integrated_payment_options()
    print(f"\n💳 Payment options available for {len(payment_options)} milestones")
    
    # Get risk analysis
    risk_summary = integration.get_integrated_risk_analysis()
    print(f"\n📊 Risk analysis complete with comprehensive metrics")
    
    # Get scenario summary
    scenario_summary = integration.get_integrated_scenario_summary()
    print(f"\n📈 Integrated Scenario Summary:")
    for key, value in scenario_summary.get('integrated_summary', {}).items():
        print(f"   {key}: {value}")
    
    # Export integrated mesh
    integration.export_integrated_mesh("comprehensive_integrated_mesh.json")
    
    return integration, integrated_data, risk_analysis

def create_visualizations(engine, mesh_data, risk_analysis):
    """Create comprehensive visualizations of the mesh system"""
    print("\n" + "="*60)
    print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Time Uncertainty Mesh Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cash Position Distribution
    cash_timeline = risk_analysis['cash_timeline']
    ax1 = axes[0, 0]
    ax1.plot(cash_timeline[:, :100], alpha=0.3, linewidth=0.5)  # Plot first 100 scenarios
    ax1.plot(np.mean(cash_timeline, axis=1), 'r-', linewidth=2, label='Mean')
    ax1.plot(np.percentile(cash_timeline, 5, axis=1), 'g--', linewidth=2, label='5th Percentile')
    ax1.plot(np.percentile(cash_timeline, 95, axis=1), 'b--', linewidth=2, label='95th Percentile')
    ax1.set_title('Cash Position Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cash ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Net Worth Distribution
    net_worth_timeline = risk_analysis['net_worth_timeline']
    ax2 = axes[0, 1]
    ax2.plot(net_worth_timeline[:, :100], alpha=0.3, linewidth=0.5)
    ax2.plot(np.mean(net_worth_timeline, axis=1), 'r-', linewidth=2, label='Mean')
    ax2.plot(np.percentile(net_worth_timeline, 5, axis=1), 'g--', linewidth=2, label='5th Percentile')
    ax2.plot(np.percentile(net_worth_timeline, 95, axis=1), 'b--', linewidth=2, label='95th Percentile')
    ax2.set_title('Net Worth Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Net Worth ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Value at Risk Timeline
    var_95 = risk_analysis['var_95_timeline']
    var_99 = risk_analysis['var_99_timeline']
    ax3 = axes[0, 2]
    ax3.plot(var_95, 'r-', linewidth=2, label='95% VaR')
    ax3.plot(var_99, 'b-', linewidth=2, label='99% VaR')
    ax3.fill_between(range(len(var_95)), var_99, var_95, alpha=0.3, color='orange')
    ax3.set_title('Value at Risk Timeline')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('VaR ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scenario Risk Distribution
    min_cash_by_scenario = risk_analysis['min_cash_by_scenario']
    ax4 = axes[1, 0]
    ax4.hist(min_cash_by_scenario, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(min_cash_by_scenario), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.axvline(np.percentile(min_cash_by_scenario, 5), color='orange', linestyle='--', linewidth=2, label='5th Percentile')
    ax4.set_title('Distribution of Minimum Cash by Scenario')
    ax4.set_xlabel('Minimum Cash ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Drawdown Distribution
    max_drawdown_by_scenario = risk_analysis['max_drawdown_by_scenario']
    ax5 = axes[1, 1]
    ax5.hist(max_drawdown_by_scenario, bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(max_drawdown_by_scenario), color='red', linestyle='--', linewidth=2, label='Mean')
    ax5.axvline(np.percentile(max_drawdown_by_scenario, 95), color='orange', linestyle='--', linewidth=2, label='95th Percentile')
    ax5.set_title('Distribution of Maximum Drawdown by Scenario')
    ax5.set_xlabel('Maximum Drawdown ($)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Event Impact Analysis
    if hasattr(engine, 'event_vectors') and engine.event_vectors is not None:
        event_vectors = engine.event_vectors
        ax6 = axes[1, 2]
        
        # Calculate average timing and amount for each event
        avg_timings = np.mean(event_vectors.timings, axis=1)
        avg_amounts = np.mean(event_vectors.amounts, axis=1)
        
        # Create scatter plot
        scatter = ax6.scatter(avg_timings, avg_amounts, 
                             s=[prob * 1000 for prob in np.mean(event_vectors.probabilities, axis=1)],
                             alpha=0.7, c=range(len(event_vectors.event_ids)), cmap='viridis')
        
        # Add labels
        for i, event_id in enumerate(event_vectors.event_ids):
            ax6.annotate(event_id, (avg_timings[i], avg_amounts[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_title('Event Timing vs Amount Analysis')
        ax6.set_xlabel('Average Timing (Unix Timestamp)')
        ax6.set_ylabel('Average Amount ($)')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_mesh_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Visualizations saved to 'comprehensive_mesh_analysis.png'")
    
    return fig

def generate_comprehensive_report(engine, mesh_data, risk_analysis):
    """Generate a comprehensive analysis report"""
    print("\n" + "="*60)
    print("📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print("="*60)
    
    # Get scenario summary
    summary = engine.get_scenario_summary()
    
    # Calculate additional metrics
    cash_timeline = risk_analysis['cash_timeline']
    net_worth_timeline = risk_analysis['net_worth_timeline']
    
    # Calculate time-based metrics
    time_to_negative_cash = []
    for scenario in range(cash_timeline.shape[1]):
        negative_cash_indices = np.where(cash_timeline[:, scenario] < 0)[0]
        if len(negative_cash_indices) > 0:
            time_to_negative_cash.append(negative_cash_indices[0])
        else:
            time_to_negative_cash.append(len(cash_timeline))
    
    # Create comprehensive report
    report = {
        'executive_summary': {
            'total_scenarios': summary['total_scenarios'],
            'worst_case_cash': summary['worst_case_cash'],
            'best_case_cash': summary['best_case_cash'],
            'scenarios_with_negative_cash': summary['scenarios_with_negative_cash'],
            'avg_time_to_negative_cash': np.mean(time_to_negative_cash),
            'worst_scenario_id': summary['worst_scenario_id'],
            'best_scenario_id': summary['best_scenario_id']
        },
        'risk_metrics': {
            'var_95_final': float(risk_analysis['var_95_timeline'][-1]),
            'var_99_final': float(risk_analysis['var_99_timeline'][-1]),
            'max_drawdown_avg': float(np.mean(risk_analysis['max_drawdown_by_scenario'])),
            'max_drawdown_95th': float(np.percentile(risk_analysis['max_drawdown_by_scenario'], 95)),
            'negative_cash_probability': float(np.mean(risk_analysis['negative_cash_probability']))
        },
        'cash_flow_analysis': {
            'final_cash_mean': float(np.mean(cash_timeline[-1, :])),
            'final_cash_std': float(np.std(cash_timeline[-1, :])),
            'cash_volatility': float(np.std(np.diff(cash_timeline, axis=0))),
            'min_cash_percentiles': {
                '5th': float(np.percentile(risk_analysis['min_cash_by_scenario'], 5)),
                '25th': float(np.percentile(risk_analysis['min_cash_by_scenario'], 25)),
                '50th': float(np.percentile(risk_analysis['min_cash_by_scenario'], 50)),
                '75th': float(np.percentile(risk_analysis['min_cash_by_scenario'], 75)),
                '95th': float(np.percentile(risk_analysis['min_cash_by_scenario'], 95))
            }
        },
        'net_worth_analysis': {
            'final_net_worth_mean': float(np.mean(net_worth_timeline[-1, :])),
            'final_net_worth_std': float(np.std(net_worth_timeline[-1, :])),
            'net_worth_growth_rate': float((np.mean(net_worth_timeline[-1, :]) / np.mean(net_worth_timeline[0, :])) ** (1/len(net_worth_timeline)) - 1)
        },
        'scenario_analysis': {
            'total_scenarios': len(engine.scenario_weights),
            'time_horizon_years': len(mesh_data['time_steps']) / 12,
            'num_events': len(engine.event_vectors.event_ids) if engine.event_vectors else 0,
            'gpu_accelerated': engine.use_gpu
        },
        'recommendations': {
            'risk_level': 'High' if summary['scenarios_with_negative_cash'] > summary['total_scenarios'] * 0.1 else 'Medium' if summary['scenarios_with_negative_cash'] > summary['total_scenarios'] * 0.05 else 'Low',
            'cash_reserve_needed': max(0, -summary['worst_case_cash']),
            'suggested_actions': [
                'Increase emergency fund' if summary['worst_case_cash'] < 0 else 'Maintain current cash position',
                'Consider insurance products' if risk_analysis['negative_cash_probability'] > 0.1 else 'Monitor cash flow regularly',
                'Diversify investments' if np.std(net_worth_timeline[-1, :]) > np.mean(net_worth_timeline[-1, :]) * 0.3 else 'Current diversification appears adequate'
            ]
        }
    }
    
    # Save report
    with open('comprehensive_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Comprehensive analysis report saved to 'comprehensive_analysis_report.json'")
    
    # Print executive summary
    print("\n📊 EXECUTIVE SUMMARY:")
    print(f"   Total Scenarios: {report['executive_summary']['total_scenarios']:,}")
    print(f"   Worst Case Cash: ${report['executive_summary']['worst_case_cash']:,.2f}")
    print(f"   Best Case Cash: ${report['executive_summary']['best_case_cash']:,.2f}")
    print(f"   Scenarios with Negative Cash: {report['executive_summary']['scenarios_with_negative_cash']:,}")
    print(f"   Average Time to Negative Cash: {report['executive_summary']['avg_time_to_negative_cash']:.1f} months")
    print(f"   Risk Level: {report['recommendations']['risk_level']}")
    
    return report

def main():
    """Main demo function"""
    print("🚀 COMPREHENSIVE TIME UNCERTAINTY MESH SYSTEM DEMO")
    print("="*80)
    print("This demo showcases the complete vector-friendly time uncertainty mesh system")
    print("with GPU acceleration, full data rejuvenation, and comprehensive analysis.")
    print("="*80)
    
    try:
        # Demo 1: Standalone Time Uncertainty Mesh
        engine, mesh_data, risk_analysis = demo_standalone_time_uncertainty()
        
        # Demo 2: Integrated System
        integration, integrated_data, integrated_risk = demo_integrated_system()
        
        # Create visualizations
        fig = create_visualizations(engine, mesh_data, risk_analysis)
        
        # Generate comprehensive report
        report = generate_comprehensive_report(engine, mesh_data, risk_analysis)
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE DEMO COMPLETE!")
        print("="*80)
        print("📁 Generated Files:")
        print("   - comprehensive_mesh_data.json (Standalone mesh data)")
        print("   - comprehensive_integrated_mesh.json (Integrated mesh data)")
        print("   - comprehensive_mesh_analysis.png (Visualizations)")
        print("   - comprehensive_analysis_report.json (Analysis report)")
        print("\n🎯 Key Features Demonstrated:")
        print("   ✅ Vectorized GBM time uncertainty modeling")
        print("   ✅ GPU acceleration (Metal/CUDA/CPU)")
        print("   ✅ Integration with existing stochastic mesh")
        print("   ✅ Comprehensive risk analysis")
        print("   ✅ Full data export/import capabilities")
        print("   ✅ Scenario management and visualization")
        
        return engine, integration, mesh_data, risk_analysis
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    main() 