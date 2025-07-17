#!/usr/bin/env python3
"""
Enhanced Mesh System Demo

This demo showcases the state-space mesh approach where:
- Each node stores complete cash flow series (not just snapshots)
- The mesh represents all possible financial evolutions
- Path-dependent analysis is supported through cash flow series comparison
- Similarity is calculated over entire cash flow histories
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_mesh_integration import EnhancedMeshIntegration
from src.unified_cash_flow_model import CashFlowEvent

def create_realistic_cash_flow_scenario():
    """Create a realistic cash flow scenario for demonstration"""
    print("🎯 Creating Realistic Cash Flow Scenario")
    print("=" * 50)
    
    # Initial financial state based on Case_1 analysis
    initial_state = {
        'cash': 764560.97 * 0.0892,  # 8.92% cash allocation
        'investments': 764560.97 * 0.9554,  # 95.54% investments
        'income': 150000,  # Annual income
        'expenses': 60000,  # Annual expenses
        'real_estate': 764560.97 * 0.3,  # 30% real estate
        'mortgage': 764560.97 * 0.4,  # 40% mortgage
        'retirement': 764560.97 * 0.2  # 20% retirement
    }
    
    # Create integration system
    integration = EnhancedMeshIntegration(initial_state)
    
    # Initialize system
    init_result = integration.initialize_system()
    print(f"✅ System initialized with {init_result['enhanced_mesh_nodes']} nodes")
    
    # Create realistic cash flow events over 12 months
    events = [
        # Month 1 - January
        CashFlowEvent(
            event_id="salary_jan",
            description="January salary payment",
            estimated_date="2024-01-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_jan",
            description="January mortgage payment",
            estimated_date="2024-01-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="investment_dividend_jan",
            description="January investment dividend",
            estimated_date="2024-01-30",
            amount=2500,
            source_account="investments",
            target_account="cash",
            event_type="income"
        ),
        
        # Month 2 - February
        CashFlowEvent(
            event_id="salary_feb",
            description="February salary payment",
            estimated_date="2024-02-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_feb",
            description="February mortgage payment",
            estimated_date="2024-02-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="car_insurance_feb",
            description="Annual car insurance payment",
            estimated_date="2024-02-20",
            amount=-1200,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        
        # Month 3 - March
        CashFlowEvent(
            event_id="salary_mar",
            description="March salary payment",
            estimated_date="2024-03-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_mar",
            description="March mortgage payment",
            estimated_date="2024-03-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="bonus_mar",
            description="Q1 performance bonus",
            estimated_date="2024-03-30",
            amount=5000,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        
        # Month 4 - April
        CashFlowEvent(
            event_id="salary_apr",
            description="April salary payment",
            estimated_date="2024-04-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_apr",
            description="April mortgage payment",
            estimated_date="2024-04-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="tax_payment_apr",
            description="Tax payment",
            estimated_date="2024-04-15",
            amount=-8000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        
        # Month 5 - May
        CashFlowEvent(
            event_id="salary_may",
            description="May salary payment",
            estimated_date="2024-05-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_may",
            description="May mortgage payment",
            estimated_date="2024-05-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="investment_growth_may",
            description="Investment portfolio growth",
            estimated_date="2024-05-30",
            amount=3500,
            source_account="investments",
            target_account="cash",
            event_type="income"
        ),
        
        # Month 6 - June
        CashFlowEvent(
            event_id="salary_jun",
            description="June salary payment",
            estimated_date="2024-06-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_jun",
            description="June mortgage payment",
            estimated_date="2024-06-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="vacation_jun",
            description="Summer vacation expenses",
            estimated_date="2024-06-25",
            amount=-4000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        )
    ]
    
    # Add events to system
    print("\n📊 Adding cash flow events to enhanced mesh...")
    for i, event in enumerate(events):
        node_id = integration.add_real_cash_flow_event(event)
        print(f"  Event {i+1:2d}: {event.event_id:20s} -> Node {node_id}")
    
    return integration

def analyze_path_dependencies(integration):
    """Analyze path dependencies in the enhanced mesh"""
    print("\n🔍 Path Dependency Analysis")
    print("=" * 50)
    
    current_node_id = integration.enhanced_mesh.current_position
    path_analysis = integration.analyze_path_dependencies(current_node_id)
    
    if not path_analysis:
        print("❌ No path analysis available")
        return
    
    # Display key metrics
    print(f"📍 Current Node: {current_node_id}")
    print(f"📈 Total Cash Flow Events: {path_analysis['pattern_analysis']['total_events']}")
    print(f"💰 Net Cash Flow: ${path_analysis['pattern_analysis']['net_flow']:,.2f}")
    print(f"📊 Cash Flow Volatility: {path_analysis['pattern_analysis']['volatility']:,.2f}")
    print(f"📈 Cash Flow Trend: {path_analysis['pattern_analysis']['trend']:,.2f}")
    print(f"⚠️  Risk Score: {path_analysis['risk_analysis']['current_risk_score']:.3f}")
    print(f"✅ Utility Score: {path_analysis['risk_analysis']['utility_score']:.3f}")
    
    # Display event distribution
    event_dist = path_analysis['pattern_analysis']['event_distribution']
    if event_dist:
        print(f"\n📊 Event Distribution:")
        print(f"  Positive events: {event_dist['positive_count']}")
        print(f"  Negative events: {event_dist['negative_count']}")
        print(f"  Total positive: ${event_dist['positive_total']:,.2f}")
        print(f"  Total negative: ${event_dist['negative_total']:,.2f}")
        print(f"  Largest inflow: ${event_dist['largest_positive']:,.2f}")
        print(f"  Largest outflow: ${event_dist['largest_negative']:,.2f}")
    
    # Display similar nodes
    similar_nodes = path_analysis['similar_nodes']
    print(f"\n🔗 Similar Nodes (Top 5):")
    for i, (node_id, similarity) in enumerate(similar_nodes[:5]):
        print(f"  {i+1}. Node {node_id}: {similarity:.3f} similarity")
    
    return path_analysis

def generate_visualization_data(integration):
    """Generate data for visualization"""
    print("\n📊 Generating Visualization Data")
    print("=" * 50)
    
    viz_data = integration.generate_path_visualization_data()
    
    if not viz_data:
        print("❌ No visualization data available")
        return
    
    print(f"✅ Generated data for {len(viz_data['path_nodes'])} path nodes")
    
    # Create output directory
    output_dir = Path("data/outputs/enhanced_mesh_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization data
    viz_file = output_dir / "path_visualization_data.json"
    with open(viz_file, 'w') as f:
        json.dump(viz_data, f, indent=2, default=str)
    
    print(f"💾 Saved visualization data to {viz_file}")
    
    # Create summary statistics
    path_nodes = viz_data['path_nodes']
    if path_nodes:
        net_worths = [node['net_worth'] for node in path_nodes]
        risk_scores = [node['risk_score'] for node in path_nodes]
        utility_scores = [node['utility_score'] for node in path_nodes]
        
        summary_stats = {
            'total_nodes': len(path_nodes),
            'net_worth_stats': {
                'min': min(net_worths),
                'max': max(net_worths),
                'mean': np.mean(net_worths),
                'std': np.std(net_worths)
            },
            'risk_stats': {
                'min': min(risk_scores),
                'max': max(risk_scores),
                'mean': np.mean(risk_scores),
                'std': np.std(risk_scores)
            },
            'utility_stats': {
                'min': min(utility_scores),
                'max': max(utility_scores),
                'mean': np.mean(utility_scores),
                'std': np.std(utility_scores)
            }
        }
        
        stats_file = output_dir / "path_summary_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"📈 Summary statistics:")
        print(f"  Net worth range: ${summary_stats['net_worth_stats']['min']:,.0f} - ${summary_stats['net_worth_stats']['max']:,.0f}")
        print(f"  Risk score range: {summary_stats['risk_stats']['min']:.3f} - {summary_stats['risk_stats']['max']:.3f}")
        print(f"  Utility score range: {summary_stats['utility_stats']['min']:.3f} - {summary_stats['utility_stats']['max']:.3f}")
    
    return viz_data

def demonstrate_cash_flow_series_analysis(integration):
    """Demonstrate cash flow series analysis capabilities"""
    print("\n📈 Cash Flow Series Analysis")
    print("=" * 50)
    
    current_node = integration.enhanced_mesh.nodes[integration.enhanced_mesh.current_position]
    cf_series = current_node.cash_flow_series
    
    print(f"📊 Cash Flow Series Analysis for Node {current_node.node_id}:")
    print(f"  Total events: {len(cf_series.amounts)}")
    print(f"  Net flow: ${cf_series.get_net_flow():,.2f}")
    print(f"  Volatility: {cf_series.get_volatility():,.2f}")
    print(f"  Trend: {cf_series.get_trend():,.2f}")
    
    # Show cumulative flow
    cumulative = cf_series.get_cumulative_flow()
    print(f"  Cumulative flow: {cumulative.tolist()}")
    
    # Demonstrate similarity calculation
    print(f"\n🔍 Similarity Analysis:")
    
    # Find similar nodes
    similar_nodes = integration.enhanced_mesh.find_similar_nodes(
        current_node.node_id, method='cash_flow', threshold=0.3
    )
    
    print(f"  Nodes with similar cash flow patterns: {len(similar_nodes)}")
    
    for i, (node_id, similarity) in enumerate(similar_nodes[:3]):
        similar_node = integration.enhanced_mesh.nodes[node_id]
        similar_cf = similar_node.cash_flow_series
        print(f"    {i+1}. Node {node_id}: {similarity:.3f} similarity")
        print(f"       Events: {len(similar_cf.amounts)}, Net: ${similar_cf.get_net_flow():,.2f}")

def export_system_data(integration):
    """Export comprehensive system data"""
    print("\n💾 Exporting System Data")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("data/outputs/enhanced_mesh_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export integration data
    integration_file = output_dir / "enhanced_mesh_integration.json"
    integration.export_integration_data(str(integration_file))
    print(f"✅ Exported integration data to {integration_file}")
    
    # Export mesh data
    mesh_file = output_dir / "enhanced_mesh_export.json"
    integration.enhanced_mesh.export_mesh(str(mesh_file))
    print(f"✅ Exported mesh data to {mesh_file}")
    
    # Get system summary
    summary = integration.get_system_summary()
    summary_file = output_dir / "system_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Exported system summary to {summary_file}")
    
    # Display summary
    print(f"\n📊 System Summary:")
    print(f"  Total nodes: {summary['enhanced_mesh_stats']['total_nodes']}")
    print(f"  Total scenarios: {summary['scenario_count']}")
    print(f"  Total cash flow events: {summary['total_cash_flow_events']}")
    print(f"  Average path length: {summary['average_path_length']:.1f}")
    print(f"  Current node: {summary['current_node']}")

def main():
    """Main demonstration function"""
    print("🚀 Enhanced Mesh System Demonstration")
    print("=" * 60)
    print("This demo showcases the state-space mesh approach with:")
    print("- Full cash flow series tracking at each node")
    print("- Path-dependent analysis using cash flow histories")
    print("- Similarity calculation over entire cash flow series")
    print("- Integration with existing time uncertainty mesh")
    print("=" * 60)
    
    try:
        # Create realistic scenario
        integration = create_realistic_cash_flow_scenario()
        
        # Analyze path dependencies
        path_analysis = analyze_path_dependencies(integration)
        
        # Demonstrate cash flow series analysis
        demonstrate_cash_flow_series_analysis(integration)
        
        # Generate visualization data
        viz_data = generate_visualization_data(integration)
        
        # Export system data
        export_system_data(integration)
        
        print("\n✅ Enhanced Mesh System Demo Completed Successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  ✓ Full cash flow series stored at each node")
        print("  ✓ Path-dependent analysis using complete histories")
        print("  ✓ Similarity calculation over cash flow series")
        print("  ✓ Risk and utility scoring based on cash flow patterns")
        print("  ✓ Integration with existing mesh systems")
        print("  ✓ Export capabilities for further analysis")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 