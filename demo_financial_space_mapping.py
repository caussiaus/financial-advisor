#!/usr/bin/env python3
"""
Demo: Financial Space Mapping with Clustering Detection

Showcases how clustering detection around commutators creates a topological map
of the financial state space, visualizing feasible vs infeasible regions.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unified_api import UnifiedFinancialEngine
from src.layers.financial_space_mapper import FinancialSpaceMapper, FinancialState, ClusterType


def demo_financial_space_mapping():
    """Demo the financial space mapping with clustering detection"""
    print("\n" + "="*80)
    print("🗺️ FINANCIAL SPACE MAPPING DEMO")
    print("="*80)
    print("This demo shows how clustering detection around commutators")
    print("creates a topological map of the financial state space.")
    print("="*80)
    
    # Initialize space mapper
    space_mapper = FinancialSpaceMapper()
    
    # Create sample commutators (simplified)
    from src.layers.recommendation_engine import Commutator, FinancialMove, MoveType
    
    sample_commutators = [
        Commutator(
            sequence_id="demo_comm_1",
            move_a=FinancialMove("increase_cash", MoveType.REALLOCATE, to_asset="cash"),
            move_b=FinancialMove("decrease_stocks", MoveType.REALLOCATE, from_asset="stocks"),
            inverse_a=FinancialMove("decrease_cash", MoveType.REALLOCATE, from_asset="cash"),
            inverse_b=FinancialMove("increase_stocks", MoveType.REALLOCATE, to_asset="stocks"),
            description="Cash-Stock Reallocation"
        ),
        Commutator(
            sequence_id="demo_comm_2",
            move_a=FinancialMove("increase_bonds", MoveType.REALLOCATE, to_asset="bonds"),
            move_b=FinancialMove("decrease_real_estate", MoveType.REALLOCATE, from_asset="real_estate"),
            inverse_a=FinancialMove("decrease_bonds", MoveType.REALLOCATE, from_asset="bonds"),
            inverse_b=FinancialMove("increase_real_estate", MoveType.REALLOCATE, to_asset="real_estate"),
            description="Bond-Real Estate Reallocation"
        )
    ]
    
    # Initial financial state
    initial_state = {
        'cash': 100000,
        'stocks': 300000,
        'bonds': 200000,
        'real_estate': 100000
    }
    
    print(f"💰 Initial State: ${sum(initial_state.values()):,.0f}")
    for asset, value in initial_state.items():
        print(f"  {asset}: ${value:,.0f}")
    
    # Generate financial states from commutators
    print(f"\n🔄 Generating financial states from {len(sample_commutators)} commutators...")
    states = space_mapper.generate_financial_states_from_commutators(
        sample_commutators, initial_state, num_samples=200
    )
    
    print(f"✅ Generated {len(states)} financial states")
    
    # Analyze state distribution
    feasibility_scores = [state.feasibility_score for state in states]
    avg_feasibility = sum(feasibility_scores) / len(feasibility_scores)
    print(f"📊 Average feasibility score: {avg_feasibility:.3f}")
    
    # Detect clusters
    print(f"\n🔍 Detecting clusters in financial state space...")
    clusters = space_mapper.detect_clusters(states, algorithm='kmeans')
    
    print(f"✅ Detected {len(clusters)} clusters:")
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id}: {cluster.cluster_type.value}")
        print(f"    States: {len(cluster.states)}")
        print(f"    Avg feasibility: {sum(s.feasibility_score for s in cluster.states) / len(cluster.states):.3f}")
        print(f"    Density: {cluster.density:.3f}")
    
    # Create financial space map
    print(f"\n🗺️ Creating financial space map...")
    space_map = space_mapper.create_financial_space_map(states)
    
    print(f"✅ Created space map:")
    print(f"  Total clusters: {len(space_map.clusters)}")
    print(f"  Feasible regions: {len(space_map.feasible_regions)}")
    print(f"  Infeasible regions: {len(space_map.infeasible_regions)}")
    print(f"  Coverage score: {space_map.coverage_score:.3f}")
    print(f"  Optimal paths: {len(space_map.optimal_paths)}")
    
    # Get cluster analysis
    cluster_analysis = space_mapper.get_cluster_analysis(space_map)
    print(f"\n📋 Cluster Analysis:")
    print(f"  Total clusters: {cluster_analysis['total_clusters']}")
    print(f"  Feasible clusters: {cluster_analysis['feasible_clusters']}")
    print(f"  Infeasible clusters: {cluster_analysis['infeasible_clusters']}")
    print(f"  Coverage score: {cluster_analysis['coverage_score']:.3f}")
    print(f"  Optimal paths: {cluster_analysis['optimal_paths']}")
    
    # Visualize the space map
    print(f"\n📈 Generating visualization...")
    try:
        html = space_mapper.visualize_financial_space(space_map, "financial_space_map.html")
        print(f"✅ Generated visualization: financial_space_map.html")
        print(f"   HTML size: {len(html)} characters")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    return space_map, states


def demo_unified_engine_with_space_mapping():
    """Demo the unified engine with space mapping"""
    print("\n" + "="*80)
    print("🚀 UNIFIED ENGINE WITH SPACE MAPPING DEMO")
    print("="*80)
    
    # Initialize unified engine
    engine = UnifiedFinancialEngine()
    
    # Create sample document
    sample_document = """
    John Smith Financial Profile
    
    Current Assets:
    - Cash: $150,000
    - Stocks: $300,000
    - Bonds: $200,000
    - Real Estate: $100,000
    
    Financial Milestones:
    - Daughter Sarah's college education: $50,000 in 2025
    - Home renovation project: $75,000 in 2024
    - Retirement planning: $200,000 target by 2030
    
    Risk Profile: Moderate
    Time Horizon: 10 years
    """
    
    # Write sample document
    doc_file = "demo_financial_profile.txt"
    with open(doc_file, 'w') as f:
        f.write(sample_document)
    
    try:
        # Run complete analysis with space mapping
        print("🔄 Running complete analysis with space mapping...")
        analysis_result = engine.process_document_and_analyze(
            doc_file, 'moderate', include_space_mapping=True
        )
        
        print(f"\n✅ Analysis complete with space mapping!")
        print(f"📊 Analysis ID: {analysis_result.analysis_id}")
        print(f"📄 Milestones extracted: {len(analysis_result.milestones)}")
        print(f"👥 Entities identified: {len(analysis_result.entities)}")
        print(f"💰 Net worth: ${analysis_result.financial_statement['summary']['net_worth']:,.2f}")
        print(f"🎯 Recommendation confidence: {analysis_result.recommendation.confidence:.1%}")
        
        # Space mapping results
        if analysis_result.financial_space_map:
            space_map = analysis_result.financial_space_map
            print(f"\n🗺️ Space Mapping Results:")
            print(f"  Total clusters: {len(space_map.clusters)}")
            print(f"  Feasible regions: {len(space_map.feasible_regions)}")
            print(f"  Infeasible regions: {len(space_map.infeasible_regions)}")
            print(f"  Coverage score: {space_map.coverage_score:.3f}")
            print(f"  Optimal paths: {len(space_map.optimal_paths)}")
            
            # Get cluster analysis
            cluster_analysis = engine.get_cluster_analysis()
            print(f"\n📋 Detailed Cluster Analysis:")
            for detail in cluster_analysis['cluster_details']:
                print(f"  Cluster {detail['cluster_id']} ({detail['cluster_type']}):")
                print(f"    States: {detail['num_states']}")
                print(f"    Avg feasibility: {detail['avg_feasibility']:.3f}")
                print(f"    Avg volatility: {detail['avg_volatility']:.3f}")
                print(f"    Density: {detail['density']:.3f}")
        
        # Visualize financial space
        print(f"\n📈 Generating financial space visualization...")
        try:
            html = engine.visualize_financial_space(output_file="unified_space_map.html")
            print(f"✅ Generated visualization: unified_space_map.html")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
        
        # Find optimal path example
        print(f"\n🛤️ Finding optimal path example...")
        try:
            from_state = {'cash': 100000, 'stocks': 300000, 'bonds': 200000, 'real_estate': 100000}
            to_state = {'cash': 200000, 'stocks': 200000, 'bonds': 300000, 'real_estate': 100000}
            
            path = engine.find_optimal_path_in_space(from_state, to_state)
            print(f"✅ Found optimal path with {len(path)} intermediate states")
            for i, state in enumerate(path):
                total = sum(state.values())
                print(f"  Step {i+1}: ${total:,.0f} total")
        except Exception as e:
            print(f"⚠️ Path finding failed: {e}")
        
        # Get analysis summary
        summary = engine.get_analysis_summary()
        print(f"\n📋 Complete Analysis Summary:")
        for key, value in summary.items():
            if key == 'space_mapping':
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        return analysis_result
        
    finally:
        # Clean up
        if os.path.exists(doc_file):
            os.remove(doc_file)


def demo_clustering_insights():
    """Demo insights from clustering analysis"""
    print("\n" + "="*80)
    print("🔍 CLUSTERING INSIGHTS DEMO")
    print("="*80)
    
    # Initialize space mapper
    space_mapper = FinancialSpaceMapper()
    
    # Generate diverse financial states
    print("🔄 Generating diverse financial states...")
    states = []
    
    # Conservative states (high cash, low risk)
    for i in range(50):
        state = {
            'cash': 200000 + i * 1000,
            'stocks': 100000 + i * 500,
            'bonds': 300000 + i * 1000,
            'real_estate': 50000 + i * 500
        }
        states.append(space_mapper._create_financial_state(state, f"conservative_{i}"))
    
    # Aggressive states (low cash, high stocks)
    for i in range(50):
        state = {
            'cash': 50000 + i * 500,
            'stocks': 400000 + i * 2000,
            'bonds': 100000 + i * 500,
            'real_estate': 150000 + i * 1000
        }
        states.append(space_mapper._create_financial_state(state, f"aggressive_{i}"))
    
    # Balanced states
    for i in range(50):
        state = {
            'cash': 100000 + i * 1000,
            'stocks': 250000 + i * 1000,
            'bonds': 200000 + i * 1000,
            'real_estate': 100000 + i * 500
        }
        states.append(space_mapper._create_financial_state(state, f"balanced_{i}"))
    
    print(f"✅ Generated {len(states)} diverse financial states")
    
    # Detect clusters
    print(f"\n🔍 Detecting clusters...")
    clusters = space_mapper.detect_clusters(states, algorithm='kmeans')
    
    print(f"✅ Detected {len(clusters)} clusters:")
    
    # Analyze each cluster
    for cluster in clusters:
        print(f"\n📊 Cluster {cluster.cluster_id} ({cluster.cluster_type.value}):")
        
        # Calculate cluster statistics
        feasibility_scores = [s.feasibility_score for s in cluster.states]
        volatilities = [s.risk_metrics['volatility'] for s in cluster.states]
        cash_ratios = [s.risk_metrics['cash_ratio'] for s in cluster.states]
        stock_ratios = [s.risk_metrics['stock_ratio'] for s in cluster.states]
        
        print(f"  States: {len(cluster.states)}")
        print(f"  Avg feasibility: {sum(feasibility_scores) / len(feasibility_scores):.3f}")
        print(f"  Avg volatility: {sum(volatilities) / len(volatilities):.3f}")
        print(f"  Avg cash ratio: {sum(cash_ratios) / len(cash_ratios):.3f}")
        print(f"  Avg stock ratio: {sum(stock_ratios) / len(stock_ratios):.3f}")
        print(f"  Density: {cluster.density:.3f}")
        print(f"  Radius: {cluster.radius:.3f}")
        
        # Identify cluster characteristics
        if cluster.cluster_type == ClusterType.FEASIBLE:
            print(f"  💚 This is a FEASIBLE region - good for operations")
        elif cluster.cluster_type == ClusterType.INFEASIBLE:
            print(f"  ❌ This is an INFEASIBLE region - avoid operations")
        elif cluster.cluster_type == ClusterType.HIGH_RISK:
            print(f"  ⚠️ This is a HIGH RISK region - proceed with caution")
        elif cluster.cluster_type == ClusterType.LOW_RISK:
            print(f"  🛡️ This is a LOW RISK region - safe for operations")
        elif cluster.cluster_type == ClusterType.TRANSITION:
            print(f"  🔄 This is a TRANSITION region - intermediate states")
    
    # Create space map
    space_map = space_mapper.create_financial_space_map(states)
    
    print(f"\n🗺️ Space Map Summary:")
    print(f"  Total clusters: {len(space_map.clusters)}")
    print(f"  Feasible regions: {len(space_map.feasible_regions)}")
    print(f"  Infeasible regions: {len(space_map.infeasible_regions)}")
    print(f"  Coverage score: {space_map.coverage_score:.3f}")
    print(f"  Optimal paths: {len(space_map.optimal_paths)}")
    
    return space_map, clusters


def main():
    """Run all financial space mapping demos"""
    print("🗺️ FINANCIAL SPACE MAPPING WITH CLUSTERING DETECTION")
    print("="*80)
    print("This demo showcases how clustering detection around commutators")
    print("creates a topological map of the financial state space.")
    print("="*80)
    
    # Run individual demos
    demo_financial_space_mapping()
    demo_unified_engine_with_space_mapping()
    demo_clustering_insights()
    
    print("\n" + "="*80)
    print("✅ ALL FINANCIAL SPACE MAPPING DEMOS COMPLETE!")
    print("="*80)
    print("\nKey insights from clustering-based space mapping:")
    print("✓ Clustering detection reveals feasible vs infeasible regions")
    print("✓ Topological mapping shows optimal paths through financial space")
    print("✓ Risk clusters help identify safe vs dangerous operations")
    print("✓ Connectivity graph enables pathfinding between states")
    print("✓ Coverage analysis shows how well the space is explored")
    print("✓ Visualization makes the mesh intuitive to understand")


if __name__ == "__main__":
    main() 