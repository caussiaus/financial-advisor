#!/usr/bin/env python3
"""
Demo: Refactored Financial Engine Architecture

Showcases the five-layer modular architecture:
1. PDF Processor Layer - Document processing and milestone extraction
2. Mesh Engine Layer - Stochastic mesh core with GBM paths
3. Accounting Layer - Financial state tracking and reconciliation
4. Recommendation Engine Layer - Commutator-based portfolio optimization
5. UI Layer - Web interface and visualization
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unified_api import UnifiedFinancialEngine, EngineConfig
from src.layers.mesh_engine import MeshConfig


def demo_pdf_processor_layer():
    """Demo the PDF processor layer"""
    print("\n" + "="*60)
    print("📄 PDF PROCESSOR LAYER DEMO")
    print("="*60)
    
    from src.layers.pdf_processor import PDFProcessorLayer
    
    # Initialize PDF processor
    pdf_processor = PDFProcessorLayer()
    
    # Sample document text (simulating PDF extraction)
    sample_text = """
    John Smith has a net worth of $750,000 with the following assets:
    - Cash: $150,000
    - Stocks: $300,000  
    - Bonds: $200,000
    - Real Estate: $100,000
    
    Financial milestones:
    - Education expenses for daughter Sarah: $50,000 in 2025
    - Home renovation: $75,000 in 2024
    - Retirement planning: $200,000 target by 2030
    """
    
    # Create temporary file for demo
    temp_file = "temp_demo_document.txt"
    with open(temp_file, 'w') as f:
        f.write(sample_text)
    
    try:
        # Process document
        milestones, entities = pdf_processor.process_document(temp_file)
        
        print(f"✅ Extracted {len(milestones)} milestones:")
        for milestone in milestones:
            print(f"  - {milestone.description} (${milestone.financial_impact:,.0f})")
        
        print(f"\n✅ Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type})")
            
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def demo_mesh_engine_layer():
    """Demo the mesh engine layer"""
    print("\n" + "="*60)
    print("🌐 MESH ENGINE LAYER DEMO")
    print("="*60)
    
    from src.layers.mesh_engine import MeshEngineLayer, MeshConfig
    
    # Initialize mesh engine
    config = MeshConfig(
        time_horizon_years=5.0,
        num_paths=100,  # Reduced for demo
        use_acceleration=False  # CPU only for demo
    )
    mesh_engine = MeshEngineLayer(config)
    
    # Sample initial state
    initial_state = {
        'cash': 150000,
        'bonds': 200000,
        'stocks': 300000,
        'real_estate': 100000
    }
    
    # Sample milestones
    milestones = [
        {'id': 'education_2025', 'amount': 50000, 'date': '2025-01-01'},
        {'id': 'renovation_2024', 'amount': 75000, 'date': '2024-06-01'}
    ]
    
    # Initialize mesh
    mesh_status = mesh_engine.initialize_mesh(initial_state, milestones)
    
    print(f"✅ Mesh initialized with {mesh_status['total_nodes']} nodes")
    print(f"📊 Current financial state: ${sum(initial_state.values()):,.0f}")
    print(f"🎯 Mesh status: {mesh_status['status']}")
    
    # Run performance benchmark
    benchmarks = mesh_engine.benchmark_performance()
    print(f"⚡ Performance: {benchmarks['path_generation_time']:.3f}s path generation")
    print(f"🔧 Optimization: {benchmarks['optimization_time']:.3f}s mesh optimization")


def demo_accounting_layer():
    """Demo the accounting layer"""
    print("\n" + "="*60)
    print("💰 ACCOUNTING LAYER DEMO")
    print("="*60)
    
    from src.layers.accounting import AccountingLayer, Transaction, TransactionType
    
    # Initialize accounting layer
    accounting = AccountingLayer()
    
    # Create some sample transactions
    transactions = [
        Transaction(
            transaction_id="demo_1",
            timestamp=datetime.now(),
            transaction_type=TransactionType.TRANSFER,
            amount=50000,
            from_account="cash_checking",
            to_account="investments_stocks",
            description="Investment in stocks",
            category="investment"
        ),
        Transaction(
            transaction_id="demo_2",
            timestamp=datetime.now(),
            transaction_type=TransactionType.PAYMENT,
            amount=25000,
            from_account="cash_checking",
            to_account=None,
            description="Education payment",
            category="education"
        )
    ]
    
    # Process transactions
    for transaction in transactions:
        success = accounting.process_transaction(transaction)
        print(f"✅ Transaction {transaction.transaction_id}: {transaction.description}")
    
    # Generate financial statement
    statement = accounting.generate_financial_statement()
    
    print(f"\n📋 Financial Statement Summary:")
    print(f"  Total Assets: ${statement['summary']['total_assets']:,.2f}")
    print(f"  Total Liabilities: ${statement['summary']['total_liabilities']:,.2f}")
    print(f"  Net Worth: ${statement['summary']['net_worth']:,.2f}")
    print(f"  Liquidity Ratio: {statement['summary']['liquidity_ratio']:.2%}")
    print(f"  Stress Level: {statement['summary']['stress_level']:.2%}")


def demo_recommendation_engine_layer():
    """Demo the recommendation engine layer"""
    print("\n" + "="*60)
    print("🎯 RECOMMENDATION ENGINE LAYER DEMO")
    print("="*60)
    
    from src.layers.recommendation_engine import RecommendationEngineLayer
    
    # Initialize recommendation engine
    recommendation_engine = RecommendationEngineLayer()
    
    # Sample current state
    current_state = {
        'cash': 100000,
        'bonds': 150000,
        'stocks': 400000,
        'real_estate': 100000
    }
    
    # Generate recommendation
    recommendation = recommendation_engine.generate_recommendation(
        current_state, risk_preference='moderate'
    )
    
    print(f"✅ Generated recommendation: {recommendation.description}")
    print(f"🎯 Target state: {recommendation.target_state.name}")
    print(f"📊 Confidence: {recommendation.confidence:.1%}")
    print(f"⚠️ Risk score: {recommendation.risk_score:.1%}")
    print(f"⏱️ Expected duration: {recommendation.expected_duration} days")
    
    print(f"\n🔄 Commutator sequence ({len(recommendation.commutator_sequence)} steps):")
    for i, commutator in enumerate(recommendation.commutator_sequence, 1):
        print(f"  Step {i}: {commutator.description}")
    
    # Generate recursive commutators
    recursive_commutators = recommendation_engine.generate_recursive_commutators(depth=2)
    print(f"\n🔄 Generated {len(recursive_commutators)} recursive commutators")


def demo_ui_layer():
    """Demo the UI layer"""
    print("\n" + "="*60)
    print("📈 UI LAYER DEMO")
    print("="*60)
    
    from src.layers.ui import UILayer
    
    # Initialize UI layer
    ui_layer = UILayer()
    
    # Sample dashboard data
    dashboard_data = {
        'allocation': {
            'Cash': 0.15,
            'Bonds': 0.25,
            'Stocks': 0.45,
            'Real Estate': 0.15
        },
        'timeline': [
            {'timestamp': '2024-01-01', 'net_worth': 750000},
            {'timestamp': '2024-06-01', 'net_worth': 780000},
            {'timestamp': '2024-12-01', 'net_worth': 820000}
        ],
        'risk_analysis': {
            'x_labels': ['Cash', 'Bonds', 'Stocks', 'Real Estate'],
            'y_labels': ['Risk', 'Liquidity', 'Return'],
            'values': [
                [0.0, 0.2, 0.6, 0.4],
                [1.0, 0.8, 0.9, 0.3],
                [0.02, 0.04, 0.08, 0.06]
            ]
        },
        'commutators': {
            'steps': ['Step 1', 'Step 2', 'Step 3'],
            'impacts': [0.1, 0.15, 0.08],
            'descriptions': [
                'Increase cash position',
                'Reduce stock allocation',
                'Add bond exposure'
            ]
        }
    }
    
    # Generate dashboard
    dashboard_html = ui_layer.build_dashboard(dashboard_data)
    
    print(f"✅ Generated dashboard HTML ({len(dashboard_html)} characters)")
    
    # Export dashboard
    dashboard_file = "demo_dashboard.html"
    ui_layer.export_dashboard(dashboard_html, dashboard_file)
    print(f"📁 Exported dashboard to {dashboard_file}")
    
    # Generate interactive dashboard
    interactive_html = ui_layer.create_interactive_dashboard(dashboard_data)
    interactive_file = "demo_interactive_dashboard.html"
    ui_layer.export_dashboard(interactive_html, interactive_file)
    print(f"📁 Exported interactive dashboard to {interactive_file}")


def demo_unified_api():
    """Demo the unified API"""
    print("\n" + "="*60)
    print("🚀 UNIFIED API DEMO")
    print("="*60)
    
    # Initialize unified engine
    engine = UnifiedFinancialEngine()
    
    # Create sample document for demo
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
        # Run complete analysis pipeline
        print("🔄 Running complete analysis pipeline...")
        analysis_result = engine.process_document_and_analyze(doc_file, 'moderate')
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Analysis ID: {analysis_result.analysis_id}")
        print(f"📄 Milestones extracted: {len(analysis_result.milestones)}")
        print(f"👥 Entities identified: {len(analysis_result.entities)}")
        print(f"🌐 Mesh status: {analysis_result.mesh_status['status']}")
        print(f"💰 Net worth: ${analysis_result.financial_statement['summary']['net_worth']:,.2f}")
        print(f"🎯 Recommendation confidence: {analysis_result.recommendation.confidence:.1%}")
        
        # Get analysis summary
        summary = engine.get_analysis_summary()
        print(f"\n📋 Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Export analysis
        export_file = "demo_analysis_export.json"
        engine.export_analysis(export_file)
        print(f"\n📁 Exported analysis to {export_file}")
        
        # Run performance benchmarks
        benchmarks = engine.benchmark_performance()
        print(f"\n⚡ Performance Benchmarks:")
        for layer, metrics in benchmarks.items():
            print(f"  {layer}: {metrics}")
            
    finally:
        # Clean up
        if os.path.exists(doc_file):
            os.remove(doc_file)


def main():
    """Run all demos"""
    print("🎯 REFACTORED FINANCIAL ENGINE ARCHITECTURE DEMO")
    print("="*80)
    print("This demo showcases the five-layer modular architecture:")
    print("1. PDF Processor Layer - Document processing and milestone extraction")
    print("2. Mesh Engine Layer - Stochastic mesh core with GBM paths")
    print("3. Accounting Layer - Financial state tracking and reconciliation")
    print("4. Recommendation Engine Layer - Commutator-based portfolio optimization")
    print("5. UI Layer - Web interface and visualization")
    print("="*80)
    
    # Run individual layer demos
    demo_pdf_processor_layer()
    demo_mesh_engine_layer()
    demo_accounting_layer()
    demo_recommendation_engine_layer()
    demo_ui_layer()
    
    # Run unified API demo
    demo_unified_api()
    
    print("\n" + "="*80)
    print("✅ ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nKey improvements in the refactored architecture:")
    print("✓ Modular design with clear separation of concerns")
    print("✓ SOLID principles applied throughout")
    print("✓ Type hints and clean APIs")
    print("✓ Commutator-based recommendation engine")
    print("✓ Performance benchmarks and acceleration")
    print("✓ Interactive dashboard generation")
    print("✓ Unified API for easy integration")


if __name__ == "__main__":
    main() 