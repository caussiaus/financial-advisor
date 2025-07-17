"""
Test script for evaluating the optimized Omega mesh system.
"""
import os
from datetime import datetime, timedelta
from src.stochastic_mesh_engine import StochasticMeshEngine
from src.mesh_memory_manager import MeshMemoryManager
from src.adaptive_mesh_generator import AdaptiveMeshGenerator
from src.vectorized_accounting import VectorizedAccountingEngine
from src.enhanced_pdf_processor import EnhancedPDFProcessor


def test_optimized_mesh():
    """
    Test the optimized mesh system with the provided PDF
    """
    print("🔬 Testing Optimized Omega Mesh System")
    
    # Initialize components
    initial_state = {
        'total_wealth': 2000000.0,
        'cash': 500000.0,
        'investments': 1500000.0,
        'debts': 0.0
    }
    
    mesh_engine = StochasticMeshEngine(initial_state)
    pdf_processor = EnhancedPDFProcessor()
    
    # Process PDF
    pdf_path = os.path.join('data', 'uploads', 'Case_1_IPS_Individual.pdf')
    print(f"\n📄 Processing PDF: {pdf_path}")
    
    milestones = pdf_processor.process_pdf(pdf_path)
    print(f"✅ Extracted {len(milestones)} milestones")
    
    # Print milestone details
    print("\n🎯 Extracted Milestones:")
    for i, milestone in enumerate(milestones, 1):
        print(f"\n{i}. {milestone.event_type.upper()}")
        print(f"   Description: {milestone.description}")
        print(f"   Date: {milestone.timestamp.strftime('%Y-%m-%d')}")
        print(f"   Impact: ${milestone.financial_impact:,.2f}" if milestone.financial_impact else "   Impact: TBD")
        print(f"   Probability: {milestone.probability:.1%}")
    
    # Filter out milestones with no financial impact
    milestones_with_impact = [m for m in milestones if m.financial_impact is not None]
    print(f"\n📈 Found {len(milestones_with_impact)} milestones with financial impact")
    
    # Initialize mesh
    print("\n🌐 Initializing Omega Mesh...")
    start_time = datetime.now()
    mesh_status = mesh_engine.initialize_mesh(milestones_with_impact, time_horizon_years=5)  # Reduced from 10 to 5 years
    end_time = datetime.now()
    
    # Print mesh statistics
    print("\n📊 Mesh Statistics:")
    print(f"Total Nodes: {mesh_status['total_nodes']:,}")
    print(f"Solidified Nodes: {mesh_status['solidified_nodes']:,}")
    print(f"Visible Future Nodes: {mesh_status['visible_future_nodes']:,}")
    print(f"Current Wealth: ${mesh_status['current_wealth']:,.2f}")
    print(f"Initialization Time: {(end_time - start_time).total_seconds():.2f} seconds")
    
    # Test payment execution
    print("\n💰 Testing Payment Execution...")
    
    # Find a milestone with financial impact
    test_milestone = next((m for m in milestones_with_impact if m.financial_impact > 0), None)
    
    if test_milestone:
        payment_amount = test_milestone.financial_impact * 0.01  # 1% payment
        
        success = mesh_engine.execute_payment(
            milestone_id=f"{test_milestone.event_type}_{test_milestone.timestamp.year}",
            amount=payment_amount,
            payment_date=datetime.now()
        )
        
        if success:
            print(f"✅ Successfully executed payment of ${payment_amount:,.2f}")
            
            # Get updated mesh status
            new_status = mesh_engine.get_mesh_status()
            print("\n📈 Updated Mesh Status:")
            print(f"Total Nodes: {new_status['total_nodes']:,}")
            print(f"Solidified Nodes: {new_status['solidified_nodes']:,}")
            print(f"Visible Future Nodes: {new_status['visible_future_nodes']:,}")
            print(f"Current Wealth: ${new_status['current_wealth']:,.2f}")
        else:
            print("❌ Payment execution failed")
    else:
        print("❌ No milestones with financial impact found")
    
    # Memory usage statistics
    print("\n💾 Memory Usage Statistics:")
    print(f"Compressed Nodes: {len(mesh_engine.memory_manager.node_cache.cache):,}")
    
    print("\n✨ Test Complete!")


if __name__ == "__main__":
    test_optimized_mesh() 