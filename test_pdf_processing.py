#!/usr/bin/env python3
"""
Test PDF processing to debug the "no milestones found" issue

from src.omega_mesh_integration import OmegaMeshIntegration

def test_pdf_processing():
    print("Testing PDF processing...")
    
    # Initialize system
    initial_state = {'total_wealth': 200    system = OmegaMeshIntegration(initial_state)
    
    # Test with existing PDF
    pdf_path = 'data/inputs/uploads/Case_1_Clean.pdf'
    print(f"Processing: {pdf_path}")
    
    try:
        milestones, entities = system.process_ips_document(pdf_path)
        print(f"✅ Found {len(milestones)} milestones and {len(entities)} entities")
        
        if milestones:
            print("\nFirst few milestones:)
            for i, m in enumerate(milestones[:3]):
                print(f  {i+1}. {m.event_type}: {m.description[:50]}...")
        else:
            print("❌ No milestones found!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    test_pdf_processing() 