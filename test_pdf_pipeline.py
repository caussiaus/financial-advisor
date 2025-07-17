#!/usr/bin/env python3
"""
Comprehensive test of the PDF processing pipeline end-to-end
"""
import os
import sys
import json
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.append('src')

from src.enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone, FinancialEntity
from src.stochastic_mesh_engine import StochasticMeshEngine
from src.accounting_reconciliation import AccountingReconciliationEngine
from src.omega_mesh_integration import OmegaMeshIntegration

def test_pdf_pipeline():
    """Test the complete PDF processing pipeline"""
    print("🔍 COMPREHENSIVE PDF PIPELINE TEST")
    print("=" * 50)
    
    pdf_path = 'data/inputs/uploads/Case_1_Clean.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Processing: {pdf_path}")
    
    # Step 1: Extract milestones and entities
    print("\n🔧 Step 1: Extracting milestones and entities...")
    processor = EnhancedPDFProcessor()
    
    try:
        milestones, entities = processor.process_pdf(pdf_path)
        print(f"✅ Extracted {len(milestones)} milestones and {len(entities)} entities")
        
        # Analyze entities for reasonableness
        print("\n📊 Entity Analysis:")
        entity_types = Counter([e.entity_type for e in entities])
        print(f"   Entity types: {dict(entity_types)}")
        
        # Show sample entities
        print(f"\n   Sample entities (first 10):")
        for i, entity in enumerate(entities[:10]):
            print(f"     {i+1}. {entity.name} ({entity.entity_type})")
        
        # Check for problematic entities
        problematic_entities = [e for e in entities if len(e.name) < 3 or e.name.isdigit()]
        if problematic_entities:
            print(f"\n   ⚠️  Found {len(problematic_entities)} potentially problematic entities:")
            for entity in problematic_entities[:5]:
                print(f"     - '{entity.name}' ({entity.entity_type})")
        
        # Analyze milestones
        print(f"\n📈 Milestone Analysis:")
        milestone_types = Counter([m.event_type for m in milestones])
        print(f"   Milestone types: {dict(milestone_types)}")
        
        print(f"\n   Sample milestones (first 5):")
        for i, milestone in enumerate(milestones[:5]):
            print(f"     {i+1}. {milestone.event_type.upper()}")
            print(f"        Description: {milestone.description[:100]}...")
            print(f"        Date: {milestone.timestamp.strftime('%Y-%m-%d')}")
            print(f"        Impact: ${milestone.financial_impact:,.2f}" if milestone.financial_impact else "        Impact: TBD")
            print(f"        Probability: {milestone.probability:.1%}")
            print()
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 2: Initialize mesh engine
    print("\n🌐 Step 2: Initializing mesh engine...")
    initial_state = {
        'total_wealth': 2000000.0,
        'cash': 500000.0,
        'investments': 1500000.0,
        'debts': 0.0
    }
    
    mesh_engine = StochasticMeshEngine(initial_state)
    
    try:
        # Initialize mesh with milestones
        mesh_status = mesh_engine.initialize_mesh(milestones)
        print(f"✅ Mesh initialized with status: {mesh_status}")
        
        # Get mesh statistics
        mesh_stats = mesh_engine.get_mesh_status()
        print(f"   Total nodes: {mesh_stats.get('total_nodes', 0)}")
        print(f"   Visible nodes: {mesh_stats.get('visible_nodes', 0)}")
        print(f"   Solidified nodes: {mesh_stats.get('solidified_nodes', 0)}")
        
    except Exception as e:
        print(f"❌ Error initializing mesh: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 3: Initialize accounting engine
    print("\n💰 Step 3: Initializing accounting engine...")
    accounting_engine = AccountingReconciliationEngine()
    
    try:
        # Register entities with accounting
        for entity in entities:
            if entity.entity_type == 'person':
                # Register main person
                accounting_engine.register_entity(entity.name, 'person')
                if entity.initial_balances:
                    for account_type, balance in entity.initial_balances.items():
                        if balance > 0:
                            accounting_engine.set_account_balance(f"{entity.name}_{account_type}", balance)
        
        print(f"✅ Accounting engine initialized with {len(entities)} entities")
        
        # Show accounting summary
        accounts = accounting_engine.get_all_accounts()
        print(f"   Total accounts: {len(accounts)}")
        
    except Exception as e:
        print(f"❌ Error initializing accounting: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 4: Initialize Omega Mesh Integration
    print("\n🔄 Step 4: Initializing Omega Mesh Integration...")
    try:
        omega_system = OmegaMeshIntegration(initial_state)
        
        # Process the PDF through the integration system
        processed_milestones, processed_entities = omega_system.process_ips_document(pdf_path)
        
        print(f"✅ Omega system processed: {len(processed_milestones)} milestones, {len(processed_entities)} entities")
        
        # Get system status
        system_status = omega_system.get_system_status()
        print(f"   System status: {system_status}")
        
    except Exception as e:
        print(f"❌ Error in Omega integration: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 5: Evaluate results
    print("\n📊 Step 5: Evaluating results...")
    
    # Entity quality assessment
    print(f"\n🔍 Entity Quality Assessment:")
    print(f"   Total entities: {len(entities)}")
    print(f"   Unique entity names: {len(set(e.name for e in entities))}")
    print(f"   Average entity name length: {sum(len(e.name) for e in entities) / len(entities):.1f}")
    
    # Check for duplicate or similar entities
    entity_names = [e.name.lower().strip() for e in entities]
    duplicates = [name for name, count in Counter(entity_names).items() if count > 1]
    if duplicates:
        print(f"   ⚠️  Found {len(duplicates)} duplicate entity names")
    
    # Milestone quality assessment
    print(f"\n🎯 Milestone Quality Assessment:")
    print(f"   Total milestones: {len(milestones)}")
    print(f"   Milestones with financial impact: {len([m for m in milestones if m.financial_impact])}")
    print(f"   Average probability: {sum(m.probability for m in milestones) / len(milestones):.2f}")
    
    # Timeline analysis
    if milestones:
        dates = [m.timestamp for m in milestones]
        date_range = max(dates) - min(dates)
        print(f"   Timeline span: {date_range.days} days")
        print(f"   Earliest milestone: {min(dates).strftime('%Y-%m-%d')}")
        print(f"   Latest milestone: {max(dates).strftime('%Y-%m-%d')}")
    
    # Step 6: Recommendations
    print(f"\n💡 Recommendations:")
    
    if len(entities) > 100:
        print(f"   ❌ Too many entities ({len(entities)}) - NLP extraction needs refinement")
        print(f"      - Consider filtering by entity type")
        print(f"      - Add minimum name length requirements")
        print(f"      - Implement entity deduplication")
    
    if len(milestones) < 5:
        print(f"   ⚠️  Few milestones ({len(milestones)}) - may need better extraction")
    
    if len(milestones) > 100:
        print(f"   ⚠️  Many milestones ({len(milestones)}) - may need filtering")
    
    # Check for reasonable financial impacts
    financial_milestones = [m for m in milestones if m.financial_impact]
    if financial_milestones:
        avg_impact = sum(m.financial_impact for m in financial_milestones) / len(financial_milestones)
        print(f"   Average financial impact: ${avg_impact:,.2f}")
        
        if avg_impact > 1000000:
            print(f"   ⚠️  High average impact - may need scaling")
        elif avg_impact < 1000:
            print(f"   ⚠️  Low average impact - may need scaling")
    
    print(f"\n✅ Pipeline test completed!")
    print(f"📁 Results saved to: {pdf_path}_results.json")

if __name__ == '__main__':
    test_pdf_pipeline() 