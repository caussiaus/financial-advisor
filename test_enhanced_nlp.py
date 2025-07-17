#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from enhanced_pdf_processor import EnhancedPDFProcessor

def test_enhanced_nlp():
    print("Testing Enhanced NLP-based milestone and entity extraction")
    
    pdf_path = 'data/inputs/uploads/Case_1_Clean.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Processing: {pdf_path}")
    
    # Initialize processor
    processor = EnhancedPDFProcessor()
    
    # Process PDF
    try:
        milestones, entities = processor.process_pdf(pdf_path)
        
        print(f"\nExtracted {len(milestones)} milestones and {len(entities)} entities:")
        
        if not milestones:
            print("No milestones found!")
        else:
            for i, milestone in enumerate(milestones, 1):
                print(f"  {i}. {milestone.event_type.upper()}")
                print(f"     Description: {milestone.description[:100]}...")
                print(f"     Date: {milestone.timestamp.strftime('%Y-%m-%d')}")
                if milestone.financial_impact:
                    print(f"     Impact: ${milestone.financial_impact:,.2f}")
                else:
                    print(f"     Impact: TBD")
                print(f"     Probability: {milestone.probability:.1%}")
                if milestone.entity:
                    print(f"     Entity: {milestone.entity}")
                print()
        
        if not entities:
            print("No entities found!")
        else:
            for i, entity in enumerate(entities, 1):
                print(f"  {i}. {entity.name} ({entity.entity_type})")
                print(f"     Initial balances: {entity.initial_balances}")
                print()
        
        # Check if results file was created
        results_file = pdf_path + '_results.json'
        if os.path.exists(results_file):
            print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    test_enhanced_nlp() 