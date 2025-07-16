#!/usr/bin/env python
"""
Test Timeline Estimation
Author: ChatGPT 2025-07-16

Tests the timeline estimation functionality for financial events.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from enhanced_chunked_processor import TimelineEstimator, EnhancedChunkedProcessor

def test_timeline_estimator():
    """Test the timeline estimator functionality"""
    print("🧪 Testing Timeline Estimator...")
    
    # Initialize timeline estimator
    estimator = TimelineEstimator()
    
    # Test client profile estimation
    test_text = """
    John Smith is a 42-year-old executive with a high income of $150,000 per year.
    He has been working in the technology sector for 15 years and recently purchased
    a house for $500,000. He has two children and is planning for retirement.
    """
    
    print("\n📊 Testing Client Profile Estimation...")
    profile = estimator.estimate_client_profile(test_text)
    print(f"Estimated Age: {profile['age']}")
    print(f"Life Stage: {profile['life_stage']}")
    print(f"Income Level: {profile['income_level']}")
    print(f"Confidence: {profile['confidence']:.2f}")
    
    # Test event timeline estimation
    print("\n📅 Testing Event Timeline Estimation...")
    test_events = [
        {'type': 'education', 'description': 'MBA Degree', 'amount': 50000, 'confidence': 0.8},
        {'type': 'work', 'description': 'Promotion to Director', 'amount': 20000, 'confidence': 0.9},
        {'type': 'housing', 'description': 'House Purchase', 'amount': 500000, 'confidence': 0.95},
        {'type': 'family', 'description': 'Child Birth', 'amount': 5000, 'confidence': 0.7},
        {'type': 'financial', 'description': 'Investment Portfolio', 'amount': 100000, 'confidence': 0.8}
    ]
    
    timeline_events = estimator.estimate_event_timeline(test_events, profile)
    
    print("\n📈 Timeline Results:")
    for event in timeline_events:
        print(f"  {event['description']}")
        print(f"    Estimated Date: {event['estimated_date']}")
        print(f"    Years Ago: {event['years_ago']}")
        print(f"    Estimated Age: {event['estimated_age']}")
        print(f"    Life Stage: {event['life_stage_when_occurred']}")
        print(f"    Timeline Confidence: {event['timeline_confidence']:.2f}")
        print()
    
    return timeline_events

def test_enhanced_processor():
    """Test the enhanced processor with timeline estimation"""
    print("\n🔧 Testing Enhanced Processor...")
    
    # Initialize processor
    processor = EnhancedChunkedProcessor("TEST_CLIENT")
    
    # Test with sample text
    test_text = """
    Sarah Johnson is a 35-year-old marketing manager earning $85,000 annually.
    She recently completed her MBA at Stanford University, which cost $60,000.
    She is planning to buy a house in the next year and has been saving for a down payment.
    Her investment portfolio is currently valued at $75,000.
    """
    
    # Create a temporary file for testing
    test_file = "test_client_data.txt"
    with open(test_file, 'w') as f:
        f.write(test_text)
    
    try:
        # Process the document
        results = processor.process_document(test_file)
        
        print("\n📊 Processing Results:")
        print(f"Client Profile: {results.get('client_profile', {})}")
        print(f"Total Events: {results['events']['total_events']}")
        
        if results['events']['event_details']:
            print("\n📅 Timeline Events:")
            for event in results['events']['event_details']:
                print(f"  {event['description']}")
                print(f"    Date: {event['estimated_date']}")
                print(f"    Years Ago: {event['years_ago']}")
                print(f"    Age: {event['estimated_age']}")
                print()
        
        return results
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Main test function"""
    print("🚀 Starting Timeline Estimation Tests...")
    
    # Test timeline estimator
    timeline_events = test_timeline_estimator()
    
    # Test enhanced processor
    results = test_enhanced_processor()
    
    print("✅ Timeline estimation tests completed!")
    print(f"Generated {len(timeline_events)} timeline events")
    print(f"Processed {results['events']['total_events']} events with timeline data")

if __name__ == "__main__":
    main() 